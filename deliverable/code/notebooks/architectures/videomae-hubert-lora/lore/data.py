"""Data access for the `lore` direction.

Splits, labels, subjects and genders come from the team's ``lie-detector/lib/dataset.py``
(``Protocol.load`` verifies the split hashes; ``Clips.load`` reads ``dataset.csv``) — the same
objects every direction uses, so the four results stay comparable.

Media come from this direction's own inputs:

* video: the face-frame cache (one sprite JPEG per clip, see ``faces.py``);
* audio: ``<audio dir>/<clip_id>.wav`` read directly. ``dataset.csv`` has an empty ``audio_path``
  for 153 usable clips although the file exists, so ``Clip.audio`` cannot be trusted.

The subject protocol (the board) never returns the virgin test set: there is no method for it.
The *official* protocol (opt-in: ``Study(paths, "official")``) is the paper's clip-level 3-fold
split, for a separate reproduction of the paper only (``lie-detector/docs/evaluation.md``). It is
clip-level by construction, so its folds contain clips of the virgin-test subjects too.
"""

from __future__ import annotations

import json
import math
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Paths, use_team_lib
from .faces import read_image, split_sprite

IMAGENET_MEAN = (0.485, 0.456, 0.406)    # VideoMAE preprocessor values (checked on the HF config)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAMPLE_RATE = 16_000
OFFICIAL_VAL_FRACTION = 0.15             # the paper's folds have no val: carved from their train
OFFICIAL_VAL_SEED = 10_000               # fixed (not cfg.seed): every run and branch shares one val


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
class Study:
    """The frozen protocol and the clip table, through the team's lib (never copied)."""

    def __init__(self, paths: Paths, protocol: str = "subject"):
        use_team_lib(paths.lib)
        from lib.dataset import Clips, Protocol

        if protocol not in ("subject", "official"):
            raise ValueError(f"unknown protocol {protocol!r}")
        self.paths = paths
        self.name = protocol
        self.protocol = Protocol.load(paths.data / "splits")      # raises if a split file was edited
        self.clips = Clips.load(paths.data)
        manifest = json.loads((paths.data / "splits" / "manifest.json").read_text(encoding="utf-8"))
        self.split_hashes = manifest.get("files", {})
        if protocol == "subject":
            self.n_folds = self.protocol.folds
        else:
            self.n_folds = sum(1 for name in self.split_hashes if name.startswith("official_fold"))

    def fold(self, k: int, debug: bool = False, limit: int | None = None,
             seed: int = 0) -> tuple[list[str], list[str], list[str]]:
        """``(train, val, holdout)`` of fold ``k``.

        subject protocol: train already includes the two captains. official protocol: ``holdout`` is
        the paper's test fold, ``val`` a fixed stratified 15% of the paper's train (see :meth:`official`).
        """
        if self.name == "official":
            train, val, holdout = self.official(k)
        else:
            train, val, holdout = self.protocol.subject(k)
        absent = self.clips.missing_from(train + val + holdout)
        if absent:
            raise RuntimeError(f"{len(absent)} split clips are not in dataset.csv (first: {absent[:3]}): "
                               "the splits and the data do not match")
        if limit:
            sizes = (limit, limit, limit)
        elif debug:
            sizes = (160, 40, 40)
        else:
            return train, val, holdout
        rng = np.random.default_rng(seed + 1000 * k)
        return tuple(self._stratified(ids, n, rng) for ids, n in zip((train, val, holdout), sizes))

    def official(self, k: int) -> tuple[list[str], list[str], list[str]]:
        """The paper's fold ``k`` as ``(train, val, test)``.

        * the 0-1 clips that the paper lists on both sides stay in ``test`` only (never trained on);
        * the paper has no validation set (its code picks the best epoch on the test fold), so 15% of
          its train is set aside, clip-level and label-stratified like the rest of this split, for early
          stopping and the threshold. The test fold is untouched.
        """
        train, test = self.protocol.official(k)
        in_test = set(test)
        train = [c for c in train if c not in in_test]
        rng = np.random.default_rng(OFFICIAL_VAL_SEED + k)
        val = self._stratified(train, round(len(train) * OFFICIAL_VAL_FRACTION), rng)
        in_val = set(val)
        return [c for c in train if c not in in_val], val, list(test)

    def _stratified(self, ids: list[str], n: int, rng: np.random.Generator) -> list[str]:
        if n >= len(ids):
            return list(ids)
        y = self.labels(ids)
        pos, neg = [c for c, t in zip(ids, y) if t == 1], [c for c, t in zip(ids, y) if t == 0]
        n_pos = min(len(pos), max(1, round(n * len(pos) / len(ids))))
        n_neg = min(len(neg), n - n_pos)
        picked = set(rng.choice(pos, n_pos, replace=False)) | set(rng.choice(neg, n_neg, replace=False))
        return [c for c in ids if c in picked]

    def working_ids(self) -> list[str]:
        """Every clip any fold of the active protocol uses (subject: never the virgin test set)."""
        seen: dict[str, None] = {}
        for k in range(1, self.n_folds + 1):
            parts = self.official(k) if self.name == "official" else self.protocol.subject(k)
            for part in parts:
                seen.update(dict.fromkeys(part))
        return list(seen)

    def labels(self, ids: list[str]) -> np.ndarray:
        return self.clips.labels(ids)

    def subjects(self, ids: list[str]) -> list[str]:
        return [self.clips[c].subject for c in ids]

    def genders(self, ids: list[str]) -> list[str]:
        return [self.clips[c].gender for c in ids]

    def describe(self, ids: list[str]) -> dict:
        return self.clips.describe(ids)

    def identity(self) -> dict:
        """What a result must record to be reproducible (team rule 4)."""
        return {"dataset_csv_sha256_12": self.clips.fingerprint, "split_sha256": self.split_hashes}


def subject_weights(subjects: list[str]) -> np.ndarray:
    """Per-clip weights ``1/sqrt(clips of that subject)``, mean 1: the captains stop dominating."""
    names, counts = np.unique(subjects, return_counts=True)
    count = dict(zip(names, counts))
    weights = np.array([1.0 / math.sqrt(count[s]) for s in subjects], dtype=np.float32)
    return weights / weights.mean()


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
class FaceClips(Dataset):
    """One item = ``(frames (T,3,S,S) float, label, position)`` from the face-frame cache.

    The cache holds ``cached`` frames spread over the whole clip; ``num_frames`` of them are used,
    one every ``stride = cached // num_frames``. Training draws a random offset in ``[0, stride)``
    (temporal jitter) and applies the same random resized crop and flip to every frame;
    evaluation uses a fixed ``offset``.
    """

    def __init__(self, ids: list[str], labels: np.ndarray, faces_dir: Path, num_frames: int = 16,
                 train: bool = False, offset: int = 0, rrc_scale=(0.8, 1.0),
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.faces_dir = Path(faces_dir)
        config = json.loads((self.faces_dir / "cache_config.json").read_text(encoding="utf-8"))
        self.cached, self.size = int(config["frames"]), int(config["size"])
        if self.cached % num_frames:
            raise ValueError(f"cache has {self.cached} frames, not a multiple of num_frames={num_frames}")
        self.num_frames = num_frames
        self.stride = self.cached // num_frames
        if not 0 <= offset < self.stride:
            raise ValueError(f"offset must be in [0, {self.stride})")
        self.offset = offset
        self.train = train
        missing = [c for c in self.ids if not (self.faces_dir / f"{c}.jpg").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips have no sprite in {self.faces_dir} "
                                    f"(first: {missing[:3]}) — build or unzip the face cache first")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.augment = None
        if train:
            from torchvision.transforms import v2
            self.augment = v2.Compose([
                v2.RandomResizedCrop(self.size, scale=tuple(rrc_scale), ratio=(0.9, 1.1), antialias=True),
                v2.RandomHorizontalFlip(),
            ])

    def __len__(self) -> int:
        return len(self.ids)

    def frames(self, index: int) -> np.ndarray:
        """All cached frames of one clip, ``(cached, S, S, 3)`` RGB uint8."""
        sprite = read_image(self.faces_dir / f"{self.ids[index]}.jpg")
        return split_sprite(sprite, self.cached, self.size)[..., ::-1]

    def __getitem__(self, index: int):
        frames = self.frames(index)
        offset = int(torch.randint(self.stride, ()).item()) if self.train else self.offset
        picked = np.ascontiguousarray(frames[offset::self.stride][: self.num_frames])
        x = torch.from_numpy(picked).permute(0, 3, 1, 2).float().div_(255.0)    # (T, 3, S, S)
        if self.augment is not None:
            x = self.augment(x)                    # one set of crop/flip parameters for all frames
        x = (x - self.mean) / self.std
        return x, torch.tensor(self.labels[index]), index


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM wav -> int16 mono array (stdlib only, no audio backend needed)."""
    with wave.open(str(path), "rb") as handle:
        rate, channels, width, n = (handle.getframerate(), handle.getnchannels(),
                                    handle.getsampwidth(), handle.getnframes())
        raw = handle.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples


class AudioStore:
    """Every waveform of the protocol in RAM as int16 (~240 MB), read once with a thread pool.

    Threads make the one-off read fast even from the Drive mount; nothing is read per epoch.
    """

    def __init__(self, audio_dir: Path, ids: list[str], threads: int = 16):
        self.audio_dir = Path(audio_dir)
        missing = [c for c in ids if not (self.audio_dir / f"{c}.wav").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips have no wav in {self.audio_dir} (first: {missing[:3]})")
        with ThreadPoolExecutor(threads) as pool:
            arrays = list(pool.map(lambda c: read_wav(self.audio_dir / f"{c}.wav"), ids))
        self._data = dict(zip(ids, arrays))

    def __getitem__(self, clip_id: str) -> np.ndarray:
        return self._data[clip_id]

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self._data


class AudioClips(Dataset):
    """One item = ``(waveform float, length, label, position)``.

    Training takes a random ``train_seconds`` window, evaluation the first ``eval_seconds``; every
    waveform is normalised to zero mean / unit variance, as the HuBERT checkpoint expects
    (``do_normalize=True`` in its preprocessor config).
    """

    def __init__(self, ids: list[str], labels: np.ndarray, store: AudioStore, train: bool = False,
                 train_seconds: float = 8.0, eval_seconds: float = 10.0):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.store = store
        self.train = train
        self.window = int((train_seconds if train else eval_seconds) * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.ids)

    def lengths(self) -> list[int]:
        return [min(len(self.store[c]), self.window) for c in self.ids]

    def __getitem__(self, index: int):
        samples = self.store[self.ids[index]]
        n = len(samples)
        if n > self.window:
            start = int(torch.randint(n - self.window + 1, ()).item()) if self.train else 0
            samples = samples[start:start + self.window]
        x = samples.astype(np.float32) / 32768.0
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        return torch.from_numpy(x), len(x), torch.tensor(self.labels[index]), index


def audio_collate(batch):
    """Zero-pad to the longest waveform of the batch; lengths drive the masked pooling."""
    waves, lengths, labels, positions = zip(*batch)
    out = torch.zeros(len(waves), max(lengths))
    for row, wav in enumerate(waves):
        out[row, : len(wav)] = wav
    return out, torch.tensor(lengths, dtype=torch.long), torch.stack(labels), torch.tensor(positions)


class BucketBatchSampler(Sampler):
    """Batches of similar length (little padding). Shuffled per epoch when ``shuffle``."""

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0, bucket: int = 50):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket = bucket * batch_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        order = list(range(len(self.lengths)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        batches = []
        for start in range(0, len(order), self.bucket):
            chunk = sorted(order[start:start + self.bucket], key=lambda i: self.lengths[i], reverse=True)
            batches += [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
