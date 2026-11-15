"""Dataset access layer for the lie-detection project.

This module is deliberately architecture-agnostic: it knows where the clips are,
which split a clip belongs to, and what its label is. It does **not** know how a
clip becomes a tensor — that is the modelling choice each direction owns.

It reads only the artifacts produced by the two scripts:

    data/processed/dataset.csv      <- scripts/build_dataset.py
    data/processed/splits/*.json    <- scripts/make_splits.py

Typical use in a notebook::

    import sys; sys.path.insert(0, "/content/lie-detector")
    from lib.dataset import Protocol, Clips, CachedFeatures, make_loader

    folds = Protocol.load()                      # verifies the manifest hashes
    clips = Clips.load()                         # only the clips actually on disk

    train_ids, val_ids, hold_ids = folds.subject(1)
    print(clips.describe(train_ids))

    # 1) raw media: your notebook decodes the video/audio itself
    ds = clips.paths(train_ids)

    # 2) cached features: whatever your preprocessing saved as <clip_id>.npy
    ds = CachedFeatures(train_ids, clips, cache="data/processed/audio")
    dl = make_loader(ds, batch_size=16, shuffle=True, seed=0)
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:  # torch is optional for the inspection helpers
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    DataLoader = None

ROOT = Path(__file__).resolve().parents[1]      # lie-detector/


CACHE_ROOT = "cached_features"


def cache_dir(encoder: str, data_dir: Path | str | None = None) -> Path:
    """Where the features produced by ``encoder`` live.

    One folder per encoding model, shared by everybody who uses it:

        <data>/cached_features/wav2vec2-base/<clip_id>.npy
        <data>/cached_features/vit-b16/<clip_id>.npy

    Name it after the model, not after your direction: two people using the same
    encoder should end up in the same folder for free.
    """
    base = Path(data_dir) if data_dir is not None else ROOT / "data" / "processed"
    return base / CACHE_ROOT / encoder


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("this helper needs torch: pip install torch")


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Clip:
    clip_id: str
    y: int
    label: str
    gender: str
    subject: str
    video: Path
    audio: Path | None
    has_media: bool = False


class Clips:
    """The clips that are actually available on disk."""

    def __init__(self, clips: dict[str, Clip], missing: list[str],
                 frame: pd.DataFrame | None = None, mumin_columns: list[str] | None = None,
                 fingerprint: str = ""):
        self._clips = clips
        self.missing = missing
        self._frame = frame
        self._mumin_columns = mumin_columns or []
        self.fingerprint = fingerprint

    def describe_source(self) -> str:
        """One line identifying the exact table this loader is reading.

        Post it in the group chat: two members with different fingerprints are not
        training on the same dataset, and their numbers are not comparable.
        """
        return (f"dataset.csv sha256:{self.fingerprint}  "
                f"{len(self._clips)} clips  {len(self.available())} with media")

    # -- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, data_dir: Path | str | None = None, require_audio: bool = False) -> "Clips":
        """Read ``dataset.csv`` from the dataset folder.

        ``data_dir`` defaults to ``data/processed``. Point it at the shared folder
        (``/content/drive/MyDrive/dolos``) or at a local copy: it must contain
        ``dataset.csv``, ``clips/`` and ``audio/``.
        """
        data_dir = Path(data_dir) if data_dir is not None else ROOT / "data" / "processed"
        table = pd.read_csv(data_dir / "dataset.csv", dtype=str)
        mumin_columns = [c for c in table.columns if c.startswith("mumin_")]
        clips: dict[str, Clip] = {}
        missing: list[str] = []

        for row in table.itertuples(index=False):
            clip_id = str(row.clip_id)
            video = Path(row.video_path) if isinstance(row.video_path, str) and row.video_path else None
            audio = Path(row.audio_path) if isinstance(row.audio_path, str) and row.audio_path else None
            video_path = data_dir / video if video is not None else Path("")
            audio_path = data_dir / audio if audio is not None else None
            has_media = bool(video) and video_path.exists()
            if audio_path is not None and not audio_path.exists():
                audio_path = None
            if require_audio and (not has_media or audio_path is None):
                has_media = False
                missing.append(clip_id)
            clips[clip_id] = Clip(
                clip_id=clip_id,
                y=int(row.y),
                label=str(row.label),
                gender=str(row.gender),
                subject=str(row.subject),
                video=video_path,
                audio=audio_path,
                has_media=has_media,
            )
        fingerprint = hashlib.sha256((data_dir / "dataset.csv").read_bytes()).hexdigest()[:12]
        return cls(clips, missing, table, mumin_columns, fingerprint)

    # -- access ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, clip_id: str) -> Clip:
        return self._clips[clip_id]

    def __contains__(self, clip_id: str) -> bool:
        return clip_id in self._clips

    def ids(self) -> list[str]:
        return list(self._clips)

    def keep(self, clip_ids: list[str]) -> list[str]:
        """Intersect a split with the annotated clips, preserving split order.

        Use this for cache-based pipelines: the features exist even if the
        original ``.mp4`` was deleted.
        """
        return [c for c in clip_ids if c in self._clips]

    def with_media(self, clip_ids: list[str]) -> list[str]:
        """Intersect a split with the clips whose video is actually on disk."""
        return [c for c in clip_ids if c in self._clips and self._clips[c].has_media]

    def missing_from(self, clip_ids: list[str]) -> list[str]:
        """Split clips that are not in ``dataset.csv``.

        Anything here is a protocol/dataset mismatch: the splits were built on a
        different table, and the affected clips would silently disappear from
        training and evaluation.
        """
        known = set(self._clips)
        return [c for c in clip_ids if c not in known]

    def available(self) -> list[str]:
        return [c for c, clip in self._clips.items() if clip.has_media]

    def labels(self, clip_ids: list[str]) -> np.ndarray:
        return np.array([self._clips[c].y for c in clip_ids], dtype=np.int64)

    def class_weights(self, clip_ids: list[str]):
        """``pos_weight`` for ``BCEWithLogitsLoss`` on deception as the positive class."""
        _require_torch()
        y = self.labels(clip_ids)
        positive = int(y.sum())
        negative = len(y) - positive
        return torch.tensor(negative / max(positive, 1), dtype=torch.float32)

    def majority_accuracy(self, clip_ids: list[str]) -> float:
        """Accuracy of always predicting the majority class *of this subset*.

        On a subject-level fold this is ~48%, not the 53.7% of the whole dataset:
        the baseline has to be computed on the same clips you evaluate on.
        """
        y = self.labels(clip_ids)
        if len(y) == 0:
            return float("nan")
        return max(y.mean(), 1 - y.mean())

    def mumin(self, clip_ids: list[str]) -> np.ndarray | None:
        """The 41 MUMIN features of these clips, shape ``(n, 41)``, or ``None``.

        Hand annotations, not something a model can compute on a new video: use
        them as auxiliary multi-task targets, not as an input.
        """
        if not self._mumin_columns or self._frame is None:
            return None
        index = self._frame.set_index("clip_id")
        rows = [c for c in clip_ids if c in index.index]
        if not rows:
            return np.zeros((0, len(self._mumin_columns)), dtype=np.float32)
        block = index.loc[rows, self._mumin_columns].apply(pd.to_numeric, errors="coerce")
        return block.fillna(0).to_numpy(dtype=np.float32)

    def describe(self, clip_ids: list[str]) -> dict:
        y = self.labels(clip_ids)
        subjects = {self._clips[c].subject for c in clip_ids}
        return {
            "clips": len(y),
            "subjects": len(subjects),
            "deception": round(float(y.mean()) * 100, 1) if len(y) else 0.0,
            "pos_weight": round(float(len(y) - y.sum()) / max(float(y.sum()), 1), 3) if len(y) else 0.0,
            "female": round(
                float(np.mean([self._clips[c].gender == "female" for c in clip_ids])) * 100, 1
            ) if clip_ids else 0.0,
            "majority_baseline": round(float(self.majority_accuracy(clip_ids)) * 100, 1),
        }

    # -- datasets ---------------------------------------------------------- #
    def paths(self, clip_ids: list[str]) -> "ClipPaths":
        """Raw media: returns paths, the caller decides how to decode them."""
        return ClipPaths(self, clip_ids)


class ClipPaths(Dataset):
    """One item = one clip, as paths plus metadata. No tensors, no transforms."""

    def __init__(self, clips: Clips, clip_ids: list[str]):
        self.clips = clips
        self.clip_ids = clips.with_media(clip_ids)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> dict:
        clip = self.clips[self.clip_ids[index]]
        return {
            "clip_id": clip.clip_id,
            "y": clip.y,
            "video": str(clip.video),
            "audio": str(clip.audio) if clip.audio else "",
            "subject": clip.subject,
            "gender": clip.gender,
        }


class CachedFeatures(Dataset):
    """One item = the array your own preprocessing saved for that clip.

    ``cache`` holds ``<clip_id>.npy`` (or ``<clip_id>.npz`` with ``key``). The
    array is returned as a float32 tensor with no interpretation attached —
    whatever your encoder produced, this loader hands it back.
    """

    def __init__(self, clip_ids: list[str], clips: Clips, cache: Path | str,
                 key: str | None = None, dtype=None, preload: bool | None = None):
        _require_torch()
        self.clips = clips
        self.cache = Path(cache)
        self.key = key
        self.dtype = dtype or torch.float32
        if not self.cache.is_dir():
            raise FileNotFoundError(f"cache directory not found: {self.cache}")

        suffix = ".npz" if key else ".npy"
        wanted = clips.keep(clip_ids)
        self.clip_ids = [c for c in wanted if (self.cache / f"{c}{suffix}").exists()]
        absent = [c for c in wanted if c not in set(self.clip_ids)]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)}/{len(wanted)} clips have no cache entry in {self.cache} "
                f"(first: {absent[:3]}). Run your preprocessing on the whole split."
            )

        # Reading one file per sample over a Drive mount costs a network round-trip,
        # so: preload when the cache is small enough to sit in RAM.
        if preload is None:
            preload = sum(p.stat().st_size for p in self.cache.iterdir()) < 1 << 30
        self._memory: dict[str, "torch.Tensor"] = {}
        if preload:
            for clip_id in self.clip_ids:
                self._memory[clip_id] = self._read(clip_id)

    def _read(self, clip_id: str) -> "torch.Tensor":
        if self.key:
            with np.load(self.cache / f"{clip_id}.npz") as data:
                array = data[self.key]
        else:
            array = np.load(self.cache / f"{clip_id}.npy")
        return torch.as_tensor(np.asarray(array), dtype=self.dtype)

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> tuple:
        clip_id = self.clip_ids[index]
        x = self._memory[clip_id] if clip_id in self._memory else self._read(clip_id)
        y = torch.tensor(self.clips[clip_id].y, dtype=torch.long)
        return x, y


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
class Protocol:
    """The frozen splits, with the manifest hash checked on load."""

    def __init__(self, splits: dict[str, dict], directory: Path, verified: bool):
        self._splits = splits
        self.directory = directory
        self.verified = verified

    @classmethod
    def load(cls, splits_dir: Path | str | None = None, verify: bool = True) -> "Protocol":
        """Read the frozen protocol.

        ``splits_dir`` defaults to ``data/processed/splits`` locally; in Colab pass
        the shared folder: ``Protocol.load(DATA / "splits")``.
        """
        splits_dir = Path(splits_dir) if splits_dir is not None else ROOT / "data" / "processed" / "splits"
        directory = Path(splits_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — run `python scripts/make_splits.py` first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        splits: dict[str, dict] = {}
        for name, digest in manifest.get("files", {}).items():
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(f"split file missing: {path}")
            if verify:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != digest:
                    raise RuntimeError(
                        f"{name} does not match the manifest hash. The split was regenerated "
                        f"or edited by hand; regenerating it invalidates earlier results."
                    )
            splits[name] = json.loads(path.read_text(encoding="utf-8"))

        if not verify:
            warnings.warn("split hashes not verified", stacklevel=2)
        return cls(splits, directory, verify)

    @property
    def folds(self) -> int:
        return sum(1 for name in self._splits if name.startswith("subject_fold"))

    def test(self) -> list[str]:
        """The virgin test set. Open it once, at the very end."""
        return self._splits["subject_test.json"]["clips"]

    def subject(self, fold: int) -> tuple[list[str], list[str], list[str]]:
        """``(train, val, holdout)`` for one cross-validation fold."""
        payload = self._splits[f"subject_fold{fold}.json"]
        train = payload["train"]["clips"] + payload["always_train"]["clips"]
        return train, payload["val"]["clips"], payload["holdout"]["clips"]

    def official(self, fold: int) -> tuple[list[str], list[str]]:
        """``(train, test)`` of the paper's clip-level split."""
        payload = self._splits[f"official_fold{fold}.json"]
        return payload["train"]["clips"], payload["test"]["clips"]


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def path_collate(batch: list[dict]) -> dict:
    """Collate for :class:`ClipPaths`: strings stay lists, labels become a tensor."""
    _require_torch()
    return {
        "clip_id": [item["clip_id"] for item in batch],
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "video": [item["video"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "subject": [item["subject"] for item in batch],
        "gender": [item["gender"] for item in batch],
    }


def make_loader(dataset, batch_size: int = 8, shuffle: bool = False,
                num_workers: int = 0, seed: int | None = None):
    """``DataLoader`` with the right collate function and a reproducible shuffle."""
    _require_torch()
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    collate = path_collate if isinstance(dataset, ClipPaths) else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate, generator=generator)


def fold_loaders(folds: Protocol, clips: Clips, fold: int, cache: Path | str | None = None,
                 batch_size: int = 8, num_workers: int = 0, seed: int = 0) -> dict:
    """Three loaders for one fold, from raw paths or from a feature cache."""
    train_ids, val_ids, holdout_ids = folds.subject(fold)
    loaders = {}
    for name, ids, shuffle in (("train", train_ids, True), ("val", val_ids, False),
                               ("holdout", holdout_ids, False)):
        dataset = (CachedFeatures(ids, clips, cache) if cache else clips.paths(ids))
        loaders[name] = make_loader(dataset, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers, seed=seed + fold)
    return loaders


# --------------------------------------------------------------------------- #
# cli: quick inspection
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the frozen splits against the local clips.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args(argv)

    folds = Protocol.load(args.data_dir / "splits")
    clips = Clips.load(args.data_dir)
    print(f"clips: {len(clips)} annotated, {len(clips.available())} with media on disk")
    print(f"folds available: {folds.folds}\n")

    rows = [("test", folds.test())]
    for i in range(1, folds.folds + 1):
        train, val, holdout = folds.subject(i)
        rows += [(f"fold{i} train", train), (f"fold{i} val", val), (f"fold{i} holdout", holdout)]
    for name, ids in rows:
        stats = clips.describe(clips.keep(ids))
        on_disk = len(clips.with_media(ids))
        majority = stats["majority_baseline"]
        print(f"{name:16} {on_disk:4d}/{len(ids):<5d} clip on disk  {stats['subjects']:3d} sogg  "
              f"deception {stats['deception']:5.1f}%  majority "
              f"{majority:5.1f}%" if majority == majority else
              f"{name:16} {on_disk:4d}/{len(ids):<5d} clip on disk  {stats['subjects']:3d} sogg  "
              f"deception {stats['deception']:5.1f}%  majority   n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
