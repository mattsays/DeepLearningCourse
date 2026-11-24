"""The `tia` recipe in one place, so a whole campaign is one call.

Until now the loop lived inside [`train.ipynb`](train.ipynb): one configuration per kernel,
because switching `SPLIT` rebuilds the feature store and switching `VARIANT` rebuilds the net.
Running the six board configurations that way is six restarts and six manual exports, and every
one of them can disagree with the others by a stale global.

Here the same loop takes the split and the variant as arguments. [`campaign.ipynb`](campaign.ipynb)
calls :func:`run_campaign` once and gets every `results/<run>/` directory; `train.ipynb` keeps its
interactive cells for trying one thing, and the recipe itself is this module.

[`test.ipynb`](test.ipynb) is the one thing this module does that the campaign must not:
:func:`run_test` trains the same folds with ``split="test"`` and scores the closed test set with
them, once, at the end.

Two things this file does *not* decide: what a run is worth (that is `results/README.md` and the
direction README) and what the split contains (`lib.dataset.Protocol`).
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import export
import model
from lib.dataset import cache_dir, make_loader
from lib.metrics import best_threshold, evaluate


@dataclass
class Encoders:
    """Which cache each stream reads; the widths are measured from the arrays, not assumed."""

    video: str = "dinov2-small"
    audio: str = "wavlm-base-plus"


@dataclass
class Settings:
    """Everything a run needs that is not data: the training recipe and the protocol."""

    data: Path
    encoders: Encoders = field(default_factory=Encoders)
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    folds: int = 5
    epochs: int = 30
    patience: int = 8
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-2
    read_workers: int = 64
    protocol: str = "v1"                 # "v1" board protocol, "v2" purity-balanced ablation
    captains: str = "include"            # "exclude": the 468 clips of the two recurring captains
    device: torch.device | None = None

    def splits(self) -> Path:
        return Path(self.data) / ("splits_v2" if self.protocol == "v2" else "splits")

    def torch_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ---------------------------------------------------------------- the feature store


class Features:
    """Both caches in RAM, keyed by clip id, read once — in parallel.

    ``CachedFeatures`` stats every clip it is handed and reads whatever split it is given, so
    building one per split (and two per fold) re-reads a 330 MB cache dozens of times over a
    latency-bound Drive mount. One pass here, then every loader is a dict lookup.
    """

    def __init__(self, ids, settings: Settings):
        self.video, missing_video = self._read_all(cache_dir(settings.encoders.video, settings.data),
                                                   ids, "video", settings.read_workers)
        self.audio, missing_audio = self._read_all(cache_dir(settings.encoders.audio, settings.data),
                                                   ids, "audio", settings.read_workers)
        self.clip_ids = [c for c in ids if c in self.video and c in self.audio]
        self.index = {c: i for i, c in enumerate(self.clip_ids)}
        print(f"  video {len(self.video)} read, {missing_video} missing | "
              f"audio {len(self.audio)} read, {missing_audio} missing")
        print(f"  usable: {len(self.clip_ids)}/{len(ids)} clips")

    @staticmethod
    def _read(cache: Path, clip_id: str) -> torch.Tensor:
        return torch.from_numpy(np.load(cache / f"{clip_id}.npy")).float()

    def _read_all(self, cache: Path, ids, label: str, workers: int):
        def load(clip_id):
            try:
                return clip_id, self._read(cache, clip_id)
            except OSError:
                return clip_id, None

        out, missing = {}, 0
        with ThreadPoolExecutor(workers) as pool:
            for clip_id, tensor in pool.map(load, ids):
                if tensor is None:
                    missing += 1
                else:
                    out[clip_id] = tensor
        return out, missing

    def subset(self, ids):
        return [c for c in ids if c in self.index]

    def widths(self) -> dict[str, int]:
        sample = self.clip_ids[0]
        return {"video": self.video[sample].shape[-1],
                "audio": self.audio[sample].shape[-1]}


class Paired(Dataset):
    """``(video, audio) -> label`` for the clips the store holds."""

    def __init__(self, features: Features, ids, clips):
        self.features = features
        self.clips = clips
        self.clip_ids = list(ids)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, index):
        clip_id = self.clip_ids[index]
        return ((self.features.video[clip_id], self.features.audio[clip_id]),
                torch.tensor(self.clips[clip_id].y, dtype=torch.long))


# ---------------------------------------------------------------- one split, ready to train


class Data:
    """A split with its features in RAM: the fold parts and the loaders for one protocol."""

    def __init__(self, clips, folds, settings: Settings, split: str):
        self.clips, self.folds, self.settings, self.split = clips, folds, settings, split
        self.n_folds = 3 if split == "official" else folds.folds
        ids = self.working_ids()
        print(f"[{split}] {len(ids)} clips in the protocol, reading both caches once")
        self.features = Features(clips.keep(ids), settings)
        self.widths = self.features.widths()
        model.configure(video_dim=self.widths["video"], audio_dim=self.widths["audio"])

    def working_ids(self) -> list[str]:
        """Every clip the protocol touches, once, in a stable order.

        In official mode this is the union of the paper's train/test pairs, which covers the
        whole dataset; the subject folds leave the virgin test set out. The ``test`` split is
        the subject folds plus that set: it trains on the same clips and scores the closed
        one, which is why it needs its features in the same store.
        """
        ids = []
        for fold in range(1, self.n_folds + 1):
            parts = (self.folds.official(fold) if self.split == "official"
                     else self.folds.subject(fold))
            for part in parts:
                ids += part
        if self.split == "test":
            ids += self.folds.test()
        return list(dict.fromkeys(ids))

    def fold_parts(self, fold: int):
        """``(train, val, holdout)``; the official split has no validation set."""
        if self.split == "official":
            train_ids, test_ids = self.folds.official(fold)
            return train_ids, None, test_ids
        train_ids, val_ids, hold_ids = self.folds.subject(fold)
        if self.settings.captains == "exclude":
            pinned = set(self.folds._splits[f"subject_fold{fold}.json"]["always_train"]["clips"])
            train_ids = [c for c in train_ids if c not in pinned]
        return train_ids, val_ids, hold_ids

    def loader(self, ids, shuffle: bool, seed: int = 0):
        dataset = Paired(self.features, self.features.subset(ids), self.clips)
        return make_loader(dataset, batch_size=self.settings.batch_size, shuffle=shuffle,
                           seed=seed)


# ---------------------------------------------------------------- the loop


def predict(net, data: Data, ids=None, loader=None) -> tuple[list[str], list[int], list[float]]:
    """``(clip ids, labels, probabilities)`` for one fitted model over one set of clips.

    Pass ``loader`` when the caller already built it (the validation loop rebuilds nothing);
    otherwise the ids are wrapped in a fresh, unshuffled loader.
    """
    device = data.settings.torch_device()
    dl = loader if loader is not None else data.loader(ids, False)
    probs, targets = [], []
    net.eval()
    with torch.no_grad():
        for (video, audio), y in dl:
            probs += torch.sigmoid(net(video.to(device), audio.to(device))).tolist()
            targets += y.tolist()
    return dl.dataset.clip_ids, targets, probs


def train_one_fold(data: Data, variant: str, fold: int, seed: int, verbose: bool = True,
                   return_net: bool = False) -> dict:
    """Train one fold and return its metrics plus the holdout predictions.

    Early stopping on validation AUC, ``pos_weight`` to compensate the class prior, and the
    decision threshold chosen on ``val`` — never on the holdout. The official split has no
    validation set, so it runs the fixed epoch budget at threshold 0.5, as the paper does.
    ``return_net`` hands the fitted net back: that is what scores the test set in
    :func:`run_test`.
    """
    settings, device = data.settings, data.settings.torch_device()
    train_ids, val_ids, hold_ids = data.fold_parts(fold)
    dl_train = data.loader(train_ids, True, seed=seed)
    dl_val = data.loader(val_ids, False) if val_ids is not None else None
    dl_hold = data.loader(hold_ids, False)
    if len(dl_train.dataset) == 0 or len(dl_hold.dataset) == 0:
        raise RuntimeError(f"fold {fold} of {data.split} has an empty split: "
                           f"{len(dl_train.dataset)} train, {len(dl_hold.dataset)} holdout")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = model.VARIANTS[variant]().to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=settings.lr,
                                  weight_decay=settings.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=data.clips.class_weights(train_ids)).to(device)

    best_score, best_state, best_thr, waited = -1.0, None, 0.5, 0
    started = time.perf_counter()
    epoch = -1
    for epoch in range(settings.epochs):
        net.train()
        for (video, audio), y in dl_train:
            video, audio, y = (video.to(device), audio.to(device),
                                     y.to(device))
            optimiser.zero_grad()
            loss = loss_fn(net(video, audio), y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

        if dl_val is None:                    # official split: no early stopping, no val threshold
            continue
        _, targets, probs = predict(net, data, loader=dl_val)
        score = evaluate(targets, probs)["auc"]
        score = score if score == score else 0.0          # NaN if the val has one class

        if score > best_score:
            best_score, best_state, waited = score, copy.deepcopy(net.state_dict()), 0
            best_thr = best_threshold(targets, probs)
        else:
            waited += 1
            if waited >= settings.patience:
                break
    seconds = time.perf_counter() - started

    if best_state is not None:
        net.load_state_dict(best_state)
    else:
        best_thr = 0.5                        # the paper's operating point: argmax of the logits
    ids, targets, probs = predict(net, data, loader=dl_hold)
    assert len(ids) == len(probs) == len(targets), (len(ids), len(probs), len(targets))

    metrics = evaluate(targets, probs, threshold=best_thr)
    metrics["seconds"] = seconds
    metrics["epochs"] = epoch + 1
    if verbose:
        print(f"    fold {fold} seed {seed}: ACC {metrics['acc'] * 100:5.1f}  "
              f"AUC {metrics['auc'] * 100:5.1f}  thr {best_thr:.2f}  {seconds:4.0f}s")
    out = {"metrics": metrics, "ids": ids, "targets": targets, "probs": probs}
    if return_net:
        out["net"] = net
    return out


def run_one(data: Data, variant: str, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every seed × fold of one variant on one split: the per-fold table and the pooled preds."""
    rows, pooled = [], []
    for seed in data.settings.seeds:
        for fold in range(1, data.n_folds + 1):
            out = train_one_fold(data, variant, fold, seed, verbose=verbose)
            rows.append({"seed": seed, "fold": fold, **out["metrics"]})
            pooled.append(pd.DataFrame({"clip_id": out["ids"], "y_true": out["targets"],
                                        "y_prob": out["probs"], "seed": seed, "fold": fold}))
    summary = pd.DataFrame(rows)
    per_clip = pd.concat(pooled, ignore_index=True)
    # one prediction per clip, seeds averaged: what the paired comparison with the other
    # directions expects, and what the README's pooled number quotes
    per_clip = (per_clip.groupby("clip_id", as_index=False)
                        .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean")))
    return summary, per_clip


def run_test(data: Data, variant: str,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train every seed × fold and score the closed test set with each fitted model.

    The number this returns is the one ``docs/evaluation.md`` reserves for the end: the test
    set is scored, never used. Each model keeps the threshold it chose on its own ``val`` —
    nothing about the test set picks a threshold, an epoch or a variant — so an unusual
    number here is a result, not something to adjust.

    Returns ``(test_summary, test_preds, holdout_summary)``: one row per seed × fold, one row
    per clip and model, and the holdout metrics those same folds scored (which is what
    ``results/<variant>-subject/folds.csv`` holds, so the retraining can be checked against
    the run the board quotes).
    """
    if data.split != "test":
        raise ValueError(f"run_test scores the closed set: build Data with split='test', "
                         f"not {data.split!r}")
    test_ids = data.folds.test()
    rows, pooled, holdout_rows = [], [], []
    for seed in data.settings.seeds:
        for fold in range(1, data.n_folds + 1):
            out = train_one_fold(data, variant, fold, seed, verbose=verbose, return_net=True)
            holdout_rows.append({"seed": seed, "fold": fold, **out["metrics"]})

            threshold = out["metrics"]["threshold"]
            ids, targets, probs = predict(out["net"], data, test_ids)
            metrics = evaluate(targets, probs, threshold=threshold)
            metrics["seconds"] = out["metrics"]["seconds"]
            metrics["epochs"] = out["metrics"]["epochs"]
            rows.append({"seed": seed, "fold": fold, **metrics})
            pooled.append(pd.DataFrame({"clip_id": ids, "y_true": targets, "y_prob": probs,
                                        "seed": seed, "fold": fold}))
            if verbose:
                print(f"    fold {fold} seed {seed} on the test set: "
                      f"ACC {metrics['acc'] * 100:5.1f}  AUC {metrics['auc'] * 100:5.1f}  "
                      f"thr {threshold:.2f}")

    return (pd.DataFrame(rows), pd.concat(pooled, ignore_index=True),
            pd.DataFrame(holdout_rows))


def config_for(data: Data, variant: str, settings: Settings, direction: str = "tia",
               owner: str = "mattsays") -> dict:
    """The provenance of one run: what was launched, on which data, on which commit."""
    net = model.VARIANTS[variant]()
    config = export.run_config(
        run=export.run_name(variant, data.split, settings.protocol, settings.captains),
        direction=direction,
        variant=variant,
        split=data.split,
        protocol=settings.protocol,
        captains=settings.captains,
        owner=owner,
        folds=data.n_folds,
        seeds=list(settings.seeds),
        encoders={"video": settings.encoders.video, "audio": settings.encoders.audio},
        widths={"video": model.VIDEO_DIM, "audio": model.AUDIO_DIM},
        trainable_params=sum(p.numel() for p in net.parameters() if p.requires_grad),
        training={"batch_size": settings.batch_size, "epochs": settings.epochs,
                  "patience": settings.patience, "lr": settings.lr,
                  "weight_decay": settings.weight_decay, "optimiser": "AdamW",
                  "loss": "BCEWithLogitsLoss(pos_weight=train-fold prevalence)"},
        device=str(settings.torch_device()),
    )
    config.update(export.describe(data.clips, data.folds))
    return config


def run_campaign(clips, folds, settings: Settings, runs, root: str | Path,
                 direction: str = "tia", owner: str = "mattsays") -> pd.DataFrame:
    """Run every ``(variant, split)`` in ``runs`` and write one ``results/<run>/`` each.

    The splits are grouped so the feature store is built once per split instead of once per
    run: the caches are the expensive read, not the training.
    """
    order = {"subject": 0, "official": 1}
    runs = sorted(runs, key=lambda pair: (order.get(pair[1], 9), pair[0]))
    print(f"{len(runs)} runs x {len(settings.seeds)} seeds x folds: "
          f"{', '.join(f'{v}/{s}' for v, s in runs)}")

    tables, current_split, data = [], None, None
    for index, (variant, split) in enumerate(runs, 1):
        if split != current_split:
            current_split, data = split, Data(clips, folds, settings, split)
        print(f"\n[{index}/{len(runs)}] {variant} on {split} — "
              f"{data.n_folds} folds x {len(settings.seeds)} seeds")
        summary, per_clip = run_one(data, variant)
        config = config_for(data, variant, settings, direction, owner)
        export.save_run(summary, per_clip, config, root=root)
        tables.append({"run": config["run"],
                       **{m: summary[m].mean() for m in ("acc", "f1", "auc", "balanced_acc")},
                       "acc_std": summary.acc.std(),
                       "params": config["trainable_params"]})

    board = pd.DataFrame(tables)
    print("\n=== campaign summary (mean over folds and seeds) ===")
    print(board.round(3).to_string(index=False))
    return board
