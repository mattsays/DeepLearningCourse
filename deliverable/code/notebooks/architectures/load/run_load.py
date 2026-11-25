"""Headless run of the `load` baseline on every clip, both protocols.

    python run_load.py --data ../../../../dolos

Same features (``features.acoustic_features``), same ``LoadHead`` and the same training rules as
``train_load.ipynb``:

- official 3-fold: no validation set, a fixed 60 epochs, threshold 0.5 (the paper's operating point);
- subject 5-fold: early stopping on validation AUC (patience 10, at most 60 epochs), threshold that
  maximises validation balanced accuracy, scored on the holdout;
- 3 seeds, AdamW 1e-3 / wd 1e-2, batch 32, clip 1.0, class-weighted BCE, scaler fitted on the
  training fold only.

Two differences from the notebook, both about coverage: the feature cache covers **every** clip
with audio (the notebook's cache held only the subject working set, which left 295 official test
clips unscored), and NaNs left by Praat are imputed with the column median over all cached clips
(feature marginals only, never labels). Clips listed in both train and test of an official fold
(dataset README §11) are scored, never trained on.

Writes ``results/load-<protocol>/{folds.csv, preds.csv, summary.json}`` next to this file and the
cache to ``<data>/cached_features/acoustic-load-all/``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))           # lie-detector/, for lib

from features import FEATURE_NAMES, acoustic_features  # noqa: E402
from lib.metrics import best_threshold, evaluate  # noqa: E402

EPOCHS, PATIENCE, BATCH = 60, 10, 32


class LoadHead(nn.Module):
    """Standardise, then a small MLP (identical to the notebook)."""

    def __init__(self, n_features, hidden=32, drop=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sigma", torch.ones(n_features))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(n_features, hidden), nn.GELU(),
                                  nn.Dropout(drop), nn.Linear(hidden, 1))

    def fit_scaler(self, features):
        self.mu.copy_(features.mean(dim=0))
        self.sigma.copy_(features.std(dim=0).clamp(min=1e-6))

    def forward(self, x):
        return self.head((x - self.mu) / self.sigma).squeeze(-1)


def _one(args):
    clip_id, wav = args
    try:
        return clip_id, acoustic_features(wav), None
    except Exception as error:                      # one broken clip must not kill the run
        return clip_id, None, f"{type(error).__name__}: {error}"[:200]


def build_cache(data: Path, clip_ids: list[str], cache: Path, workers: int) -> dict[str, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    todo = [(c, str(data / "audio" / f"{c}.wav")) for c in clip_ids
            if not (cache / f"{c}.npy").exists() and (data / "audio" / f"{c}.wav").exists()]
    print(f"{len(clip_ids)} clips, {len(todo)} to compute")
    failures = {}
    with ProcessPoolExecutor(workers) as pool:
        for i, (clip_id, vec, err) in enumerate(pool.map(_one, todo, chunksize=8), 1):
            if err:
                failures[clip_id] = err
            else:
                np.save(cache / f"{clip_id}.npy", vec)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"{len(failures)} failed" + (f": {dict(list(failures.items())[:3])}" if failures else ""))
    present = [c for c in clip_ids if (cache / f"{c}.npy").exists()]
    matrix = np.stack([np.load(cache / f"{c}.npy") for c in present]).astype(np.float32)
    median = np.nanmedian(matrix, axis=0)
    nans = int(np.isnan(matrix).any(axis=1).sum())
    matrix = np.where(np.isnan(matrix), median, matrix)
    print(f"cache: {len(present)} clips x {matrix.shape[1]}; {nans} vectors imputed")
    return dict(zip(present, matrix))


def batches(ids, X, y, shuffle, seed):
    order = np.arange(len(ids))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        yield torch.from_numpy(X[idx]), torch.from_numpy(y[idx])


def train_fold(feat, labels, train_ids, val_ids, hold_ids, seed, epoch_seed_offset):
    def arrays(ids):
        ids = [c for c in ids if c in feat]
        return ids, np.stack([feat[c] for c in ids]), np.array([labels[c] for c in ids], np.float32)

    tr_ids, Xtr, ytr = arrays(train_ids)
    ho_ids, Xho, yho = arrays(hold_ids)
    va = arrays(val_ids) if val_ids is not None else None

    torch.manual_seed(seed)
    net = LoadHead(len(FEATURE_NAMES))
    net.fit_scaler(torch.from_numpy(Xtr))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pos = ytr.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - pos) / max(pos, 1)))

    best, best_state, thr, waited, epoch = -1.0, None, 0.5, 0, -1
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        net.train()
        for x, yy in batches(tr_ids, Xtr, ytr, True, seed * 1000 + epoch + epoch_seed_offset):
            opt.zero_grad()
            loss_fn(net(x), yy).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if va is None:
            continue
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(torch.from_numpy(va[1]))).numpy()
        score = evaluate(va[2], pv)["auc"]
        score = score if score == score else 0.0
        if score > best:
            best, best_state, waited = score, copy.deepcopy(net.state_dict()), 0
            thr = best_threshold(va[2], pv)
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ph = torch.sigmoid(net(torch.from_numpy(Xho))).numpy()
    m = evaluate(yho, ph, threshold=thr)
    m.update(seconds=round(time.perf_counter() - t0, 2), epochs=epoch + 1,
             n_train=len(tr_ids), n_val=0 if va is None else len(va[0]))
    return m, ho_ids, yho, ph


def run(protocol, data, feat, labels, seeds, out):
    splits = data / "splits"
    rows, preds = [], []
    nf = 3 if protocol == "official" else 5
    for seed in seeds:
        for k in range(1, nf + 1):
            if protocol == "official":
                d = json.loads((splits / f"official_fold{k}.json").read_text())
                test = d["test"]["clips"]
                both = set(d["train"]["clips"]) & set(test)
                train = [c for c in d["train"]["clips"] if c not in both]
                val = None
            else:
                d = json.loads((splits / f"subject_fold{k}.json").read_text())
                train = d["train"]["clips"] + d["always_train"]["clips"]
                val, test = d["val"]["clips"], d["holdout"]["clips"]
            m, ids, y, p = train_fold(feat, labels, train, val, test, seed, k * 100)
            m.pop("confusion")
            rows.append({"seed": seed, "fold": k, **m})
            preds += [{"clip_id": c, "y_true": int(t), "y_prob": float(q), "seed": seed, "fold": k}
                      for c, t, q in zip(ids, y, p)]
            print(f"{protocol} seed {seed} fold {k}: n={m['n']} ACC {m['acc']*100:5.1f} "
                  f"AUC {m['auc']*100:5.1f} thr {m['threshold']:.2f}")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "folds.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "preds.csv", index=False)
    keys = ["acc", "f1", "auc", "balanced_acc"]
    summary = {"protocol": protocol, "seeds": seeds, "n_features": len(FEATURE_NAMES),
               "trainable_params": sum(p.numel() for p in LoadHead(len(FEATURE_NAMES)).parameters()),
               **{k: float(df[k].mean()) for k in keys},
               **{f"{k}_std": float(df[k].std()) for k in keys},
               "per_fold": df.groupby("fold")[["acc", "auc", "n"]].mean().round(4)
                             .reset_index().to_dict("records"),
               "per_seed": df.groupby("seed")[["acc", "auc"]].mean().round(4)
                             .reset_index().to_dict("records"),
               "seconds_per_fold": float(df.seconds.mean())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"== {protocol}: ACC {summary['acc']*100:.1f} ± {summary['acc_std']*100:.1f}  "
          f"F1 {summary['f1']*100:.1f}  AUC {summary['auc']*100:.1f}  "
          f"bal {summary['balanced_acc']*100:.1f}  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dolos/ (dataset.csv, audio/, splits/)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    meta = pd.read_csv(a.data / "dataset.csv")
    meta = meta[meta.status == "ok"]
    labels = dict(zip(meta.clip_id, meta.y.astype(np.float32)))
    feat = build_cache(a.data, list(meta.clip_id), a.data / "cached_features" / "acoustic-load-all",
                       a.workers)
    seeds = [int(s) for s in a.seeds.split(",")]
    for protocol in ("official", "subject"):
        run(protocol, a.data, feat, labels, seeds, HERE / "results" / f"load-{protocol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
