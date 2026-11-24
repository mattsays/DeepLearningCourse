"""TemporalCNN on the shared protocol files: official 3-fold and subject 5-fold.

    python train_official_temporal.py --protocol official \
        --data ../../../../../dolos --cache ../../../../../dolos/cached_features/teo-landmarks-egemaps

Why a second runner: ``train_fold_temporal.py`` reads only the subject folds and maps clips to
speakers through the clip-id prefix (``an_wilty_...`` -> ``an``), which is the uploader tag, not
the speaker. This runner takes the clip lists straight from ``dolos/splits/*.json`` and the
labels from ``dolos/dataset.csv``, and reuses the training code of ``train_fold_temporal.py``
unchanged (``train_one_fold``: AdamW, positive-class weighting, LR halving and early stopping on
validation AUC, threshold chosen on validation).

- ``official``: ``official_fold{1,2,3}.json``. The files have no validation set, so a stratified
  15% clip-level split of each training fold is held out for early stopping and the threshold —
  the rule VideoMAE + HuBERT and MARLIN + ConvNeXt use. Scored on ``test``.
- ``subject``: ``subject_fold{1..5}.json`` — ``train`` + ``always_train`` (the two captains),
  ``val``, scored on ``holdout``. ``subject_test.json`` is never read.

Features: ``{clip_id}_feats.npy`` (T x 53) written by ``features/extract_features.py``. The
loader also needs ``{clip_id}_mumin.npy``; they are written from the ``mumin_*`` columns of
``dataset.csv`` (the auxiliary MUMIN loss weight is 0 by default, as in teo's scripts).

Writes ``teo/results/temporal-<protocol>/``: ``folds.csv`` (one row per seed x fold),
``preds.csv`` (clip_id, y_true, y_prob, seed, fold), ``summary.json``. Checkpoints go to
``--runs`` (default: next to the cache, outside git).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from model_temporal import TemporalCNN
from train_fold_temporal import compute_metrics, evaluate, make_loader, train_one_fold

HERE = Path(__file__).resolve().parent


def load_folds(splits: Path, protocol: str, labels: dict, seed: int) -> list[dict]:
    folds = []
    if protocol == "official":
        for k in (1, 2, 3):
            d = json.loads((splits / f"official_fold{k}.json").read_text())
            test = [c for c in d["test"]["clips"] if c in labels]
            # The official files list a few clips in both train and test of the same fold
            # (dataset README §11); they are scored, never trained on.
            both = set(d["train"]["clips"]) & set(test)
            if both:
                print(f"official_fold{k}: {len(both)} clip(s) in train and test, dropped from train")
            train = [c for c in d["train"]["clips"] if c in labels and c not in both]
            y = [labels[c] for c in train]
            tr, va = train_test_split(train, test_size=0.15, stratify=y, random_state=seed + k)
            folds.append({"train": set(tr), "val": set(va), "holdout": set(test)})
    else:
        for k in range(1, 6):
            d = json.loads((splits / f"subject_fold{k}.json").read_text())
            pick = lambda key: {c for c in d[key]["clips"] if c in labels}  # noqa: E731
            folds.append({"train": pick("train") | pick("always_train"),
                          "val": pick("val"), "holdout": pick("holdout")})
    for k, f in enumerate(folds, 1):
        assert not (f["train"] & f["holdout"]) and not (f["val"] & f["holdout"]), f"fold {k} leaks"
    return folds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("official", "subject"), required=True)
    ap.add_argument("--data", type=Path, required=True, help="dolos/ (dataset.csv, splits/)")
    ap.add_argument("--cache", type=Path, required=True, help="folder of *_feats.npy")
    ap.add_argument("--runs", type=Path, default=None, help="checkpoints (default: <cache>/../teo-runs)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mumin_weight", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    a = ap.parse_args()

    out = a.out or HERE.parent / "results" / f"temporal-{a.protocol}"
    runs = a.runs or a.cache.parent / "teo-runs" / a.protocol
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(a.data / "dataset.csv")
    meta = meta[meta["status"] == "ok"]
    mumin_cols = [c for c in meta.columns if c.startswith("mumin_")]
    labels = {}
    for row in meta.itertuples(index=False):
        cid = row.clip_id
        if not (a.cache / f"{cid}_feats.npy").exists():
            continue
        mpath = a.cache / f"{cid}_mumin.npy"
        if not mpath.exists():
            np.save(mpath, meta.loc[meta.clip_id == cid, mumin_cols].to_numpy(np.float32)[0])
        labels[cid] = float(row.y)
    print(f"{len(labels)} clips with features ({len(meta) - len(labels)} ok clips without)")

    device = torch.device("cpu")
    fold_rows, pred_rows = [], []
    t0 = time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        folds = load_folds(a.data / "splits", a.protocol, labels, seed)
        for k, fold in enumerate(folds, 1):
            torch.manual_seed(seed * 100 + k)
            np.random.seed(seed * 100 + k)
            args = SimpleNamespace(out=str(runs / f"seed{seed}"), batch_size=a.batch_size,
                                   hidden=a.hidden, dropout=a.dropout, lr=a.lr, epochs=a.epochs,
                                   mumin_weight=a.mumin_weight, patience=a.patience)
            t = time.time()
            r = train_one_fold(k, fold, str(a.cache), labels, {}, args, device)
            model = TemporalCNN(hidden=a.hidden, dropout=a.dropout).to(device)
            model.load_state_dict(r["model_state"])
            ids = sorted(fold["holdout"])
            loader = make_loader(ids, str(a.cache), labels, a.batch_size, shuffle=False)
            y, s = evaluate(model, loader, device)          # same order as ids (no shuffle)
            m = compute_metrics(y, s, r["threshold"])
            fold_rows.append({"seed": seed, "fold": k, **m, "threshold": r["threshold"],
                              "n_train": len(fold["train"]), "n_val": len(fold["val"]),
                              "n_test": len(ids), "best_val_auc": r["best_val_auc"],
                              "seconds": round(time.time() - t, 1)})
            pred_rows += [{"clip_id": c, "y_true": int(yy), "y_prob": float(ss),
                           "seed": seed, "fold": k} for c, yy, ss in zip(ids, y, s)]

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out / "folds.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "preds.csv", index=False)
    keys = ["acc", "f1", "auc", "bal_acc", "majority"]
    summary = {
        "protocol": a.protocol, "seeds": a.seeds, "n_folds": int(folds_df.fold.nunique()),
        **{k: float(folds_df[k].mean()) for k in keys},
        **{f"{k}_std": float(folds_df[k].std(ddof=0)) for k in keys},
        "per_fold": folds_df.groupby("fold")[["acc", "auc", "n_test"]].mean().round(4)
                            .reset_index().to_dict("records"),
        "per_seed": folds_df.groupby("seed")[["acc", "auc"]].mean().round(4)
                            .reset_index().to_dict("records"),
        "trainable_params": sum(p.numel() for p in
                                TemporalCNN(hidden=a.hidden, dropout=a.dropout).parameters()),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "config": {k: v for k, v in vars(a).items() if not isinstance(v, Path)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== {a.protocol}: mean over {len(folds_df)} seed x fold runs ===")
    print("  " + "  ".join(f"{k}={summary[k]*100:.1f}±{summary[k + '_std']*100:.1f}" for k in keys))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
