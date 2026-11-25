"""
train_temporal_cv.py — subject-level 5-fold CV for DOLOS.

Requires a folder containing subject_fold{1..5}.json and subject_test.json.
official_fold{1..3}.json are intentionally ignored (clip-level, they leak).

For each fold:
  - train a fresh TemporalCNN on that fold's `train` subjects
  - early-stop on `val` subjects by val AUC
  - pick a decision threshold on `val` via best_threshold (Youden's J)
  - evaluate on `holdout` at that threshold (never seen during training)
The results-board row is the mean over the five holdouts.

Virgin test (subject_test.json) is CLOSED by default; pass
--evaluate-test to open it (ensembles the 5 fold models, thresholds
at the mean of the five fold thresholds).
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
)

from model_temporal import TemporalCNN


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(r"^([^_]+)_")


def subject_of(clip_id: str) -> str:
    """'an_wilty_ep15_lie10' -> 'an'."""
    m = _SUBJECT_RE.match(clip_id)
    return m.group(1).lower() if m else clip_id.lower()


def _find_key(d: dict, *candidates):
    for c in candidates:
        if c in d:
            return c
    lc = {k.lower(): k for k in d}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def _value_to_clips(value, subject_to_clips, all_clips):
    """Normalize a JSON split value to a set of clip IDs.

    Accepts list of subjects, list of clips, dict {subject: [...]}, or
    dict {clip_id: ...}. Unknown ids are ignored.
    """
    out = set()
    if value is None:
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            ks = str(k)
            if ks in all_clips:
                out.add(ks)
            elif ks.lower() in subject_to_clips:
                out.update(subject_to_clips[ks.lower()])
            if isinstance(v, list):
                for x in v:
                    xs = str(x)
                    if xs in all_clips:
                        out.add(xs)
                    elif xs.lower() in subject_to_clips:
                        out.update(subject_to_clips[xs.lower()])
    elif isinstance(value, list):
        for x in value:
            xs = str(x)
            if xs in all_clips:
                out.add(xs)
            elif xs.lower() in subject_to_clips:
                out.update(subject_to_clips[xs.lower()])
    return out


def load_splits(folder, subject_to_clips, all_clips):
    folds = []
    for i in range(1, 6):
        path = os.path.join(folder, f"subject_fold{i}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        train_k = _find_key(d, "train", "train_subjects", "train_clips")
        val_k = _find_key(d, "val", "valid", "validation",
                          "val_subjects", "val_clips")
        hold_k = _find_key(d, "holdout", "test", "holdout_subjects",
                           "holdout_clips")
        if not (train_k and val_k and hold_k):
            raise ValueError(
                f"{path}: expected train/val/holdout keys, got {list(d.keys())}"
            )
        folds.append({
            "train": _value_to_clips(d[train_k], subject_to_clips, all_clips),
            "val": _value_to_clips(d[val_k], subject_to_clips, all_clips),
            "holdout": _value_to_clips(d[hold_k], subject_to_clips, all_clips),
        })
        f = folds[-1]
        print(f"  subject_fold{i}.json: train={len(f['train'])} "
              f"val={len(f['val'])} holdout={len(f['holdout'])}")

    test = None
    test_path = os.path.join(folder, "subject_test.json")
    if os.path.exists(test_path):
        d = json.load(open(test_path))
        if isinstance(d, list):
            test = _value_to_clips(d, subject_to_clips, all_clips)
        else:
            k = _find_key(d, "test", "test_subjects", "test_clips",
                          "holdout", "clips")
            if k is None:
                raise ValueError(
                    f"{test_path}: cannot find test key in {list(d.keys())}"
                )
            test = _value_to_clips(d[k], subject_to_clips, all_clips)
        print(f"  subject_test.json: {len(test)} clips (closed by default)")

    return folds, test


# ---------------------------------------------------------------------------
# Dataset / loaders
# ---------------------------------------------------------------------------

class CachedTemporalDataset(Dataset):
    def __init__(self, clip_ids, cache_dir, labels):
        self.clip_ids = list(clip_ids)
        self.cache_dir = cache_dir
        self.labels = labels

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        vid = self.clip_ids[i]
        feats = np.load(os.path.join(self.cache_dir, f"{vid}_feats.npy"))
        mumin = np.load(os.path.join(self.cache_dir, f"{vid}_mumin.npy"))
        if mumin.ndim == 2:
            mumin = mumin.mean(axis=0)
        return (
            torch.from_numpy(feats.astype(np.float32)),
            torch.tensor(self.labels[vid], dtype=torch.float),
            torch.from_numpy(mumin.astype(np.float32)),
        )


def collate_temporal(batch):
    feats_list, labels, mumins = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    max_len = int(lengths.max())
    D = feats_list[0].shape[1]
    padded = torch.zeros(len(batch), max_len, D)
    for i, f in enumerate(feats_list):
        padded[i, : f.shape[0]] = f
    return padded, torch.stack(labels), torch.stack(mumins), lengths


def make_loader(clip_ids, cache_dir, labels, batch_size, shuffle):
    ds = CachedTemporalDataset(clip_ids, cache_dir, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_temporal, num_workers=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_threshold(y_true, y_score):
    """Val-only threshold that maximizes balanced accuracy (Youden's J).

    Used instead of 0.5 because the paper notes train folds are ~56%
    deception and holdouts ~47%, so 0.5 is biased.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([y_score, [0.0, 1.0]]))
    best_t, best_j = 0.5, -1.0
    for t in candidates:
        pred = (y_score >= t).astype(int)
        try:
            j = balanced_accuracy_score(y_true, pred)
        except ValueError:
            continue
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def compute_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_score >= threshold).astype(int)
    auc = (float(roc_auc_score(y_true, y_score))
           if len(np.unique(y_true)) > 1 else float("nan"))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc": auc,
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "majority": float(max(np.mean(y_true), 1.0 - np.mean(y_true))),
    }


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y, _, lengths in loader:
            x, y = x.to(device), y.to(device)
            lie_logit = model(x, lengths, return_mumin=False)
            scores.append(torch.sigmoid(lie_logit).cpu())
            labels.append(y.cpu())
    return np.concatenate(labels), np.concatenate(scores)


def train_one_fold(fold_idx, fold, cache_dir, labels, gender, args, device):
    out_dir = os.path.join(args.out, f"fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    train_loader = make_loader(fold["train"], cache_dir, labels,
                               args.batch_size, shuffle=True)
    val_loader = make_loader(fold["val"], cache_dir, labels,
                             args.batch_size, shuffle=False)
    hold_loader = make_loader(fold["holdout"], cache_dir, labels,
                              args.batch_size, shuffle=False)

    train_labels = np.array([labels[c] for c in fold["train"]])
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float)

    model = TemporalCNN(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5,
    )

    best_val_auc = -float("inf")
    best_state = None
    patience_ctr = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, mumin, lengths in train_loader:
            x, y, mumin = x.to(device), y.to(device), mumin.to(device)
            optimizer.zero_grad()
            lie_logit, mumin_pred = model(x, lengths, return_mumin=True)
            loss = (
                F.binary_cross_entropy_with_logits(lie_logit, y,
                                                   pos_weight=pos_weight)
                + args.mumin_weight * F.mse_loss(mumin_pred, mumin)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_y, val_s = evaluate(model, val_loader, device)
        val_auc = (roc_auc_score(val_y, val_s)
                   if len(np.unique(val_y)) > 1 else 0.0)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                break

    model.load_state_dict(best_state)
    val_y, val_s = evaluate(model, val_loader, device)
    t_val = best_threshold(val_y, val_s)
    val_metrics = compute_metrics(val_y, val_s, t_val)

    hold_y, hold_s = evaluate(model, hold_loader, device)
    hold_metrics = compute_metrics(hold_y, hold_s, t_val)

    print(f"[fold {fold_idx}] best_ep={best_epoch:02d} t_val={t_val:.3f} "
          f"val_auc={val_metrics['auc']:.3f} | "
          f"holdout: acc={hold_metrics['acc']:.3f} "
          f"f1={hold_metrics['f1']:.3f} "
          f"auc={hold_metrics['auc']:.3f} "
          f"bacc={hold_metrics['bal_acc']:.3f} "
          f"maj={hold_metrics['majority']:.3f}")

    # Optional per-gender breakdown on holdout
    if gender:
        hold_ids = hold_loader.dataset.clip_ids
        for g in sorted(set(gender.values())):
            idx = [i for i, c in enumerate(hold_ids) if gender.get(c) == g]
            if len(idx) >= 3:
                gacc = accuracy_score(
                    hold_y[idx].astype(int),
                    (hold_s[idx] >= t_val).astype(int),
                )
                print(f"  {g}: acc={gacc:.3f} n={len(idx)}")

    torch.save(
        {"model_state": best_state,
         "threshold": t_val,
         "hidden": args.hidden, "dropout": args.dropout,
         "fold": fold_idx, "best_epoch": best_epoch,
         "best_val_auc": best_val_auc},
        os.path.join(out_dir, "best.pt"),
    )

    return {
        "fold": fold_idx, "threshold": t_val,
        "val": val_metrics, "holdout": hold_metrics,
        "model_state": best_state,
        "best_val_auc": best_val_auc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-folder", required=True,
                    help="Folder with subject_fold{1..5}.json and subject_test.json")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/cv")
    ap.add_argument("--gender-csv", default=None,
                    help="Optional CSV with video_id,gender for per-gender ACC")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mumin_weight", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate-test", action="store_true",
                    help="Open the virgin test set (ensemble of fold models)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load labels; keep only clips with both caches
    df = pd.read_csv(args.labels)
    labels = {}
    for _, r in df.iterrows():
        vid = str(r["video_id"])
        if (os.path.exists(os.path.join(args.cache, f"{vid}_feats.npy"))
                and os.path.exists(os.path.join(args.cache, f"{vid}_mumin.npy"))):
            labels[vid] = float(r["label"])
    all_clips = set(labels.keys())

    subject_to_clips = {}
    for c in all_clips:
        subject_to_clips.setdefault(subject_of(c), set()).add(c)
    print(f"loaded {len(all_clips)} clips across "
          f"{len(subject_to_clips)} subjects")

    gender = {}
    if args.gender_csv and os.path.exists(args.gender_csv):
        gdf = pd.read_csv(args.gender_csv)
        for _, r in gdf.iterrows():
            g = str(r["gender"]).strip().upper()[:1]
            if g in ("M", "F"):
                gender[str(r["video_id"])] = g

    print(f"loading splits from {args.splits_folder}:")
    folds, test_clips = load_splits(args.splits_folder, subject_to_clips, all_clips)
    if not folds:
        raise SystemExit(
            f"no subject_foldN.json files found in {args.splits_folder}"
        )

    # Sanity: no subject overlap between train and val/holdout
    for i, f in enumerate(folds, 1):
        subj_train = {subject_of(c) for c in f["train"]}
        subj_val = {subject_of(c) for c in f["val"]}
        subj_hold = {subject_of(c) for c in f["holdout"]}
        leak = (subj_train & subj_val) | (subj_train & subj_hold)
        if leak:
            print(f"  WARNING fold {i}: subjects on both sides: "
                  f"{sorted(leak)[:5]}{'...' if len(leak) > 5 else ''}")

    results = []
    for i, fold in enumerate(folds, 1):
        if not (fold["train"] and fold["val"] and fold["holdout"]):
            print(f"[fold {i}] skipping — empty split "
                  f"(train={len(fold['train'])} val={len(fold['val'])} "
                  f"holdout={len(fold['holdout'])})")
            continue
        results.append(train_one_fold(i, fold, args.cache, labels, gender,
                                      args, device))

    if not results:
        raise SystemExit("no fold completed")

    keys = ["acc", "f1", "auc", "bal_acc", "majority"]
    print("\n=== mean over folds (results board row) ===")
    means = {k: float(np.mean([r["holdout"][k] for r in results])) for k in keys}
    stds = {k: float(np.std([r["holdout"][k] for r in results])) for k in keys}
    print("  " + "  ".join(f"{k}={means[k]:.3f}±{stds[k]:.3f}" for k in keys))
    print(f"  folds used: {len(results)} / {len(folds)}")

    if test_clips:
        if not args.evaluate_test:
            print(f"\n(virgin test set: {len(test_clips)} clips — closed; "
                  f"pass --evaluate-test to open)")
            return
        print(f"\n=== virgin test ({len(test_clips)} clips, "
              f"ensemble of {len(results)} fold models) ===")
        test_loader = make_loader(list(test_clips), args.cache, labels,
                                  args.batch_size, shuffle=False)
        per_fold_scores = []
        for r in results:
            model = TemporalCNN(hidden=args.hidden, dropout=args.dropout).to(device)
            model.load_state_dict(r["model_state"])
            y_test, s_test = evaluate(model, test_loader, device)
            per_fold_scores.append(s_test)
        test_scores = np.mean(per_fold_scores, axis=0)
        test_thresh = float(np.mean([r["threshold"] for r in results]))
        test_metrics = compute_metrics(y_test, test_scores, test_thresh)
        print(f"  threshold (mean of fold t)={test_thresh:.3f}")
        for k in keys:
            print(f"  {k}={test_metrics[k]:.3f}")


if __name__ == "__main__":
    main()