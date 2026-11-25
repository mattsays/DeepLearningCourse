"""GAT on subject-level 5-fold CV. Mirrors train_temporal_cv.py protocol."""
import os, argparse, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

from config import Config
from graph_construction import build_window_graphs
from model import DualHeadGAT
from new_train_temporal import (
    subject_of, load_splits, best_threshold, compute_metrics,
)


class GATFoldDataset(Dataset):
    def __init__(self, clip_ids, cache_dir, labels, cfg):
        self.clip_ids = list(clip_ids)
        self.cache_dir = cache_dir
        self.labels = labels
        self.cfg = cfg

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        vid = self.clip_ids[i]
        feats = np.load(os.path.join(self.cache_dir, f"{vid}_feats.npy"))
        mumin = np.load(os.path.join(self.cache_dir, f"{vid}_mumin.npy"))
        if mumin.ndim == 2:
            mumin = mumin.mean(axis=0)
        ws = build_window_graphs(feats, self.cfg)
        if not ws:
            ws = [Data(x=torch.zeros(1, self.cfg.node_feat_dim),
                       edge_index=torch.zeros(2, 0, dtype=torch.long),
                       node_type=torch.zeros(1, dtype=torch.long))]
        xs, ets, nts, off = [], [], [], 0
        for g in ws:
            xs.append(g.x); ets.append(g.edge_index + off); nts.append(g.node_type)
            off += g.num_nodes
        d = Data(x=torch.cat(xs, 0), edge_index=torch.cat(ets, 1),
                 node_type=torch.cat(nts, 0))
        d.y = torch.tensor([self.labels[vid]], dtype=torch.float)
        d.mumin = torch.from_numpy(mumin.astype(np.float32)).unsqueeze(0)
        return d


def make_gat_loader(clip_ids, cache_dir, labels, cfg, bs, shuffle):
    ds = GATFoldDataset(clip_ids, cache_dir, labels, cfg)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def evaluate(model, loader, device):
    model.eval()
    s, y = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            s.append(torch.sigmoid(model(b, return_mumin=False)).cpu())
            y.append(b.y.cpu())
    return np.concatenate(y), np.concatenate(s)


def train_one_fold(fold_idx, fold, cache_dir, labels, cfg, args, device):
    out_dir = os.path.join(args.out, f"fold{fold_idx}")
    os.makedirs(out_dir, exist_ok=True)

    tr = make_gat_loader(fold["train"], cache_dir, labels, cfg, args.batch_size, True)
    va = make_gat_loader(fold["val"], cache_dir, labels, cfg, args.batch_size, False)
    ho = make_gat_loader(fold["holdout"], cache_dir, labels, cfg, args.batch_size, False)

    tr_labels = np.array([labels[c] for c in fold["train"]])
    n_pos = float(tr_labels.sum()); n_neg = float(len(tr_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float)

    model = DualHeadGAT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3, min_lr=1e-5)
    best_auc, best_state, best_ep, pat = -1e9, None, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for b in tr:
            b = b.to(device)
            opt.zero_grad()
            ll, mp = model(b, return_mumin=True)
            loss = (F.binary_cross_entropy_with_logits(ll, b.y, pos_weight=pos_weight)
                    + args.mumin_weight * F.mse_loss(mp, b.mumin.view(-1, cfg.num_mumin)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        vy, vs = evaluate(model, va, device)
        vauc = roc_auc_score(vy, vs) if len(np.unique(vy)) > 1 else 0.0
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, best_ep, pat = vauc, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= args.patience:
                break

    model.load_state_dict(best_state)
    vy, vs = evaluate(model, va, device)
    t = best_threshold(vy, vs)
    vm = compute_metrics(vy, vs, t)
    hy, hs = evaluate(model, ho, device)
    hm = compute_metrics(hy, hs, t)
    print(f"[fold {fold_idx}] best_ep={best_ep:02d} t={t:.3f} "
          f"val_auc={vm['auc']:.3f} | hold: acc={hm['acc']:.3f} "
          f"f1={hm['f1']:.3f} auc={hm['auc']:.3f} bacc={hm['bal_acc']:.3f} "
          f"maj={hm['majority']:.3f}")
    torch.save({"model_state": best_state, "threshold": t,
                "cfg": cfg.__dict__, "fold": fold_idx,
                "best_epoch": best_ep, "best_val_auc": best_auc},
               os.path.join(out_dir, "best.pt"))
    return {"fold": fold_idx, "threshold": t, "val": vm,
            "holdout": hm, "model_state": best_state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-folder", required=True)
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/cv_gat")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mumin_weight", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = pd.read_csv(args.labels)
    labels = {}
    for _, r in df.iterrows():
        vid = str(r["video_id"])
        if (os.path.exists(os.path.join(args.cache, f"{vid}_feats.npy"))
                and os.path.exists(os.path.join(args.cache, f"{vid}_mumin.npy"))):
            labels[vid] = float(r["label"])
    all_clips = set(labels)
    subject_to_clips = {}
    for c in all_clips:
        subject_to_clips.setdefault(subject_of(c), set()).add(c)
    print(f"loaded {len(all_clips)} clips / {len(subject_to_clips)} subjects")

    folds, test_clips = load_splits(args.splits_folder, subject_to_clips, all_clips)

    cfg = Config()
    cfg.hidden = args.hidden
    cfg.dropout = args.dropout
    cfg.mumin_loss_weight = args.mumin_weight

    results = []
    for i, fold in enumerate(folds, 1):
        if not (fold["train"] and fold["val"] and fold["holdout"]):
            continue
        results.append(train_one_fold(i, fold, args.cache, labels, cfg, args, device))
    if not results:
        raise SystemExit("no fold completed")

    keys = ["acc", "f1", "auc", "bal_acc", "majority"]
    print("\n=== mean over folds (results board row) ===")
    for k in keys:
        vals = [r["holdout"][k] for r in results]
        print(f"  {k}={np.mean(vals):.3f}±{np.std(vals):.3f}")
    print(f"  folds used: {len(results)}")

    if test_clips and args.evaluate_test:
        tl = make_gat_loader(list(test_clips), args.cache, labels, cfg,
                             args.batch_size, False)
        scores = []
        for r in results:
            m = DualHeadGAT(cfg).to(device)
            m.load_state_dict(r["model_state"])
            _, s = evaluate(m, tl, device)
            scores.append(s)
            if "y" not in locals():
                y = _
        s = np.mean(scores, 0)
        t = float(np.mean([r["threshold"] for r in results]))
        print(f"\nvirgin test: t={t:.3f}")
        for k, v in compute_metrics(y, s, t).items():
            print(f"  {k}={v:.3f}")


if __name__ == "__main__":
    main()