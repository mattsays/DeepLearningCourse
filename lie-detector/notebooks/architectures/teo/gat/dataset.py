import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from config import Config
from graph_construction import build_window_graphs


class DolosDataset(Dataset):
    """
    Each item = one clip, represented as a single Data object whose nodes are
    all window-graph nodes concatenated. Edges are intra-window only.
    Clips missing either {stem}_feats.npy or {stem}_mumin.npy are skipped
    at construction time.
    """

    def __init__(self, labels_csv: str, cache_dir: str, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.cache_dir = cache_dir

        df = pd.read_csv(labels_csv)
        all_ids = df["video_id"].astype(str).tolist()
        all_labels = df["label"].astype(np.float32).tolist()

        self.video_ids, self.labels = [], []
        n_feats_missing = 0
        n_mumin_missing = 0
        for vid, lbl in zip(all_ids, all_labels):
            feats_path = os.path.join(cache_dir, f"{vid}_feats.npy")
            mumin_path = os.path.join(cache_dir, f"{vid}_mumin.npy")
            if not os.path.exists(feats_path):
                n_feats_missing += 1
                continue
            if not os.path.exists(mumin_path):
                n_mumin_missing += 1
                continue
            self.video_ids.append(vid)
            self.labels.append(lbl)

        print(f"[{split}] kept {len(self.video_ids)} clips "
              f"(skipped {n_feats_missing} missing feats, "
              f"{n_mumin_missing} missing mumin)")
        if len(self.video_ids) == 0:
            raise RuntimeError(
                f"No usable clips for split '{split}' — check that cache dir "
                f"'{cache_dir}' contains both *_feats.npy and *_mumin.npy "
                f"files matching data/labels.csv."
            )

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Data:
        vid = self.video_ids[idx]
        feats = np.load(os.path.join(self.cache_dir, f"{vid}_feats.npy"))
        mumin = np.load(os.path.join(self.cache_dir, f"{vid}_mumin.npy"))

        # Accept both per-clip (41,) and legacy per-frame (T, 41) layouts
        if mumin.ndim == 2:
            mumin = mumin.mean(axis=0)
        mumin = np.asarray(mumin, dtype=np.float32).reshape(-1)

        window_graphs = build_window_graphs(feats, self.cfg)
        if not window_graphs:
            window_graphs = [Data(
                x=torch.zeros(1, self.cfg.node_feat_dim),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                node_type=torch.zeros(1, dtype=torch.long),
            )]

        xs, ets, nts = [], [], []
        offset = 0
        for g in window_graphs:
            xs.append(g.x)
            ets.append(g.edge_index + offset)
            nts.append(g.node_type)
            offset += g.num_nodes

        data = Data(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(ets, dim=1),
            node_type=torch.cat(nts, dim=0),
        )
        data.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        data.mumin = torch.from_numpy(mumin).unsqueeze(0)   # (1, num_mumin)
        return data


def make_loaders(labels_csv: str, cache_dir: str, cfg: Config,
                 val_split: float = 0.15, seed: int = 42):
    """
    Builds train/val loaders. The underlying dataset is split with Subset,
    so both loaders share the same filtered list of video_ids.
    """
    full = DolosDataset(labels_csv, cache_dir, cfg)
    n = len(full)
    n_val = max(1, int(n * val_split))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = torch.utils.data.Subset(full, train_idx)
    val_ds = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader