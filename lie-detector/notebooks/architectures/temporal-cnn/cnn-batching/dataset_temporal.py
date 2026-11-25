import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TemporalDataset(Dataset):
    """
    Returns (feats [T, D], label scalar, mumin [41]).
    Skips clips missing either _feats.npy or _mumin.npy.
    """

    def __init__(self, labels_csv: str, cache_dir: str):
        df = pd.read_csv(labels_csv)
        self.cache_dir = cache_dir
        self.video_ids, self.labels = [], []
        n_missing_feats = n_missing_mumin = 0

        for _, row in df.iterrows():
            vid = str(row["video_id"])
            feat_p = os.path.join(cache_dir, f"{vid}_feats.npy")
            mum_p  = os.path.join(cache_dir, f"{vid}_mumin.npy")
            if not os.path.exists(feat_p):
                n_missing_feats += 1; continue
            if not os.path.exists(mum_p):
                n_missing_mumin += 1; continue
            self.video_ids.append(vid)
            self.labels.append(float(row["label"]))

        print(f"kept {len(self.video_ids)} clips "
              f"(skipped {n_missing_feats} no-feats, {n_missing_mumin} no-mumin)")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        feats = np.load(os.path.join(self.cache_dir, f"{vid}_feats.npy"))
        mumin = np.load(os.path.join(self.cache_dir, f"{vid}_mumin.npy"))
        if mumin.ndim == 2:
            mumin = mumin.mean(axis=0)
        return (
            torch.from_numpy(feats.astype(np.float32)),          # [T, 53]
            torch.tensor(self.labels[idx], dtype=torch.float),    # scalar
            torch.from_numpy(mumin.astype(np.float32)),           # [41]
        )


def collate_temporal(batch):
    """Pad variable-length sequences to max T in batch."""
    feats_list, labels, mumins = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    max_len = int(lengths.max())
    D = feats_list[0].shape[1]
    padded = torch.zeros(len(batch), max_len, D)
    for i, f in enumerate(feats_list):
        padded[i, :f.shape[0]] = f
    return padded, torch.stack(labels), torch.stack(mumins), lengths


def make_temporal_loaders(labels_csv, cache_dir, batch_size=16,
                          val_split=0.15, seed=42):
    full = TemporalDataset(labels_csv, cache_dir)
    n = len(full)
    n_val = max(1, int(n * val_split))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = torch.utils.data.Subset(full, train_idx)
    val_ds   = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_temporal, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_temporal, num_workers=0)

    train_labels = np.array([full.labels[i] for i in train_idx], dtype=np.float32)
    return train_loader, val_loader, train_labels