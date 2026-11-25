import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """
    A small 1D CNN over per-frame features [T, 53].
    Learns temporal filters (e.g., "jaw tightens over 0.3s") that the
    averaged GAT representation could not express.
    """

    def __init__(self, feat_dim=53, hidden=64, num_mumin=41, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.lie_head   = nn.Linear(hidden, 1)
        self.mumin_head = nn.Linear(hidden, num_mumin)

    def forward(self, x, lengths, return_mumin=False):
        # x: [B, T, D]  lengths: [B]
        h = self.net(x.transpose(1, 2))            # [B, hidden, T]
        # Masked mean over time (ignores padded frames)
        mask = (torch.arange(h.size(2), device=h.device)[None, :]
                < lengths[:, None]).float().unsqueeze(1)   # [B, 1, T]
        h = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [B, hidden]

        lie = self.lie_head(h).squeeze(-1)
        if return_mumin:
            return lie, self.mumin_head(h)
        return lie