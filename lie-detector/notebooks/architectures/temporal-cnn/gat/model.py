# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import Config


class DualHeadGAT(nn.Module):
    """
    Dual-head GAT with per-branch normalization.

    Fixes applied:
      - LayerNorm after landmark projection and after audio projection,
        separately. Landmark coords are ~30x smaller than openSMILE LLDs,
        so without this the audio branch dominates gradients.
      - Pre-norm residual structure (h + F(...)) instead of post-norm,
        which is far more stable for small models on small batches.
      - Dropout between GAT layers to slow memorization of train noise.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.node_feat_dim

        # Per-branch projection + normalization
        self.lm_proj = nn.Linear(d, cfg.hidden)
        self.au_proj = nn.Linear(d, cfg.hidden)
        self.lm_norm = nn.LayerNorm(cfg.hidden)
        self.au_norm = nn.LayerNorm(cfg.hidden)
        self.type_emb = nn.Embedding(2, cfg.hidden)

        # GAT stack
        self.gat1 = GATConv(cfg.hidden, cfg.hidden // cfg.heads,
                            heads=cfg.heads, dropout=cfg.dropout)
        self.gat2 = GATConv(cfg.hidden, cfg.hidden, heads=1,
                            dropout=cfg.dropout)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

        # Lie head (training + inference)
        self.lie_head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, 1),
        )

        # Aux MUMIN head (training only)
        self.mumin_head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.num_mumin),
        )

    def encode(self, data):
        x, edge_index, batch, node_type = (
            data.x, data.edge_index, data.batch, data.node_type
        )
        lm_mask = node_type == 0
        au_mask = node_type == 1

        h = torch.zeros(x.size(0), self.cfg.hidden, device=x.device)
        if lm_mask.any():
            h[lm_mask] = self.lm_norm(self.lm_proj(x[lm_mask]))
        if au_mask.any():
            h[au_mask] = self.au_norm(self.au_proj(x[au_mask]))
        h = h + self.type_emb(node_type)

        # Pre-norm residual
        h = h + self.drop1(F.elu(self.gat1(h, edge_index)))
        h = h + self.drop2(F.elu(self.gat2(h, edge_index)))

        return global_mean_pool(h, batch)

    def forward(self, data, return_mumin: bool = False):
        clip_emb = self.encode(data)
        lie_logit = self.lie_head(clip_emb).squeeze(-1)
        if return_mumin:
            return lie_logit, self.mumin_head(clip_emb)
        return lie_logit

    @torch.no_grad()
    def predict_proba(self, data):
        return torch.sigmoid(self.forward(data, return_mumin=False))