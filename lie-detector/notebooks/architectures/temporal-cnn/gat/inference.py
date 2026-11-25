import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from config import Config
from graph_construction import build_window_graphs
from model import DualHeadGAT


@torch.no_grad()
def score_video(checkpoint_path: str, video_id: str, cache_dir: str) -> float:
    """
    Loads a trained DualHeadGAT and returns a lie probability for the clip.
    Assumes cached features exist — swap this out for live extraction later.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**ckpt["cfg"])
    model = DualHeadGAT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feats = np.load(os.path.join(cache_dir, f"{video_id}_feats.npy"))
    graphs = build_window_graphs(feats, cfg)
    if not graphs:
        return 0.5

    xs, ets, nts = [], [], []
    offset = 0
    for g in graphs:
        xs.append(g.x)
        ets.append(g.edge_index + offset)
        nts.append(g.node_type)
        offset += g.num_nodes

    data = Data(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(ets, dim=1),
        node_type=torch.cat(nts, dim=0),
    )
    batch = Batch.from_data_list([data])
    prob = torch.sigmoid(model(batch, return_mumin=False)).item()
    return prob


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--video_id", required=True)
    p.add_argument("--cache", default="data/cache")
    args = p.parse_args()
    print(f"lie_probability={score_video(args.ckpt, args.video_id, args.cache):.4f}")