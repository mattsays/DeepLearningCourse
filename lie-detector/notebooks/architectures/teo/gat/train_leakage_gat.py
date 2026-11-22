import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.loader import DataLoader

from config import Config
from dataset import make_loaders
from model import DualHeadGAT


def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            lie_logit, mumin_pred = model(batch, return_mumin=True)
            loss = (F.binary_cross_entropy_with_logits(
                        lie_logit, batch.y, pos_weight=pos_weight)
                    + 0.3 * F.mse_loss(mumin_pred, batch.mumin))
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
            all_scores.append(torch.sigmoid(lie_logit).cpu())
            all_labels.append(batch.y.cpu())
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(labels, (scores > 0.5).astype(int))
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return total_loss / n, acc, auc


def _labels_from_subset(loader) -> np.ndarray:
    """
    Reads the label list of a DataLoader whose dataset is a Subset, without
    triggering __getitem__ (which would load and graph-ify every clip).
    """
    subset = loader.dataset           # torch.utils.data.Subset
    base = subset.dataset             # DolosDataset
    indices = subset.indices
    return np.array([base.labels[i] for i in indices], dtype=np.float32)


def train(cfg: Config, labels_csv: str, cache_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(42)

    train_loader, val_loader = make_loaders(labels_csv, cache_dir, cfg)

    # pos_weight from the filtered train labels — instant, no data loading
    train_labels = _labels_from_subset(train_loader)
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float)
    print(f"train: pos={int(n_pos)}, neg={int(n_neg)}, "
          f"pos_weight={pos_weight.item():.3f}")

    model = DualHeadGAT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float("inf")
    best_epoch = -1
    patience_count = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            lie_logit, mumin_pred = model(batch, return_mumin=True)
            loss = (F.binary_cross_entropy_with_logits(
                        lie_logit, batch.y, pos_weight=pos_weight)
                    + cfg.mumin_loss_weight * F.mse_loss(mumin_pred, batch.mumin))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            n_train += batch.num_graphs
        train_loss = train_loss / max(n_train, 1)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, pos_weight)
        scheduler.step()

        print(f"[epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.3f} "
              f"val_auc={val_auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }, os.path.join(out_dir, "best.pt"))
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_loss {best_val_loss:.4f})")
                break

    print(f"done. best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
    return best_epoch, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--out", default="runs/gat_baseline")
    args = parser.parse_args()

    cfg = Config()
    train(cfg, args.labels, args.cache, args.out)