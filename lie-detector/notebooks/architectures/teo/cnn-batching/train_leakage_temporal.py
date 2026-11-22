import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset_temporal import make_temporal_loaders
from model_temporal import TemporalCNN


def evaluate(model, loader, device, pos_weight):
    model.eval()
    all_scores, all_labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y, mumin, lengths in loader:
            x, y, mumin, lengths = x.to(device), y.to(device), mumin.to(device), lengths
            lie_logit, mumin_pred = model(x, lengths, return_mumin=True)
            loss = (F.binary_cross_entropy_with_logits(lie_logit, y, pos_weight=pos_weight)
                    + 0.3 * F.mse_loss(mumin_pred, mumin))
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            all_scores.append(torch.sigmoid(lie_logit).cpu())
            all_labels.append(y.cpu())
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(labels, (scores > 0.5).astype(int))
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return total_loss / max(n, 1), acc, auc


def train(labels_csv, cache_dir, out_dir, epochs=30, lr=3e-4,
          mumin_weight=0.0, patience=10, batch_size=16, hidden=64, dropout=0.3):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cpu")

    train_loader, val_loader, train_labels = make_temporal_loaders(
        labels_csv, cache_dir, batch_size=batch_size)
    n_pos = float((train_labels == 1).sum())
    n_neg = float((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float)
    print(f"train: pos={int(n_pos)} neg={int(n_neg)} pos_weight={pos_weight.item():.3f}")

    model = TemporalCNN(hidden=hidden, dropout=dropout).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    best_val = float("inf"); best_ep = -1; patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        for x, y, mumin, lengths in train_loader:
            x, y, mumin = x.to(device), y.to(device), mumin.to(device)
            optimizer.zero_grad()
            lie_logit, mumin_pred = model(x, lengths, return_mumin=True)
            loss = (F.binary_cross_entropy_with_logits(lie_logit, y, pos_weight=pos_weight)
                    + mumin_weight * F.mse_loss(mumin_pred, mumin))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * y.size(0); n_train += y.size(0)
        train_loss /= max(n_train, 1)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, pos_weight)
        scheduler.step(val_loss)

        print(f"[epoch {epoch:03d}] train={train_loss:.4f} "
              f"val={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f}")

        if val_loss < best_val:
            best_val, best_ep, patience_ctr = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(),
                        "hidden": hidden, "dropout": dropout,
                        "epoch": epoch, "val_loss": val_loss,
                        "val_acc": val_acc, "val_auc": val_auc},
                       os.path.join(out_dir, "best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stop at epoch {epoch} (best {best_ep})")
                break

    print(f"done. best epoch={best_ep} val_loss={best_val:.4f}")
    return best_ep, best_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="runs/cnn_baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mumin_weight", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()
    train(args.labels, args.cache, args.out, epochs=args.epochs, lr=args.lr,
          mumin_weight=args.mumin_weight, patience=args.patience,
          batch_size=args.batch_size, hidden=args.hidden, dropout=args.dropout)