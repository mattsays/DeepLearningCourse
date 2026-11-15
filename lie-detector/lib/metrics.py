"""Metrics shared by every direction, so the results board is comparable.

``evaluate`` always returns the same keys, whatever the model: if four people
compute F1 in four ways, the table means nothing.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def majority_baseline(y_true) -> float:
    """Accuracy of always predicting the majority class *of these clips*."""
    y = np.asarray(y_true)
    if y.size == 0:
        return float("nan")
    return float(max(y.mean(), 1 - y.mean()))


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Binary deception metrics. Positive class is ``deception`` (y=1).

    Returns accuracy, F1, AUC, balanced accuracy, precision, recall and the
    confusion matrix. AUC is NaN if only one class is present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else float("nan"),
        "balanced_acc": float((recall_pos + recall_neg) / 2),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_pos),
        "threshold": float(threshold),
        "n": int(y_true.size),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_threshold(y_true, y_prob) -> float:
    """Threshold maximising balanced accuracy on a *validation* split.

    Use it instead of 0.5 when the split is imbalanced (our training folds are
    ~56% deception, the holdouts ~47%).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.unique(np.round(y_prob, 4))
    best, best_score = 0.5, -1.0
    for thr in grid:
        score = evaluate(y_true, y_prob, threshold=float(thr))["balanced_acc"]
        if score > best_score:
            best, best_score = float(thr), score
    return best


def format_row(direction: str, results: dict, *, owner: str = "—", method: str = "—",
               params: int | str = "—", majority: float | None = None,
               runtime: str = "—") -> str:
    """One markdown row for the results board, in the board's column order.

    The board lives in ``README.md``, section ``## Results board``. ``results`` holds the
    cross-validation summary in 0-1: ``acc`` (mean), optional ``acc_std`` (spread over the
    folds), ``f1``, ``auc`` and ``balanced_acc``. ``majority`` is the baseline computed on
    the same clips, also 0-1.

    >>> format_row("tia", {"acc": .61, "acc_std": .03, "f1": .62, "auc": .65,
    ...                    "balanced_acc": .60}, owner="A", method="PECL",
    ...            params=1200000, majority=.524, runtime="9 min")
    '| tia | A | PECL | 1,200,000 | 61.0 ± 3.0 | 62.0 | 65.0 | 60.0 | 52.4 | 9 min |'
    """
    spread = results.get("acc_std")
    acc = f"{results['acc'] * 100:.1f}"
    if spread is not None:
        acc += f" ± {spread * 100:.1f}"
    maj = "—" if majority is None else f"{majority * 100:.1f}"
    shown = f"{params:,}" if isinstance(params, int) else str(params)
    return (f"| {direction} | {owner} | {method} | {shown} | {acc} | {results['f1'] * 100:.1f} "
            f"| {results['auc'] * 100:.1f} | {results['balanced_acc'] * 100:.1f} | {maj} | {runtime} |")
