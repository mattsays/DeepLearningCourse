"""Pilot: train fold 1 only, look at it, and only then start folds 2-5 (by hand).

    python scripts/check_pilot.py --runs video-lora-r8 audio-lora-r8

:func:`describe_fold` is the performance report of the fold (metrics, confusion matrix, learning
curve). :func:`check_pilot` adds sanity checks — a fold *passes* when it shows that the pipeline works, not that the model is good:

* blocking  the fold finished (``metrics.json``); the train loss went down; the predictions are not
            constant; the best val AUC is at least ``min_val_auc`` (default 0.5 = better than chance).
* warning   holdout AUC below 0.5, or holdout accuracy below the majority baseline (one fold of ~280
            clips is noisy: worth a look, not a reason to stop).

It also estimates how long the remaining folds will take from the pilot fold's real time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PilotReport:
    run: str
    fold: int
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    seconds: float = 0.0                                  # real time of the fold (estimate input)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def lines(self) -> list[str]:
        head = f"{self.run} fold {self.fold}: {'PASS' if self.ok else 'FAIL'}"
        facts = "  ".join(f"{k} {v}" for k, v in self.facts.items())
        return [head] + ([f"    {facts}"] if facts else []) + [f"    ERROR   {e}" for e in self.errors] + \
               [f"    warning {w}" for w in self.warnings]


def check_fold(run_dir: Path | str, fold: int = 1, min_val_auc: float = 0.5) -> PilotReport:
    run_dir = Path(run_dir)
    fold_dir = run_dir / f"fold{fold}"
    report = PilotReport(run=run_dir.name, fold=fold)
    marker = fold_dir / "metrics.json"
    if not marker.exists():
        state = "interrupted (checkpoint.pt present)" if (fold_dir / "checkpoint.pt").exists() else "not run"
        report.fail(f"fold not finished: {state}")
        return report

    m = json.loads(marker.read_text(encoding="utf-8"))
    log = pd.read_csv(fold_dir / "log.csv")
    hold = m["holdout"]
    fold_seconds = m["seconds"]["train_total"] + m["seconds"]["val_per_epoch"] * m["epochs_run"]
    report.facts = {
        "epochs": f"{m['epochs_run']} (best {m['best_epoch']})",
        "val AUC": f"{m['best_val_auc']:.3f}",
        "holdout AUC": f"{hold['auc']:.3f}",
        "holdout ACC": f"{hold['acc'] * 100:.1f} (majority {m['majority_holdout'] * 100:.1f})",
        "peak GPU": f"{m['peak_gpu_gb']} GB",
        "fold time": f"{fold_seconds / 3600:.2f} h",
    }
    report.seconds = fold_seconds

    losses = log["train_loss"].to_numpy()
    if len(losses) >= 2 and not losses[1:].min() < losses[0]:
        report.fail(f"train loss never went below epoch 1 ({losses[0]:.4f}): the model is not learning")
    probs = pd.read_csv(fold_dir / "holdout_preds.csv")["prob"].to_numpy()
    if float(np.std(probs)) < 1e-4:
        report.fail(f"constant holdout predictions (p = {probs.mean():.4f} for every clip)")
    if not m["best_val_auc"] >= min_val_auc:
        report.fail(f"best val AUC {m['best_val_auc']:.3f} < {min_val_auc} (not better than chance)")

    if hold["auc"] == hold["auc"] and hold["auc"] < 0.5:
        report.warnings.append(f"holdout AUC {hold['auc']:.3f} below chance on this fold")
    if hold["acc"] < m["majority_holdout"]:
        report.warnings.append("holdout accuracy below the majority baseline on this fold")
    return report


def check_pilot(runs_dir: Path | str, runs: list[str], fold: int = 1, min_val_auc: float = 0.5,
                remaining_folds: int = 4, gpus: int = 1) -> tuple[bool, list[str]]:
    """Check fold ``fold`` of every run; returns (all passed, printable lines incl. a time estimate)."""
    reports = [check_fold(Path(runs_dir) / run, fold, min_val_auc) for run in runs]
    lines = [line for r in reports for line in r.lines()]
    seconds = sum(r.seconds for r in reports)
    if all(r.ok for r in reports) and seconds:
        hours = seconds / 3600 * remaining_folds / max(1, gpus)
        lines.append(f"estimate for the other {remaining_folds} folds of these runs on {gpus} GPU(s): "
                     f"~{hours:.1f} h")
    ok = all(r.ok for r in reports)
    lines.append("PILOT PASSED: review the performance, then start folds 2-5 by hand" if ok else
                 "PILOT FAILED: fix the errors above before spending GPU hours on folds 2-5")
    return ok, lines


def _metric_line(label: str, m: dict) -> str:
    c = m["confusion"]
    return (f"  {label:<24} ACC {m['acc'] * 100:5.1f}  F1 {m['f1'] * 100:5.1f}  AUC {m['auc'] * 100:5.1f}  "
            f"bal {m['balanced_acc'] * 100:5.1f}  P {m['precision'] * 100:5.1f}  R {m['recall'] * 100:5.1f}  "
            f"[tn {c['tn']} fp {c['fp']} fn {c['fn']} tp {c['tp']}]")


def describe_fold(run_dir: Path | str, fold: int = 1) -> str:
    """Performance of one finished fold: val/holdout metrics, baselines and the learning curve."""
    run_dir = Path(run_dir)
    fold_dir = run_dir / f"fold{fold}"
    if not (fold_dir / "metrics.json").exists():
        return f"{run_dir.name} fold {fold}: not finished"
    m = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    log = pd.read_csv(fold_dir / "log.csv")
    sizes = m["sizes"]
    lines = [f"{run_dir.name} — fold {fold}  (train {sizes['train']}, val {sizes['val']}, holdout {sizes['holdout']} "
             f"clips; positive = deception)",
             f"  best epoch {m['best_epoch']} of {m['epochs_run']}, threshold {m['threshold']:.3f} (chosen on val)",
             _metric_line("val @ threshold", m["val"]),
             _metric_line("holdout @ threshold", m["holdout"]),
             _metric_line("holdout @ 0.5", m["holdout_at_0.5"]),
             f"  majority baseline on holdout: ACC {m['majority_holdout'] * 100:.1f}",
             "  learning curve:"]
    curve = log[["epoch", "train_loss", "val_loss", "val_auc", "train_s"]].copy()
    curve["best"] = np.where(curve["epoch"] == m["best_epoch"], "*", "")
    lines += ["    " + row for row in curve.to_string(index=False).splitlines()]
    preds = pd.read_csv(fold_dir / "holdout_preds.csv")
    if "gender" in preds and preds["gender"].nunique() > 1:
        thr = m["threshold"]
        per = (preds.assign(ok=(preds["prob"] >= thr).astype(int) == preds["y_true"])
               .groupby("gender")["ok"].agg(["mean", "size"]))
        lines.append("  holdout ACC by gender: " + ", ".join(f"{g} {r['mean'] * 100:.1f} (n={int(r['size'])})"
                                                              for g, r in per.iterrows()))
    return "\n".join(lines)
