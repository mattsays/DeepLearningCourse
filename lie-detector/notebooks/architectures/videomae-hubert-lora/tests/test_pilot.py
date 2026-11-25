"""The pilot gate passes a healthy fold and blocks a broken one (synthetic run folders, no data needed).

    python tests/test_pilot.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.pilot import check_fold, check_pilot  # noqa: E402


def make_fold(runs: Path, run: str, losses, probs, val_auc=0.62, hold_auc=0.58, acc=0.6, majority=0.53):
    fold = runs / run / "fold1"
    fold.mkdir(parents=True)
    metrics = {"epochs_run": len(losses), "best_epoch": 1, "best_val_auc": val_auc,
               "holdout": {"auc": hold_auc, "acc": acc}, "majority_holdout": majority, "peak_gpu_gb": 6.1,
               "seconds": {"train_total": 600.0 * len(losses), "val_per_epoch": 60.0}}
    (fold / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "train_loss": losses}).to_csv(fold / "log.csv", index=False)
    pd.DataFrame({"prob": probs}).to_csv(fold / "holdout_preds.csv", index=False)


def test_pilot_gate():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        make_fold(runs, "good", [0.70, 0.66, 0.61], rng.uniform(0.2, 0.8, 50))
        make_fold(runs, "flat-loss", [0.69, 0.70, 0.71], rng.uniform(0.2, 0.8, 50))
        make_fold(runs, "constant", [0.70, 0.66], np.full(50, 0.47))
        make_fold(runs, "chance", [0.70, 0.66], rng.uniform(0.2, 0.8, 50), val_auc=0.45, hold_auc=0.40, acc=0.4)
        (runs / "interrupted" / "fold1").mkdir(parents=True)
        (runs / "interrupted" / "fold1" / "checkpoint.pt").write_bytes(b"x")

        good = check_fold(runs / "good")
        assert good.ok and not good.warnings, good.lines()
        assert not check_fold(runs / "flat-loss").ok
        assert not check_fold(runs / "constant").ok
        chance = check_fold(runs / "chance")
        assert not chance.ok and len(chance.warnings) == 2, chance.lines()
        assert check_fold(runs / "chance", min_val_auc=0.4).ok                    # threshold is configurable
        assert "interrupted" in check_fold(runs / "interrupted").errors[0]
        assert "not run" in check_fold(runs / "missing").errors[0]

        ok, lines = check_pilot(runs, ["good"], remaining_folds=4, gpus=2)
        assert ok and "~1.1 h" in lines[-2], lines              # (3*600 + 3*60) s = 0.55 h, * 4 folds / 2 GPUs
        ok, lines = check_pilot(runs, ["good", "constant"])
        assert not ok and lines[-1].startswith("PILOT FAILED")
    print("ok: pilot gate")


if __name__ == "__main__":
    test_pilot_gate()
