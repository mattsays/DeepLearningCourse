"""A fold cut after one epoch resumes from ``checkpoint.pt`` instead of starting over.

    python tests/test_resume.py          (CPU, audio branch, a few clips; ~2-4 min)

Skipped when the local ``dolos/`` copy is not there.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths, default_config  # noqa: E402
from lore.train import EpochBudgetReached, Runner  # noqa: E402

PATHS = Paths.resolve()
HAVE_DATA = (PATHS.data / "splits" / "manifest.json").exists()


def test_fold_resumes_from_checkpoint():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        paths = Paths.resolve(runs=tmp)
        cfg = default_config("audio", limit=8, epochs=2, patience=5, batch_size=2, grad_accum=1,
                             audio_train_seconds=2.0, audio_eval_seconds=2.0, run_name="resume-test")
        fold_dir = Path(tmp) / cfg.run_name / "fold1"

        try:                                         # session 1: killed after epoch 1
            Runner(cfg, paths, device="cpu", verbose=False, stop_after_epochs=1).run(folds=[1])
            raise AssertionError("the epoch budget should have stopped the run")
        except EpochBudgetReached:
            pass
        assert (fold_dir / "checkpoint.pt").exists()
        assert not (fold_dir / "metrics.json").exists()
        assert len(pd.read_csv(fold_dir / "log.csv")) == 1

        Runner(cfg, paths, device="cpu", verbose=False).run(folds=[1])   # session 2: resumes at epoch 2
        log = pd.read_csv(fold_dir / "log.csv")
        assert list(log["epoch"]) == [1, 2], log
        metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["epochs_run"] == 2
        assert not (fold_dir / "checkpoint.pt").exists()
    print("ok: resume from checkpoint")


def test_deadline_stops_before_an_epoch():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg = default_config("audio", limit=4, epochs=1, batch_size=2, grad_accum=1,
                             audio_train_seconds=2.0, audio_eval_seconds=2.0, run_name="deadline-test")
        runner = Runner(cfg, Paths.resolve(runs=tmp), device="cpu", verbose=False)
        runner.deadline = time.time() + 60          # less than the 45-minute first-epoch estimate
        try:
            runner.run(folds=[1])
            raise AssertionError("the deadline should have stopped the run")
        except EpochBudgetReached:
            pass
        assert not (Path(tmp) / cfg.run_name / "fold1" / "metrics.json").exists()
    print("ok: deadline stops before an epoch")


if __name__ == "__main__":
    test_deadline_stops_before_an_epoch()
    test_fold_resumes_from_checkpoint()
