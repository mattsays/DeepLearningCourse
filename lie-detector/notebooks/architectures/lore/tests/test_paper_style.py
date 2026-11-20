"""The official protocol records the paper-style test F1 and the test predictions of that epoch,
and ``fuse_runs`` turns them into a paper-style fused line.

    python tests/test_paper_style.py     (CPU, audio branch, a few clips; ~2-4 min)

Skipped when the local ``dolos/`` copy is not there.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths, default_config  # noqa: E402
from lore.fusion import fuse_runs  # noqa: E402
from lore.train import Runner  # noqa: E402

PATHS = Paths.resolve()
HAVE_DATA = (PATHS.data / "splits" / "manifest.json").exists()


def test_official_run_saves_paper_style_f1_and_preds():
    if not HAVE_DATA:
        print("skipped: no local dolos/")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cfg = default_config("audio", protocol="official", limit=8, epochs=2, patience=5, batch_size=2,
                             grad_accum=1, audio_train_seconds=2.0, audio_eval_seconds=2.0,
                             run_name="paper-style-test")
        Runner(cfg, Paths.resolve(runs=tmp), device="cpu", verbose=False).run(folds=[1])
        fold_dir = Path(tmp) / cfg.run_name / "fold1"

        metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
        style = metrics["paper_style"]
        assert style["test_f1_at_05"] is not None, style
        log = pd.read_csv(fold_dir / "log.csv")
        assert "test_f1_at_05" in log.columns

        preds = pd.read_csv(fold_dir / "paper_test_preds.csv")
        assert len(preds) == metrics["sizes"]["holdout"], (len(preds), metrics["sizes"])
        assert int(preds.epoch.iloc[0]) == style["epoch"], (preds.epoch.iloc[0], style)
        holdout = pd.read_csv(fold_dir / "holdout_preds.csv")
        assert set(preds.clip_id) == set(holdout.clip_id)

        # the same run as both branches is enough to exercise the fused paper-style line
        video_dir = Path(tmp) / "video-official"
        shutil.copytree(Path(tmp) / cfg.run_name, video_dir)
        out = fuse_runs(video_dir, Path(tmp) / cfg.run_name, Path(tmp) / "fusion-official", PATHS.lib)
        assert "  fused  ACC" in out["table"], out["table"]
        assert " F1 " in out["table"]
    print("ok: paper-style F1, test predictions and fused line")


if __name__ == "__main__":
    test_official_run_saves_paper_style_f1_and_preds()
