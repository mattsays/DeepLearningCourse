"""Cross-validate one branch of the `lore` direction on the frozen subject-level folds.

    python scripts/run_cv.py --modality video                  # main video run, folds 1-5
    python scripts/run_cv.py --modality audio                  # main audio run
    python scripts/run_cv.py --modality video --debug          # ~200 clips, 2 epochs
    python scripts/run_cv.py --modality video --folds 3-5      # resume / split across sessions
    python scripts/run_cv.py --modality video --head-only      # ablation: no LoRA
    python scripts/run_cv.py --modality video --lora-mlp-last-n 4 --dora   # more ablations
    python scripts/run_cv.py --modality audio --protocol official           # paper's 3 folds

Results: <runs>/<run name>/fold<k>/ (metrics.json, val/holdout predictions, LoRA weights, log.csv).
A fold whose metrics.json exists is skipped, so re-running the same command resumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths, default_config  # noqa: E402


def parse_folds(text: str | None) -> list[int] | None:
    if not text:
        return None
    folds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            folds += list(range(int(lo), int(hi) + 1))
        else:
            folds.append(int(part))
    return folds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=["video", "audio"], required=True)
    p.add_argument("--protocol", choices=["subject", "official"], default=None,
                   help="subject: the 5 frozen subject-level folds (default, the board); official: the "
                        "paper's 3 clip-level folds, for a separate paper comparison only")
    p.add_argument("--folds", default=None, help="e.g. '1-5', '2', '1,3' (default: all)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    # paths
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--lib-dir", type=Path, default=None)
    p.add_argument("--faces-root", type=Path, default=None)
    p.add_argument("--audio-dir", type=Path, default=None)
    p.add_argument("--runs-dir", type=Path, default=None)
    # model / LoRA
    p.add_argument("--backbone", default=None)
    p.add_argument("--head-only", action="store_true", default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument("--lora-mlp-last-n", type=int, default=None)
    p.add_argument("--dora", dest="use_dora", action="store_true", default=None)
    p.add_argument("--head-dropout", type=float, default=None)
    # data
    p.add_argument("--face-cache", default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--eval-tta", action="store_true", default=None)
    p.add_argument("--audio-train-seconds", type=float, default=None)
    p.add_argument("--audio-eval-seconds", type=float, default=None)
    p.add_argument("--subject-balanced", action="store_true", default=None)
    # optimisation
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-lora", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-checkpointing", action="store_true", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    # debugging
    p.add_argument("--debug", action="store_true", default=None)
    p.add_argument("--limit", type=int, default=None, help="N clips per split (smoke test)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in {"modality", "folds", "device", "data_dir", "lib_dir", "faces_root",
                              "audio_dir", "runs_dir"}}
    cfg = default_config(args.modality, **overrides)
    paths = Paths.resolve(data=args.data_dir, lib=args.lib_dir, faces_root=args.faces_root,
                          face_cache=cfg.face_cache, audio=args.audio_dir, runs=args.runs_dir)

    from lore.train import EpochBudgetReached, Runner

    try:
        Runner(cfg, paths, device=args.device).run(parse_folds(args.folds))
    except EpochBudgetReached as stop:
        print(f"\nSTOPPED: {stop}", flush=True)
        return 3                                      # not an error: re-run the same command to resume
    return 0


if __name__ == "__main__":
    sys.exit(main())
