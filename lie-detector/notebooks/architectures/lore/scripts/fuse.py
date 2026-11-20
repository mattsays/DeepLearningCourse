"""Late fusion + final report of the `lore` direction.

    python scripts/fuse.py --video-run video-lora-r8 --audio-run audio-lora-r8 --owner "<your name>"

Prints video-only / audio-only / fused rows (CV mean ± std, pooled, majority baseline, per gender)
and the results-board row, and writes <runs>/fusion-<video>+<audio>/:
    preds/cv_lore.csv (+ cv_lore-video.csv, cv_lore-audio.csv)   clip_id,y_true,y_prob,y_pred,fold
    board_row.md, comparison.txt, fusion_report.json
With a single --video-run or --audio-run it reports that branch alone. --folds 1 restricts it to
the pilot fold (then write it to its own --out-dir: it is not the board number).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths  # noqa: E402
from lore.fusion import fuse_runs  # noqa: E402


def parse_folds(text: str | None) -> list[int] | None:
    """'1' / '1,3' / '2-5' -> list of folds (same syntax as run_cv.py --folds)."""
    if not text:
        return None
    folds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        folds += list(range(int(lo), int(hi or lo) + 1))
    return folds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-run", default=None, help="run name (or path) of the video branch")
    p.add_argument("--audio-run", default=None, help="run name (or path) of the audio branch")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--lib-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--owner", default="—")
    p.add_argument("--folds", default=None, help="only these folds, e.g. '1' for the pilot (default: all done)")
    args = p.parse_args(argv)
    if not (args.video_run or args.audio_run):
        p.error("give --video-run and/or --audio-run")

    paths = Paths.resolve(lib=args.lib_dir, runs=args.runs_dir)

    def run_dir(name):
        if not name:
            return None
        candidate = Path(name)
        return candidate if candidate.is_dir() else paths.runs / name

    video, audio = run_dir(args.video_run), run_dir(args.audio_run)
    label = "+".join(d.name for d in (video, audio) if d)
    out_dir = args.out_dir or paths.runs / f"fusion-{label}"
    result = fuse_runs(video, audio, out_dir, paths.lib, owner=args.owner, folds=parse_folds(args.folds))
    print(result["table"])
    print("\nresults-board row (lie-detector/README.md, ## Results board):")
    print(result["board_row"])
    print(f"\nwritten to {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
