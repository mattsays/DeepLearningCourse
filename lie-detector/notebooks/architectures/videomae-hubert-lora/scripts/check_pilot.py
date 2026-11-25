"""Check the pilot fold of one or more runs before running the other folds (see ``lore/pilot.py``).

    python scripts/check_pilot.py --runs audio-lora-r8 video-lora-r8
    python scripts/check_pilot.py --runs video-lora-r8 --min-val-auc 0.45 --gpus 2

Exit code 0 when every run passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import Paths  # noqa: E402
from lore.pilot import check_pilot  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run names under the runs directory")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--min-val-auc", type=float, default=0.5)
    p.add_argument("--remaining-folds", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1, help="for the time estimate only")
    args = p.parse_args(argv)
    runs_dir = args.runs_dir or Paths.resolve().runs
    ok, lines = check_pilot(runs_dir, args.runs, args.fold, args.min_val_auc, args.remaining_folds, args.gpus)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
