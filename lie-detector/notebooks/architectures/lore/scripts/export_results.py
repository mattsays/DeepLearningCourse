"""Copy finished runs into ``results/`` — the label-free version that is committed to git.

    python scripts/export_results.py                          # every run in runs/
    python scripts/export_results.py --runs video-lora-r8 fusion-video-lora-r8+audio-lora-r8

``runs/`` (git-ignored) keeps everything a run wrote. ``results/`` gets:

* the summaries unchanged: run.json, summary.json, metrics.json, log.csv, fusion_report.json,
  comparison.txt, board_row.md, report.txt;
* every prediction CSV without the DOLOS annotation columns (``y_true``, ``subject``, ``gender``):
  the annotations are not redistributable, and anyone holding DOLOS joins them back on ``clip_id``;
* no weights or checkpoints (``*.pt``).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

LORE_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FILES = {"run.json", "summary.json", "metrics.json", "log.csv", "fusion_report.json",
                 "comparison.txt", "board_row.md", "report.txt"}
ANNOTATION_COLUMNS = {"y_true", "subject", "gender"}


def is_prediction_csv(path: Path) -> bool:
    return path.suffix == ".csv" and ("preds" in path.name or path.name.startswith("cv_"))


def strip_annotations(src: Path, dst: Path) -> None:
    with src.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        keep = [c for c in reader.fieldnames if c not in ANNOTATION_COLUMNS]
        rows = [{c: row[c] for c in keep} for row in reader]
    with dst.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=keep, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_run(run_dir: Path, out_dir: Path) -> tuple[int, int]:
    copied = stripped = 0
    for src in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        dst = out_dir / src.relative_to(run_dir)
        if src.name in SUMMARY_FILES:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1
        elif is_prediction_csv(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            strip_annotations(src, dst)
            stripped += 1
    return copied, stripped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=LORE_ROOT / "runs")
    p.add_argument("--out-dir", type=Path, default=LORE_ROOT / "results")
    p.add_argument("--runs", nargs="*", help="run names (default: every folder in --runs-dir)")
    args = p.parse_args()

    names = args.runs or sorted(d.name for d in args.runs_dir.iterdir() if d.is_dir())
    for name in names:
        run_dir = args.runs_dir / name
        if not run_dir.is_dir():
            print(f"missing run: {run_dir}", file=sys.stderr)
            return 1
        copied, stripped = export_run(run_dir, args.out_dir / name)
        print(f"{name}: {copied} summary files, {stripped} prediction CSVs (annotation columns removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
