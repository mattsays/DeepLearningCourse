#!/usr/bin/env python3
"""Measure what a subject-level split hands to each fold.

The split files say *who* is in a fold; they do not say what that fold is like.
This script reads a protocol (v1 or v2), joins it to ``dataset.csv`` and prints
the composition of every holdout:

- clips, subjects, deception rate, female share;
- **purity**: share of clips in subjects with <=10% or >=90% of one label;
- **MAD**: clip-weighted mean ``|subject rate - 0.5|``;
- **ICC**: how much of the label variance is between subjects (one-way ANOVA);
- **n_eff**: clips divided by the design effect ``1 + (m-1) * ICC`` — the
  number of independent observations the fold really has;
- **oracle AUC**: AUC of using each subject's true rate as the prediction.
  This is an oracle, not a model: it says how much of the fold is explained by
  *who is speaking*, which is the shortcut a face encoder can learn;
- **episode overlap**: holdout clips whose episode also appears in that fold's
  training clips (captains included);
- **clean episodes**: captain-free episodes with no training clip at all.

Usage
-----
    python scripts/report_splits.py data/processed/splits data/processed/splits_v2
    python scripts/report_splits.py --data-dir /content/drive/MyDrive/dolos \
        /content/drive/MyDrive/dolos/splits /content/drive/MyDrive/dolos/splits_v2 \
        --csv splits_report.csv

Two folders are compared side by side; one folder prints its own table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.dataset import Protocol  # noqa: E402

CAPTAIN_MIN_CLIPS = 50


def design_effect(frame: pd.DataFrame) -> tuple[float, float]:
    """One-way ICC of the label within subjects, and the design effect.

    ICC = (MS_between - MS_within) / (MS_between + (m-1) * MS_within).
    With m clips per subject and positive ICC, n_eff = n / (1 + (m-1) * ICC).
    """
    groups = [group.y.to_numpy() for _, group in frame.groupby("subject")]
    groups = [g for g in groups if len(g) > 0]
    n = sum(len(g) for g in groups)
    k = len(groups)
    if k < 2 or n <= k:
        return 0.0, 1.0
    grand = float(np.concatenate(groups).mean())
    ssb = sum(len(g) * (float(g.mean()) - grand) ** 2 for g in groups)
    ssw = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    sizes = np.array([len(g) for g in groups], dtype=float)
    # m0 for an unbalanced one-way layout: (n - sum(n_i^2)/n) / (k - 1)
    m = float((n - (sizes ** 2).sum() / n) / (k - 1))
    if msb + (m - 1) * msw <= 0:
        return 0.0, 1.0
    icc = (msb - msw) / (msb + (m - 1) * msw)
    icc = max(0.0, min(1.0, icc))
    deff = max(1.0, 1 + (m - 1) * icc)          # a design effect below 1 is not meaningful
    return icc, float(deff)


def fold_row(frame: pd.DataFrame, train_ids: set[str], holdout_ids: set[str],
             rates: pd.Series, captain_episodes: set[str]) -> dict:
    hold = frame[frame.clip_id.isin(holdout_ids)]
    train = frame[frame.clip_id.isin(train_ids)]
    if hold.empty:
        return {}
    clips = len(hold)
    by_subject = hold.groupby("subject").y.agg(["size", "mean"])
    pure_clips = int(by_subject.loc[(by_subject["mean"] <= 0.10) |
                                    (by_subject["mean"] >= 0.90), "size"].sum())
    icc, deff = design_effect(hold)
    oracle = roc_auc_score(hold.y, hold.subject.map(rates)) if hold.y.nunique() == 2 else float("nan")
    train_episodes = set(train.episode)
    overlap = float(hold.episode.isin(train_episodes).mean() * 100) if len(hold) else float("nan")
    clean = hold[~hold.episode.isin(train_episodes) & ~hold.episode.isin(captain_episodes)]
    return {
        "subjects": int(hold.subject.nunique()),
        "clips": clips,
        "deception": round(float(hold.y.mean() * 100), 1),
        "female": round(float(hold.female.mean() * 100), 1),
        "purity": round(pure_clips / clips * 100, 1),
        "mad": round(float((by_subject["size"] * (by_subject["mean"] - 0.5).abs()).sum() / clips), 3),
        "icc": round(icc, 3),
        "n_eff": round(clips / deff, 0),
        "oracle_auc": round(float(oracle), 3) if oracle == oracle else None,
        "episode_overlap": round(overlap, 1),
        "clean_episodes": int(clean.episode.nunique()),
        "clean_clips": int(len(clean)),
    }


def report(splits_dir: Path, frame: pd.DataFrame, rates: pd.Series,
           captain_episodes: set[str]) -> tuple[list[dict], dict]:
    protocol = Protocol.load(splits_dir)
    rows = []
    for fold in range(1, protocol.folds + 1):
        train, _, holdout = protocol.subject(fold)
        row = fold_row(frame, set(train), set(holdout), rates, captain_episodes)
        row["fold"] = fold
        rows.append(row)
    summary = {
        "splits": str(splits_dir),
        "folds": rows,
        "spread": {
            "purity": round(max(r["purity"] for r in rows) - min(r["purity"] for r in rows), 1)
            if rows else None,
            "deception": round(max(r["deception"] for r in rows) - min(r["deception"] for r in rows), 1)
            if rows else None,
            "icc": round(max(r["icc"] for r in rows) - min(r["icc"] for r in rows), 3)
            if rows else None,
        },
    }
    return rows, summary


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    header = (f"{'fold':>4} {'subj':>4} {'clip':>5} {'dec%':>5} {'fem%':>5} {'pure%':>6} "
              f"{'MAD':>5} {'ICC':>5} {'n_eff':>5} {'oracle':>6} {'ep%':>5} {'clean':>5}")
    print(header)
    for row in rows:
        oracle = "  n/a" if row["oracle_auc"] is None else f"{row['oracle_auc']:6.3f}"
        print(f"{row['fold']:>4} {row['subjects']:>4} {row['clips']:>5} {row['deception']:5.1f} "
              f"{row['female']:5.1f} {row['purity']:6.1f} {row['mad']:5.3f} {row['icc']:5.2f} "
              f"{row['n_eff']:5.0f} {oracle} {row['episode_overlap']:5.1f} "
              f"{row['clean_clips']:>5}")
    spread = (max(r["purity"] for r in rows) - min(r["purity"] for r in rows)) if rows else 0
    print(f"purity spread across folds: {spread:.1f} pp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", nargs="+", type=Path, help="one or more split folders")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--captains", default=None,
                        help="comma-separated; default: every subject with >= 50 clips")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    table = pd.read_csv(args.data_dir / "dataset.csv", dtype=str)
    table["y"] = table.y.astype(int)
    table["female"] = (table.gender == "female").astype(int)
    # same clips load_clips() keeps: a real range and a subject
    table["start_s"] = pd.to_numeric(table.start_s, errors="coerce")
    table["end_s"] = pd.to_numeric(table.end_s, errors="coerce")
    table = table[(table.end_s > table.start_s) & table.subject.notna()
                  & (table.subject != "")]
    table = table.rename(columns={"yt_id": "episode"})
    table = table[["clip_id", "subject", "y", "female", "episode"]]
    rates = table.groupby("subject").y.mean()

    if args.captains:
        captains = [s.strip() for s in args.captains.split(",") if s.strip()]
    else:
        captains = table.groupby("subject").size().loc[
            lambda s: s >= CAPTAIN_MIN_CLIPS].index.tolist()
    captain_episodes = set(table[table.subject.isin(captains)].episode)

    outputs = []
    for splits_dir in args.splits:
        rows, summary = report(splits_dir, table, rates, captain_episodes)
        print_table(splits_dir.name, rows)
        outputs.append(summary)
        if len(args.splits) > 1:
            print()

    if len(outputs) > 1:
        print("\n=== purity spread (max - min across folds) ===")
        for summary in outputs:
            print(f"  {summary['splits']:<45} {summary['spread']['purity']:>5.1f} pp")

    if args.csv:
        frame = pd.DataFrame([{"splits": s["splits"], **row}
                              for s in outputs for row in s["folds"]])
        frame.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.json:
        args.json.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
