#!/usr/bin/env python3
"""Create the evaluation splits for the project.

Two things are produced, and they answer two different questions:

1. **Official 3-fold** (`official_fold{1,2,3}.json`) — copied verbatim from the
   DOLOS protocol files. Used only to compare with the paper.
2. **Our subject-level protocol** — one virgin test set plus a k-fold
   cross-validation over the rest:

      194 subjects
      ├── test           15% of the non-captain subjects, frozen
      └── working set    everyone else
          ├── k folds   train / val / holdout, split by subject
          └── captains  always in train, never evaluated

   Each fold file holds `train`, `val` and `holdout` clip ids. A subject appears
   in exactly one `holdout`, in no `val`, and in the `train` of every other fold.

Why the two recurring captains are pinned to the training side: with 472 clips
between them they are 28% of the dataset, and a single one of them would be ~70%
of a fold's holdout. Details in `README.md` §5 of the project README.

The output is deterministic: same annotations + same `--seed` = same files. The
manifest records a sha256 per file so a notebook can refuse a mismatched split.

Usage
-----
    python scripts/make_splits.py                       # 15% test, 5 folds, 15% val
    python scripts/make_splits.py --test-fraction 0.2
    python scripts/make_splits.py --folds 3 --val-fraction 0.2
    python scripts/make_splits.py --captains "Lee Mack"  # override the auto-detection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import ROOT, load_clips  # same clip ids as the downloader

CAPTAIN_MIN_CLIPS = 50          # only the two recurring captains exceed this
OUT = ROOT / "data" / "processed" / "splits"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def subject_table(clips) -> pd.DataFrame:
    """One row per subject: clip count, deception count, female clips."""
    rows = []
    for clip in clips:
        rows.append({"subject": clip.subject, "clip_id": clip.clip_id,
                     "y": clip.y, "female": int(clip.gender == "female")})
    frame = pd.DataFrame(rows)
    return (frame.groupby("subject")
                 .agg(clips=("clip_id", "size"), pos=("y", "sum"), female=("female", "sum"))
                 .reset_index())


def pick_subset(pool: pd.DataFrame, n: int, seed: int, tries: int = 20000) -> list[str]:
    """Pick `n` subjects whose totals match the pool distribution.

    Matches the deception rate, the female share and the mean clips-per-subject,
    so the test set and the working set stay comparable.
    """
    if n >= len(pool):
        return pool.subject.tolist()
    target = (pool.pos.sum() / pool.clips.sum(),
              pool.female.sum() / pool.clips.sum(),
              pool.clips.mean())
    rng = np.random.default_rng(seed)
    best, best_cost = None, None
    idx = pool.clips.to_numpy()
    pos = pool.pos.to_numpy()
    fem = pool.female.to_numpy()
    for _ in range(tries):
        pick = rng.choice(len(pool), n, replace=False)
        c, p, f = idx[pick].sum(), pos[pick].sum(), fem[pick].sum()
        cost = (abs(p / c - target[0]) * 4 + abs(f / c - target[1]) * 4
                + abs(c / n - target[2]) / 20)
        if best_cost is None or cost < best_cost:
            best, best_cost = pick, cost
    return pool.iloc[best].subject.tolist()


def balanced_folds(pool: pd.DataFrame, k: int, seed: int, sweeps: int = 20000) -> list[list[str]]:
    """Partition subjects into k folds balancing clips, deception and gender.

    Longest-processing-time start, then pairwise swaps that reduce the cost.
    """
    rng = random.Random(seed)
    order = pool.sample(frac=1, random_state=seed).sort_values("clips", ascending=False)
    folds: list[list[str]] = [[] for _ in range(k)]
    loads = [0] * k
    for _, row in order.iterrows():
        j = min(range(k), key=lambda j: loads[j])
        folds[j].append(row.subject)
        loads[j] += int(row.clips)

    stats = pool.set_index("subject")[["clips", "pos", "female"]]
    t_c, t_p, t_f = float(pool.clips.sum()), float(pool.pos.sum()), float(pool.female.sum())

    def cost() -> float:
        total = 0.0
        for fold in folds:
            sub = stats.loc[fold]
            c, p, f = sub.clips.sum(), sub.pos.sum(), sub.female.sum()
            total += ((c - t_c / k) / t_c) ** 2 + ((p - t_p / k) / t_p) ** 2 + ((f - t_f / k) / t_f) ** 2
        return total

    current = cost()
    for _ in range(sweeps):
        a, b = rng.sample(range(k), 2)
        if not folds[a] or not folds[b]:
            continue
        i, j = rng.choice(folds[a]), rng.choice(folds[b])
        folds[a].remove(i); folds[b].remove(j); folds[a].append(j); folds[b].append(i)
        new = cost()
        if new < current:
            current = new
        else:
            folds[a].remove(j); folds[b].remove(i); folds[a].append(i); folds[b].append(j)
    return folds


def split_payload(name: str, subjects: list[str], subj_to_clips: dict[str, list[str]]) -> dict:
    clips = [c for s in sorted(subjects) for c in subj_to_clips[s]]
    return {"name": name, "subjects": sorted(subjects), "clips": clips,
            "n_subjects": len(subjects), "n_clips": len(clips)}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DOLOS")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--captains", default=None,
                        help="comma-separated subjects pinned to training "
                             f"(default: every subject with >= {CAPTAIN_MIN_CLIPS} clips)")
    args = parser.parse_args()

    clips = load_clips(args.dataset_dir)
    broken = [c for c in clips if c.duration <= 0]
    clips = [c for c in clips if c.duration > 0 and c.subject]
    print(f"annotations: {len(clips)} usable clips ({len(broken)} dropped: end <= start)")
    if not clips:
        print("[error] no usable clips")
        return 2

    table = subject_table(clips)
    subj_to_clips = pd.DataFrame([{"subject": c.subject, "clip_id": c.clip_id} for c in clips]) \
                      .groupby("subject").clip_id.apply(list).to_dict()

    if args.captains:
        captains = [s.strip() for s in args.captains.split(",") if s.strip()]
    else:
        captains = table.loc[table.clips >= CAPTAIN_MIN_CLIPS, "subject"].tolist()
    missing = [c for c in captains if c not in subj_to_clips]
    if missing:
        print(f"[error] captains not found: {missing}")
        return 2

    pool = table[~table.subject.isin(captains)].reset_index(drop=True)
    n_test = max(1, round(args.test_fraction * len(pool)))
    test_subjects = pick_subset(pool, n_test, args.seed)
    working = pool[~pool.subject.isin(test_subjects)].reset_index(drop=True)

    print(f"captains (train only): {captains} -> {int(table[table.subject.isin(captains)].clips.sum())} clips")
    print(f"test:    {len(test_subjects)} subjects / {int(pool[pool.subject.isin(test_subjects)].clips.sum())} clips")
    print(f"working: {len(working)} subjects / {int(working.clips.sum())} clips")

    folds = balanced_folds(working, args.folds, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # ---- our subject-level protocol -------------------------------------- #
    test_payload = split_payload("test", test_subjects, subj_to_clips)
    path = args.out_dir / "subject_test.json"
    write_json(path, test_payload)
    written[path.name] = sha256(path)

    fold_summary = []
    working_set = set(working.subject)
    for i, holdout in enumerate(folds, start=1):
        rest = sorted(working_set - set(holdout))
        n_val = max(1, round(args.val_fraction * len(rest)))
        val_subjects = pick_subset(working[working.subject.isin(rest)].reset_index(drop=True),
                                   n_val, args.seed + i)
        train_subjects = sorted(set(rest) - set(val_subjects))

        payload = {
            "fold": i,
            "protocol": "subject-level",
            "train": split_payload("train", train_subjects, subj_to_clips),
            "val": split_payload("val", val_subjects, subj_to_clips),
            "holdout": split_payload("holdout", holdout, subj_to_clips),
            "always_train": {"name": "captains", "subjects": captains,
                             "clips": [c for s in captains for c in subj_to_clips[s]]},
        }
        path = args.out_dir / f"subject_fold{i}.json"
        write_json(path, payload)
        written[path.name] = sha256(path)

        fold_summary.append({
            "fold": i,
            "train": (len(train_subjects), len(payload["train"]["clips"]) + len(payload["always_train"]["clips"])),
            "val": (len(val_subjects), len(payload["val"]["clips"])),
            "holdout": (len(holdout), len(payload["holdout"]["clips"])),
            "holdout_deception": round(
                sum(c.y for c in clips if c.clip_id in set(payload["holdout"]["clips"]))
                / len(payload["holdout"]["clips"]) * 100, 1),
        })

    # ---- official 3-fold, straight from the DOLOS CSVs -------------------- #
    proto_dir = args.dataset_dir / "Training_Protocols"
    by_name = {c.source_name: c.clip_id for c in clips}
    for i in (1, 2, 3):
        entries = {}
        for split in ("train", "test"):
            path_csv = proto_dir / f"{split}_fold{i}.csv"
            names = [line.split(",")[0].strip()
                     for line in path_csv.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            entries[split] = [by_name[n] for n in names if n in by_name]
        payload = {"fold": i, "protocol": "official-clip-level",
                   "train": {"name": "train", "clips": entries["train"], "n_clips": len(entries["train"])},
                   "test": {"name": "test", "clips": entries["test"], "n_clips": len(entries["test"])}}
        path = args.out_dir / f"official_fold{i}.json"
        write_json(path, payload)
        written[path.name] = sha256(path)

    manifest = {
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "folds": args.folds,
        "val_fraction": args.val_fraction,
        "captains": captains,
        "dropped_invalid_clips": [c.clip_id for c in broken],
        "counts": {
            "clips_total": len(clips),
            "test_subjects": len(test_subjects), "test_clips": test_payload["n_clips"],
            "working_subjects": len(working), "working_clips": int(working.clips.sum()),
            "captain_clips": int(table[table.subject.isin(captains)].clips.sum()),
        },
        "files": written,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    # ---- report ---------------------------------------------------------- #
    print(f"\n{'fold':4} {'train':>18} {'val':>16} {'holdout':>18} {'% decep':>8}")
    for row in fold_summary:
        print(f"{row['fold']:4} "
              f"{row['train'][0]:4d} sogg / {row['train'][1]:5d} clip "
              f"{row['val'][0]:4d} sogg / {row['val'][1]:4d} clip "
              f"{row['holdout'][0]:4d} sogg / {row['holdout'][1]:4d} clip "
              f"{row['holdout_deception']:7.1f}%")

    all_holdout = [s for f in folds for s in f]
    assert len(all_holdout) == len(set(all_holdout)) == len(working), "folds do not partition the working set"
    assert not (set(test_subjects) & working_set), "test leaked into the working set"
    assert not (set(test_subjects) & set(captains)), "a captain is in the test set"
    print(f"\nchecks: {len(working)} working subjects partitioned once, no overlap, no leak")
    print(f"written: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
