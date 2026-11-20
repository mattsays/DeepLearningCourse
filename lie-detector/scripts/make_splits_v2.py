#!/usr/bin/env python3
"""Build the v2 subject-level protocol: purity- and episode-aware folds.

Why this exists
---------------
The v1 protocol (`make_splits.py`) balances clips, deception count and female
clips per fold. Those three are under control, but they leave *subject-level
label purity* free: a fold can be 73% clips of people who are always truthful
or always lying, while another is 28%. That is not a class-balance problem —
deception stays at 47–48% — it is a problem of *who* a fold is made of. A fold
dominated by pure subjects is easier for any encoder that reads the person or
the scene, and with ~27 subjects per fold the difference is larger than the
sampling noise. `docs/splits-v2.md` measures it.

What v2 changes
---------------
1. **The virgin test set does not move.** It is read from the existing v1
   folder (`--test-from`) and copied byte-for-byte; v2 only re-partitions the
   working subjects. Re-drawing the test set would break the "opened once"
   rule and make v1 and v2 results incomparable.
2. **Purity is balanced.** Each fold must match the working set on the share
   of clips in subjects with <=10% or >=90% of one label, and on the
   clip-weighted mean distance of a subject's rate from 0.5.
3. **Episodes are kept together where the graph allows it.** Subjects that
   share an episode form a connected component. Components of at most
   `--max-group-subjects` are atomic: they are never split across folds, so
   those episodes are scene-disjoint from every training fold. The largest
   component (102 of the 151 working subjects on the full annotations) cannot
   be kept whole, and 68 of 75 episodes contain a captain who is always in
   training; the residual overlap is reported, not hidden.
4. **Fold labels are shuffled at the end.** The optimiser starts by giving the
   largest subject to fold 1; renumbering the folds afterwards removes any
   "fold 1 is special" reading that the construction order would create.
5. **Validation is stratified** on clips, deception, female, purity and MAD,
   instead of matching only the first three.

The cost is a sum of squared deviations, each divided by a tolerance, so a
term is "how many tolerances off" and the weights are readable. A small
allowance is given for clips, positives, female clips and subject count; a
much tighter one for purity. `docs/splits-v2.md` lists the defaults.

Output format is identical to v1, so `lib.dataset.Protocol.load` works
unchanged. Files are written to a *new* directory; v1 is never touched.

Usage
-----
    # on the shared folder, reusing the frozen test set
    python scripts/make_splits_v2.py \
        --available-from /content/drive/MyDrive/dolos/dataset.csv \
        --test-from /content/drive/MyDrive/dolos/splits \
        --out-dir /content/drive/MyDrive/dolos/splits_v2

    # local prototype on the annotations, no download filter
    python scripts/make_splits_v2.py \
        --test-from data/processed/splits \
        --out-dir data/processed/splits_v2

    # five repeated CV draws (rep1 is the root, rep2.. in replicates/)
    python scripts/make_splits_v2.py --test-from ... --out-dir ... --replicates 5

    # one draw from a different split seed, for a repeated-CV campaign
    python scripts/make_splits_v2.py --test-from ... --out-dir ... --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import ROOT, load_clips
from make_splits import sha256, split_payload, write_json

CAPTAIN_MIN_CLIPS = 50
DEFAULT_OUT = ROOT / "data" / "processed" / "splits_v2"

# squared-deviation tolerances: a fold this far off costs one unit
TOLERANCES = {
    "clips": 3.0,       # clips per fold
    "positives": 3.0,   # deception clips per fold
    "female": 3.0,      # female clips per fold
    "purity": 0.015,    # share of clips in >=90%-pure subjects (1.5 pp)
    "mad": 0.008,       # clip-weighted mean |rate - 0.5|
    "subjects": 2.0,    # subjects per fold
}


# --------------------------------------------------------------------------- #
# groups: subjects that share an episode stay together when small enough
# --------------------------------------------------------------------------- #
def episode_components(frame: pd.DataFrame) -> dict[str, str]:
    """Map each subject to the connected component of the episode-sharing graph.

    Two subjects are linked when at least one episode contains both. A component
    that fits in `max_group_subjects` is assigned to a single fold, which makes
    its episodes scene-disjoint from the other folds (captains excepted).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, group in frame.groupby("episode"):
        subjects = list(dict.fromkeys(group["subject"]))
        for other in subjects[1:]:
            union(subjects[0], other)
    return {subject: find(subject) for subject in frame["subject"].unique()}


def build_groups(subjects: pd.DataFrame, components: dict[str, str],
                 max_group_subjects: int) -> list[dict]:
    """Atomic groups: small episode components, individual subjects otherwise."""
    subjects = subjects.copy()
    subjects["component"] = subjects.subject.map(components)
    sizes = subjects.groupby("component").size().to_dict()
    groups: list[dict] = []
    for component, block in subjects.groupby("component"):
        blocks = [block] if sizes[component] <= max_group_subjects else \
                 [subjects.loc[[i]] for i in block.index]
        for part in blocks:
            groups.append({
                "subjects": list(part.subject),
                "clips": int(part.clips.sum()),
                "positives": int(part.positives.sum()),
                "female": int(part.female.sum()),
                "pure_clips": int((part.clips * part.pure).sum()),
                "mad_sum": float((part.clips * (part.rate - 0.5).abs()).sum()),
                "n_subjects": int(len(part)),
                "component": component,
            })
    return groups


# --------------------------------------------------------------------------- #
# the optimiser
# --------------------------------------------------------------------------- #
def optimise(groups: list[dict], k: int, seed: int, restarts: int, sweeps: int,
             tolerances: dict[str, float]) -> tuple[list[list[int]], float]:
    """Longest-processing-time start + pairwise swaps, minimising `cost`."""
    n = len(groups)
    total = {
        "clips": sum(g["clips"] for g in groups),
        "positives": sum(g["positives"] for g in groups),
        "female": sum(g["female"] for g in groups),
        "n_subjects": sum(g["n_subjects"] for g in groups),
    }
    target = {
        "clips": total["clips"] / k,
        "positives": total["positives"] / k,
        "female": total["female"] / k,
        "n_subjects": total["n_subjects"] / k,
        "purity": sum(g["pure_clips"] for g in groups) / total["clips"],
        "mad": sum(g["mad_sum"] for g in groups) / total["clips"],
    }

    def cost(sums: list[list[float]]) -> float:
        value = 0.0
        for clips, positives, female, pure_clips, mad_sum, n_subjects in sums:
            value += ((clips - target["clips"]) / tolerances["clips"]) ** 2
            value += ((positives - target["positives"]) / tolerances["positives"]) ** 2
            value += ((female - target["female"]) / tolerances["female"]) ** 2
            value += ((n_subjects - target["n_subjects"]) / tolerances["subjects"]) ** 2
            value += ((pure_clips / clips - target["purity"]) / tolerances["purity"]) ** 2
            value += ((mad_sum / clips - target["mad"]) / tolerances["mad"]) ** 2
        return value

    def apply(sums, fold, group, sign):
        sums[fold][0] += sign * group["clips"]
        sums[fold][1] += sign * group["positives"]
        sums[fold][2] += sign * group["female"]
        sums[fold][3] += sign * group["pure_clips"]
        sums[fold][4] += sign * group["mad_sum"]
        sums[fold][5] += sign * group["n_subjects"]

    best, best_cost = None, None
    for restart in range(restarts):
        rng = random.Random(seed * 1000 + restart)
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda i: -groups[i]["clips"])      # stable: ties stay random

        folds: list[list[int]] = [[] for _ in range(k)]
        loads = [0] * k
        for index in order:
            fold = min(range(k), key=lambda j: loads[j])
            folds[fold].append(index)
            loads[fold] += groups[index]["clips"]

        sums = [[0, 0, 0, 0, 0.0, 0] for _ in range(k)]
        for fold, members in enumerate(folds):
            for index in members:
                apply(sums, fold, groups[index], 1)
        current = cost(sums)

        for _ in range(sweeps):
            a, b = rng.sample(range(k), 2)
            if not folds[a] or not folds[b]:
                continue
            ia = rng.choice(folds[a])
            ib = rng.choice(folds[b])
            apply(sums, a, groups[ia], -1)
            apply(sums, b, groups[ib], -1)
            apply(sums, a, groups[ib], 1)
            apply(sums, b, groups[ia], 1)
            candidate = cost(sums)
            if candidate < current:
                current = candidate
                folds[a].remove(ia)
                folds[b].remove(ib)
                folds[a].append(ib)
                folds[b].append(ia)
            else:
                apply(sums, a, groups[ia], 1)
                apply(sums, b, groups[ib], 1)
                apply(sums, a, groups[ib], -1)
                apply(sums, b, groups[ia], -1)

        if best_cost is None or current < best_cost:
            best_cost, best = current, [list(members) for members in folds]
    assert best is not None
    return best, float(best_cost)


# --------------------------------------------------------------------------- #
# validation split
# --------------------------------------------------------------------------- #
def pick_val(pool: pd.DataFrame, n: int, seed: int, tries: int = 20000) -> list[str]:
    """Pick `n` training-side subjects matching the pool on five quantities."""
    if n >= len(pool):
        return pool.subject.tolist()
    target = {
        "mean_clips": float(pool.clips.mean()),
        "rate": float(pool.positives.sum() / pool.clips.sum()),
        "female": float(pool.female.sum() / pool.clips.sum()),
        "purity": float((pool.clips * pool.pure).sum() / pool.clips.sum()),
        "mad": float((pool.clips * (pool.rate - 0.5).abs()).sum() / pool.clips.sum()),
    }
    rng = np.random.default_rng(seed)
    clips = pool.clips.to_numpy()
    positives = pool.positives.to_numpy()
    female = pool.female.to_numpy()
    pure = pool.pure.to_numpy()
    rate = pool.rate.to_numpy()
    best, best_cost = None, None
    for _ in range(tries):
        pick = rng.choice(len(pool), n, replace=False)
        c = clips[pick].sum()
        cost = (abs(c / n - target["mean_clips"]) / 1.5
                + abs(positives[pick].sum() / c - target["rate"]) * 6
                + abs(female[pick].sum() / c - target["female"]) * 6
                + abs((clips[pick] * pure[pick]).sum() / c - target["purity"]) * 3
                + abs((clips[pick] * np.abs(rate[pick] - 0.5)).sum() / c - target["mad"]) * 3)
        if best_cost is None or cost < best_cost:
            best, best_cost = pick, cost
    return pool.iloc[best].subject.tolist()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_subjects(clips, dataset_dir: Path, available_from: Path | None,
                  statuses: str) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    broken = [c for c in clips if c.duration <= 0]
    clips = [c for c in clips if c.duration > 0 and c.subject]
    if available_from is not None:
        if not available_from.exists():
            raise SystemExit(f"[error] {available_from} not found: build dataset.csv first")
        status = pd.read_csv(available_from, dtype=str)[["clip_id", "status"]]
        allowed = {s.strip() for s in statuses.split(",") if s.strip()}
        available = set(status.loc[status.status.isin(allowed), "clip_id"])
        before = len(clips)
        clips = [c for c in clips if c.clip_id in available]
        print(f"dataset.csv: {len(available)} clips with status in {sorted(allowed)} "
              f"-> {len(clips)} of {before} annotations kept")
    if not clips:
        raise SystemExit("[error] no usable clips")

    rows = [{"clip_id": c.clip_id, "subject": c.subject, "episode": c.yt_id,
             "y": c.y, "female": int(c.gender == "female")} for c in clips]
    frame = pd.DataFrame(rows)
    subjects = (frame.groupby("subject")
                .agg(clips=("clip_id", "size"), positives=("y", "sum"),
                     female=("female", "sum"))
                .reset_index())
    subjects["rate"] = subjects.positives / subjects.clips
    subjects["pure"] = ((subjects.rate <= 0.10) | (subjects.rate >= 0.90)).astype(int)
    return frame, subjects, broken, clips


def reuse_test(test_from: Path, subjects: pd.DataFrame, clip_ids: set[str]) -> dict:
    """Read the v1 test payload and verify it still matches the current dataset."""
    path = Path(test_from) / "subject_test.json"
    if not path.exists():
        raise SystemExit(f"[error] {path} not found: pass --test-from or --new-test")
    payload = json.loads(path.read_text(encoding="utf-8"))
    known = set(subjects.subject)
    missing = [s for s in payload["subjects"] if s not in known]
    if missing:
        raise SystemExit(f"[error] {len(missing)} test subjects have no clips now: {missing[:5]}")
    missing_clips = [c for c in payload["clips"] if c not in clip_ids]
    if missing_clips:
        raise SystemExit(f"[error] {len(missing_clips)} test clips are absent from this table "
                         f"(first: {missing_clips[:3]}); use the same dataset.csv as v1")
    print(f"test: reusing {len(payload['subjects'])} subjects / {len(payload['clips'])} clips "
          f"from {test_from}")
    return payload


def draw_test(pool: pd.DataFrame, seed: int, fraction: float) -> dict:
    """Only for an explicit --new-test: a fresh 15% search, same rule as v1."""
    from make_splits import pick_subset
    n = max(1, round(fraction * len(pool)))
    # pick_subset() speaks v1's column names (pos, not positives)
    chosen = pick_subset(pool.rename(columns={"positives": "pos"})[
        ["subject", "clips", "pos", "female"]], n, seed)
    block = pool[pool.subject.isin(chosen)]
    return {"name": "test", "subjects": sorted(chosen), "clips": [],
            "n_subjects": len(chosen), "n_clips": int(block.clips.sum())}


def fold_stats(subjects: pd.DataFrame, holdout: list[str]) -> dict:
    block = subjects[subjects.subject.isin(holdout)]
    clips = int(block.clips.sum())
    return {
        "clips": clips,
        "subjects": int(len(block)),
        "deception": round(float(block.positives.sum() / clips) * 100, 1),
        "female": round(float(block.female.sum() / clips) * 100, 1),
        "purity": round(float((block.clips * block.pure).sum() / clips) * 100, 1),
        "mad": round(float((block.clips * (block.rate - 0.5).abs()).sum() / clips), 3),
    }


def write_protocol(clips_by_subject: dict[str, list[str]], subjects: pd.DataFrame,
                   test_payload: dict, folds: list[list[str]], captains: list[str],
                   val_fraction: float, seed: int, out_dir: Path,
                   config: dict, test_source: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    test_path = out_dir / "subject_test.json"
    if test_source is not None:
        # byte-for-byte: the frozen test set must hash the same in v1 and v2
        shutil.copyfile(test_source, test_path)
    else:
        write_json(test_path, test_payload)
    written[test_path.name] = sha256(test_path)

    working = set(subjects.subject) - set(test_payload["subjects"]) - set(captains)
    summary = []
    rng = random.Random(seed + 17)
    labels = list(range(len(folds)))
    rng.shuffle(labels)                                  # fold 1 is not special

    for slot, label in enumerate(labels, start=1):
        holdout = folds[label]
        rest = sorted(working - set(holdout))
        n_val = max(1, round(val_fraction * len(rest)))
        val_subjects = pick_val(subjects[subjects.subject.isin(rest)].reset_index(drop=True),
                                n_val, seed + slot)
        train_subjects = sorted(set(rest) - set(val_subjects))
        payload = {
            "fold": slot,
            "protocol": "subject-level-v2",
            "train": split_payload("train", train_subjects, clips_by_subject),
            "val": split_payload("val", sorted(val_subjects), clips_by_subject),
            "holdout": split_payload("holdout", sorted(holdout), clips_by_subject),
            "always_train": {"name": "captains", "subjects": sorted(captains),
                             "clips": [c for s in sorted(captains) for c in clips_by_subject[s]]},
        }
        path = out_dir / f"subject_fold{slot}.json"
        write_json(path, payload)
        written[path.name] = sha256(path)
        stats = fold_stats(subjects, sorted(holdout))
        stats["fold"] = slot
        stats["val_clips"] = payload["val"]["n_clips"]
        summary.append(stats)

    manifest = {
        "protocol": "subject-level-v2",
        "seed": seed,
        "val_fraction": val_fraction,
        "captains": sorted(captains),
        "config": config,
        "counts": {
            "working_subjects": len(working),
            "working_clips": int(subjects[subjects.subject.isin(working)].clips.sum()),
            "test_subjects": len(test_payload["subjects"]),
            "test_clips": test_payload["n_clips"],
        },
        "folds": summary,
        "files": written,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset" / "DOLOS")
    parser.add_argument("--available-from", type=Path, default=None)
    parser.add_argument("--statuses", default="ok")
    parser.add_argument("--test-from", type=Path, default=None,
                        help="v1 splits folder: subject_test.json is reused unchanged")
    parser.add_argument("--new-test", action="store_true",
                        help="draw a new test set (breaks comparability with v1: avoid)")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=1,
                        help="extra repeated-CV draws under <out-dir>/replicates/")
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=80000)
    parser.add_argument("--max-group-subjects", type=int, default=10,
                        help="episode components up to this size are kept whole")
    parser.add_argument("--captains", default=None)
    parser.add_argument("--tol-clips", type=float, default=TOLERANCES["clips"])
    parser.add_argument("--tol-positives", type=float, default=TOLERANCES["positives"])
    parser.add_argument("--tol-female", type=float, default=TOLERANCES["female"])
    parser.add_argument("--tol-purity", type=float, default=TOLERANCES["purity"])
    parser.add_argument("--tol-mad", type=float, default=TOLERANCES["mad"])
    parser.add_argument("--tol-subjects", type=float, default=TOLERANCES["subjects"])
    args = parser.parse_args()

    tolerances = {"clips": args.tol_clips, "positives": args.tol_positives,
                  "female": args.tol_female, "purity": args.tol_purity,
                  "mad": args.tol_mad, "subjects": args.tol_subjects}

    raw = load_clips(args.dataset_dir)
    frame, subjects, broken, clips = load_subjects(raw, args.dataset_dir,
                                                   args.available_from, args.statuses)
    print(f"annotations: {len(clips)} usable clips ({len(broken)} dropped: end <= start); "
          f"{len(subjects)} subjects")

    clips_by_subject = (frame.groupby("subject").clip_id.apply(list).to_dict())


    if args.captains:
        captains = [s.strip() for s in args.captains.split(",") if s.strip()]
        captain_source = "--captains"
    elif args.test_from is not None and (args.test_from / "subject_fold1.json").exists():
        # the same two subjects v1 pinned to training: keep them pinned, or a
        # subject that moved into the working pool could land in a holdout
        captains = json.loads((args.test_from / "subject_fold1.json").read_text(
            encoding="utf-8"))["always_train"]["subjects"]
        captain_source = f"{args.test_from}/subject_fold1.json"
    else:
        captains = sorted(subjects.loc[subjects.clips >= CAPTAIN_MIN_CLIPS, "subject"])
        captain_source = f"clips >= {CAPTAIN_MIN_CLIPS}"

    missing = [s for s in captains if s not in set(subjects.subject)]
    if missing:
        raise SystemExit(f"[error] captains not found: {missing}")

    if args.test_from is not None:
        test_payload = reuse_test(args.test_from, subjects, set(frame.clip_id))
    elif args.new_test:
        non_captains = subjects[~subjects.subject.isin(captains)]
        test_payload = draw_test(non_captains, args.seed, args.test_fraction)
        test_payload["clips"] = [c for s in test_payload["subjects"] for c in clips_by_subject[s]]
        print(f"test: new draw, {test_payload['n_subjects']} subjects / {test_payload['n_clips']} clips")
    else:
        raise SystemExit("[error] pass --test-from <v1 splits> to reuse the frozen test set "
                         "(or --new-test, which breaks comparability)")
    working = subjects[~subjects.subject.isin(set(test_payload["subjects"]) | set(captains))]
    working = working.reset_index(drop=True)
    if len(working) < args.folds:
        raise SystemExit("[error] fewer working subjects than folds")
    print(f"working: {len(working)} subjects / {int(working.clips.sum())} clips; "
          f"captains from {captain_source}: {captains}")

    components = episode_components(frame[frame.subject.isin(set(working.subject))])
    groups = build_groups(working, components, args.max_group_subjects)
    if len(groups) < args.folds:
        raise SystemExit(f"[error] {len(groups)} atomic groups cannot fill {args.folds} folds")
    print(f"groups: {len(groups)} (episode components up to {args.max_group_subjects} are atomic; "
          f"largest group {max(len(g['subjects']) for g in groups)} subjects)")

    config = {
        "tolerances": tolerances,
        "restarts": args.restarts,
        "sweeps": args.sweeps,
        "max_group_subjects": args.max_group_subjects,
        "test_from": str(args.test_from) if args.test_from else None,
        "available_from": str(args.available_from) if args.available_from else None,
    }

    folds: list[list[str]] = []
    for replicate in range(1, args.replicates + 1):
        seed = args.seed + (replicate - 1)
        split, value = optimise(groups, args.folds, seed, args.restarts, args.sweeps, tolerances)
        fold_subjects = [[s for index in members for s in groups[index]["subjects"]]
                         for members in split]
        target = args.out_dir if replicate == 1 else args.out_dir / "replicates" / f"rep{replicate}"
        manifest = write_protocol(clips_by_subject, working, test_payload, fold_subjects,
                                  captains, args.val_fraction, seed, target, config,
                                  test_source=(args.test_from / "subject_test.json"
                                               if args.test_from else None))
        all_holdout = [subject for fold in fold_subjects for subject in fold]
        assert len(all_holdout) == len(set(all_holdout)) == len(working), \
            f"replicate {replicate} does not partition the working set"
        assert not (set(all_holdout) & set(test_payload["subjects"])), \
            f"replicate {replicate}: a test subject is in a fold"
        assert not (set(all_holdout) & set(captains)), \
            f"replicate {replicate}: a captain is in a fold"
        print(f"\nreplicate {replicate} (seed {seed}) -> {target}  cost {value:.4f}")
        for row in manifest["folds"]:
            print(f"  fold {row['fold']}: {row['subjects']:3d} subjects  {row['clips']:4d} clips  "
                  f"deception {row['deception']:5.1f}%  female {row['female']:5.1f}%  "
                  f"purity {row['purity']:5.1f}%  mad {row['mad']:.3f}  val {row['val_clips']}")
        folds = fold_subjects

    # checks: partitions, disjointness, test and captains
    working_subjects = set(working.subject)
    all_holdout = [s for fold in folds for s in fold]
    assert len(all_holdout) == len(set(all_holdout)) == len(working_subjects), \
        "folds do not partition the working set"
    assert not (set(test_payload["subjects"]) & working_subjects), "test leaked into the working set"
    assert not (set(test_payload["subjects"]) & set(captains)), "a captain is in the test set"
    print(f"\nchecks: {len(working_subjects)} working subjects partitioned once, no overlap, no leak")

    # the root manifest also points at the repeated-CV draws
    root_manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
    root_manifest["replicates"] = [
        f"replicates/rep{i}" for i in range(2, args.replicates + 1)]
    write_json(args.out_dir / "manifest.json", root_manifest)
    print(f"written: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
