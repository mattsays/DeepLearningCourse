# Splits

The evaluation protocol: which clips are used for what, and why. Produced once by
[`scripts/make_splits.py`](../scripts/make_splits.py); consumed by every notebook through
`Protocol.load(DATA / "splits")`.
Read this if you are touching the splits, the seeds, or anything that could leak.

## The command

```bash
python scripts/make_splits.py \
    --available-from <shared folder>/dataset.csv \
    --test-fraction 0.15 --folds 5 --val-fraction 0.15
```

It runs **after** the download, on the clips that exist. `--available-from` keeps only the clips whose
status is `ok` in `dataset.csv`: without it the protocol references clips nobody has, every fold shrinks
by a different amount, and two members end up training on different subsets. The preprocessing notebook
passes it for you.

```
<shared folder>/splits/
├── official_fold{1,2,3}.json   # straight from the DOLOS CSVs, for the paper comparison
├── subject_test.json           # the reserved test set
├── subject_fold{1..5}.json     # train / val / holdout per fold
└── manifest.json               # seed, counts, sha256 of every file
```

Written straight into the shared folder by the preprocessing notebook (see [data.md](data.md)). **Not in git**: one copy, on
Drive, read by everybody with `Protocol.load(DATA / "splits")`.

### How the subjects are assigned

The unit is the **person**, never the clip: no subject appears on both sides of any boundary. The script
works in four steps, all deterministic given `--seed`.

1. **Two captains are pulled out of the evaluation.** Every subject with ≥ 50 clips is pinned to the
   training side of every fold — on the shipped data exactly two, `Lee Mack` (239 clips) and
   `David Mitchell` (233). Together they are 28% of the dataset and 69% of their clips are deception:
   inside a fold they would be ~70% of its holdout, so the fold would measure one person. They are never
   tested, only trained on.
2. **The test set is drawn first**, 15% of the remaining subjects, so it cannot be influenced by what the
   folds end up containing. It is chosen by random search to match the pool on deception rate, female
   share and mean clips per subject — that is why the test is 47.5% deception against 47.7% in the pool,
   not a lopsided 60/40.
3. **The rest is cut into 5 folds** with a longest-processing-time start followed by pairwise swaps that
   minimise the deviation of clips, deception count and female count per fold. The optimisation matters:
   a naive split gives folds from 42% to 66% deception, this one gives 47.3–48.0%.
4. **Inside each fold, 15% of the training subjects are moved to `val`**, with the same matching rule. The
   folds are `train` (fit) / `val` (early stopping) / `holdout` (the score of that fold).

Numbers below are measured on the **complete** annotation set (1,671 usable clips); your own run
prints its own. Every clip that fails to download removes one from these totals, and a subject who loses
all of theirs disappears from the protocol.

On the shipped annotations — 196 distinct names, and 180 subjects once the alias rule in
`scripts/build_dataset.py` (`canonical_subject`: the two recurring captains plus 14 unambiguous first
names) has merged them:

| Block | Subjects | Clips | Deception | Role |
|---|---:|---:|---:|---|
| test | 27 | 182 | 47.5% | reserved, not used by the direction notebooks |
| working, 5 folds | 151 | 1,021 | 47.7% | cross-validation: the results board |
| captains, train only | 2 | 468 | 69.0% | never evaluated |

Per fold: `train` ≈ 102 subjects / 1,160 clips (captains included), `val` ≈ 18 / 124,
`holdout` ≈ 30 / 204. Every working subject lands in exactly one `holdout` and in no `val`.

### What a fold file contains

```json
{
  "fold": 1,
  "protocol": "subject-level",
  "train":   {"name": "train",   "subjects": ["..."], "clips": ["..."], "n_subjects": 102, "n_clips": 692},
  "val":     {"name": "val",     "subjects": ["..."], "clips": ["..."], "n_subjects": 18,  "n_clips": 124},
  "holdout": {"name": "holdout", "subjects": ["..."], "clips": ["..."], "n_subjects": 31,  "n_clips": 205},
  "always_train": {"subjects": ["David Mitchell", "Lee Mack"], "clips": ["..."]}
}
```

`train.clips` and `always_train.clips` are what a notebook concatenates; the subject lists are there for
inspection. `subject_test.json` has the same shape with a single `subjects`/`clips` pair.

### Checks the script runs

```
checks: 151 working subjects partitioned once, no overlap, no leak
```

It asserts that the folds partition the working set exactly once, that no test subject appears in a fold,
and that no captain is in the test set. If any of those fail, it exits without writing.

### Integrity

Same annotations + same `--seed` = byte-identical files (verified: regenerating produces no git diff).
`manifest.json` records the sha256 of every file and `Protocol.load` verifies it, so a hand edit raises.
What it cannot catch is a *coherent regeneration* — someone re-running the script with another seed — which
is why the shared folder is **view-only** for the rest of the team.

