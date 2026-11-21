# `tia` results — what is saved, and what each file means

Every run writes one directory here, through [`export.py`](../export.py). The README quotes
numbers, this folder proves them: a row without a directory is a number nobody can check.

The rule the layout encodes: **a run is one variant on one protocol**, and the protocol it ran
on is in the name, because the same model scores ~53 and ~72 ACC depending on it (see
[`docs/protocol-effect.md`](../docs/protocol-effect.md)). Two protocols, two directories, never
one file holding both.

```
results/
├── README.md                 ← this file: the layout, and the rules for reading it
├── diagnostics_mumin.csv     ← diagnostics.ipynb: cue-by-cue probe AUC, video and audio
├── oracle_mumin.csv          ← diagnostics.ipynb: MUMIN + logistic regression, subject split
└── <variant>-<split>[-<ablation>][+<protocol>]/
    ├── run.json              settings, encoders, fingerprints, trainable parameters
    ├── summary.json          mean and spread over the folds and seeds
    ├── folds.csv             one row per (seed, fold): the numbers summary.json averages
    ├── preds.csv             one row per clip: clip_id, y_true, y_prob
    └── board_row.md          the results-board line, or why this run has none
```

Run names are built by `export.run_name`: `late-subject`, `late-official`, `audio-subject`,
`late-subject-nocaptains`, `late-subject+v2`. `subject` is our subject-independent 5-fold
protocol — the one the board reads. `official` is the paper's clip-level 3-fold, which leaks
speaker and round; it is reproduced to measure the leak, never to compete.

## The files

| File | Owner of | Reading it |
|---|---|---|
| `run.json` | every setting a run cannot be reconstructed without | `variant`, `split`, `protocol`, `seeds`, `encoders`, `widths`, `trainable_params`, `training`, plus `dataset.fingerprint` (`dataset.csv` sha256), `protocol_files.manifest_sha256`, `git_commit`, `git_dirty` |
| `folds.csv` | the per-fold numbers | one row per seed × fold: `acc`, `f1`, `auc`, `balanced_acc`, `precision`, `recall`, `threshold`, the confusion counts, `seconds`, `epochs` |
| `summary.json` | the mean and the spread | `acc`/`f1`/`auc`/`balanced_acc` with `_std`, the same per seed, `threshold_range`, `trainable_params`, `train_minutes` |
| `preds.csv` | what the paired comparisons read | `clip_id`, `y_true`, `y_prob`; with several seeds `y_prob` is the per-clip mean, which is what McNemar against another direction needs |
| `board_row.md` | the one line for `README.md` §Results board | a markdown row, or the sentence saying this split is not a board row |

`preds.csv` is the file to quote when comparing two directions: same clips, one probability
each. `folds.csv` is the file to quote when asking whether a difference is larger than the
fold-to-fold noise.

## How a run is written

[`campaign.ipynb`](../campaign.ipynb) runs the whole set: one call to
`training.run_campaign`, which loops the `(variant, split)` pairs, builds each split's feature
store once, trains the seeds and folds, and calls `export.save_run` per run. It is the batch
path, and the one that should produce this folder.

[`train.ipynb`](../train.ipynb) cell **9 — Save the run** writes a single configuration from the
interactive kernel, with the same writer and the same file set. Use it for one-off variants; it
refuses to write anything when the notebook ran on pseudo-features (no cache), because those
numbers are invented by construction.

The thresholds, epochs and confusion counts come from the same `lib.metrics.evaluate` calls the
console printed, so the file and the screen cannot disagree. Re-running one configuration
overwrites its directory: one run, one version, identified by the git commit inside `run.json`.
The file set is the contract — `export.py` is the only writer, and nothing here is edited by
hand.

## Which directories the current text needs

The rows in [`../README.md`](../README.md) §Results and in
[`../docs/protocol-effect.md`](../docs/protocol-effect.md) are backed by these runs. The campaign
runs **three seeds everywhere**, controls included: a one-seed row cannot be told apart from a
lucky draw, and on this dataset the seed spread is larger than the gap between variants.

| Directory | `VARIANT` | `SPLIT` | `SEEDS` | Backs |
|---|---|---|---|---|
| `late-subject` | `late` | `subject` | `[0, 1, 2]` | the board row (pool + concat + MLP) |
| `audio-subject` | `audio` | `subject` | `[0, 1, 2]` | "audio carries the signal" |
| `video-subject` | `video` | `subject` | `[0, 1, 2]` | "video at 16 frames is chance" |
| `late-official` | `late` | `official` | `[0, 1, 2]` | the leaky side of the protocol effect |
| `audio-official` | `audio` | `official` | `[0, 1, 2]` | the leak does not travel through voice |
| `video-official` | `video` | `official` | `[0, 1, 2]` | the leak is the face |

`PROTOCOL = "v1"`, `CAPTAINS = "include"`, `FOLDS = 5` for all of them. Where the direction
README still quotes a one-seed number (`audio` 54.7 / 58.9, `video` 52.2 / 53.7), the campaign
replaces it with the three-seed measurement: `summary.json` is the measurement, and the text is
corrected to the run, never the other way round.

`congruence` is not in the campaign. It is a probe — *is the decision driven only by
disagreement?* — not a candidate for the board, and it is reported from the interactive notebook.
If it ever needs an artifact, running it once from `train.ipynb` writes `congruence-subject/`
with the same layout.

Two flat files sit beside the directories, written by
[`diagnostics.ipynb`](../diagnostics.ipynb) rather than `export.py`:
`diagnostics_mumin.csv` (per-cue probe AUC against a prevalence-matched noise band) and
`oracle_mumin.csv` (the 41 annotations through a logistic regression, subject split — the
non-deployable reference at 51.1 ACC / 52.5 AUC).

## What is not here

- **Checkpoints.** The trainable part is 10⁵ parameters; retraining a fold is cheaper than
  storing one, and `run.json` + the caches reproduce it.
- **Features.** `cached_features/` lives in the shared Drive folder and is 330 MB.
- **The virgin test set.** `subject_test.json` is opened once, for the final number, and no run
  in this folder touches it.
