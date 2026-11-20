# Splits v2: purity- and episode-aware folds

`docs/splits.md` owns the v1 protocol. This document owns the v2 protocol, which
exists because v1 balances three quantities and leaves a fourth free. It is a
**separate protocol**: it does not replace v1, it does not change the board, and
v1 and v2 numbers are not comparable. Adopting it is a group decision.

Files: [`scripts/make_splits_v2.py`](../scripts/make_splits_v2.py) builds the
protocol, [`scripts/report_splits.py`](../scripts/report_splits.py) measures it,
[`lib/metrics.py`](../lib/metrics.py) has the subject-level summary and the
cluster bootstrap. Read this if you are deciding whether to run the board on v2,
regenerating the folds, or reporting a per-fold number.

## The problem v1 leaves open

v1 matches clips, deception count and female count per fold. It does not match
*who* the clips belong to. On the full-annotation regeneration of v1 (the one
`splits.md` documents), fold 1 has

- **73.2%** of its holdout clips in subjects with ≤10% or ≥90% of one label,
  against 27.9–50.5% in the other folds;
- an intraclass correlation of **0.70** (the share of label variance that is
  between subjects) against 0.36–0.53;
- therefore **n_eff ≈ 42** independent clips out of 205, against 51–67 elsewhere;
- a subject-rate oracle AUC of **0.969** against 0.872–0.933: the fold is almost
  solved by *who is speaking*.

That is not a deception/truth imbalance: all five folds sit at 47.3–48.0%
deception, 28.8–29.4% female, 204–205 clips and 5.03–5.50 s mean duration. It is
a **subject-composition** imbalance, and it is not random: the v1 optimiser
(the LPT start in `balanced_folds`) gives the largest working subject to fold 1,
and the swap step compensates its 29 truth clips with pure-deception subjects.
Re-running v1 with 2,000 different seeds on that pool puts the largest subject
in fold 1 in 85% of draws and makes fold 1 the purest fold in 77–78%; the fold
index is not arbitrary, it is the sink of the largest subject. On a 1,452-clip
available subset the bias is weaker (fold 1 is the purest in 54% of draws) and
the lucky index can move, but the spread across folds does not: the fold that
inherits the largest subject is still the outlier, whichever number it gets.

The same composition makes the fold's clip-level estimate noisy: with n_eff 42,
the standard error of an accuracy near 55% is ~7.7 pp, not the ~3.5 pp a
binomial count on 205 clips suggests. A per-fold difference of a few points is
therefore not evidence, whichever metric it is computed on.

## What v2 changes

1. **The virgin test set does not move.** `--test-from <v1 folder>` copies
   `subject_test.json` byte-for-byte and hashes the copy, so v1 and v2 share
   the same test file. v2 only re-partitions the working subjects. The script
   refuses to run without `--test-from` unless `--new-test` is passed;
   `--new-test` is a comparability break, not a convenience, and at the default
   seed it can reproduce v1's draw exactly — which makes the break invisible.
2. **Purity is balanced explicitly.** Every fold must match the working set on
   the share of pure-subject clips and on the clip-weighted MAD
   `mean(|subject rate − 0.5|)`.
3. **Small episode components are atomic.** Two subjects that share an episode
   are linked; connected components of at most `--max-group-subjects` (10) go to
   one fold as a unit, so those episodes are scene-disjoint from training. The
   largest component is 102 of the 151 working subjects and cannot be kept
   whole; the 14 small components cover 49 subjects and are kept whole. 68 of
   75 episodes contain a captain who is always in training. The
   residual overlap is measured by `report_splits.py`, not hidden: v2 lowers it
   from 93–100% to 88–100%, and the captain channel remains.
4. **Fold labels are shuffled after construction.** The optimiser starts by
   giving the largest subject to fold 1; the final numbering is drawn from the
   seed, so no fold index is special by construction.
5. **Validation is stratified on five quantities** (clips, deception, female,
   purity, MAD), not three. The val subjects are still disjoint from the
   holdout and from training, and the notebook still early-stops on them.

The cost is a sum of squared deviations, each divided by a tolerance, so each
term reads as "how many tolerances off" and no term silently dominates:

| Term | Tolerance | Why |
|---|---:|---|
| clips per fold | 3 | keep the fold size comparable |
| deception clips | 3 | the class balance v1 already had |
| female clips | 3 | the per-gender ratio v1 already had |
| purity share | **0.015** | the quantity that made fold 1 easy |
| MAD | **0.008** | continuous version of the same quantity |
| subjects per fold | 2 | avoid a 28-subject fold next to a 33-subject one |

The first three keep the v1 guarantees; the fourth and fifth are the new
constraint. The defaults live in `TOLERANCES` and are copied into every
`manifest.json`, so a run with different tolerances is a different protocol and
is not silently comparable.

## Measured result

Full annotation set (1,671 clips, 151 working subjects), seed 0, `report_splits.py`
on both protocols. This is the local prototype; the Drive run reuses the frozen
test set and the clips with `status=ok`, so its counts are smaller (see
"What to check before running" below).

| Fold | Subjects | Clips | Deception | Female | Purity | MAD | ICC | n_eff | Oracle AUC | Episode overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **v1** 1 | 31 | 205 | 47.3 | 28.8 | **73.2** | 0.383 | **0.70** | **42** | **0.969** | 100.0 |
| v1 2 | 30 | 204 | 47.5 | 28.9 | 27.9 | 0.294 | 0.36 | 67 | 0.872 | 95.6 |
| v1 3 | 30 | 204 | 47.5 | 28.9 | 50.5 | 0.348 | 0.53 | 51 | 0.933 | 94.6 |
| v1 4 | 30 | 204 | 48.0 | 29.4 | 43.1 | 0.328 | 0.48 | 55 | 0.910 | 100.0 |
| v1 5 | 30 | 204 | 47.5 | 29.4 | 40.2 | 0.289 | 0.39 | 64 | 0.885 | 93.1 |
| **v2** 1 | 31 | 204 | 48.0 | 29.4 | 47.1 | 0.328 | 0.49 | 55 | 0.924 | 87.7 |
| v2 2 | 30 | 205 | 47.8 | 28.8 | 46.8 | 0.329 | 0.47 | 55 | 0.916 | 92.2 |
| v2 3 | 30 | 204 | 47.1 | 28.9 | 47.5 | 0.328 | 0.50 | 53 | 0.925 | 92.6 |
| v2 4 | 30 | 204 | 47.5 | 28.9 | 46.6 | 0.328 | 0.51 | 54 | 0.926 | 100.0 |
| v2 5 | 30 | 204 | 47.5 | 29.4 | 47.1 | 0.328 | 0.50 | 53 | 0.925 | 100.0 |

The headline is the spread, not the mean:

| Quantity | v1 | v2 |
|---|---:|---:|
| purity, max − min | **45.3 pp** | **0.9 pp** |
| ICC, max − min | 0.34 | 0.03 |
| n_eff | 42–67 | 53–55 |
| oracle AUC | 0.872–0.969 | 0.916–0.926 |
| deception, max − min | 0.7 pp | 0.9 pp |
| clips | 204–205 | 204–205 |

Every fold of v2 is now the same kind of fold. That is the point: a fold
difference can no longer be "fold 1 is made of pure subjects".

The same run on a 1,455-clip subset (ten random episodes removed, which is the
size of the shipped download) gives folds of 176–178 clips and 27–29 subjects:
purity spread **1.4 pp** for v2 against **12.4 pp** for the v1 regenerated on
that same subset, ICC 0.48–0.51, n_eff 47–51. The bias survives the data loss;
only the absolute counts change. That subset is a simulation, not the shipped
files — regenerate on Drive for the real numbers.

The episode term buys less than it promises, and the table says so: the
overlap stays at 88–100% because the two captains appear in 68 of 75 episodes
and are in every training fold. Keeping small components whole is still worth
doing — it is free — but v2 does **not** make the protocol episode-independent.
An episode-level protocol would have to drop the captain episodes, which costs
more data than the project can spare.

## How to run it

On the shared folder, after the v1 splits exist and `dataset.csv` has the
`status` column:

```bash
python scripts/make_splits_v2.py \
    --available-from /content/drive/MyDrive/dolos/dataset.csv \
    --test-from      /content/drive/MyDrive/dolos/splits \
    --out-dir        /content/drive/MyDrive/dolos/splits_v2
```

The script writes `subject_test.json` (copied), `subject_fold{1..5}.json` and a
`manifest.json` with the seed, the tolerances, the per-fold summary and the
sha256 of every file. `Protocol.load(DATA / "splits_v2")` verifies the hashes and
works unchanged.

Repeated CV:

```bash
python scripts/make_splits_v2.py ... --out-dir .../splits_v2 --replicates 5
```

`rep1` is the root; `rep2..rep5` are under `splits_v2/replicates/`. Each
replicate re-partitions the working subjects with its own seed while keeping the
same test set. Report the mean ± sd **across replicates**, not only across folds
of one replicate: one draw of five ~30-subject holdouts is not a protocol estimate.

## What to check before running

- **The v1 folder you pass is the one you think it is.** `splits.md` quotes the
  1,671-annotation run (fold 1: 205 clips / 31 subjects), while the shipped v1
  files were built on the 1,452 clips with `status=ok` (fold 1: 177 clips / 27
  subjects). Use the same `dataset.csv` snapshot for v1 and v2; otherwise the
  working pool changes and the comparison is confounded.
- **The test set is copied, not re-drawn.** `subject_test.json` must have the
  same sha256 in v1 and v2. If it does not, stop: the virgin set moved.
- **Do not regenerate v1.** Its hash is in every result already on the board.
  v2 is a new folder.
- **The local `dataset.csv` is a plan, not the download.** A dry run writes
  `status=planned`, so `--available-from --statuses ok` keeps zero clips
  locally; omit `--available-from` for the prototype, or pass
  `--statuses planned`.

## How to evaluate on v2

In the notebooks, `PROTOCOL = "v2"` makes `Protocol.load(SPLITS)` read `splits_v2/` instead of
`splits/` (`architecture_template.ipynb`, `train.ipynb`, `train_load.ipynb`). In the fusion and
load notebooks also set `SPLIT = "subject"`: the official 3-fold exists only in v1 and the
notebook refuses the combination. The feature cache does not need rebuilding — the working set
and the test set are the same subjects, only the fold assignment changes. Predictions are written
with a `_v2` suffix, so v1 files are not overwritten.

The split is only half of the fix. On v2, report:

```python
from lib.metrics import cluster_bootstrap, evaluate, subject_level

scores = evaluate(y_true, y_prob, threshold=best_thr)      # clip-level, as before
per_subject = subject_level(y_true, y_prob, subjects, threshold=best_thr)
ci = cluster_bootstrap(y_true, y_prob, subjects, n_boot=1000, seed=0,
                       threshold=best_thr)
```

- `per_subject["subject_acc"]` gives every subject the same weight; a 29-clip
  subject no longer contributes 14% of the score.
- `cluster_bootstrap` resamples subjects, not clips, and returns percentile CIs
  for acc, F1, AUC and balanced accuracy. On v1 fold 1 the design effect is 4.8,
  so a binomial CI is roughly twice as narrow as the honest one; on v2 the
  design effect is ~3.7, so the correction is smaller but still needed.
- Keep AUC as the primary comparison: the threshold is still an argmax on
  ~100 validation clips.
- The epoch budget is still a confound. v2 stratifies val, but a fold whose val
  plateaus at epoch 9 and a fold that runs to the 30-epoch cap are not trained
  the same. Either fix the budget (`PATIENCE = EPOCHS`) or use nested CV; a
  fold-level difference that disappears under a fixed budget was optimisation,
  not data.

## Reproduce the table

```bash
python scripts/make_splits.py --out-dir /tmp/proto/splits_v1
python scripts/make_splits_v2.py \
    --test-from /tmp/proto/splits_v1 \
    --out-dir   /tmp/proto/splits_v2
python scripts/report_splits.py \
    /tmp/proto/splits_v1 /tmp/proto/splits_v2 \
    --csv /tmp/proto/report.csv
```

These write to `/tmp`, never to the shared `splits/`: regenerating v1 in place
would invalidate every stored hash. `report_splits.py` also works on the Drive
folders; pass `--data-dir` at the folder that holds `dataset.csv`.

## What this does not fix

- **Episode leakage through the captains.** 68/75 episodes have a captain in
  training. Closing that channel means dropping those episodes from evaluation,
  which is a different (and much smaller) protocol.
- **The frozen test set is still 27 subjects.** v2 does not make the test
  estimate sharper; it only makes the five fold estimates comparable.
- **The thresholds and the epoch budget.** v2 changes the folds, not the
  training loop.
- **A genuine difference between people.** Balancing purity makes folds
  comparable; it does not make the task easier or harder in a way that
  transfers to a new person. The estimate is still "a model on 27 unseen
  subjects".
