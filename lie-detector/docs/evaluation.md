# Evaluation protocol

Read this if: you are about to produce a number, or to change how numbers are computed.

One split feeds the results board, one is reserved for the very end, and every number has a baseline
it has to beat.

## The protocol

**Primary split — our subject-level protocol.** `subject_test.json` + `subject_fold{1..5}.json`, built
by `scripts/make_splits.py` and described in [splits.md](splits.md):

- **working set** (151 subjects): 5-fold cross-validation. Each fold is `train` / `val` (early stopping)
  / `holdout`; the score of a direction is the **mean over the five holdouts**, and that is what goes
  on the results board.
- **virgin test set** (27 subjects): closed. It is opened once, at the end, by one person, with the
  configuration chosen before opening it. Nobody trains or evaluates on it during the project.

**The official 3-fold is not used.** `official_fold{1,2,3}.json` ships with the protocol and
`Protocol.official(fold)` can read it, but it is **clip-level**: `train_fold1` and `test_fold1` share
147 subjects and all 75 episodes, and 1–2 clips appear on both sides. A number computed on it measures
leakage, not skill. The files stay in the folder for a single reproduction of the paper, reported
separately, never on the board.

Only the cross-validation estimate enters the board:

| Estimate | Subjects | 95% CI on 60% ACC |
|---|---:|---:|
| working set, mean of 5 folds | 151 | ±7.8 pp |
| virgin test set | 27 | ±18.5 pp |

The width of that second interval is the reason the test set is opened once: a single configuration on
27 subjects cannot distinguish 55% from 65%.

## Metrics

`lib/metrics.py` computes all of these; the template prints them in board order.

| Column | What it is | Why it is on the board |
|---|---|---|
| ACC | share of correct decisions at the chosen threshold | the number the paper reports |
| F1 | harmonic mean of precision and recall on `deception` | the classes are not balanced |
| AUC | threshold-free ranking quality | shows separability when the threshold is wrong |
| Balanced ACC | mean of the two per-class recalls | accuracy alone flatters the majority class |
| Majority | most frequent class on the same clips | the floor a model has to beat |

Two rules that decide the numbers more than the architecture does:

- **Pick the threshold on `val`**, with `best_threshold`, never on the holdout and never 0.5 by
  default: the training folds are ~56% deception and the holdouts ~47%, so 0.5 is already biased.
- **Report per-gender ACC too** (`clips.describe(ids)`, and the per-gender cell in the template): male
  clips are 79% of the data and show 56.6% deception against 43.8%, so a model can look balanced and be
  useless.

## Calibration

| Reference | ACC |
|---|---:|
| majority class, whole dataset | 53.7% |
| majority class, working set | 52.4% |
| best PECL model in the paper, official clip-level folds | 64.75% |

The paper's number is orientation only: it is computed on the clip-level folds we do not use, so the
board is not expected to match it. The dataset README (§11.6) is blunt about the alternative — a
claimed 90% is almost always subject or episode leakage, not a better model.

## Results

The board lives in the project README, section `## Results board`: one row per configuration, filled by
pasting the row the template prints. `lib/metrics.py::format_row` produces it in the board's column
order, so do not hand-format one.
