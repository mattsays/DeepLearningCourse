# Evaluation protocol

Two splits, one number to report, and the baselines it has to beat.

## The protocol

**Primary split — the official 3-fold** (`dataset/DOLOS/Training_Protocols/train_fold*.csv` /
`test_fold*.csv`). Report mean ± std across the three folds. This is what makes our numbers comparable
with the paper.

**Secondary split — our subject-level protocol** (`subject_test.json` + `subject_fold*.json`, see [splits.md](splits.md)).
⚠️ The official folds are **clip-level**: train and test share 147 subjects and all 75 episodes, and 1–2
clips leak across the boundary. Any suspiciously high accuracy there is leakage, not skill.

Our protocol works the other way round: one **virgin test set** of 27 subjects, and a **working set** of
151 subjects used for 5-fold cross-validation. Different subjects in every split, so a model cannot
recognise the person. Report both protocols: the gap between them is one of the most interesting results
this project can produce, and it is the honest measure of generalisation.

| Estimate | Subjects | 95% CI on 60% ACC |
|---|---:|---:|
| working set, mean of 5 folds | 151 | ±7.8 pp |
| virgin test set | 27 | ±18.5 pp |

**The results board is filled from the cross-validation.** Every direction reports the mean over the five
fold holdouts; nobody trains or evaluates on the test set during the project. It stays in the protocol as
the only uncontaminated check left: if the group wants one final number, it is produced once, at the end,
by one person, with a configuration chosen before opening it.

Metrics, in this order: **ACC, F1, AUC**, then per-gender ACC (male clips are 79% of the data and show
56.6% deception vs 43.8% for female clips — a model that ignores this will look fine and be useless).

Calibration, for orientation only:

| Reference | ACC |
|---|---|
| Majority class (`deception`) | 53.7% |
| Best model in the paper, on these folds | 64.75% |

## Results

The board lives in the project README — it is the one table everybody edits as directions land.
