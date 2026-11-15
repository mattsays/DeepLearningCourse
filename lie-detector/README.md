# Lie Detection on DOLOS

Deception detection from short game-show clips: a clip goes in, `truth` or `deception` comes out.
Four people, four directions, one frozen evaluation protocol.

| | |
|---|---|
| **Input** | one video clip from DOLOS (`.mp4`, 2–19 s) |
| **Output** | `lie` / `not lie` — binary `deception` (1) vs `truth` (0) |
| **Team** | 4 members, one research direction each |
| **Compute** | Google Colab only — nobody has a usable local GPU |
| **Deadline** | **26 November** |
| **Dataset** | [`dataset/README.md`](dataset/README.md) — read it before writing any code |

Repository-wide rules (language, dataset handling, submodules) are in [`../AGENTS.md`](../AGENTS.md);
project-level ones (data, notebooks, shared files, reproducibility) in [`AGENTS.md`](AGENTS.md).

## What the model does

```
          ┌────────────┐
clip ────►│   model    │────►  lie / not lie
(.mp4)    └────────────┘
```

- **Input**: one video clip — the 2–19 s segment of a game-show contestant answering a question, exactly
  the object described by one row of `Dolos.xlsx`.
- **Output**: a binary decision, `lie` (`deception` = 1) or `not lie` (`truth` = 0). The label is game
  ground truth, not an annotator's opinion ([`dataset/README.md` §2](dataset/README.md)).
- **Hard constraint**: everything the model consumes must be **computable from the clip at inference
  time**. A model that needs a human annotation to make a prediction is not a deception detector: the 41
  MUMIN columns in `train_dolos.xlsx` are annotations, not inputs.

Metrics: **accuracy, F1, AUC** — the same three the paper reports, so our numbers stay comparable.

**How the clip is turned into a prediction is each member's own research decision.** This document fixes
the problem, the data and the evaluation, and deliberately says nothing about models.

## Quick start

**Whoever prepares the data** — once, on the machine that holds the DOLOS annotations:

1. `zip -r dolos_annotations.zip DOLOS` from `dataset/`, upload it to the shared Drive folder;
2. open [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb) on Colab (CPU runtime),
   run it top to bottom: splits, 1,671 clips, audio, `dataset.csv`, all written into that folder;
3. share the folder **view-only** with the other three, plus write access on `cached_features/`;
4. post in the chat how many clips are `ok` and the sha256 of `dataset.csv`.

**Everybody else** — for each direction:

1. copy [`notebooks/architecture_template.ipynb`](notebooks/architecture_template.ipynb) into
   `notebooks/architectures/<direction>/train_<x>.ipynb`;
2. add a Drive shortcut named `dolos`, set `DIRECTION` and `ENCODER`, produce your features into
   `cached_features/<ENCODER>/`;
3. run the five folds, paste the results row into the board below.

## Layout

```
lie-detector/
├── README.md                  # this file
├── docs/                      # the long version of every section below
├── dataset/                   # DOLOS annotations (not in git) + their documentation
├── notebooks/
│   ├── preprocessing.ipynb    # splits + download + publish, one member once
│   ├── architecture_template.ipynb
│   └── architectures/<direction>/
├── scripts/                   # build_dataset.py, make_splits.py
├── lib/                       # dataset.py, metrics.py — imported by the notebooks
├── data/processed/            # local scratch copy of the shared folder
└── vendor/, baselines/        # submodules: tooling and the paper's code
```

## Documentation

| Page | What it answers |
|---|---|
| [`docs/data.md`](docs/data.md) | how the clips, the audio and `dataset.csv` are produced, and where they live on Drive |
| [`docs/splits.md`](docs/splits.md) | how the test set and the five folds are built, step by step |
| [`docs/notebooks.md`](docs/notebooks.md) | what a notebook reads, the `lib/` API, the rules per direction |
| [`docs/evaluation.md`](docs/evaluation.md) | what to measure, the baselines, the precision you can claim |
| [`docs/colab.md`](docs/colab.md) | Colab limits, session setup, the two things that bite |
| [`dataset/README.md`](dataset/README.md) | the DOLOS dataset itself: formats, statistics, 16 pitfalls |

## Results board

Filled in as directions land, from the **cross-validation** numbers the template prints. One row per
configuration; the best per direction in bold.

| Direction | Owner | Method | Trainable params | ACC (CV) | F1 | AUC | Balanced ACC | Majority | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| *reference* | — | majority class | — | 52.4 | — | — | 50.0 | — | — |

`ACC (CV)` is the mean ± spread over the five fold holdouts. `Majority` is the baseline **on the same
clips** (52.4% on the working set — not the 53.7% of the whole dataset): if a row does not beat it, the
model is worse than guessing the most frequent class.

The official 3-fold split is not run by the template. It stays available (`official_fold*.json`) for a
single reproduction of the paper's protocol, reported separately if someone does it.

---

## Plan to 26 November

| Dates | What | Output |
|---|---|---|
| 15–17 Nov | Directions assigned, `preprocessing.ipynb` written and run on the full set | artifacts + split files |
| 17–23 Nov | Each member trains and iterates on their own direction | one notebook + one result row each |
| 23–25 Nov | Cross-direction comparison, error analysis, subject-split numbers | results board complete |
| 25–26 Nov | Report, figures, freeze the repo | final deliverable |

The risky dependency is the first row: nothing can start before the artifacts exist. If preprocessing
slips, anyone can bootstrap on a ~300-clip subset with their own extraction — that is a debugging path,
not a result, because those numbers would not be comparable with the others.

## References

- Paper: Guo et al., *Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal
  Learning*, ICCV 2023 — <https://arxiv.org/abs/2303.12745>
- Dataset page: <https://rose1.ntu.edu.sg/dataset/DOLOS/>
- Official code (submodule): [`baselines/DOLOS-PECL`](baselines/DOLOS-PECL)
- Download / pre-processing tools (submodule): [`vendor/AV-Data-Processing`](vendor/AV-Data-Processing)
