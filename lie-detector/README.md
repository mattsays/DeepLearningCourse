# Lie Detection on DOLOS

Project README for `lie-detector/`: the problem, the shared data pipeline, and the rules that keep four
independent experiments comparable.

| | |
|---|---|
| **Input** | one video clip from DOLOS (`.mp4`, 2–19 s) |
| **Output** | `lie` / `not lie` — binary `deception` (1) vs `truth` (0) |
| **Team** | 4 members, one research direction each |
| **Compute** | Google Colab only — nobody has a usable local GPU |
| **Deadline** | **26 November** (11 days from kickoff on 15 November) |
| **Dataset** | [`dataset/README.md`](dataset/README.md) — read it before writing any code |

Repository-wide rules (language, dataset handling, submodules) are in [`../AGENTS.md`](../AGENTS.md);
project-level ones (data, notebooks, shared files, reproducibility) in [`AGENTS.md`](AGENTS.md). Short
version: everything in English, the DOLOS folder is **never committed**, third-party code stays in
`vendor/` as submodules.

---

## 1. What the model does

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

---

## 2. Repository layout

```
lie-detector/
├── README.md                  # this file
├── AGENTS.md                  # project-level rules
├── dataset/
│   ├── README.md              # dataset documentation (formats, statistics, pitfalls)
│   └── DOLOS/                 # annotations — NOT tracked by git, no videos inside
├── notebooks/
│   ├── preprocessing.ipynb    # SHARED: clips → decoded frames + audio for everybody
│   └── architectures/         # one sub-folder per research direction, named by its author
├── data/                      # generated artifacts — NOT tracked
├── clips/                     # downloaded .mp4 — NOT tracked
└── vendor/                    # git submodules: AV-Data-Processing, DOLOS-PECL
```

Two things are strictly one-per-person, one-per-team:

- **`notebooks/preprocessing.ipynb` is shared and singular.** It is the only place where the raw data
  work happens: download, clip cutting, decoding, split generation. Nobody re-downloads or re-decodes
  clips inside an architecture notebook.
- **`notebooks/architectures/<direction>/` is private to one member**, and the folder name is theirs
  too. Each folder holds that person's Colab notebook plus a one-page summary of results.

---

## 3. Shared preprocessing (single notebook)

Goal: **after preprocessing, every member works from the same decoded clips**, which is the only way
four people without GPUs can iterate before 26 November.

The shared notebook does *data* work only. It stops at decoded, ready-to-load material:

```
clips/<file_name>.mp4                  # the clip itself, cut from the episode
data/processed/frames/<file_name>/     # decoded RGB frames
data/processed/audio/<file_name>.wav   # 16 kHz mono
data/processed/mumin.parquet           # annotations (auxiliary / reference only)
data/processed/splits/*.json           # official 3-fold + subject-level split
data/processed/manifest.csv            # processed / failed clips, runtime per stage
```

Proposed contract — to be confirmed when the notebook is written:

| Artifact | Content |
|---|---|
| `clips/` | one `.mp4` per clip that downloaded successfully |
| `frames/` | RGB frames at a documented, fixed rate, same naming as the clip |
| `audio/` | 16 kHz mono wav extracted from the clip's audio track |
| `mumin.parquet` | `file_name, y, gender` + the 41 MUMIN columns — **never an inference input** (§1) |
| `splits/` | clip lists for the official 3-fold **and** a subject-level split |
| `manifest.csv` | which clips succeeded/failed and why, plus runtime |

**Why it stops there.** Any further step — cropping faces, sampling frames differently, computing
features, caching embeddings — already encodes a modelling choice, and this project is precisely about
letting four different choices compete. Each member does that inside their own notebook and caches
whatever they like under `data/processed/<their-direction>/`. Two members who happen to want the same
intermediate representation can share it privately; the shared notebook does not pick one.

Rules for the notebook:

1. **Idempotent and resumable** — it must skip clips already processed, because Colab sessions die.
2. **Mount Drive first** (`from google.colab import drive; drive.mount('/content/drive')`) and keep
   artifacts there; `/content` is wiped on disconnect.
3. **Deterministic** — seed everything, sort file lists before iterating, no shuffling when writing.
4. **Validated** — drop the 7 clips with `end ≤ start`, and report how many of the 75 episodes are still
   downloadable. The 1,678 annotation rows shrink in practice; record the final count.
5. **One run produces everything.**
6. **Same clip set for everybody** — if a clip fails processing, it is dropped from *all* directions, so
   the four results stay comparable.

---

## 4. Individual notebooks (one per direction)

Each member owns one folder under `notebooks/architectures/` and delivers one notebook that runs
top-to-bottom on a fresh Colab runtime, plus a short `README.md` with their result row and a paragraph of
interpretation.

Minimum content of a notebook:

1. **Setup cell** — mount Drive, install pinned dependencies, define `DATA_DIR`.
2. **Data cell** — load the shared artifacts, assert clip counts; print the split sizes.
3. **Model cell** — the method, with the number of **trainable** parameters printed.
4. **Training cell** — fixed seeds, early stopping, checkpoint saved to Drive.
5. **Evaluation cell** — ACC / F1 / AUC on the test fold, plus a confusion matrix and a per-gender
   breakdown.
6. **Results cell** — the row to paste into the results board (§7).

Rules that keep the four results comparable:

- **Consume the shared artifacts; do not re-download or re-decode** clips in this notebook.
- **Report cost, not just score**: trainable parameters, wall-clock per epoch, and the Colab GPU used.
  At this scale "same accuracy, 5× cheaper" is a legitimate result.
- **Keep it runnable on a ~200-clip subset** behind a flag, so debugging costs 2 minutes and not 2 hours.

---

## 5. Evaluation protocol

**Primary split — the official 3-fold** (`dataset/DOLOS/Training_Protocols/train_fold*.csv` /
`test_fold*.csv`). Report mean ± std across the three folds. This is what makes our numbers comparable
with the paper.

**Secondary split — subject-level.** ⚠️ The official folds are **clip-level**: train and test share 147
subjects and all 75 episodes, and 1–2 clips leak across the boundary. Any suspiciously high accuracy is
leakage, not skill. So the preprocessing notebook must also emit a split grouped by *subject*, and every
member reports both numbers. The gap between the two is one of the most interesting results this project
can produce.

Metrics, in this order: **ACC, F1, AUC**, then per-gender ACC (male clips are 79% of the data and show
56.6% deception vs 43.8% for female clips — a model that ignores this will look fine and be useless).

Calibration, for orientation only:

| Reference | ACC |
|---|---|
| Majority class (`deception`) | 53.7% |
| Best model in the paper, on these folds | 64.75% |

---

## 6. Compute on Colab

Nobody has a local GPU, so the workflow is built around Colab's limits: a free T4, sessions that die
after a few idle hours, and a 12-hour ceiling.

- **Save everything to Drive**: checkpoints, caches, logs, result JSONs. `/content` is disposable.
- **Pin versions** (`%pip install -q transformers==<x> torch==<y>`) — Colab images change without notice
  and a notebook that breaks the day before the deadline is a lost direction.
- **Small first.** Debug on 200 clips, then scale to the full split.
- **Record the runtime** (GPU type, package versions) in your result row; it is part of reproducibility.
- Stay within roughly 8 GB of GPU memory and a single-digit number of hours per experiment, or the
  experiment will not be reproducible by someone else in the group.

---

## 7. Results board

Filled in as directions land. One row per configuration; the best per direction in bold.

| Direction | Owner | Method | Trainable params | ACC (3-fold) | F1 | AUC | ACC (subject split) | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| *reference* | — | majority class | — | 53.70 | — | — | — | — |

---

## 8. Plan to 26 November

| Dates | What | Output |
|---|---|---|
| 15–17 Nov | Directions assigned, `preprocessing.ipynb` written and run on the full set | artifacts + split files |
| 17–23 Nov | Each member trains and iterates on their own direction | one notebook + one result row each |
| 23–25 Nov | Cross-direction comparison, error analysis, subject-split numbers | results board complete |
| 25–26 Nov | Report, figures, freeze the repo | final deliverable |

The risky dependency is the first row: nothing can start before the artifacts exist. If preprocessing
slips, anyone can bootstrap on a ~300-clip subset with their own extraction — that is a debugging path,
not a result, because those numbers would not be comparable with the others.

---

## 9. References

- Dataset documentation: [`dataset/README.md`](dataset/README.md)
- Paper: Guo et al., *Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal
  Learning*, ICCV 2023 — <https://arxiv.org/abs/2303.12745>
- Dataset page: <https://rose1.ntu.edu.sg/dataset/DOLOS/>
- Official code (submodule): [`vendor/DOLOS-PECL`](vendor/DOLOS-PECL)
- Download / pre-processing tools (submodule): [`vendor/AV-Data-Processing`](vendor/AV-Data-Processing)
