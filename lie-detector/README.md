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
├── vendor/                    # submodule: download / extraction tools
└── baselines/                 # submodule: official reference implementation
```

Two things are strictly one-per-person, one-per-team:

- **`notebooks/preprocessing.ipynb` is shared and singular.** It is the only place where the raw data
  work happens: download, clip cutting, decoding, split generation. Nobody re-downloads or re-decodes
  clips inside an architecture notebook.
- **`notebooks/architectures/<direction>/` is private to one member**, and the folder name is theirs
  too. Each folder holds that person's Colab notebook plus a one-page summary of results.

---

## 3. Shared preprocessing

Goal: **after preprocessing, every member works from the same clips and the same splits**, which is the
only way four people without GPUs can iterate before 26 November.

The work is implemented by two scripts; the notebook only drives them.

```bash
python scripts/build_dataset.py --workers 4     # clips + audio + metadata  (§3.1)
python scripts/make_splits.py                   # frozen splits            (§3.2)
```

```
data/
└── processed/
    ├── clips/<clip_id>.mp4      # the annotated range, video + audio track
    ├── audio/<clip_id>.wav      # 16 kHz mono
    ├── dataset.csv              # one row per clip: label, subject, paths, 41 MUMIN features
    ├── splits/*.json            # frozen protocol — published with the dataset
    └── <direction>/             # a member's own caches
```

| Artifact | Content |
|---|---|
| `clips/` | one `.mp4` per clip that downloaded successfully |
| `audio/` | 16 kHz mono wav extracted from the clip's audio track |
| `dataset.csv` | label, `y`, gender, subject, both paths, download status, **41 MUMIN columns** |
| `splits/` | the virgin test set and the 5 cross-validation folds |
| `cached_features/<encoder>/` | features produced by one encoding model (see §3.5) |

**Why it stops there.** Any further step — cropping faces, sampling frames differently, computing
features, caching embeddings — already encodes a modelling choice, and this project is precisely about
letting four different choices compete. Each member does that inside their own notebook and caches
whatever they like under `data/processed/<direction>/`. Two members who happen to want the same
intermediate representation can share it privately; the shared pipeline does not pick one.

Rules:

1. **Only the scripts touch the raw data.** Nobody re-downloads or re-decodes clips inside an architecture
   notebook.
2. **Idempotent and resumable** — `build_dataset.py` skips what is already on disk, because Colab sessions
   die mid-run.
3. **Mount Drive first** (`from google.colab import drive; drive.mount('/content/drive')`) and keep
   artifacts there; `/content` is wiped on disconnect.
4. **Same clip set for everybody** — filter by `metadata.status == "ok"` at load time, always the same way,
   or two members train on different subsets and nothing is comparable.

### 3.1 Canonical dataset (`scripts/build_dataset.py`)

One script turns the annotations into the files every notebook reads:

```
clips/<clip_id>.mp4                   # annotated range only, video + audio track
data/processed/audio/<clip_id>.wav    # 16 kHz mono
data/processed/dataset.csv            # one row per clip: metadata + MUMIN — the source of truth
```

`clip_id` is the lower-case slug of the DOLOS `file_name`, so it can always be recomputed from the
dataset files alone: `AN_WILTY_EP15_truth1` → `an_wilty_ep15_truth1`. The single duplicated name in
DOLOS becomes `yw_wilty_ep46_truth1` and `yw_wilty_ep46_truth1_2`.

`dataset.csv` carries, per clip: the 17 metadata columns above (`clip_id, source_name, yt_id, start_s,
end_s, duration_expected, duration_actual, label, y, gender, subject, test_fold, protocols, video_path,
audio_path, status, error`) plus the **41 binary MUMIN features** as `mumin_*` columns. Paths are relative
to the folder that contains the CSV, so the archive can be extracted anywhere.

`dataset.csv`, `clips/` and `audio/` are published together in the shared Drive folder (§3.5).

```bash
python scripts/build_dataset.py --dry-run                  # writes dataset.csv, downloads nothing
python scripts/build_dataset.py --limit 20                 # smoke test
python scripts/build_dataset.py --workers 4                # full run
```

The script is idempotent (valid existing clips are skipped), resumable, and marks broken annotations
(`end ≤ start`, 7 rows) as `invalid_range` instead of downloading them. Budget roughly 15 s per clip:
a full run is a couple of hours with 4 workers.

### 3.2 Splits (`scripts/make_splits.py`)

```bash
python scripts/make_splits.py --test-fraction 0.15 --folds 5 --val-fraction 0.15
```

```
<shared folder>/splits/
├── official_fold{1,2,3}.json   # straight from the DOLOS CSVs, for the paper comparison
├── subject_test.json           # the reserved test set
├── subject_fold{1..5}.json     # train / val / holdout per fold
└── manifest.json               # seed, counts, sha256 of every file
```

Written straight into the shared folder by the preprocessing notebook (§3.5). **Not in git**: one copy, on
Drive, read by everybody with `Protocol.load(DATA / "splits")`.

#### How the subjects are assigned

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

On the shipped annotations — 180 distinct subjects after merging the name aliases:

| Block | Subjects | Clips | Deception | Role |
|---|---:|---:|---:|---|
| test | 27 | 182 | 47.5% | reserved, not used by the direction notebooks |
| working, 5 folds | 151 | 1,021 | 47.7% | cross-validation: the results board |
| captains, train only | 2 | 468 | 69.0% | never evaluated |

Per fold: `train` ≈ 102 subjects / 1,160 clips (captains included), `val` ≈ 18 / 124,
`holdout` ≈ 30 / 204. Every working subject lands in exactly one `holdout` and in no `val`.

#### What a fold file contains

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

#### Checks the script runs

```
checks: 151 working subjects partitioned once, no overlap, no leak
```

It asserts that the folds partition the working set exactly once, that no test subject appears in a fold,
and that no captain is in the test set. If any of those fail, it exits without writing.

#### Integrity

Same annotations + same `--seed` = byte-identical files (verified: regenerating produces no git diff).
`manifest.json` records the sha256 of every file and `Protocol.load` verifies it, so a hand edit raises.
What it cannot catch is a *coherent regeneration* — someone re-running the script with another seed — which
is why the shared folder is **view-only** for the rest of the team.

### 3.3 Loading in a notebook (`lib/dataset.py`)

```python
import sys; sys.path.insert(0, "/content/lie-detector")
from lib.dataset import Protocol, Clips, CachedFeatures, make_loader

folds = Protocol.load()          # verifies the manifest hashes, raises if the split was edited
clips = Clips.load()             # annotated clips; clips.available() = files actually on disk

train, val, holdout = folds.subject(1)
ds = CachedFeatures(val, clips, cache="data/processed/<your-features>")
dl = make_loader(ds, batch_size=16)
```

The module hands out **paths** (`clips.paths(ids)`) or **cached arrays** (`CachedFeatures`), never a
fixed tensor pipeline: decoding a clip, picking frames, resizing, normalising is each direction's job.

Two helpers that prevent the usual mistakes:

- `clips.describe(ids)` — clips, subjects, % deception and the **majority baseline of that subset**;
- `clips.class_weights(ids)` — `pos_weight` ready for `BCEWithLogitsLoss`.

⚠️ On our splits the training set is **56.3% deception** while the holdout is **47.3%**: the two captains
(69% deception) are 40% of the training clips. A plain BCE trained on that over-predicts `deception`.
Either pass `pos_weight`, or pick the decision threshold on `val` instead of using 0.5.

`python -m lib.dataset --fold 1` prints the composition of every split against what is on disk.

### 3.4 The two notebooks

| Notebook | Who runs it | What it does |
|---|---|---|
| `notebooks/preprocessing.ipynb` | **one member, once** | runs both scripts with explicit paths, so clips, audio, splits and `dataset.csv` are written straight into the shared folder |
| `notebooks/architecture_template.ipynb` | everybody | skeleton to copy into `architectures/<direction>/` |

The template does the plumbing once: mount Drive, point at the shared folder, load the frozen splits, five
folds with early stopping on `val`, metrics from `lib/metrics.py`, per-gender breakdown, and predictions
saved to `preds/cv_<direction>.csv`. Members replace only the model and the feature loader, then paste the
results row it prints.

### 3.5 The shared folder on Drive

One folder, one path, for everybody:

```
dolos/                          # /content/drive/MyDrive/dolos
├── dataset.csv                 # 1 MB
├── clips/                      # ~510 MB, 1,671 files
├── audio/                      # ~270 MB, 1,671 files
├── splits/                     # the frozen protocol
└── cached_features/            # one sub-folder per encoding model
    ├── wav2vec2-base/<clip_id>.npy
    └── vit-b16/<clip_id>.npy
```

The owner shares it, everybody else adds a **shortcut** named `dolos` in their own My Drive, so the path
`/content/drive/MyDrive/dolos` is identical in every notebook. There is no archive to extract: the folder
*is* the dataset.

`cached_features/` is named after the **encoder**, not after the direction: two people using the same model
land in the same folder for free, and a third can see what already exists before recomputing it.

The rule for reading it: **anything read once per epoch must come from RAM or local disk**. `dataset.csv`
and the feature caches are fine over the mount — `CachedFeatures` preloads a cache into RAM when it is
under 1 GB, so features cost nothing per epoch. The 1,671 `mp4`/`wav` files do not: decoding video from the
mount pays a network round-trip per file, so a video direction should copy `clips/` to local disk first.

Since nothing is versioned, the loader prints the identity of the table it is reading:

```
dataset.csv sha256:a1027d7ef46d  1678 clips  2 with media
```

Post it in the chat. Two members with different fingerprints are not training on the same dataset, and
nothing they report is comparable.

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

**Secondary split — our subject-level protocol** (`subject_test.json` + `subject_fold*.json`, see §3.2).
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
- Official code (submodule): [`baselines/DOLOS-PECL`](baselines/DOLOS-PECL)
- Download / pre-processing tools (submodule): [`vendor/AV-Data-Processing`](vendor/AV-Data-Processing)
