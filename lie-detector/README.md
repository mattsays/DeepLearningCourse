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

Repository-wide rules (language, dataset handling, submodules) are in [`../AGENTS.md`](../AGENTS.md),
and §3 there is the routing map for the whole repository. Project-level rules (data, notebooks, shared
files, reproducibility) are in [`AGENTS.md`](AGENTS.md).

## State

No generated artifact is in the repository. `data/processed/` does not exist; `dataset.csv`, `clips/`,
`audio/` and `splits/` live only in the shared Drive folder (see [`docs/data.md`](docs/data.md)). One
draft direction notebook exists, `notebooks/architectures/pecl.ipynb`; the results board below has one row so far (`lore`).
Anything that reads `data/processed/` fails until the preprocessing has run — that is expected.

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

Metrics: **accuracy, F1, AUC** — the same three the paper reports, so our numbers stay comparable. What
we add on top (balanced accuracy, the majority baseline, per-gender accuracy) and how each is computed is
[`docs/evaluation.md`](docs/evaluation.md).

**How the clip is turned into a prediction is each member's own research decision.** This document fixes
the problem, the data and the evaluation, and deliberately says nothing about models.

## Quick start

**Whoever prepares the data** — once, on the machine that holds the DOLOS annotations:

1. `zip -r dolos_annotations.zip DOLOS` from `dataset/`, upload it to the shared Drive folder;
2. open [`notebooks/preprocessing.ipynb`](notebooks/preprocessing.ipynb) on Colab (CPU runtime),
   run it top to bottom: clips, audio, `dataset.csv`, then the splits — all into that folder;
3. share the folder **view-only** with the other three, plus write access on `cached_features/`;
4. post in the chat how many clips are `ok` and the sha256 of `dataset.csv`.

**Everybody else** — for each direction:

1. copy [`notebooks/architecture_template.ipynb`](notebooks/architecture_template.ipynb) to
   `notebooks/architectures/<direction>/train_<direction>.ipynb`;
2. add a Drive shortcut named `dolos`, set `DIRECTION` and `ENCODER`, produce your features into
   `cached_features/<ENCODER>/`;
3. run the five folds and paste the row printed by `lib/metrics.py::format_row` into the board below.

## Layout

```
lie-detector/
├── README.md                  # this file
├── AGENTS.md                  # project rules
├── docs/                      # the long version of every section below
├── dataset/                   # DOLOS annotations (not in git) + their documentation
├── notebooks/
│   ├── preprocessing.ipynb    # splits + download + publish, one member once
│   ├── architecture_template.ipynb
│   └── architectures/<direction>/train_<direction>.ipynb
├── scripts/                   # build_dataset.py, make_splits.py
├── lib/                       # dataset.py, metrics.py — imported by the notebooks
├── data/processed/            # written by the scripts — absent until you run them
└── vendor/, baselines/        # submodules: tooling and the paper's code
```

## Read this, not everything

Each row names the document that owns the answer. Do not read the dataset README for a question about
metrics, and do not restate a number that one of these files owns.

| If you are… | Read | Section |
|---|---|---|
| writing a loader, or touching a DOLOS column | [`dataset/README.md`](dataset/README.md) | §4, §8, §10 |
| running the preprocessing | [`docs/data.md`](docs/data.md), [`docs/colab.md`](docs/colab.md) | all |
| touching splits, seeds or leakage | [`docs/splits.md`](docs/splits.md) | all |
| reporting a number | [`docs/evaluation.md`](docs/evaluation.md) | all |
| writing a direction notebook | [`docs/notebooks.md`](docs/notebooks.md) | all |
| choosing an architecture | [`docs/architectures.md`](docs/architectures.md) | all (report material, not a spec) |
| asking what the dataset actually contains | [`dataset/README.md`](dataset/README.md) | §5, §6 |

## Directions

The short name is what goes in `DIRECTION` and in the notebook folder. Backbones are listed for
orientation; the shared feature cache is named after the encoder that produced it
(`cached_features/<ENCODER>/`, one folder per encoding model).

| Direction | Paradigm | Backbones | Owner | Notebook (target) |
|---|---|---|---|---|
| `tia` | 1 — frozen backbones + cross-modal fusion | `dinov2-small` / `siglip-base`, `wavlm-base-plus` | — | `cross-modal-fusion/train.ipynb` |
| `lore` | 2 — spatio-temporal Transformer + LoRA | `videomae-base`, `hubert-base` | Lorenzo | `videomae-hubert-lora/train_lore.ipynb` |
| `fab` | 3 — 3D CNN + temporal head | `marlin-vit-base` / `convnext_tiny`, mel-spectrogram | Fabio | `marlin-convnext/train_marlin.ipynb` |

Only three paradigms are documented so far ([`docs/architectures.md`](docs/architectures.md)); add the
fourth row here when it is assigned. Fill the Owner column in the same commit. `notebooks/architectures/pecl.ipynb`
is an early draft of paradigm 1 that predates this naming: its owner moves it to `tia/train_tia.ipynb`
before the next run, rather than starting a second notebook for the same direction.

## Results board

Filled in as directions land, from the **cross-validation** numbers the template prints. One row per
configuration; the best per direction in bold.

| Direction | Owner | Method | Trainable params | ACC (CV) | F1 | AUC | Balanced ACC | Majority | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`lore`** | Lorenzo | VideoMAE-B + HuBERT-B · LoRA · late fusion | 594,434 | 56.1 ± 7.3 | 55.1 | 60.6 | 56.2 | 52.1 | 114 min train (Tesla T4) |
| **`fab`** | Fabio | MARLIN-B (Faces) + ConvNeXt-T · TCN · Mean Pool | 97,857 | 54.1 ± 3.2 | 58.5 | 56.4 | 54.7 | 52.4 | ≈15 s (Tesla T4) |
| | | | | | | | | | |
| | | | | | | | | | |
| *reference* | — | majority class | — | 52.4 | — | — | 50.0 | — | — |

`ACC (CV)` is the mean ± spread over the five fold holdouts. `Majority` is the baseline **on the same
clips** (52.4% on the working set — not the 53.7% of the whole dataset): if a row does not beat it, the
model is worse than guessing the most frequent class.

The official 3-fold split is not run by the template and is **not** part of this board. It stays
available (`official_fold*.json`, `Protocol.official(fold)`) for a single reproduction of the paper's
protocol, reported separately if someone does it — see [`docs/evaluation.md`](docs/evaluation.md).

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
