# Notebooks and the `lib/` API

What a notebook reads, which files it may touch, and the two notebooks that exist.
Read this if you are about to write or change a direction notebook.

## Loading in a notebook (`lib/dataset.py`)

```python
import sys; sys.path.insert(0, "/content/DeepLearningCourse/lie-detector")
from lib.dataset import Protocol, Clips, CachedFeatures, make_loader

folds = Protocol.load()          # verifies the manifest hashes, raises if the split was edited
clips = Clips.load()             # annotated clips; clips.available() = files actually on disk

train, val, holdout = folds.subject(1)
ds = CachedFeatures(val, clips, cache="data/processed/<your-features>")
dl = make_loader(ds, batch_size=16)
```

The module hands out **paths** (`clips.paths(ids)`) or **cached arrays** (`CachedFeatures`), never a
fixed tensor pipeline: decoding a clip, picking frames, resizing, normalising is each direction's job.

The two modes differ in what the loader yields, and that reaches into your training loop:
`CachedFeatures` gives `(x, y)` pairs, `clips.paths(ids)` gives one dict per clip
(`{clip_id, y, video, audio}`). `make_loader` picks the right collate on its own, but a loop written for
`for x, y in dl` needs a branch to read the dict — the template implements the tensor path only, so a
direction that decodes video replaces the loop, not just the model.

Two helpers that prevent the usual mistakes:

- `clips.describe(ids)` — clips, subjects, % deception and the **majority baseline of that subset**;
- `clips.class_weights(ids)` — `pos_weight` ready for `BCEWithLogitsLoss`.

⚠️ On our splits the training set is **56.3% deception** while the holdout is **47.3%**: the two captains
(69% deception) are 40% of the training clips. A plain BCE trained on that over-predicts `deception`.
Either pass `pos_weight`, or pick the decision threshold on `val` instead of using 0.5.

`python -m lib.dataset --fold 1` prints the composition of every split against what is on disk.

## The two notebooks

| Notebook | Who runs it | What it does |
|---|---|---|
| `notebooks/preprocessing.ipynb` | **one member, once** | runs both scripts with explicit paths, so clips, audio, splits and `dataset.csv` are written straight into the shared folder |
| `notebooks/architecture_template.ipynb` | everybody | skeleton to copy into `architectures/<direction>/` |

The template does the plumbing once: mount Drive, point at the shared folder, load the frozen splits, seed
everything, five folds with early stopping on `val`, metrics from `lib/metrics.py`, per-gender breakdown,
predictions saved to `preds/cv_<direction>.csv`, and the board row printed by
`lib/metrics.py::format_row`. Members replace only the model and the feature loader, then paste that row.

## Individual notebooks (one per direction)

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
6. **Results cell** — the row to paste into the results board ([evaluation.md](evaluation.md)).

Rules that keep the four results comparable:

- **Consume the shared artifacts; do not re-download or re-decode** clips in this notebook.
- **Report cost, not just score**: trainable parameters, wall-clock per epoch, and the Colab GPU used.
  At this scale "same accuracy, 5× cheaper" is a legitimate result.
- **Keep it runnable on a ~200-clip subset** behind a flag, so debugging costs 2 minutes and not 2 hours.

