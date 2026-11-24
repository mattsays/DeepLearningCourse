# `load` — cognitive load and vocal arousal

A direction that tests a **theory** rather than a backbone: seventeen acoustic features chosen because
the deception literature says they should differ between truth and lies, given to a 600-parameter
classifier. The comparison is against `tia`, where a frozen 94M-parameter WavLM reaches 58.9 AUC on
the same split.

Why these features, and what the evidence actually supports: [`docs/theory.md`](docs/theory.md).
What each feature measures: [`docs/features.md`](docs/features.md).

**Status: run, three seeds, both protocols.**

## Results

| Protocol | ACC | AUC | Per-seed AUC | Trainable |
|---|---:|---:|---|---:|
| subject (5 fold) | 53.1 ± 4.7 | 53.6 | 57.0 · 51.6 · 52.2 | 609 |
| official (3 fold) | 56.5 ± 1.7 | 59.2 | 59.6 · 59.2 · 58.9 | 609 |
| *majority class* | 52.0 | — | — | — |
| *MUMIN oracle* (41 hand annotations) | 51.1 | 52.5 | — | — |

These rows come from [`run_load.py`](run_load.py) (`results/load-{official,subject}/`): the same
features, model and training rules as the notebook, with the cache built for **all** 1,452 clips. The
notebook's cache held only the 1,021 working-set clips, so its runs trained without the two captains
and scored 1,132 of the 1,427 official test clips (50.8 / 54.0 subject, 55.1 / 58.8 official).

## What it means

**The protocol effect almost disappears.** Seventeen acoustic features gain **+3.4 ACC and +5.6 AUC**
from the leaky split; the frozen-encoder fusion gains **+19.2 and +25.2** on the same folds and seeds.
That is the cleanest evidence available that the 19 points were never a property of the dataset: they
were a property of looking at the pixels of a person the model had already met. Acoustic statistics
carry no face, no set, no costume, so there is nothing to leak. The full comparison is in
[`docs/protocol-effect.md`](../cross-modal-fusion/docs/protocol-effect.md).

**The features do not beat the encoder, but they come close for free.** Subject AUC is 53.6 against
55.2 for the fusion and 58.9 for WavLM alone — so a 94M-parameter frozen encoder buys roughly 1 to 5
points over 17 hand-chosen numbers and 609 parameters. Both are within noise of each other at this
sample size.

**One fold collapses.** On fold 3 the AUC is below 0.5 on all three seeds (48.3, 44.9, 47.6): the
features rank that fold's speakers the wrong way round. In the notebook's run the same fold scored
47.6, 45.7 and 45.7 ACC against a 54.3 majority, so the failure does not depend on the cache.

## Honest reading

At 53.6 subject AUC against a 52.5 oracle, these features are doing as well as the annotated
behaviour allows and no better. That is consistent with everything else measured on this dataset: the
behaviour is visible, and it does not carry the label. It is *not* evidence that the cognitive-load
theory is wrong — the theory's stronger half is eliciting cues through the interview, which this
notebook cannot do, because the questions are not in the data.

## The two theories

| Theory | Cue | Measured by |
|---|---|---|
| Cognitive load (Vrij) | more and longer pauses, slower speech | `pause_*`, `onsets_per_s`, `voiced_ratio` |
| Vocal arousal | higher and more variable pitch, noisier voice | `f0_*`, `jitter`, `shimmer`, `hnr_mean` |

The honest caveat is in the theory note: Vrij's approach works best when the interviewer **elicits**
load. A classifier can only observe, and this direction tests the weaker half.

## Files

| File | What it is |
|---|---|
| [`features.py`](features.py) | the 17 features, one function, no GPU |
| [`train_load.ipynb`](train_load.ipynb) | builds the cache, trains, evaluates, prints the board row |
| [`run_load.py`](run_load.py) | the same pipeline headless, on every clip and both protocols; writes `results/` |
| [`docs/theory.md`](docs/theory.md) | the theories, the evidence, the ceiling that applies |
| [`docs/features.md`](docs/features.md) | every feature, its units and expected direction |

## How to run

```
train_load.ipynb   # sections 3 (cache), 6 (one fold), 7 (all folds)
```

The cache goes to `cached_features/acoustic-load/` in the shared folder, next to the DINOv2 and WavLM
ones — same convention, so `CachedFeatures` reads it without knowing what is inside.

Requires `praat-parselmouth` (installed by the notebook). The energy family works without it, and the
Praat family returns `NaN` if it is missing.

## Reproducing

```
train_load.ipynb   # section 3 builds the cache, 6 one fold, 7 all folds and seeds
```

Predictions land in `preds/cv_load_<split>.csv`. The board row is printed by the last cell and goes
into `lie-detector/README.md` through a pull request, since that file is shared.
