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
| subject (5 fold) | 50.8 ± 4.1 | 54.0 | 56.4 · 53.1 · 52.7 | 609 |
| official (3 fold) | 55.1 ± 2.1 | 58.8 | 58.5 · 59.1 · 58.6 | 609 |
| *majority class* | 52.0 | — | — | — |
| *MUMIN oracle* (41 hand annotations) | 51.1 | 52.5 | — | — |

## What it means

**The protocol effect almost disappears.** Seventeen acoustic features gain **+4.3 ACC and +4.7 AUC**
from the leaky split; the frozen-encoder fusion gains **+19.2 and +25.2** on the same folds and seeds.
That is the cleanest evidence available that the 19 points were never a property of the dataset: they
were a property of looking at the pixels of a person the model had already met. Acoustic statistics
carry no face, no set, no costume, so there is nothing to leak. The full comparison is in
[`docs/protocol-effect.md`](../cross-modal-fusion/docs/protocol-effect.md).

**The features do not beat the encoder, but they come close for free.** Subject AUC is 54.0 against
55.2 for the fusion and 58.9 for WavLM alone — so a 94M-parameter frozen encoder buys roughly 1 to 5
points over 17 hand-chosen numbers and 609 parameters. Both are within noise of each other at this
sample size.

**One fold collapses.** On fold 3 the model scores 47.6, 45.7 and 45.7 ACC against a majority
baseline of 54.3 — worse than always answering "truth", on all three seeds. That fold has 45.7%
deception, the lowest of the five, and the model over-predicts deception there. The `f1` column in
the same folds (0.28, 0.61, 0.28) shows the same thing: the operating point, not the ranking.

## Honest reading

At 54.0 subject AUC against a 52.5 oracle, these features are doing as well as the annotated
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
