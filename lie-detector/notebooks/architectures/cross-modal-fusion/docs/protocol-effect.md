# The protocol effect: one model, two splits

Read this if: you are comparing our numbers with the paper's, writing the evaluation section, or
wondering why the same model scores 56 on one split and 74 on another.

[`docs/evaluation.md`](../../../../docs/evaluation.md) states the rule — the official 3-fold is
clip-level and measures leakage, so it is never used for the board. This note is the **measurement**
behind that rule: what the leakage is worth, in points, on our own data, and which channel carries it.

## What was measured

One model family, one seed, two protocols, nothing else changed. Same feature caches, same training
loop, same number of epochs, same augmentation (none).

- **Models**: `late`, `video` and `audio` in [`model.py`](../model.py) — DINOv2 + WavLM frozen, mean
  over time, projection to 128, `LayerNorm`, MLP. 164,737 / 57,857 / 107,009 trainable parameters.
- **Data**: the clips with both cached arrays, ~1,158 of the 1,675 in the annotation set (the rest
  either failed to download or have no audio).
- **Seed**: 0. **Epochs**: 30. Early stopping on val in the subject protocol; fixed 30 epochs and
  threshold 0.5 in the official one, because that split has no validation set.
- **Command**: `VARIANT` as above, `SEEDS = [0]`, `SPLIT = "subject"` or `"official"` in
  [`train.ipynb`](../train.ipynb).

> An earlier run of `late` on the official split gave 65.5 / 71.9. It ran against a feature store
> built for the subject protocol, so the official test folds were partial: those numbers are
> **superseded** by the table below, which comes from a run with the store rebuilt. Switching
> `SPLIT` requires re-running the feature cell (or a restart), or the store still holds the wrong
> clip set.

| Variant | Seeds | Subject ACC | Subject AUC | Official ACC | Official AUC | Δ ACC | Δ AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `late` (fusion) | 3 | 53.0 ± 7.7 | 55.2 | 72.2 ± 3.0 | 80.4 | **+19.2** | **+25.2** |
| `video` (DINOv2) | 1 | 52.2 ± 5.5 | 53.7 | 72.8 ± 2.8 | 79.5 | **+20.6** | **+25.8** |
| `audio` (WavLM) | 1 / 3 | 54.7 ± 3.9 | 58.9 | 58.8 ± 2.0 | 62.5 | +4.1 | +3.6 |
| `load` (17 acoustic features, 609 params) | 3 | 50.8 ± 4.1 | 54.0 | 55.1 ± 2.1 | 58.8 | **+4.3** | **+4.8** |
| PECL PAVF (paper, Tab. 2) | — | — | 64.75 | 62.71 | — | — |
| PECL PAVF + multi-task (paper, Tab. 2) | — | — | 66.84 | 64.58 | — | — |

The paper's rows are quoted from its Table 2, not measured here: their backbone, their subset, their
20 epochs. Our rows are measured on our subset with the command above.

## The size of the gap

With the model, the code and the seed held constant, the official split is worth **+19 points of ACC
and +25 points of AUC** for the fusion, measured over three seeds on both sides. That difference is
the protocol, not the method.

The stability is part of the same story: the subject split gives a per-seed AUC of 52.5, 55.4 and
57.7, while the official split gives 79.6, 79.8 and 81.8. The leaky split is not only easier, it is
more reproducible — identity is a stable signal, behaviour is not.

## The gap is the visual channel, not the dataset

Two of the four rows barely move, and they are the two that never look at a face.

| Representation | Δ AUC from the leaky split | Parameters |
|---|---:|---:|
| DINOv2 + WavLM fusion | +25.2 | 165k trainable + 116M frozen |
| DINOv2 alone | +25.8 | 58k trainable + 22M frozen |
| WavLM alone | **+3.6** | 107k trainable + 94M frozen |
| 17 hand-chosen acoustic features | **+4.8** | 609, no frozen encoder |

Same dataset, same folds, same seeds, same training loop. What changes is what the model is allowed
to see, and the split between the two groups is a factor of six.

**The capacity explanation is ruled out by the audio rows.** WavLM carries 150,000 times more
parameters than the hand-crafted features — 94M frozen against 609 — and both gain about four points.
If the gap were a matter of model size, the encoder would have exploited the leak far more than the
17 numbers did. It does not, because there is nothing in the acoustics to exploit.

**What is in the pixels is the person.** The clip-level split places siblings of the same round on
both sides of the boundary: same speaker, same set, same costume, same twenty minutes of video. A
face encoder reads that identity directly. The 19–25 points were never a property of DOLOS; they were
a property of looking at the pixels of someone the model had already met.

It is not spread evenly. **The video channel gains +20.6 ACC and +25.8 AUC; the audio channel gains
+4.6 and +3.8.** Whatever the official split rewards, the visual stream sees it and the acoustic one
mostly does not.

## Why: the round is the unit

A DOLOS clip is one answer inside one round of the game show. The annotation set has **1,678 clips in
149 groups** of (episode, label) — a median of **11 clips per group**, and every clip belongs to a
group with at least two members. All the clips of a group share the label, the speaker, the clothes,
the set and often the same few seconds of video.

The official split is random over clips, so most test clips have a sibling of the same group in train.
A frozen face encoder only has to recognise the scene to read the label of the whole group — which is
exactly why the gain is visual. The audio carries the voice, not the set, and gains a seventh as much.

## The fusion adds nothing, on either split

The three-seed runs settle a question the one-seed runs left open.

| Split | `video` alone (1 seed) | `late`, video + audio (3 seeds) | Fusion advantage |
|---|---:|---:|---:|
| official | 72.8 | 72.2 | −0.6 |
| subject | 52.2 | 53.0 | +0.8 |

On the leaky split the fusion is *below* the visual branch alone; on the honest split it is half a
point above. Both differences are inside the fold-to-fold spread, so the honest statement is that
**combining the two channels neither helps nor hurts** — and the earlier reading, that fusion pays off
when the split is honest, came from single-seed runs and does not survive.

## The same effect on a model with no network at all

| Baseline | ACC | AUC |
|---|---:|---:|
| MUMIN + MLP, paper (Tab. 4) | 58.84 | — |
| MUMIN + logistic regression, our subject split | 51.1 | 52.5 |

Same 41 annotations, same model family, different split: **−7.7 points**. The paper's number is quoted
from its Table 4; ours comes from [`diagnostics.ipynb`](../diagnostics.ipynb).

## The paper already reports the honest number

Its Table 3 cross-tests PECL on other datasets, where the subjects are not the same:

| Test set | ACC |
|---|---:|
| Bag-of-Lies | 57.54 |
| Real-Life Trials | 56.17 |

Those sit in the same range as our subject-level estimate (56.5). The 64.75 is a *within-dataset*
figure, and the paper's own cross-dataset table shows what happens when the subjects change.

## What this does not show

- **Not that PECL is weak.** Our 73.8 is far above their 64.75 on the same split, and that is not a
  victory: a frozen DINOv2 is a much stronger identity encoder than a four-layer ViT-B/16, so our
  model exploits the leakage more. Comparing methods across different backbones on a leaky split
  measures the backbone, not the method.
- **Not intent.** The clip-level 3-fold is the convention of the field: Box of Lies, Real-Life Trials
  and MDPE all split with subjects on both sides. This is a systematic evaluation problem, not one
  group's shortcut.
- **Not a board number.** The official run reproduces the paper's protocol and is reported
  separately; `train.ipynb` refuses to print a board row for it.

## How to reproduce

```python
SPLIT = "official"     # settings cell
VARIANT = "late"       # or "video", "audio"
SEEDS = [0]
```

Then **restart and run all**: the feature store is built from the protocol's clip list, so switching
`SPLIT` without rebuilding it silently scores a partial test set. The campaign cell prints the variant
and its parameter count before the first fold — if that line does not say what you expect, the kernel
still holds the previous value.

Predictions land in `preds/cv_tia_<variant>_<split>.csv`, so the two protocols never overwrite each
other.

## Wording for the report

> On the paper's 3-fold clip-level split, our late-fusion baseline reaches 72.2 ACC / 80.4 AUC over
> three seeds, while the same model on a subject-independent split reaches 53.0 / 55.2. The gain is
> confined to the visual channel: DINOv2 alone moves from 52.2 / 53.7 to 72.8 / 79.5, while WavLM
> alone moves only from 54.7 / 58.9 to 58.8 / 62.5, and 17 hand-chosen acoustic features with 609
> parameters move from 50.8 / 54.0 to 55.1 / 58.8. The dataset has 1,678 clips in 149 episode-label
> groups (median 11 clips per group), so a random clip-level split places same-episode, same-label
> siblings on both sides of the boundary, and a frozen face encoder only has to recognise the round.
> The paper's own cross-dataset results (57.54 on Bag-of-Lies, 56.17 on Real-Life Trials) sit in the
> same range as our subject-level estimate, which suggests the in-dataset figure is a within-dataset
> benchmark rather than a generalisation estimate.
