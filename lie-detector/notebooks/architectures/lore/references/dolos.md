# DOLOS — Audio-Visual Deception Detection

Guo et al., *Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal
Learning*, ICCV 2023 · [arXiv 2303.12745](https://arxiv.org/abs/2303.12745) ·
[PDF](Audio-Visual-Deception-Detection-DOLOS.pdf)

## The idea

Deception research has two bottlenecks:

- **Data.** Datasets are small, often staged in a lab, and say little about *what* the liar does.
- **Method.** Large pretrained audio and vision models exist, but fine-tuning all their weights on a few
  hundred clips overfits.

The paper attacks both:

1. **A dataset.** The paper builds a new dataset from a TV game show, where people lie for real and the
   truth is always revealed.
2. **A model, PECL.** It keeps two big pretrained encoders frozen and trains only small modules added
   inside them. Some modules teach the encoders to look at *time*; others let audio and video talk to
   each other.

## The dataset

The show has a simple, useful structure:

1. A contestant reads a statement about their own life from a card. It may be true or false, and only
   they know which.
2. The opposing team questions them, and the contestant must defend the story.
3. The team guesses "truth" or "lie", and the answer is revealed.

The game gives three things a lab cannot:

- **Real motivation**: the contestant wants to win the round.
- **Spontaneous conversation**: the lie is built live, under questioning.
- **Certain labels**: the truth is always revealed at the end of the round.

**Clip selection.** A clip is kept only if the speaker talks clearly and the face is visible.

**Annotations.** Each clip is also labelled with non-verbal behaviours from the MUMIN coding scheme:

- facial: smile, frown, gaze direction, blinking, lip movements…
- vocal: pauses, repetitions, filled pauses ("ehm"), loudness…

These labels let a model learn *how* people behave, not only *whether* they lie.

**Protocols.** The paper defines three ways to split the data:

- a 3-fold train/test split;
- a duration split (short clips against long clips);
- a gender split (train on one gender, test on the other).

## The model (PECL)

```
face frames ─► 2D CNN ─► ViT layer 1 ─► … ─► ViT layer 4 ─┐
                             ⇅ PAVF          ⇅ PAVF         ├─► classifier ─► truth / lie
raw audio ──► 1D CNN ─► W2V2 layer 1 ─► … ─► W2V2 layer 4 ─┘   (+ MUMIN labels)
```

- **Frozen backbones.**
  - Video: ViT, an image model that sees each face frame separately.
  - Audio: wav2vec 2.0, a self-supervised speech model.
  - Only their **first 4 layers** are kept. The authors argue that deception cues are low-level
    (micro-expressions, voice quality) and that deeper layers add little.
- **UT-Adapter: teaching time.** Self-attention mixes all positions globally. Deceptive cues, though,
  are often *local in time*: a blink, a pause, a pitch jump. The UT-Adapter is a small side path
  placed beside each attention and MLP block:
  - squeeze the features to a small size;
  - run a 1D convolution **along time**, so each step sees its neighbours;
  - expand back to the original size.

  Only these adapters are trained, not the backbone.
- **PAVF: letting audio and video talk.** After each layer, the model learns a *correlation matrix*:
  which video frames relate to which audio frames. For example, a facial expression may matter only
  while a certain word is spoken. Each modality then attends to the other through this matrix. The
  fused features of all four layers feed the classifier.
- **Multi-task learning.** Besides "truth or lie", the model also predicts the MUMIN behaviours. This
  pushes it to notice smiles, pauses and so on, which in turn helps detect deception.

**Main message.** Small trainable modules plus cross-modal fusion beat full fine-tuning: about 7 % of the
parameters are trained, and accuracy is better.

## How it is used in `lore`

DOLOS is both the **data** and the **reference point** of `lore`:

- **Data.** The clips, labels and subject IDs come from the DOLOS annotations, through the shared
  `dataset.csv` (`lore/data.py`, paths in `lore/config.py`).
- **Same recipe.** `lore` follows the PECL recipe: frozen pretrained audio and video encoders, a few
  trainable parameters, and fusion. Each piece is swapped for a simpler or newer one:

  | PECL | `lore` | Where |
  |---|---|---|
  | ViT on single frames | VideoMAE on clips: time is built in ([videomae.md](videomae.md)) | `lore/models.py` · `VideoBranch` |
  | wav2vec 2.0 | HuBERT ([hubert.md](hubert.md)) | `lore/models.py` · `AudioBranch` |
  | UT-Adapter | LoRA on attention Q/V ([lora.md](lora.md)) | `lore/models.py` · `wrap_lora` |
  | PAVF cross-attention inside the encoders | **late fusion**: each branch is trained alone, and their scores are averaged with a weight chosen on validation | `lore/fusion.py` · `fuse_fold` |
  | MTCNN face crops | YuNet + SFace crops that follow *the speaker* only | `lore/faces.py` |
  | MUMIN multi-task | not used | — |
- **Protocol.** `lore` reruns the paper's 3-fold split (`--protocol official`), so the two can be
  compared fairly.
- **Main board.** It also uses a split where **test speakers never appear in training**. On that
  split the video branch falls to chance, which suggests that much of the paper's video score comes
  from recognising *who* is speaking. See the [`lore` README](../README.md#what-it-means) and
  [`docs/report.md`](../docs/report.md) §7.
