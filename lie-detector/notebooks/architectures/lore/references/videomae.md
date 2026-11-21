# VideoMAE — Masked Autoencoders for Video

Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video
Pre-Training*, NeurIPS 2022 · [arXiv 2203.12602](https://arxiv.org/abs/2203.12602) ·
[PDF](VideoMAE.pdf)

## The idea

Vision Transformers need a lot of data. For video, the usual trick is to start from a model trained on
millions of labelled *images*, but then the model has never really learned about *motion*. VideoMAE
learns from video directly, **without labels**, with a simple game:

> Hide almost all of a video, and ask the model to redraw the missing parts.

To redraw what is hidden from the little that is visible, the model must understand the scene: what
objects are there, how they move, and what happens next. The representation it builds along the way is
what we reuse.

## How it works

```
16 frames ─► cut into 2×16×16 cubes ─► hide 90 % (tube mask) ─► encoder (ViT) ─► decoder ─► rebuild pixels
                                                    │
                                    after pretraining: keep only the encoder
```

- **Cubes, not patches.** An image ViT cuts a picture into 16×16 patches. VideoMAE cuts the clip into
  small **cubes**: 2 frames × 16 × 16 pixels. Each cube is one token, so a token already contains a bit
  of motion. A 16-frame 224×224 clip becomes 8 × 14 × 14 = 1568 tokens.
- **Space-time attention.** Every token can attend to every other token, in any frame. The model
  therefore relates *where* and *when* together. A per-frame image model cannot do this.
- **Hide 90 % or more.** In images, hiding 75 % is already hard. Video, though, is very *redundant*:
  consecutive frames are almost identical. With a normal ratio, the task would be too easy, so VideoMAE
  hides 90–95 %.
- **Tube masking.** If each frame were masked differently, the model could cheat: it would copy a
  hidden patch from the same spot in the next frame, where it happens to be visible. VideoMAE hides the
  **same spatial positions in every frame** (a "tube" through time). A hidden region is then hidden in
  all frames, and the model must actually *reason* about content and movement to fill it.
- **Cheap encoder, throwaway decoder.** The encoder only processes the ~10 % of visible tokens, which
  makes pretraining fast. A small decoder rebuilds the pixels. After pretraining, the decoder is
  discarded and the **encoder is the useful model**.

## What the paper shows

- **Data-efficient.** This pretraining works even on small video datasets (a few thousand clips).
  Contrastive methods and training from scratch fail there.
- **Quality over quantity.** Pretraining on video *similar to the target* matters more than
  pretraining on *more* video. A **domain gap** between pretraining and target data hurts.
- **Strong results.** With a plain ViT, it beats previous methods on action recognition benchmarks
  (Kinetics-400, Something-Something).

## How it is used in `lore`

VideoMAE is the **video branch** of `lore`: it turns the speaker's face over time into one vector.

- **Checkpoint.** `lore/config.py` sets `VIDEO_BACKBONE = "MCG-NJU/videomae-base-finetuned-kinetics"`:
  the ViT-Base encoder, pretrained as above on Kinetics-400 and then fine-tuned on its action labels.
- **Input (`lore/faces.py`).** It does not see the full TV frame:
  - YuNet detects faces;
  - SFace finds which face is the speaker;
  - the speaker's face is cropped and tracked over the clip;
  - 16 face crops of 224×224 are given to VideoMAE.

  The full pipeline is in [`docs/report.md`](../docs/report.md) §4.1.
- **Branch (`lore/models.py` · `VideoBranch`):**
  - **Load.** It loads the model with `VideoMAEForVideoClassification`, keeps the encoder
    (`base.videomae`) and its final norm (`fc_norm`), and drops the Kinetics classifier.
  - **Freeze and adapt.** The encoder is frozen and adapted with LoRA on query/value
    ([lora.md](lora.md)).
  - **Embed (`embed`).** The 1568 output tokens are **averaged** into one 768-d vector, then
    normalised.
  - **Classify.** A dropout + linear layer produces one logit: deception or truth.
- **Why VideoMAE rather than DOLOS's ViT.** The ViT in [DOLOS](dolos.md) looks at frames one by one
  and needs an extra adapter to model time. VideoMAE has time built in through its cubes and
  space-time attention.
- **A caveat.** The paper's own lesson about domain gaps applies here. VideoMAE learned from human
  *actions* (Kinetics), not from talking faces. In `lore`, the video branch is near chance on unseen
  speakers ([`lore` README](../README.md#what-it-means)). The domain gap is a plausible reason, but it
  has not been tested.
