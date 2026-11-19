# Why AV-HuBERT, and what it is not

## The model

AV-HuBERT (Shi et al., 2022) is a self-supervised **audio-visual speech** encoder. It is
trained with masked multimodal cluster prediction: units are clustered from the audio stream,
and the model has to predict the missing ones from the visible mouth region (and the other way
round). The result is a representation of **what a mouth looks like while producing speech**.

That is a different prior from every other encoder in this project:

| Encoder | Trained on | What its features encode |
|---|---|---|
| DINOv2 | still images | appearance, face geometry |
| V-JEPA 2 | video, no audio | generic motion and action |
| WavLM (`tia`) | audio | speech content and paralinguistics |
| **AV-HuBERT (video pass)** | mouth + speech, aligned | **speech-shaped facial dynamics** |
| **AV-HuBERT (audio pass)** | mouth + speech, aligned | speech, but shaped by having to predict the mouth |

## Why that could matter for deception

The hypothesis behind this direction is that deception shows up in the *relation* between
what a face is doing and what a voice is doing — the incongruence PECL-style fusion has to
learn from scratch on 1,600 clips. AV-HuBERT is the only encoder here whose visual features
were shaped by having to predict speech, so part of that relation is already in the
representation.

It is also the only direction where both streams come from the **same checkpoint**: the mouth
pass and the audio pass are two heads of one model that was trained to make them predict each
other. Two consequences:

- the streams are the same width (768), so `congruence` compares representations that live in
  the same space by construction, not two encoders that never met;
- `unified` fuses *outside* the model, which is a choice: letting AV-HuBERT fuse them internally
  is one forward pass and one sequence, and then there is nothing left to compare. The split is
  what makes the incongruence test possible.

## What it is not

- **Not a deception model.** Nothing in the pre-training knows what a lie is.
- **Not a face encoder.** It sees a 96×96 mouth crop, not the eyes or the head. Whatever
  gaze or eyebrow cues matter are invisible to it.
- **Not the standard preprocessing.** The official `align_mouth.py` stabilises the crop
  against a mean face; this pipeline takes the mouth bounding box from dlib's 68 landmarks
  and follows the head. The difference is recorded in the manifest.
- **Not permissive.** The weights are under Meta's AV-HuBERT licence: non-commercial research
  only, and it explicitly forbids **biometric processing** and surveillance uses. Deception
  detection on video is arguably behavioural biometrics. This is a decision for the group, not
  for a notebook — see `docs/plan.md`.

## Domain fit

LRS3 is TED talks; VoxCeleb2 is interview footage. Both are frontal, well-lit, single-speaker
talking heads — closer to a BBC studio than AudioSet is. The risk here is not the domain, it
is the preprocessing: a mouth crop that drifts during laughter or head turns is worse than no
crop at all, which is why the notebook draws a grid of frames before spending an hour.

## The claim this direction tests

*Does a speech-conditioned facial encoder see something a generic one does not?* It is
falsifiable with the variants in `model.py`: if the mouth stream carries deception signal, the
`video`-only row leaves chance; if the relation matters, `congruence` beats both single
streams. A negative answer is reportable — and it would say that on DOLOS the face is a weak
channel regardless of how it is encoded.
