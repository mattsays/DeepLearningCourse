# HuBERT — Hidden-Unit BERT for Speech

Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden
Units*, IEEE/ACM TASLP 2021 · [arXiv 2106.07447](https://arxiv.org/abs/2106.07447) ·
[PDF](HuBERT.pdf)

## The idea

BERT learned language by hiding words in a sentence and guessing them. HuBERT wants to do the same
with **raw speech**, but speech has no words to guess:

- **No vocabulary.** Audio is a continuous signal, with no list of discrete units to predict.
- **No boundaries.** Nobody marks where one sound ends and the next begins.
- **Many sounds per input.** One recording contains many sounds, not one object as in an image.

HuBERT's trick: **invent the vocabulary first, then play BERT's game with it.**

1. **Invent labels.** Group short slices of audio into clusters with plain k-means. Each cluster ID
   acts as a crude "pseudo-phoneme", a *hidden unit*.
2. **Mask and guess.** Hide parts of the audio and train a Transformer to predict the cluster IDs of
   the hidden parts.
3. **Repeat with better labels.** Re-cluster the model's own features (better than the original ones),
   and train again.

The labels are noisy, and that is fine. What matters is that they are *consistent*: the same sound
always gets the same ID. Then the model can learn the structure of speech from them.

## How it works

```
raw audio (16 kHz) ─► CNN ─► one vector per 20 ms ─► mask some spans ─► Transformer ─► predict cluster IDs
                                                                                      (hidden frames only)
targets: round 1 = k-means on MFCC features
         round 2 = k-means on HuBERT's own middle-layer features
```

- **Front-end.** A convolutional encoder turns the waveform into one feature vector every 20 ms, about
  50 per second.
- **Masking.** Random spans of these frames are replaced by a mask token.
- **Prediction.** A BERT-style Transformer (12 layers in the Base model) sees the masked sequence and
  predicts the cluster ID of every frame.
- **Loss on hidden frames only.** This is the key choice:
  - To guess a hidden frame, the model must use the **context** around it, like a language model. It
    must also understand what the visible frames sound like, like an acoustic model.
  - If the loss also covered visible frames, the model could just learn to copy the clustering
    algorithm, including its mistakes.
- **Iterative refinement.** After the first round, the middle layers of the model represent speech much
  better than the original MFCC features. Clustering *them* gives better targets for a second round.

## What the paper shows

- **On par with or better than wav2vec 2.0.** HuBERT matches or beats wav2vec 2.0, the previous best
  self-supervised speech model, when fine-tuned for speech recognition, especially with very little
  labelled data.
- **Simpler.** It needs no contrastive loss and none of the tricks that come with one.
- **Rich representations.** As the introduction stresses, good speech representations capture not
  only *what* is said but also *how*: speaker identity, emotion, hesitation, interruptions.

## How it is used in `lore`

HuBERT is the **audio branch** of `lore`: it turns the speaker's voice into one vector.

- **Checkpoint.** `lore/config.py` sets `AUDIO_BACKBONE = "facebook/hubert-base-ls960"`: the Base
  model pretrained without labels on LibriSpeech (960 h of read audiobooks).
- **Input (`lore/data.py`).**
  - The clip's audio is converted to 16 kHz mono and normalised, which is what HuBERT expects.
  - Training uses a random window of the clip; evaluation uses its beginning.

  See [`docs/report.md`](../docs/report.md) §4.2.
- **Branch (`lore/models.py` · `AudioBranch`):**
  - **Load.** It loads the encoder with `HubertModel`.
  - **Freeze and adapt.** The encoder is frozen and adapted with LoRA on `q_proj`/`v_proj`
    ([lora.md](lora.md)).
  - **Embed (`embed`).** The ~50 vectors per second are **averaged over the real audio only**. Padded
    silence is excluded with a mask built from `_get_feat_extract_output_lengths`. The result is
    normalised.
  - **Classify.** A dropout + linear layer produces one logit.
- **Why HuBERT for deception.** What the paper says the model captures besides words (hesitation,
  pauses, emotion) is exactly the vocal cues annotated in DOLOS. In `lore`, the audio branch is the
  only one above chance on unseen speakers ([`lore` README](../README.md#what-it-means)).
- **Two limits to keep in mind:**
  - **Speaker identity.** HuBERT also encodes *who* is speaking, which a model can exploit as a
    shortcut when the same people appear in training and test. This is why `lore` reports
    speaker-independent splits.
  - **Domain.** It was pretrained on read speech, not on spontaneous game-show conversation.
- **Why HuBERT rather than DOLOS's wav2vec 2.0.** The architecture is the same, but HuBERT's
  pretraining target is simpler and its results are equal or better ([dolos.md](dolos.md)).
