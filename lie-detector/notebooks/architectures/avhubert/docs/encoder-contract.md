# Encoder contract: mouth crops, tokens, cache, checkpoint

Everything here is a constant of the **cache**: change one and the arrays are a different
dataset. The manifest refuses to mix them.

## Preprocessing

AV-HuBERT is a lip-reading model, so the input is a **mouth region**, not a face.

| | Value |
|---|---|
| Frame rate | 25 fps (the model's native rate) |
| Crop | 96 × 96 grayscale |
| Landmarks | dlib 68 points; mouth = points 48–67 |
| Box | mouth bounding box, side = `max(1.6 × width, 2.2 × height)`, squared, clipped |
| Face detection | dlib HOG, re-run once per second; landmarks every frame |
| Alignment | **none** — no mean-face stabilisation, the crop follows the head |
| Range | `[0, 1]` float |

The official `align_mouth.py` warps each frame to a mean face using five stable points. That
removes head rotation and is what the checkpoint expects; the simplified crop keeps the head's
motion instead. It is a real deviation, recorded as `alignment` in the manifest, and the reason
the notebook draws a frame grid before encoding 1,675 clips.

## Tokens

| | Value |
|---|---|
| Encoder | `base_vox_iter5.pt` (LRS3 + VoxCeleb2, no finetuning) |
| Modality | one pass per stream: `["video"]`, `["audio"]` |
| Raw output | `(B, T, 768)` at 25 fps; `T` between 50 (2 s) and 475 (19 s) |
| Cached | `(64, 768)` — adaptive average pooling over time |
| Bytes/clip | 96 KB (fp16) → ~160 MB for the protocol |

A fixed token count keeps the cache rectangular and the probe free of padding masks; the cost
is temporal detail on the long clips. `64` is the same count on both streams, so they line up
one-to-one in the `congruence` variant.

## Cache layout

```
<DATA>/cached_features/avhubert-video/<clip_id>.npy     float16  (64, 768)
<DATA>/cached_features/avhubert-audio/<clip_id>.npy     float16  (64, 768)
<DATA>/cached_features/avhubert-video/manifest.json
```

`manifest.json`: `settings` (`checkpoint`, `fps`, `crop`, `tokens`, `width`, `alignment`,
`dtype`), `weights`, `dataset_fingerprint`, `created`, `count`, `failures`. A folder whose
`settings` differ is refused; `FORCE = True` deletes its arrays first.

Both streams are produced here, by the **same checkpoint** in two passes. AV-HuBERT was trained
with modality dropout, so `modalities = ["video"]` and `modalities = ["audio"]` are both
in-distribution: the encoder is being used the way it was trained, not as a hacked face model.
The model's own fused pass (`["audio", "video"]`) emits a single sequence and is deliberately not
cached — it would leave nothing for a two-stream probe to compare.

## Audio branch

| | Value |
|---|---|
| Input | the 16 kHz mono wav the preprocessing wrote next to the clip |
| Features | `python_speech_features.logfbank`, 26 bins, 25 ms window, 10 ms hop |
| Stacking | four consecutive frames → **104 dims at 25 fps** (`stack_order_audio: 4`) |
| Normalisation | `F.layer_norm(feats, (104,))` per frame — `hubert_dataset.py`'s own line |
| Alignment | padded / trimmed to the video's frame count |

No design decision here: this reproduces `hubert_dataset.py`, including the odd per-frame
layer norm over the 104 stacked bins. A different fbank (torchaudio's Kaldi frontend, say)
produces different numbers and a silently different model input.

## How it runs

Two interpreters, one artefact:

| Step | Interpreter | Command |
|---|---|---|
| crop + filterbank + encoder | `/content/avh8/bin/python` (3.8) | `scripts/extract_avhubert.py` |
| verification, probe training | the Colab kernel (3.13) | `train.ipynb` |

The split exists because fairseq has no wheel past cp38. The script imports `model.py` and
`preprocess.py` from this folder, so there is still one implementation of the crop and one of
the forward pass; only the interpreter differs.

## Checkpoint

| Model | URL |
|---|---|
| base, LRS3 + VoxCeleb2 | `https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt` |
| large, LRS3 + VoxCeleb2 | `…/large_vox_iter5.pt` |

`model.load_avhubert` calls `fairseq.checkpoint_utils.load_model_ensemble_and_task` with
`task.data` pointed at the checkpoint's folder (fairseq insists on one), then sets
`model.modalities = ["video"]`. The encoder is frozen.

**Dependencies.** fairseq — the fork bundled inside `vendor/av_hubert/fairseq` — plus dlib.
The official code targets **Python 3.8**; the bundled fairseq may not build against a current
Colab image. `check_avhubert_environment()` runs before the download and prints python,
fairseq version and whether the model classes resolve. If that import fails, nothing below it
can work; `docs/plan.md` lists the exits.

## Forward

```python
features, _ = model.extract_features(source={"video": video}, padding_mask=None, mask=False)
# video: (1, T, 1, 96, 96) float in [0, 1]  ->  features: (1, T, 768)
pooled = pool_avhubert(features, 64)          # (1, 64, 768)
```
