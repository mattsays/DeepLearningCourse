# `avh` — AV-HuBERT mouth-region encoder, one probe

Direction `avh` of the [lie-detector project](../../../README.md#directions). AV-HuBERT
(Shi et al., 2022) encodes **both streams**: the mouth region at 25 fps and the same speech as
a 104-dim filterbank, each through the same checkpoint in its own modality. The only trainable
part is a small attentive probe. Reasoning in
[`docs/theory.md`](docs/theory.md), tensor contract in
[`docs/encoder-contract.md`](docs/encoder-contract.md).

**Status: written, not run.** Three things gate it — the AV-HuBERT licence (non-commercial,
no biometric processing), a fairseq dependency that targets Python 3.8, and a simplified mouth
alignment. Read [`docs/plan.md`](docs/plan.md) before spending a session.

## Results

| Variant | Seeds | ACC | AUC | Trainable | What it is |
|---|---:|---:|---:|---:|---|
| *not run* | — | — | — | — | awaiting the first extraction pass |
| *majority class* | — | — | ≈52 | — | the floor to beat (board, *reference* row) |

Reference points from the same folds, `tia` direction: audio-only 58.9 AUC, video-only 53.7
(DINOv2), three-seed late fusion 55.2 (52.5–57.7 per seed).

## Files

| File | What it is |
|---|---|
| [`extract_features.ipynb`](extract_features.ipynb) | mouth ROI + filterbank → AV-HuBERT (two passes) → two `(64, 768)` caches |
| [`train.ipynb`](train.ipynb) | subject-disjoint five-fold training, evaluation, board row |
| [`model.py`](model.py) | probe variants, `load_avhubert`, `pool_avhubert`, `check_avhubert_environment` |
| [`docs/theory.md`](docs/theory.md) | why a speech-conditioned facial encoder might see deception |
| [`docs/encoder-contract.md`](docs/encoder-contract.md) | crops, tokens, cache, checkpoint, forward |
| [`docs/plan.md`](docs/plan.md) | blockers (licence, fairseq), ladder, kill criteria |

## How to run

```bash
git submodule update --init --recursive --depth 1 lie-detector/vendor/av_hubert
export AVHUBERT_CKPT=/content/drive/MyDrive/dolos/checkpoints/base_vox_iter5.pt
```

Weights: [base_vox_iter5.pt](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt)
(LRS3 + VoxCeleb2, ~1.2 GB).

Then on Colab: `extract_features.ipynb` — the environment cell fails fast if fairseq cannot
import — and after the frame grid looks right, `train.ipynb` with `VARIANT = "unified"`, then
`video`, `audio`, `late`, `congruence`.

`SPLIT = "official"` reproduces the paper's protocol and must be reported separately: the
official split is random over clips and shares speakers across the boundary
([`docs/splits.md`](../../../docs/splits.md)).

## What this direction can and cannot claim

It tests whether a **speech-conditioned** representation carries deception signal that a
generic one did not — on the face (mouth dynamics), on the voice (audio encoded by a model that
also sees the mouth), and in their agreement. It says nothing about text, and the visual stream
sees only the mouth: gaze and eyebrows are outside its input.
