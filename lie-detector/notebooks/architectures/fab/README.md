# Direction 3 (`fab`) — Spatio-Temporal Convolutions & Sequential Modelling

Direction 3 of the DOLOS Multimodal Deception Detection project.

## Overview

Unlike foundation Vision Transformers that lack strong inductive biases on small datasets, Direction 3 exploits
the structural inductive bias of spatio-temporal convolutions to extract localized short-term features, paired with
a dedicated sequential model for global temporal reasoning across the full dialogue clip.

```
Visual Stream (16 frames)  ──► Swin3D / X3D      ──► (B, T_v, 768) ──┐
                                                                     ├──► Interp & Concat ──► Bi-GRU / TCN ──► Attn Pool ──► Logit
Acoustic Stream (3-ch Mel) ──► ConvNeXt-Tiny     ──► (B, T_a, 768) ──┘
```

1. **Visual Stream**: `swin3d_t` (or `x3d_m`) preserves the temporal sequence axis ($T_v \approx 8$) by averaging across spatial dimensions ($H, W$) only.
2. **Acoustic Stream**: Raw audio is converted to a 3-channel Log-Mel spectrogram containing static energy ($S$), velocity ($\Delta$), and acceleration ($\Delta^2$). Passing this through `convnext_tiny` with frequency-only pooling yields $(T_a, 768)$ ($T_a \approx 31$).
3. **Temporal Alignment & Fusion**: 1D temporal linear interpolation resamples audio from $T_a$ to $T_v$, projects both to $d=128$, and concatenates to $d=256$.
4. **Sequential Modelling**: A 2-layer Bidirectional GRU (or Dilated TCN) models the psychological trajectory over time.
5. **Temporal Attention Pooling**: Dynamically weights salient emotional micro-leakage instants ($\alpha_t$) rather than washing out the signal with a uniform average.

## Notebooks & Approaches

1. **`train_fab.ipynb`**: Original Swin3D (scene video) + ConvNeXt-3ch (audio) + Bi-GRU / TCN with Two-Stream LayerNorm projection.
2. **`train_marlin.ipynb`**: **MARLIN** (Masked Autoencoder for facial video Representation LearnINg, CVPR 2023) on MTCNN face crops + ConvNeXt-3ch (audio) + Bi-GRU / TCN.

## Feature Pre-Extraction (Phase 2)

To avoid network I/O bottlenecks across Google Drive, multimodal sequence descriptors are pre-extracted
once into the shared cache:

```bash
# 1. Swin3D + ConvNeXt-3ch (scene video + audio):
python scripts/extract_features_fab.py --data-dir /content/drive/MyDrive/dolos --encoder swin3d_convnext_3ch

# 2. MARLIN (MTCNN face crops) + ConvNeXt-3ch (facial video + audio):
python scripts/extract_features_marlin.py --data-dir /content/drive/MyDrive/dolos --encoder marlin_convnext --marlin-model marlin_vit_base_ytf
```

- **Output structure**: `DATA/cached_features/<ENCODER>/<clip_id>.npy`
- **Output tensor shape**: `(T_v=8, 1536)` where `[0:768]` is visual/face features and `[768:1536]` is ConvNeXt acoustic features.
- **RAM Preloading**: Total disk space is ~400 MB. `CachedFeatures` loads all arrays directly into RAM at initialization, reducing epoch training time to ~2 seconds.

## Experimental Matrix (Ablation Study)

| Config | Video Backbone | Audio Representation | Sequence Head & Pooling | Primary Hypothesis |
|---|---|---|---|---|
| **V3.0 (Baseline)** | `swin3d_t` (Frozen) | Log-Mel 1-channel + `convnext_tiny` (Frozen) | Bi-GRU + Global Average Pooling | Establish baseline performance of frozen convolutional backbones. |
| **V3.1 (+ Attention)** | `swin3d_t` (Frozen) | Log-Mel 1-channel + `convnext_tiny` (Frozen) | Bi-GRU + **Temporal Attention Pooling** | Measure benefit of isolating micro-leakage timestamps vs uniform mean. |
| **V3.2 (+ Mel 3Ch)** | `swin3d_t` (Frozen) | **Mel 3-Channel ($S, \Delta, \Delta^2$)** + `convnext_tiny` | Bi-GRU + Temporal Attention Pooling | Evaluate pitch velocity and vocal jitter cues matching ImageNet RGB filters. |
| **V3.3 (All-Conv TCN)** | `x3d_m` (Frozen) | Mel 3-Channel + `convnext_tiny` | **TCN (Dilated Temporal Conv1D)** + Attn | Test all-convolutional inductive bias and parallel sequence processing. |
| **V3.4 (Sliding Chunks)** | `swin3d_t` (Chunks) | Mel 3-Channel + `convnext_tiny` | Bi-GRU / TCN + Attention Pooling | High-resolution dense chunk sampling to prevent losing 50ms micro-expressions. |
| **V3.5 (Fine-Tuning)** | `swin3d_t` (Last Stage Unfrozen) | `convnext_tiny` (Last Stage Unfrozen) | Best Sequential Head | Domain adaptation of high-level representations via differential learning rates. |

## Results Board

### 1. Subject-Independent 5-Fold Cross-Validation (Official Project Protocol)
Evaluated under the official 5-Fold Subject-Independent Cross-Validation protocol (`Protocol.load(DATA / "splits")`). The majority baseline on the working set is **52.4%**.

| Direction | Owner | Method | Trainable params | ACC (CV) | F1 | AUC | Balanced ACC | Majority | Runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`fab`** | Fabio | MARLIN-B (Faces) + ConvNeXt-T · TCN · Mean Pool | 97,857 | **54.1 ± 3.2** | **58.5** | **56.4** | **54.7** | 52.4 | ≈15 s (Tesla T4) |

*Out-of-fold predictions per clip are saved to `preds/cv_fab_marlin.csv`.*

### 2. Official 3-Fold Benchmark (Direct Comparison with DOLOS Paper / PECL)
Evaluated under the original DOLOS paper clip-level 3-fold split (`folds.official(fold)`):

| Benchmark | Method | Trainable params | ACC | F1 | AUC | Balanced ACC |
|---|---|---:|---:|---:|---:|---:|
| **PECL (Reported in Paper)** | Wav2Vec2 + ViT + UT-Adapter + PAVF | 5.06M | 64.75 | 71.20 | 62.71 | — |
| **MARLIN + ConvNeXt (Ours)** | MARLIN-B + ConvNeXt-T · TCN · Mean Pool | **97,857** | **59.1 ± 3.2** | **61.4** | **60.3** | **60.6** |

*Official 3-fold predictions per clip are saved to `preds/official_3fold_marlin_tcn.csv`.*
