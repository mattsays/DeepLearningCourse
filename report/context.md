# Report Context: Architectures, Design Choices & Verified Results

Internal reference compiled from `lie-detector/notebooks/architectures/` to keep all `.typ` report sections aligned with the latest code, checkpoints, and committed `results/` files.

---

## 1. Evaluation Protocols & Baselines

- **Dataset**: DOLOS (*Would I Lie To You?*). Paper reports 1,675 clips (213 subjects, 84 episodes); local `dataset.csv` has **1,678 rows** (901 `deception` / 777 `truth`, 53.7% / 46.3%), **1,452 downloadable clips** (`status = ok`), **151 (episode, label) groups** (median 10 clips/group).
- **Official 3-Fold Clip-Level Split** (`official_fold{1..3}.json`): paper's benchmark. Random over clips, so **95.0%–99.6%** of test clips share a speaker with train and **100%** share an episode. Measures within-corpus/speaker-familiarity performance. Majority class: **52.4%–53.3%** depending on subset.
- **Subject-Independent 5-Fold CV** (`subject_fold{1..5}.json` + locked `subject_test.json`): strict speaker-disjoint split. The two recurring captains (*Lee Mack*, *David Mitchell*, 468 clips, 69.0% deception) are pinned to `train`; 27 subjects (182 clips) are held out in `subject_test.json`; 151 subjects (1,021 clips, 47.7% deception) form the 5-fold CV working set (`train` / `val` / `holdout`). Decision thresholds are tuned on `val` (never fixed at 0.50). Majority class on holdouts: **52.0%–52.6%**.
- **Published Paper Benchmarks (PECL, Guo et al., ICCV 2023, 3-fold official)**:
  - Visual only (ViT + UT-Adapter): **61.44% ACC**, 69.42% F1, 58.89% AUC
  - Audio only (W2V2 + UT-Adapter): **59.19% ACC**, 73.46% F1, 52.54% AUC
  - PECL (PAVF, 5.06M trainable / 71.08M total): **64.75% ACC**, 71.20% F1, 62.71% AUC
  - PECL (PAVF + 25-cue MUMIN multi-task): **66.84% ACC**, 73.35% F1, 64.58% AUC
  - MUMIN + MLP (41 hand annotations): **58.84% ACC**
  - Cross-dataset unseen-subject tests (Table 3): **57.54% ACC** (Bag-of-Lies), **56.17% ACC** (Real-Life Trials).

---

## 2. Theory-Driven Baseline (`load`)

- **Path**: `lie-detector/notebooks/architectures/load/` (`features.py`, `train_load.ipynb`).
- **Theoretical Basis**: Cognitive Load (Vrij — hesitations, pauses, speech rate) & Vocal Arousal (pitch, micro-instability, HNR).
- **Features**: **17 hand-crafted acoustic features** from 16 kHz audio (zero GPU):
  - *Energy & Pause (9, NumPy)*: `duration_s`, `rms_mean`, `rms_std`, `energy_range_db`, `voiced_ratio`, `pause_count`, `pause_ratio`, `pause_mean_ms`, `onsets_per_s`.
  - *Pitch & Voice Quality (8, Praat/`parselmouth`)*: `f0_mean`, `f0_std`, `f0_range_st`, `jitter_local`, `shimmer_local`, `hnr_mean`, `intensity_mean_db`, `intensity_std_db`.
- **Architecture (`LoadHead`)**: Train-split z-standardization → `Dropout → Linear(17, 32) → GELU → Dropout → Linear(32, 1)` (**609 trainable params**).
- **Results (3 seeds)**:
  - Subject 5-fold: **50.8 ± 4.1% ACC**, **54.0% AUC** (per-seed AUC: 56.4, 53.1, 52.7) vs. 52.0% majority and 51.1% ACC / 52.5% AUC MUMIN oracle.
  - Official 3-fold: **55.1 ± 2.1% ACC**, **58.8% AUC** (per-seed AUC: 58.5, 59.1, 58.6).
  - Protocol gap: **+4.3 ACC / +4.8 AUC** (minimal, as 17 summary statistics carry no visual identity).

---

## 3. Direction 1: Frozen Foundation Models + Cross-Modal Fusion (`cross-modal-fusion` / `tia`)

- **Path**: `lie-detector/notebooks/architectures/cross-modal-fusion/` (`model.py`, `training.py`, `campaign.ipynb`, `results/`).
- **Design Choices**:
  - 100% frozen backbones pre-extracted to disk; only a lightweight fusion/projection head is trained (~30× smaller than PECL's 5.06M).
  - Default visual: `facebook/dinov2-small` on 16 MTCNN face crops → `(16, 384)`. (Optional caches in code: `vjepa2-vitl` `(32, 1024)` over 64 frames; `hsemotion-b0-vgaf` `(16, 1280)` or `(16, 8)`).
  - Default audio: `microsoft/wavlm-base-plus` last hidden state average-pooled to 64 tokens → `(64, 768)`.
  - Projection to $d = 128$, `LayerNorm`, `Dropout(0.3)`, `Linear → GELU → Dropout(0.3) → Linear(64, 1)`.
  - 11 variants in `model.py`: `late` (164,737 params), `video` (57,857), `audio` (107,009), `congruence` (`1 - cos(v,a)`, 147,714), `audio_text`, `text`, `video_delta`, `late_delta`, `video_attention`, `audio_attention`, `late_attention`.
- **Committed Results (`results/*/summary.json`, 3 seeds [0, 1, 2])**:
  - **`late-official`**: **72.18 ± 2.96% ACC**, **80.41 ± 2.93% AUC**, 73.09% F1, 72.19% bal. ACC (164,737 params). Beats PECL PAVF by **+7.4 ACC / +17.7 AUC**.
  - **`video-official`**: **73.15 ± 1.71% ACC**, **80.24 ± 1.79% AUC**, 74.76% F1, 73.02% bal. ACC (57,857 params).
  - **`audio-official`**: **58.84 ± 1.97% ACC**, **62.49 ± 1.72% AUC**, 61.58% F1, 58.67% bal. ACC (107,009 params).
  - **`late-subject`**: **56.09 ± 5.89% ACC**, **57.34 ± 8.34% AUC**, 53.37% F1, 55.85% bal. ACC. *(Note: supersedes earlier 1-seed/preliminary 53.0 / 55.2 in `README.md`.)*
  - **`video-subject`**: **49.89 ± 5.85% ACC**, **51.00 ± 6.56% AUC**, 44.11% F1, 50.41% bal. ACC.
  - **`audio-subject`**: **54.90 ± 3.76% ACC**, **58.29 ± 4.18% AUC**, 57.19% F1, 55.67% bal. ACC.
- **Key Findings**:
  - Protocol gap on `late`: **+16.1 ACC / +23.1 AUC**; on `video`: **+23.3 ACC / +29.2 AUC**; on `audio`: only **+3.9 ACC / +4.2 AUC**.
  - MUMIN linear probes on frozen embeddings recover annotated cues (mean AUC 0.665 vs 0.567 null for DINOv2; 0.585 vs 0.539 for WavLM), ruling out an encoder bottleneck.

---

## 4. Direction 2: Spatio-Temporal Transformers with LoRA (`lore`)

- **Path**: `lie-detector/notebooks/architectures/lore/` (`lore/`, `scripts/`, `results/`).
- **Design Choices**:
  - Video branch: `MCG-NJU/videomae-base-finetuned-kinetics` (86.5M) on 16 × 224×224 speaker-centred face crops (YuNet detection + SFace 128-d clustering + median-smoothed boxes).
  - Audio branch: `facebook/hubert-base-ls960` (94.7M) on raw 16 kHz z-normalised waveform (8 s random train window / first 10 s eval).
  - Adaptation: LoRA ($r = 8, \alpha = 16$) on attention `Q` and `V` across all 12 layers + `LayerNorm`/`fc_norm` + linear head → **297,217 trainable params per branch** (0.31–0.34%; **594,434** total). Includes `assert_lora_active` gradient/forward check.
  - Late fusion (`lore/fusion.py`): z-score each branch's logits using `val` stats, combine $s = w z_v + (1-w) z_a$ with $w \in \{0, 0.05, \dots, 1\}$ tuned on `val` (also reports $w = 0.5$).
- **Committed Results (`results/*/summary.json` & `comparison.txt`, seed 0)**:
  - **Subject 5-fold**:
    - Video only: **51.58 ± 4.60% ACC**, **53.62 ± 8.61% AUC**, 41.70% F1, 51.51% bal. ACC.
    - Audio only: **56.10 ± 5.51% ACC**, **59.54 ± 7.39% AUC**, 56.05% F1, 56.25% bal. ACC (95% subject-bootstrap CI: 51.4–60.8%, binomial $p = 0.010$ vs 52.1% majority).
    - Fused ($w$ tuned on `val`): **56.1 ± 7.3% ACC**, **60.6% AUC**, 55.1% F1, 56.2% bal. ACC ($w = 0$ in 3/5 folds).
    - Fused ($w = 0.5$): **55.4 ± 7.7% ACC**, **59.7% AUC**, 60.5% F1.
  - **Official 3-fold**:
    - Video only: **65.38 ± 0.83% ACC**, **71.15 ± 2.47% AUC**, 64.55% F1, 65.73% bal. ACC.
    - Audio only: **62.77 ± 0.88% ACC**, **66.97 ± 3.51% AUC**, 65.06% F1, 62.54% bal. ACC.
    - Fused ($w$ tuned on `val`): **66.3 ± 1.5% ACC**, **73.1% AUC**, 65.9% F1, 66.6% bal. ACC (paper-style best-test-epoch @ 0.5: **68.0 ± 1.8% ACC**).
  - Protocol gap: **−13.8 ACC / −17.5 AUC** on video vs **−6.7 ACC / −7.5 AUC** on audio.

---

## 5. Direction 3: Spatio-Temporal Convolutions & Sequential Modelling (`fab`)

- **Path**: `lie-detector/notebooks/architectures/fab/` (`README.md`, `train_fab.ipynb`, `train_marlin.ipynb`, `train_marlin_kaggle.ipynb`).
- **Design Choices**:
  - **Visual stream**: `swin3d_t` or `x3d_m` on 16 scene frames (`train_fab.ipynb`), OR `marlin_vit_base_ytf` (MARLIN facial masked autoencoder, CVPR 2023) on MTCNN face crops (`train_marlin.ipynb` / `train_marlin_kaggle.ipynb`), producing `(T_v=8, 768)`.
  - **Acoustic stream**: 3-channel Log-Mel spectrogram ($S, \Delta, \Delta^2$) through `convnext_tiny` with frequency-only pooling → `(T_a≈31, 768)`, linearly interpolated to `T_v=8`.
  - **Sequence head**: Two-stream `LayerNorm` projection to $d=128$, concat to $256$, **2-layer Bi-GRU** or **Dilated TCN** + **Temporal Attention Pooling** → binary logit.
- **Status**: Feature extraction started in `train_fab.ipynb` (stopped during Swin3D extraction); `train_marlin.ipynb` and `train_marlin_kaggle.ipynb` have 0 cell outputs. **No training/evaluation results committed.**

---

## 6. Direction 4: From-Scratch Landmark & Acoustic Models (`teo`)

- **Path**: `lie-detector/notebooks/architectures/teo/` (`features/`, `cnn-batching/`, `gat/`, `runs/`).
- **Shared Feature Pipeline (`features/extract_features.py`)**:
  - Per-frame `(T, 53)` array at **10 fps**, zero pre-trained deep backbones:
    - **Visual (45 dims)**: MediaPipe `FaceLandmarker` (15 pooled 3D facial regions: eyes, eyebrows, nose bridge/tip, inner/outer mouth, left/right jaw, chin, left/right cheeks, forehead, iris centers; mean $(x, y, z)$ per region).
    - **Acoustic (8 dims)**: first 8 `openSMILE` `eGeMAPSv02` Low-Level Descriptors interpolated to the 10 fps grid.
  - Optional auxiliary multi-task supervision using 41 per-clip MUMIN cues (`mumin_head`, MSE loss).
- **Architecture 4A — `TemporalCNN` (`cnn-batching/model_temporal.py`)**:
  - **57,322 trainable params**.
  - 4-layer 1D CNN (`Conv1d(53→64, k=5)` + 3× `Conv1d(64→64, k=3)` with `BatchNorm1d`, `ReLU`, `Dropout(0.3)`), masked temporal mean pooling, dual heads (`lie_head`: `Linear(64, 1)`; `mumin_head`: `Linear(64, 41)`).
  - **Checkpoints (`runs/`)**:
    - `runs/cnn_baseline/best.pt` (leaky random split, ep 37): `val_acc = 62.9%`, `val_auc = 64.8%`.
    - `runs/cv_cnn/fold{1..5}/best.pt` (5-fold subject CV, `train_fold_temporal.py`): `best_val_auc` = `0.544, 0.578, 0.640, 0.500, 0.518` (mean **0.556**). *No holdout metrics JSON/CSV committed.*
- **Architecture 4B — `DualHeadGAT` (`gat/model.py`, newly added)**:
  - **18,410 trainable params** at `hidden=64, heads=4, dropout=0.3` (5,642 at `hidden=32`).
  - Converts each clip into sliding-window intra-window graphs (`window_sec=1.5`, `fps=10`, `overlap=0.5`) with landmark nodes (`node_type=0`) and audio nodes (`node_type=1`).
  - Separate `lm_proj` + `LayerNorm` and `au_proj` + `LayerNorm` (compensating for ~30× scale difference between landmark coords and openSMILE LLDs) + `type_emb`, 2-layer pre-norm residual `GATConv` (4 heads → 1 head, ELU, dropout 0.3), `global_mean_pool`, and dual heads (`lie_head` 2-layer MLP, `mumin_head` 2-layer MLP).
  - **Checkpoints (`runs/`)**:
    - `runs/gat_baseline/best.pt` (leaky random split, `mumin_loss_weight=0.3`, ep 7): `val_acc = 47.9%`, `val_auc = 50.4%`.
    - `runs/cv_gat/fold{1..5}/best.pt` (5-fold subject CV, `train_fold_gat.py`, `mumin_loss_weight=0.5`): `best_val_auc` = `0.468, 0.631, 0.709, 0.675, 0.602` (mean **0.617**). *No holdout metrics JSON/CSV committed.*

---

## 7. Reference Papers Used Across Directions

### Dataset, Benchmarks & Psychological Foundations
- **DOLOS / PECL** (`2303.12745v2.pdf` in `cross-modal-fusion/reference paper/` & `lore/references/`, `guo2023dolos` in `refs.bib`): Guo et al., *Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning*, ICCV 2023 (arXiv:2303.12745). Defines DOLOS, the official 3-fold split, MUMIN multi-tasking, UT-Adapter, PAVF, and cross-dataset unseen-subject evaluations.
- **Prior Deception Datasets** (`refs.bib`): *Real-Life Trials* (Pérez-Rosas et al., ICMI 2015, `perezrosas2015trial`), *Bag-of-Lies* (Gupta et al., CVPRW 2019, `gupta2019bagoflies`), *Box of Lies* (Soldner et al., NAACL-HLT 2019, `soldner2019boxoflies`).
- **Cognitive Load & Behavioural Cues** (`load/docs/theory.md`, `refs.bib`):
  - Vrij et al., *A cognitive load approach to lie detection*, J. Investigative Psychology and Offender Profiling, 2008 (`vrij2008cognitive`).
  - DePaulo et al., *Cues to deception*, Psychological Bulletin, 2003 (`depaulo2003cues`) — meta-analysis of 158 cues.
  - Bond & DePaulo (2006) — meta-analysis (>200 studies) placing human accuracy at ~54% (cited in `load/docs/theory.md`).
  - Jordan et al. (2019) — direct test of Ekman's Micro-Expressions Training Tool (cited in `load/docs/theory.md`).
  - Allwood et al., *The MUMIN coding scheme...*, LRE 2007 (`allwood2007mumin`).

### `cross-modal-fusion` (`tia`, 12 PDFs in `cross-modal-fusion/reference paper/`)
- **Backbones**:
  - **DINOv2**: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024 (arXiv:2304.07193, `dinov2-2304.07193.pdf`, `oquab2024dinov2`).
  - **WavLM**: Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*, IEEE JSTSP 2022 (arXiv:2110.13900, `wavlm-2110.13900.pdf`, `chen2022wavlm`).
  - **MiniLM**: Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, NeurIPS 2020 (arXiv:2002.10957, `minilm-2002.10957.pdf`).
  - **V-JEPA 2**: Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025 (arXiv:2506.09985, `vjepa2-2506.09985.pdf`).
  - *(HSEmotion: Savchenko `face-emotion-recognition` GitHub repo, Apache-2.0; no PDF.)*
- **Fusion, Adaptation & Methodology**:
  - **MulT**: Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 (arXiv:1906.00295, `mult-1906.00295.pdf`).
  - **Perceiver**: Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021 (arXiv:2103.03206, `perceiver-2103.03206.pdf`).
  - **GMU**: Arevalo et al., *Gated Multimodal Units for Information Fusion*, 2017 (arXiv:1702.01992, `gmu-1702.01992.pdf`).
  - **Adapters**: Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019 (arXiv:1902.00751, `adapters-1902.00751.pdf`).
  - **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv:2106.09685, `lora-2106.09685.pdf`, `hu2022lora`).
  - **Multimodal ML Survey**: Baltrušaitis, Ahuja & Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 (arXiv:1705.09406, `multimodal-ml-survey-1705.09406.pdf`).
  - **Shortcut Learning**: Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020 (arXiv:2004.07780, `shortcut-learning-2004.07780.pdf`).

### `lore` (4 PDFs in `lore/references/`)
- **DOLOS / PECL**: `Audio-Visual Deception Detection...pdf` (Guo et al., ICCV 2023).
- **VideoMAE**: Tong et al., *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*, NeurIPS 2022 (`VideoMAE.pdf`, `tong2022videomae`).
- **HuBERT**: Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, IEEE/ACM TASLP 2021 (`HuBERT.pdf`, `hsu2021hubert`).
- **LoRA**: Hu et al., ICLR 2022 (`LoRA Low Rank Adaptation...pdf`, `hu2022lora`).

### `fab` (`README.md`, `refs.bib`)
- **MARLIN**: Cai et al., *MARLIN: Masked Autoencoder for Facial Video Representation LearnINg*, CVPR 2023 (`cai2023marlin`).
- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s*, CVPR 2022 (`liu2022convnext`).
- *(Video backbones `swin3d_t` / `x3d_m`, Bi-GRU, and Dilated TCN are used via standard torchvision/PyTorch architectures without dedicated PDFs in `fab/`.)*

### `teo` (`features/extract_features.py`, `gat/model.py`, `refs.bib`)
- **MediaPipe**: Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines*, arXiv:1906.08172, 2019 (`lugaresi2019mediapipe`).
- **eGeMAPS / openSMILE**: Eyben et al., *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing*, IEEE TAFFC 2016 (`eyben2016gemaps`).
- *(Note: `GATConv` from PyTorch Geometric — Veličković et al., *Graph Attention Networks*, ICLR 2018 — is used in `gat/model.py`, though not yet in `refs.bib`.)*

### `load` (`features.py`, `refs.bib`)
- **Parselmouth / Praat**: Jadoul, Thompson & de Boer, *Introducing Parselmouth: A Python interface to Praat*, Journal of Phonetics 2018 (`jadoul2018parselmouth`).
