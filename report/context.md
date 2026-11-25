# Report Context: Editorial Framing, Architectures & Verified Results

Internal reference compiled from `lie-detector/notebooks/architectures/*/results/` and `report/` to keep all `.typ` sections aligned with the code, committed result files, and the editorial scope of the report.

---

## 0. Report Framing & Editorial Policy

- **Compiled output**: `report/report.typ` → `report/report.pdf` (20 pages, compiled with Python `typst`).
- **Main body (`00-title.typ` through `08-conclusion.typ`)**: evaluates strictly on the **official 3-fold clip-level protocol** (`official_fold{1..3}.json`) for direct comparability with prior work (PECL). Limitations in `07-discussion.typ` cover only general methodological limits (small/imbalanced data, heterogeneous evaluation rules/coverage, temporal resolution, external validity, ethics). **No speaker-overlap or identity-leakage claims appear in the main body.**
- **Appendix A (`sections/appendix/a-subject-independent.typ`)**: treats the speaker overlap in the official clip-level folds and the 5-fold subject-independent re-evaluation as an **open methodological question** ("An open question about the official folds"), complementary to—not replacing—the official-split comparisons in the main text.
- **Visualisations**: native Typst `#hbar-chart` in `report/template.typ` renders `@fig:main-chart` (official ACC/AUC in §5.5); `report/figures/make_figures.py` generates `fig4-efficiency.pdf` (`@fig:efficiency`, AUC vs trainable params & runtime in §6) and `fig1-protocol-gap.pdf` (`@fig:protocol-gap`, official vs subject-independent ACC/AUC in Appendix A).

---

## 1. Evaluation Protocols & Benchmarks

- **Dataset**: DOLOS (*Would I Lie To You?*). Paper reports 1,675 clips (899 `deception`, 776 `truth`), 213 subjects, 84 episodes, duration 2–19 s (median 5 s). Local `dataset.csv` has **1,678 rows** (1,677 unique clip names; 901 `deception` / 777 `truth`, 53.7% / 46.3%), **180 subjects** (after alias merging), **75 distinct videos**, duration **2–48 s** (median 5 s; 5 clips > 19 s), **79.1% male / 20.9% female** (deception rates **56.4% / 43.6%**), **149 (episode, label) rounds** (median 11 clips/round), and **1,452 downloadable clips** (`status = ok`). Two recurring captains contribute 468 clips (28%, 69.0% deceptive).
- **Official 3-Fold Clip-Level Split** (`official_fold{1..3}.json`): re-divides 1,646 of the 1,677 distinct clips into 3 folds (1,099/549, 1,095/553, 1,100/548 in paper files; 1,650 total test entries due to a duplicate in fold 1). Downloadable test clips: **483 / 483 / 461** (1,427 total; 1,426 deduplicated). Majority class: **52.4%–53.3%**.
- **Subject-Independent 5-Fold Split (Appendix A)** (`subject_fold{1..5}.json` + locked `subject_test.json`): two captains pinned to `train`; 27 subjects (182 annotated / 24 subjects, 155 downloadable) held out in untouched `subject_test.json`; 151 subjects (1,021 annotated, 47.7% deception / 135 subjects, 886 downloadable) split into 5 subject-disjoint folds, with 15% of train subjects as `val` for early stopping and thresholding. Majority class on holdouts: **52.1%–52.6%**.
- **Published Paper Benchmarks (PECL, Guo et al., ICCV 2023, 3-fold official)**:
  - Visual only (ViT + UT-Adapter): **61.44% ACC**, 69.42% F1, 58.89% AUC
  - Audio only (W2V2 + UT-Adapter): **59.19% ACC**, 73.46% F1, 52.54% AUC
  - Concatenation (last layer): **61.62% ACC**, 70.20% F1, 60.50% AUC
  - PECL (PAVF, 5.06M trainable / 71.08M total): **64.75% ACC**, 71.20% F1, 62.71% AUC
  - PECL (PAVF + 25-cue MUMIN multi-task): **66.84% ACC**, 73.35% F1, 64.58% AUC
  - MUMIN + MLP (41 hand annotations): **58.84% ACC**
  - Cross-dataset external tests (Table 3): **57.54% ACC** (Bag-of-Lies), **56.17% ACC** (Real-Life Trials).

---

## 2. Theory-Driven Baseline (`load`)

- **Path**: `lie-detector/notebooks/architectures/load/` (`features.py`, `run_load.py`, `results/load-{official,subject}/`).
- **Theoretical Basis**: Cognitive Load (Vrij — hesitations, pauses, speech rate) & Vocal Arousal (pitch, cycle-to-cycle instability, HNR).
- **Features**: **17 hand-crafted acoustic features** from 16 kHz audio (`@tab:load-feats` in `04-baseline.typ`):
  - *Energy & Pauses (9, NumPy)*: `duration_s`, `rms_mean`, `rms_std`, `energy_range_db`, `voiced_ratio`, `pause_count`, `pause_ratio`, `pause_mean_ms`, `onsets_per_s`.
  - *Pitch & Voice Quality (8, Praat/`parselmouth`)*: `f0_mean`, `f0_std`, `f0_range_st`, `jitter_local`, `shimmer_local`, `hnr_mean`, `intensity_mean_db`, `intensity_std_db`.
- **Architecture (`LoadHead`)**: Train-split z-standardisation → `Dropout(0.3) → Linear(17, 32) → GELU → Dropout(0.3) → Linear(32, 1)` (**609 trainable params**).
- **Committed Results (`results/load-{official,subject}/summary.json`, 3 seeds [0, 1, 2])**:
  - **Official 3-fold** (1,427 clips, fixed 60 epochs, thr 0.5): **56.46 ± 1.71% ACC**, **60.16% F1**, **59.23 ± 1.71% AUC**, **56.10% bal. ACC** (reported as **56.5 ± 1.7 / 59.2**; per-fold ACC/AUC: 54.9/57.2, 56.7/60.3, 57.9/60.2; per-seed AUC: 59.6, 59.2, 58.9).
  - **Subject 5-fold** (Appendix A): **53.05 ± 4.65% ACC**, **53.94% F1**, **53.59 ± 5.45% AUC**, **53.55% bal. ACC** (reported as **53.1 / 53.6**).
  - Protocol shift: **−3.4 ACC / −5.6 AUC**.

---

## 3. Direction 1: Frozen Foundation Models + Cross-Modal Fusion (`cross-modal-fusion` / `tia`)

- **Path**: `lie-detector/notebooks/architectures/cross-modal-fusion/` (`model.py`, `training.py`, `campaign.ipynb`, `results/`).
- **Design Choices**:
  - 100% frozen backbones pre-extracted to disk; lightweight head trained in ~8 s/fold on T4 GPU (~30× fewer trainable parameters than PECL's 5.06M).
  - Visual: `facebook/dinov2-small` `[CLS]` tokens on 16 MTCNN face crops (224×224, 1.5× margin) → `(16, 384)`.
  - Audio: `microsoft/wavlm-base-plus` last hidden state adaptive average-pooled to 64 tokens → `(64, 768)`.
  - Projection to $d = 128$, temporal mean pooling, concatenation (256-d), `LayerNorm`, `Dropout(0.3)`, `Linear(256, 64) → GELU → Dropout(0.3) → Linear(64, 1)`.
  - Additional exploratory variants in `model.py` (`@tab:variants`, not run on official folds): `congruence` (`1 - cos(v, a)`, 147,714), `video_delta` (115,585), `late_delta` (222,465), `video_attention` (66,178), `audio_attention` (115,330), `late_attention` (173,058), plus `audio_text` / `text`.
- **Committed Results (`results/*/summary.json`, 3 seeds [0, 1, 2])**:
  - **`late-official`** (164,737 params, 1,274 test clips, fixed 30 epochs, thr 0.5): **72.18 ± 2.96% ACC**, **73.09% F1**, **80.41 ± 2.93% AUC**, **72.19% bal. ACC** (reported **72.2 ± 3.0 / 80.4**; per-fold: 69.0/77.1, 73.4/81.3, 74.2/82.9; per-seed: 73.8/81.8, 70.5/79.8, 72.2/79.6). Beats reported PECL PAVF by **+7.4 ACC / +17.7 AUC** (under a related, non-identical subset evaluation).
  - **`video-official`** (57,857 params): **73.15 ± 1.71% ACC**, **74.76% F1**, **80.24 ± 1.79% AUC**, **73.02% bal. ACC** (reported **73.2 / 80.2**).
  - **`audio-official`** (107,009 params): **58.84 ± 1.97% ACC**, **61.58% F1**, **62.49 ± 1.72% AUC**, **58.67% bal. ACC** (reported **58.8 / 62.5**).
  - **`late-subject`** (800/886 holdout clips): **56.09 ± 5.89% ACC**, **53.37% F1**, **57.34 ± 8.34% AUC**, **55.85% bal. ACC** (reported **56.1 / 57.3**; Δ: −16.1 ACC / −23.1 AUC).
  - **`video-subject`**: **49.89 ± 5.85% ACC**, **44.11% F1**, **51.00 ± 6.56% AUC**, **50.41% bal. ACC** (reported **49.9 / 51.0**; Δ: −23.3 ACC / −29.2 AUC).
  - **`audio-subject`**: **54.90 ± 3.76% ACC**, **57.19% F1**, **58.29 ± 4.18% AUC**, **55.67% bal. ACC** (reported **54.9 / 58.3**; Δ: −3.9 ACC / −4.2 AUC).
- **Subgroup & Probe Findings (`06-best-model.typ`, `a-subject-independent.typ`)**:
  - Gender (seed-averaged pooled `late-official`, 73.3% ACC / 81.3% AUC): 70.3% ACC / 79.3% AUC on 256 female clips vs 74.1% ACC / 81.9% AUC on 1,016 male clips.
  - Duration tertiles (2–4 s, 4–5 s, 5–48 s): 71.2%, 71.7%, 77.1% ACC (AUC 79.0, 80.4, 84.2).
  - MUMIN linear probes on frozen embeddings (5 subject folds, seed 0): 24/26 frequent cues above noise band for DINOv2 (mean AUC 0.665 vs 0.567) and 18/26 for WavLM (0.585 vs 0.539), while MUMIN + logistic regression reaches only 51.1% ACC / 52.5% AUC on the subject split (vs 58.84% paper Table 4).
  - Identity vs. scene linear probes on time-averaged DINOv2 vectors (`episode_probe.py`, 1,299 cached clips, `@tab:probes` in Appendix A): **96.3% speaker ACC** (chance 6.8%, 145 classes, random folds); **88.0% episode ACC** (random folds, chance 1.7%, 59 classes) dropping to **7.4% episode ACC** when speaker is held out; within-speaker episode ACC **87.1%** (Lee Mack, 186 clips) and **81.4%** (David Mitchell, 172 clips).

---

## 4. Direction 2: Spatio-Temporal Transformers with LoRA (`lore`)

- **Path**: `lie-detector/notebooks/architectures/lore/` (`lore/`, `scripts/`, `results/`).
- **Design Choices**:
  - Video: `MCG-NJU/videomae-base-finetuned-kinetics` (86.5M) on 16 × 224×224 speaker-centred face crops (YuNet detection + SFace 128-d clustering + median-smoothed boxes; 32 cached frames/clip).
  - Audio: `facebook/hubert-base-ls960` (94.7M) on raw 16 kHz z-normalised waveform (8 s random train window / first 10 s eval).
  - Adaptation: LoRA ($r = 8, \alpha = 16$, dropout 0.1) on `Q` and `V` projections across all 12 layers + norm + linear head → **297,217 trainable params per branch** (0.31–0.34%; **594,434** total). Verified via `assert_lora_active`.
  - Late fusion (`lore/fusion.py`): z-score each branch's logits using `val` stats, combine $s = w z_v + (1-w) z_a$ with $w \in \{0, 0.05, \dots, 1\}$ tuned on `val` (official fold weights: 0.50, 0.80, 0.45).
- **Committed Results (`results/*/summary.json` & `comparison.txt`, seed 0)**:
  - **Official 3-fold** (1,426 test clips; 15% clip-level train split as `val`):
    - Video only: **65.38 ± 0.83% ACC**, **64.55% F1**, **71.15 ± 2.47% AUC**, **65.73% bal. ACC** (reported **65.4 / 71.1**).
    - Audio only: **62.77 ± 0.88% ACC**, **65.06% F1**, **66.97 ± 3.51% AUC**, **62.54% bal. ACC** (reported **62.8 / 67.0**).
    - Fused ($w$ tuned on `val`): **66.3 ± 1.5% ACC**, **65.9% F1**, **73.1% AUC**, **66.6% bal. ACC** (per-fold: 65.1/72.9, 67.9/76.0, 65.7/70.3; untuned $w = 0.5$: 67.1 ± 2.1% / 73.7%; paper-style best-test-epoch @ thr 0.5: **68.0 ± 1.8% ACC**).
  - **Subject 5-fold** (Appendix A, 886 clips / 135 speakers):
    - Video only: **51.58 ± 4.60% ACC**, **41.70% F1**, **53.62 ± 8.61% AUC**, **51.51% bal. ACC** (reported **51.6 / 53.6**; Δ: −13.8 / −17.5).
    - Audio only: **56.10 ± 5.51% ACC**, **56.05% F1**, **59.54 ± 7.39% AUC**, **56.25% bal. ACC** (reported **56.1 / 59.5**; Δ: −6.7 / −7.4).
    - Fused ($w$ tuned on `val`): **56.1 ± 7.3% ACC**, **55.1% F1**, **60.6% AUC**, **56.2% bal. ACC** (Δ: −10.2 / −12.5).
  - **Speaker-cluster bootstrap (10,000 resamples, `lore/scripts/cluster_test.py`)**: HuBERT pooled ACC 56.1% (+4.0 vs 52.1% majority, 95% CI −4.7 to +9.1, one-sided $p = 0.21$), pooled AUC 58.7 (+8.7, CI +2.8 to +14.7, $p = 0.001$); fused AUC 60.4 (CI +4.1 to +16.5); video AUC 53.5 (CI −3.2 to +10.3).

---

## 5. Direction 3: MARLIN + ConvNeXt (`fab`)

- **Path**: `lie-detector/notebooks/architectures/fab/` (`train_marlin_kaggle.ipynb`, `results/marlin-{official,subject}/`).
- **Reported Configuration**:
  - **Visual stream**: frozen `marlin_vit_base_ytf` (CVPR 2023) on 16 MTCNN face crops (conf ≥ 0.85, 1.5× margin, 224×224) → 8 × 768 sequence.
  - **Acoustic stream**: 128-band 3-channel log-mel spectrogram ($S, \Delta, \Delta^2$) through frozen `convnext_tiny` (freq-pooled) → linearly interpolated to 8 steps × 768.
  - **Sequence head**: per-stream `LayerNorm`, `768→32` + `GELU` + dropout, concat → `64→64`, 3-layer causal dilated TCN (kernel 3, dilations 1, 2, 4, residual) + temporal mean pool + `LayerNorm → Dropout(0.3) → MLP(64→64→1)` (**97,857 trainable params**). *(Swin3D/X3D, Bi-GRU, and attention pooling exist in the codebase as non-reported options.)*
- **Results (seed 0, recomputed from `preds.csv` against `dataset.csv`)**:
  - **Official 3-fold** (1,426 clips; 15% clip-level `val`): **58.9 ± 0.9% ACC**, **59.8% F1**, **58.9% AUC**, **59.0% bal. ACC** (per-fold: 60.0/58.1, 58.4/56.8, 58.4/61.8).
  - **Subject 5-fold** (Appendix A, 800/886 clips): **54.8 ± 4.0% ACC**, **55.4% AUC** (Δ: −4.1 ACC / −3.5 AUC).
  - *(Note: committed `fab/results/*/summary.json` / `folds.csv` report 59.1 / 60.3 official and 54.1 / 56.4 subject, which `preds.csv` does not reproduce; official `preds.csv` also mislabels 17 `ls_wilty_ep73_true*` clips. The report always uses the recomputed values.)*

---

## 6. Direction 4: From-Scratch Temporal CNN (`teo`)

- **Path**: `lie-detector/notebooks/architectures/teo/` (`features/extract_features.py`, `cnn-batching/`, `results/temporal-{official,subject,official-video,official-audio}/`).
- **Scope Note**: The repository also contains an experimental `DualHeadGAT` under `teo/gat/` (18,410 params at `hidden=64`) with `.pt` checkpoints only; **it is excluded from the report and deliverable**, which evaluate only `TemporalCNN`.
- **Reported Configuration (`TemporalCNN`, `cnn-batching/model_temporal.py`)**:
  - **Input (`(T, 53)` at 10 fps, zero pre-training)**: 45 dims from MediaPipe `FaceLandmarker` (15 pooled 3D facial regions) + 8 dims from openSMILE `eGeMAPSv02` LLDs.
  - **Architecture**: `Conv1d(53→64, k=5)` + 3× `Conv1d(64→64, k=3)` with `BatchNorm1d`, `ReLU`, `Dropout(0.3)` on first two blocks, masked temporal mean pool ("time batching"), `lie_head` (`Linear(64, 1)`). Optional `mumin_head` (`Linear(64, 41)`) is switched off (`mumin_weight = 0.0`). **57,322 trainable params** (54,657 used at inference).
- **Committed Results (`teo/results/*/summary.json`, 3 seeds [0, 1, 2])**:
  - **`temporal-official`** (1,426 clips, 15% clip-level `val`): **56.33 ± 2.85% ACC**, **56.14% F1**, **59.08 ± 2.45% AUC**, **56.42% bal. ACC** (reported **56.3 ± 2.9 / 59.1**; per-fold: 55.1/56.8, 59.4/62.0, 54.5/58.4).
  - **`temporal-official-video`** (audio columns zeroed): **55.03 ± 1.76% ACC**, **57.40% F1**, **57.84 ± 1.62% AUC**, **54.51% bal. ACC** (reported **55.0 / 57.8**).
  - **`temporal-official-audio`** (landmark columns zeroed): **55.87 ± 2.65% ACC**, **55.82% F1**, **59.08 ± 2.01% AUC**, **55.75% bal. ACC** (reported **55.9 / 59.1**).
  - **`temporal-subject`** (Appendix A): **50.52 ± 5.17% ACC**, **52.90% F1**, **51.91 ± 6.67% AUC**, **50.94% bal. ACC** (reported **50.5 / 51.9**; Δ: −5.8 ACC / −7.2 AUC).

---

## 7. Reference Papers Used in the Report (`report/refs.bib`)

- **Dataset, Benchmarks & Psychology**: Guo et al., *DOLOS / PECL*, ICCV 2023 (`guo2023dolos`); Vrij et al., 2008 (`vrij2008cognitive`); DePaulo et al., 2003 (`depaulo2003cues`); Bond & DePaulo, 2006 (`bond2006accuracy`); Allwood et al., *MUMIN*, 2007 (`allwood2007mumin`); Pérez-Rosas et al., *Real-Life Trials*, ICMI 2015 (`perezrosas2015trial`); Gupta et al., *Bag-of-Lies*, CVPRW 2019 (`gupta2019bagoflies`); Soldner et al., *Box of Lies*, NAACL-HLT 2019 (`soldner2019boxoflies`).
- **Baselines & Backbones**: Dosovitskiy et al., *ViT*, ICLR 2021 (`dosovitskiy2021vit`); Baevski et al., *wav2vec 2.0*, NeurIPS 2020 (`baevski2020wav2vec`); Oquab et al., *DINOv2*, TMLR 2024 (`oquab2024dinov2`); Chen et al., *WavLM*, IEEE JSTSP 2022 (`chen2022wavlm`); Hsu et al., *HuBERT*, IEEE/ACM TASLP 2021 (`hsu2021hubert`); Tong et al., *VideoMAE*, NeurIPS 2022 (`tong2022videomae`); Hu et al., *LoRA*, ICLR 2022 (`hu2022lora`); Cai et al., *MARLIN*, CVPR 2023 (`cai2023marlin`); Liu et al., *ConvNeXt*, CVPR 2022 (`liu2022convnext`); Zhang et al., *MTCNN*, IEEE SPL 2016 (`zhang2016mtcnn`); Jadoul et al., *Parselmouth*, J. Phonetics 2018 (`jadoul2018parselmouth`); Eyben et al., *GeMAPS*, IEEE TAFFC 2016 (`eyben2016gemaps`); Lugaresi et al., *MediaPipe*, 2019 (`lugaresi2019mediapipe`).
- **Direction Reference Folders** (committed PDFs not all cited in `report.typ`): `cross-modal-fusion/reference paper/` (11 PDFs: DOLOS/PECL, DINOv2, WavLM, V-JEPA 2, MulT, Perceiver, GMU, Adapters, LoRA, Multimodal ML Survey, Shortcut Learning) and `lore/references/` (4 PDFs: DOLOS/PECL, VideoMAE, HuBERT, LoRA).
