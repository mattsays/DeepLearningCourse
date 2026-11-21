# Multimodal Deception Detection on the DOLOS Dataset: Architectural Analysis and Experimental Report

## 1. General Task, Dataset, and Evaluation Protocol

### 1.1 Problem Formulation and Motivation

Automated deception detection is one of the most intriguing and demanding problems at the intersection of multimodal machine learning and human behavioral analysis. Unlike standard action or speech recognition—where the target signal is explicit—deception has no single, unmistakable physical signature. Even human observers struggle to separate truth from lies in brief conversational clips, because deceptive cues are subtle, fleeting, and often hidden in the interplay between channels: a speaker may maintain a composed facial expression while betraying slight tension, hesitation, or pitch instability in their voice.

In this project, we tackle **fully automated, audio-visual deception detection** as a binary classification problem: given a short raw video clip (typically 2 to 19 seconds) of a person answering a question, a model must predict whether the statement is **`truth` (0)** or **`deception` (1)** using only signals that can be computed directly from the video and audio streams at inference time. Rather than relying on a single model family, we design a controlled comparative study across **four distinct paradigms**—spanning theory-driven acoustic statistics (`load`), frozen foundation models (`tia`), parameter-efficient transformer adaptation (`lore`), and spatio-temporal convolutions (`fab`)—all evaluated under a unified, leakage-free protocol to uncover what multimodal deep learning can genuinely learn about human deception.

### 1.2 The DOLOS Dataset

Experiments are conducted on **DOLOS** (Guo et al., ICCV 2023), currently one of the largest non-laboratory audio-visual deception datasets, built from episodes of the British television game show *Would I Lie To You?*. In each round, a contestant reads a personal statement from a card (either true or false, known only to them) and must defend it under questioning from the opposing team. Because the truth of the card is revealed at the end of the round, every clip belonging to that round inherits an objective **game ground truth** rather than a subjective annotator's guess.

While the original paper reports 1,675 clips (899 deceptive, 776 truthful) across 213 participants and 84 episodes (median duration 5 seconds), the local annotation files contain **1,678 rows** (1,677 unique clip names; 901 `deception` / 777 `truth`, i.e., **53.7% vs. 46.3%**), spanning **75 distinct YouTube episode IDs** and **180 unique subjects** after merging name aliases. The corpus is heavily skewed by gender (**79.1% male** vs. **20.9% female** clips, with deception rates of 56.6% and 43.8%, respectively).

Two structural properties of the dataset strongly shape our experimental design:
1. **Dominant recurring speakers**: two permanent team captains (*Lee Mack* with 239 clips and *David Mitchell* with 233 clips) account for 468 clips (~28% of the dataset) and are deceptive 69.0% of the time. If placed in a test set, a model could achieve high accuracy simply by recognizing their faces or voices.
2. **Round-level grouping**: clips are not independent events. The 1,678 clips belong to **151 episode–label groups** (a median of 10 clips per group), where all clips from the same round share the same speaker, clothing, lighting, studio background, and label.

### 1.3 Evaluation Protocols: Subject-Independent 5-Fold CV vs. Official 3-Fold Clip-Level Split

A critical methodological finding of this project concerns how the data is split between training and testing:

- **Official 3-Fold Split (Clip-Level, Paper Benchmark)**: this is the standard benchmark distributed by the DOLOS authors—consistent with prior deception datasets such as *Box of Lies* and *Real-Life Trials*—which partitions individual clips randomly into 3 folds. Because clips are sampled independently, **95.3% to 98.7% of test clips involve speakers who also appear in the training set**, often from the same game-show round. This protocol therefore measures **within-corpus / within-speaker** performance (including familiarity with a speaker's baseline demeanor and studio setting) rather than generalization to brand-new individuals. We run this split separately solely to ensure a fair, apples-to-apples comparison with the published ICCV 2023 results.
- **Subject-Independent 5-Fold Cross-Validation (Our Primary, Stricter Protocol)**: to go beyond speaker familiarity and test whether a model genuinely learns *deceptive behavior* on unseen people, we designed a much stricter **subject-disjoint** evaluation where no speaker ever crosses the boundary between training, validation, and testing:
  - The two recurring captains (468 clips) are pinned exclusively to the **training** side and are never evaluated.
  - **27 subjects** (182 clips, 47.5% deception) are set aside in a locked *virgin test set* for final evaluation only.
  - The remaining **151 subjects** (1,021 clips, 47.7% deception) form the **5-fold cross-validation working set**. Each fold assigns ~102 subjects (~1,160 clips, including captains) to `train`, ~18 subjects (~124 clips) to `val` (used for early stopping and hyperparameter/threshold tuning), and ~30 subjects (~204 clips) to `holdout`.

Every direction in this project reports its main results as the **mean ± standard deviation across the 5 subject-independent fold holdouts**. When the official 3-fold split is also run, it is reported separately solely to quantify this *protocol effect* (identity and round leakage).

### 1.4 Metrics, Threshold Selection, and Majority Baselines

Because deception detection on DOLOS is both noisy and slightly class-imbalanced, no single number tells the whole story. Each architecture is evaluated using five complementary metrics:

- **Accuracy (ACC)**: the fraction of clips classified correctly; reported for direct comparability with the DOLOS paper.
- **F1-score**: the harmonic mean of precision and recall on the `deception` class, capturing the trade-off between catching lies and avoiding false accusations.
- **AUC (Area Under the ROC Curve)**: a threshold-free measure of ranking quality (0.50 = random guessing, 1.00 = perfect separation) that indicates whether deceptive clips tend to receive higher scores than truthful ones regardless of the decision cutoff. Because validation sets are small (~124 clips), AUC is the most stable metric for comparing models.
- **Balanced Accuracy**: the average recall across both classes (`truth` and `deception`), ensuring that a model cannot look strong merely by over-predicting the majority class.
- **Majority Baseline**: the accuracy achieved by always predicting the most frequent class on the exact same set of clips (**53.7%** on the full dataset, **52.4%** on the 5-fold working set, and **52.0%–52.1%** on subsets with complete cached audio/video). Any useful detector must beat this floor.

Finally, because the training folds contain ~56% deceptive clips (due to the two captains) while the holdouts contain ~47%, using a fixed `0.50` probability cutoff systematically over-predicts deception. To avoid this bias, **the classification threshold is always selected on each fold's validation set (`val`)** and then applied unchanged to that fold's `holdout`.

---

## 2. Theory-Driven Mathematical Baseline (`load`: Cognitive Load & Vocal Arousal)

### 2.1 Theoretical Foundation: Cognitive Load and Vocal Arousal

Before introducing deep neural architectures with millions of parameters, we establish a lightweight, transparent baseline called `load`. Rather than learning representations from raw pixels or waveforms, `load` relies exclusively on **17 hand-crafted acoustic measurements** computed from the clip's 16 kHz audio track, grounded in two established psychological theories of deception:

1. **Cognitive Load (Vrij's approach)**: fabricating and maintaining a lie while monitoring an interlocutor's reaction requires greater mental effort than recalling the truth. This extra cognitive load can slow down speech production and increase hesitations and pauses. *(An important caveat applies here: Vrij's paradigm works best when an interviewer actively elicits load through unexpected questions; in DOLOS, our classifier can only observe short answers passively without access to the question).*
2. **Vocal Arousal (Voice-Stress research)**: concealed emotion and stress subtly affect autonomic control of respiration and vocal-fold tension, manifesting as shifts in vocal pitch, cycle-to-cycle vocal instability, and reduced harmonic clarity.

### 2.2 Acoustic Feature Extraction (17 Hand-Crafted Mathematical & Physical Features)

Each clip is converted into a single 17-dimensional vector in a few milliseconds on CPU (no GPU required), combining two complementary feature families:

- **Energy & Pause Structure (9 features, NumPy)**: the waveform is analyzed over 25 ms sliding windows with a 10 ms step. Any frame whose energy falls more than 25 dB below the clip's 90th-percentile loudness is marked as silence, and continuous silent segments of at least 200 ms are counted as pauses (shorter gaps reflect normal consonant closures). From this envelope, we extract `duration_s`, `rms_mean`, `rms_std`, `energy_range_db` (95th–5th percentile across voiced frames only), `voiced_ratio`, `pause_count`, `pause_ratio`, `pause_mean_ms`, and `onsets_per_s` (silent-to-voiced transitions per second, approximating syllable/word rate).
- **Pitch & Voice Quality (8 features, Praat via `parselmouth`)**: using standard phonetic analysis (75–500 Hz search range), we extract `f0_mean` and `f0_std` (mean and standard deviation of the fundamental frequency $F_0$), `f0_range_st` (pitch range in logarithmic semitones, $12 \log_2(F_0 / 100)$, making male and female voices directly comparable), `jitter_local` (micro-instability of pitch between consecutive vocal cycles), `shimmer_local` (micro-instability of amplitude), `hnr_mean` (Harmonics-to-Noise Ratio, capturing vocal clarity versus breathy or tense noise), `intensity_mean_db`, and `intensity_std_db`.

### 2.3 Classifier Architecture (`LoadHead`, 609 Parameters)

Because the 17 features span very different physical scales (e.g., `f0_mean` around 150 Hz vs. `jitter_local` around 0.02), the classifier (`LoadHead`) first standardizes each feature using `(x - mu) / sigma`, where `mu` and `sigma` are computed **strictly on each fold's training split** to avoid any data leakage. The normalized vector is then passed to a compact two-layer Multi-Layer Perceptron (MLP):

`Dropout → Linear(17, 32) → GELU → Dropout → Linear(32, 1)`

In total, the model has only **609 trainable parameters** ($17 \times 32 + 32$ in the first layer, plus $32 \times 1 + 1$ in the output layer). It is trained with class-weighted binary cross-entropy (`BCEWithLogitsLoss`), early stopping on validation AUC, and validation-tuned thresholding on the 5-fold subject split (while using a fixed epoch budget and `0.50` threshold on the 3-fold official split, which has no validation set).

### 2.4 Results Across Protocols and Comparison with the MUMIN Oracle

Evaluated across **3 random seeds** on both protocols, `load` produces the following results:

| Protocol | ACC (%) | AUC (%) | Per-Seed AUC (%) | Trainable Params |
|---|---:|---:|---|---:|
| **Subject-Independent (5-fold)** | 50.8 ± 4.1 | **54.0** | 56.4 · 53.1 · 52.7 | 609 |
| **Official Clip-Level (3-fold)** | 55.1 ± 2.1 | **58.8** | 58.5 · 59.1 · 58.6 | 609 |
| *Majority class baseline* | 52.0 | — | — | — |
| *MUMIN oracle (41 hand annotations)* | 51.1 | 52.5 | — | 41 |

Three key insights emerge from this purely mathematical baseline:
1. **Competitive with the behavioral oracle**: on unseen speakers, 17 acoustic statistics reach **54.0% AUC**, slightly outperforming a classifier trained on all 41 human-annotated MUMIN behavioral cues (**52.5% AUC**). This shows that passive observable cues in short clips carry a real, yet modest, signal.
2. **Threshold sensitivity vs. ranking**: average accuracy (50.8%) sits just below the 52.0% majority floor, largely because one fold (fold 3, which has the lowest deception rate at 45.7%) over-predicts deception at the chosen threshold—even though the threshold-free AUC (54.0%) remains consistently above 50%.
3. **Minimal protocol gap (+4.3 ACC / +4.8 AUC)**: moving from the subject-independent split to the official clip-level split yields only a small gain. Because 17 summary acoustic numbers contain no facial appearance, clothing, or studio background, `load` cannot exploit speaker or episode familiarity. This establishes a clean, leakage-free reference floor against which all deep multimodal architectures can be judged.

---

## 3. Direction 1 (`tia` / `cross-modal-fusion`): Frozen Foundation Models & Cross-Modal Fusion

### 3.1 Reference Paper Analysis and Reported Results (DOLOS / PECL, Guo et al., ICCV 2023)

### 3.2 Deconstruction of Our Architecture and Differences from the Reference Paper

### 3.3 Results: Comparison Against `load` and the Reference Paper (Including Protocol Effect & MUMIN Diagnostics)

---

## 4. Direction 2 (`lore`): Spatio-Temporal Transformers with LoRA Adaptation

### 4.1 Reference Papers Analysis and Reported Results (VideoMAE, HuBERT, LoRA)

### 4.2 Deconstruction of Our Architecture and Differences from the Reference Papers

### 4.3 Results: Comparison Against `load`, `tia`, and the Reference Paper

---

## 5. Direction 3 (`fab`): Spatio-Temporal Convolutions & Sequential Modelling

### 5.1 Reference Papers Analysis and Reported Results (Swin3D / X3D, ConvNeXt, Bi-GRU / TCN)

### 5.2 Deconstruction of Our Architecture and Differences from the Reference Papers

### 5.3 Implementation Status and Expected Comparison Against `load` and Other Directions

---

## 6. Overall Summary and Cross-Architecture Comparison

### 6.1 Unified Results Board (Subject-Independent 5-Fold CV vs. Official 3-Fold)

### 6.2 Audio vs. Video Signal and the Protocol Effect (Identity & Round Leakage)

### 6.3 Conclusions, Limitations, and Future Work
