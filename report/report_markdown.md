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

### 2.2 Acoustic Feature Extraction (17 Hand-Crafted Mathematical & Physical Features)

### 2.3 Classifier Architecture (`LoadHead`, 609 Parameters)

### 2.4 Results Across Protocols and Comparison with the MUMIN Oracle

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
