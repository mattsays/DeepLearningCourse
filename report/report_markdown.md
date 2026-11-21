# Multimodal Deception Detection on the DOLOS Dataset: Architectural Analysis and Experimental Report

## 1. General Task, Dataset, and Evaluation Protocol

### 1.1 Problem Formulation and Inference Constraints

### 1.2 The DOLOS Dataset

### 1.3 Evaluation Protocols: Subject-Independent 5-Fold CV vs. Official 3-Fold Clip-Level Split

### 1.4 Metrics, Threshold Selection, and Majority Baselines

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
