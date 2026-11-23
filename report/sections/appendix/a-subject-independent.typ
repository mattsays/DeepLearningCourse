#import "../../template.typ": *

= Subject-Independent Evaluation <app:subject>

== Motivation: speakers on both sides of the split
The official folds are drawn at random over _clips_, so the same person, often within the same
round, appears in both training and test. @tab:leak measures this on the full annotation set. The
1,678 clips also form only 151 (episode, label) groups, with a median of 10 clips per group, and all
clips of a group share the speaker, clothing, set and label. On the downloadable clips the overlap
is the same or larger: 95.0%, 96.9% and 99.6% of the test clips come from a training speaker, and
100% from a training episode.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Fold], [Train / test clips], [Test speakers], [Also in train], [Test clips with a train speaker]),
    table.hline(stroke: 0.5pt),
    [1], [1,099 / 549], [168], [153], [95.3%],
    [2], [1,095 / 553], [162], [150], [95.7%],
    [3], [1,100 / 548], [163], [160], [98.7%],
    table.hline(),
  ),
  caption: [Speaker overlap in the official folds (`lie-detector/docs/splits.md`).],
) <tab:leak>

== Our 5-fold protocol
The unit is the person: no subject ever crosses a boundary.
+ The two team captains (468 clips, 69.0% deception) are always in training and never evaluated;
  otherwise they would make up ≈70% of a test fold and the fold would measure two people.
+ 27 subjects (182 clips) are held out as a test set that is opened only once, at the end.
+ The remaining 151 subjects (1,021 clips, 47.7% deception) are split into 5 folds. The assignment
  is optimised so that every fold has 47.3–48.0% deception and the same share of female clips; a
  naive random split would give folds from 42% to 66% deception.
+ Inside each fold, 15% of the training subjects form a validation set, used for early stopping and
  for choosing the threshold.

The script is `lie-detector/scripts/make_splits.py`; same annotations and same seed give
byte-identical files, and every split file is verified against a hash at load time.

== Results and the protocol effect
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Model], [Official ACC / AUC], [Subject ACC / AUC], [Δ ACC], [Δ AUC]),
    table.hline(stroke: 0.5pt),
    [Theory-driven baseline], [55.1 / 58.8], [50.8 / 54.0], [−4.3], [−4.8],
    [Cross-Modal Fusion, late], [72.2 / 80.4], [56.1 / 57.3], [−16.1], [−23.1],
    [#h(1em) video only], [73.2 / 80.2], [49.9 / 51.0], [−23.3], [−29.2],
    [#h(1em) audio only], [58.8 / 62.5], [54.9 / 58.3], [−3.9], [−4.2],
    [VideoMAE + HuBERT, LoRA], [66.3 / 73.1], [56.1 / 60.6], [−10.2], [−12.5],
    [#h(1em) video only], [65.4 / 71.1], [51.6 / 53.6], [−13.8], [−17.5],
    [#h(1em) audio only], [62.8 / 67.0], [56.1 / 59.5], [−6.7], [−7.5],
    [Majority class], [52.4–53.3], [52.1–52.6], [—], [—],
    table.hline(),
  ),
  caption: [The same models on the official and the subject-independent protocol. Baseline and
    Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT: seed 0. The Temporal CNN and MARLIN + ConvNeXt
    have no holdout results on this split.],
) <tab:protocol>

On unseen speakers every model falls to within a few points of the majority floor. The drop is
concentrated in the visual channel: DINOv2 alone loses 23 accuracy points and falls to chance, and
the LoRA-adapted VideoMAE loses 14, whereas WavLM, HuBERT and the 17 acoustic features lose 4–7. The
acoustic representations carry no face, clothing or set, so they have little identity to exploit.
Audio becomes the strongest single channel: HuBERT alone reaches 56.1% (95% speaker-cluster
bootstrap interval 51.4–60.8%, one-sided binomial $p = 0.010$ against the 52.1% majority), the only
single-channel result clearly above chance. The DOLOS paper's own cross-dataset results (57.54% on
Bag-of-Lies, 56.17% on Real-Life Trials, both with unseen subjects) fall in the same range as our
subject-independent estimates.

Two further observations point the same way. The fused Cross-Modal Fusion model chooses very
different thresholds on different folds and seeds (0.06–0.84), a sign that its validation scores
carry little signal. And the official split is not only easier but more reproducible: its per-seed
AUCs span 2.2 points, against 5.0 on the subject split, because identity is a stable signal and
behaviour is not.

The reserved test set of 27 subjects has not been opened: no configuration was selected with enough
confidence to justify spending it.

== MUMIN diagnostics
Linear probes on the frozen embeddings recover the 41 MUMIN annotations better than chance: mean
AUC 0.665 against a 0.567 noise band for DINOv2, and 0.585 against 0.539 for WavLM, with 25 of 26
frequent cues above their band in at least one modality. The encoders therefore _see_ the annotated
behaviours. Yet a logistic regression trained on the 41 ground-truth annotations themselves reaches
only 51.1% ACC / 52.5% AUC on the subject split, against 58.84% reported in the paper on the official
folds. The behaviours exist and are encoded; they do not predict deception on new people.
