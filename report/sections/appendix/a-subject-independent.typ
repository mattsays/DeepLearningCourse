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

The folds are subject-disjoint but not episode-disjoint: clips of different speakers recorded in the
same episode share studio, lighting and audio conditions, and can still fall on both sides of a
boundary. An episode-disjoint control has not been run.

== Results and the protocol effect
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Model], [Official ACC / AUC], [Subject ACC / AUC], [Δ ACC], [Δ AUC]),
    table.hline(stroke: 0.5pt),
    [Theory-driven baseline], [55.1 / 58.8], [50.8 / 54.0], [−4.3], [−4.8],
    [Temporal CNN], [56.3 / 59.1], [50.5 / 51.9], [−5.8], [−7.2],
    [Cross-Modal Fusion, late], [72.2 / 80.4], [56.1 / 57.3], [−16.1], [−23.1],
    [#h(1em) video only], [73.2 / 80.2], [49.9 / 51.0], [−23.3], [−29.2],
    [#h(1em) audio only], [58.8 / 62.5], [54.9 / 58.3], [−3.9], [−4.2],
    [VideoMAE + HuBERT, LoRA], [66.3 / 73.1], [56.1 / 60.6], [−10.2], [−12.5],
    [#h(1em) video only], [65.4 / 71.1], [51.6 / 53.6], [−13.8], [−17.5],
    [#h(1em) audio only], [62.8 / 67.0], [56.1 / 59.5], [−6.7], [−7.5],
    [MARLIN + ConvNeXt], [59.1 / 60.3], [54.1 / 56.4], [−5.0], [−3.9],
    [Majority class], [52.4–53.3], [52.1–52.6], [—], [—],
    table.hline(),
  ),
  caption: [The same models on the official and the subject-independent protocol. Baseline,
    Temporal CNN and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT and MARLIN + ConvNeXt: seed 0.],
) <tab:protocol>

On unseen speakers every model falls to within a few points of the majority floor. The drop is
concentrated in the visual channel: DINOv2 alone loses 23 accuracy points and falls to chance, and
the LoRA-adapted VideoMAE loses 14, whereas WavLM, HuBERT and the 17 acoustic features lose 4–7. The
acoustic representations exclude the _visual_ identity cues — face, clothing, set — but not
identity altogether: voice timbre and the recording conditions of an episode remain available, so
the smaller drop is consistent with, not proof of, less reliance on identity. The Temporal CNN,
which sees no pixels but the mean 3-D position of 15 face regions, still loses 5.8 accuracy and 7.2
AUC points and ends at chance (50.5 / 51.9): absolute landmark coordinates also encode face shape
and head placement, so they too can carry speaker-specific information.

Audio becomes the strongest single channel, but only its ranking is reliably above chance. We pooled
the 886 holdout clips of VideoMAE + HuBERT (135 speakers), each judged at its own fold's threshold,
and resampled whole speakers 10,000 times, comparing the model on every resample with the majority
rate of that same resample (`lore/scripts/cluster_test.py`). HuBERT's accuracy of 56.1% is 4.0
points above the 52.1% majority, but the 95% interval of that gain, −4.7 to +9.1, contains zero
(one-sided $p = 0.21$): its accuracy is suggestive, not significant. Its pooled AUC of 58.7 is
8.7 points above 0.5 with interval +2.8 to +14.7 ($p = 0.001$), and the fused model behaves the
same way (AUC 60.4, interval +4.1 to +16.5). The video branch is not distinguishable from chance on
either measure (AUC 53.5, interval −3.2 to +10.3). A binomial test on the clips would report
$p = 0.010$ for HuBERT's accuracy, but it treats the clips of one speaker as independent and is
therefore too optimistic. The DOLOS paper's own cross-dataset results (57.54% on
Bag-of-Lies, 56.17% on Real-Life Trials, both with unseen subjects) fall in the same range as our
subject-independent estimates.

MARLIN + ConvNeXt has the smallest protocol gap of the deep models (−5.0 ACC, −3.9 AUC), but also
the lowest official score, and it was run without single-stream variants. Whether its face-specific
encoder relies less on identity than DINOv2 or VideoMAE, or simply has less to lose, cannot be
decided from these numbers.

Two further observations point the same way. The fused Cross-Modal Fusion model chooses very
different thresholds on different folds and seeds (0.06–0.84), a sign that its validation scores
carry little signal. And the official split is not only easier but more reproducible: its per-seed
AUCs span 2.2 points, against 5.0 on the subject split, consistent with identity being a stable signal and
behaviour is not.

These five-fold numbers are _development estimates_, not untouched test results. The variants of
@tab:variants and the design choices of each model were explored on the same folds, so any choice
made by looking at them is optimistically biased; a clean estimate needs model selection nested
inside the training folds. The reserved test set of 27 subjects (182 clips) exists for this reason.
The configuration to be scored on it is frozen — Cross-Modal Fusion, late variant, seeds 0, 1 and
2, retrained on the same five folds (`cross-modal-fusion/test.ipynb`) — and the set is opened
exactly once; it was not opened for this report. With 27 subjects its interval would be about
±18.5 points (`lie-detector/docs/evaluation.md`), enough to confirm the order of magnitude of the
five-fold estimate but not to rank models.

== MUMIN diagnostics
Linear probes on the frozen embeddings recover the 41 MUMIN annotations better than chance: mean
AUC 0.665 against a 0.567 noise band for DINOv2, and 0.585 against 0.539 for WavLM, with 25 of 26
frequent cues above their band in at least one modality. The encoders therefore _see_ the annotated
behaviours. Yet a logistic regression trained on the 41 ground-truth annotations themselves reaches
only 51.1% ACC / 52.5% AUC on the subject split, against 58.84% reported in the paper on the official
folds. The behaviours exist and are encoded; they do not predict deception on new people.
