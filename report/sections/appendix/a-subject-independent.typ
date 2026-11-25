#import "../../template.typ": *

= Subject-Independent Evaluation <app:subject>

== Motivation: speakers on both sides of the split
The official folds are drawn at random over _clips_, so the same person, often within the same
round, appears in both training and test. @tab:leak measures this on the full annotation set. The
1,678 clips also form only 149 (episode, label) groups, with a median of 11 clips per group, and all
clips of a group share the speaker, clothing, set and label. On the clips that could still be
downloaded the picture is the same: 95.0%, 96.9% and 99.6% of their test clips come from a training
speaker, and 100% from a training episode.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Fold], [Train / test clips], [Test speakers], [Also in train], [Test clips with a train speaker]),
    table.hline(stroke: 0.5pt),
    [1], [1,099 / 549], [161], [149], [96.0%],
    [2], [1,095 / 553], [157], [148], [96.7%],
    [3], [1,100 / 548], [159], [156], [98.7%],
    table.hline(),
  ),
  caption: [Speaker overlap in the official folds, computed from the official fold files on the
    full annotation set.],
) <tab:leak>

== Our 5-fold protocol
The unit is the person: no subject ever crosses a boundary.
+ The two team captains (468 clips, 69.0% deception) are always in training and never evaluated;
  otherwise they would make up ≈70% of a test fold and the fold would measure two people.
+ 27 subjects (182 annotated clips; 24 subjects and 155 clips among the downloadable ones) are held
  out as a test set that is opened only once, at the end.
+ The remaining 151 subjects (1,021 annotated clips, 47.7% deception; 135 subjects and 886 clips
  downloadable) are split into 5 folds. The assignment
  is optimised so that every fold has 47.3–48.0% deception and the same share of female clips; a
  naive random split would give folds from 42% to 66% deception.
+ Inside each fold, 15% of the training subjects form a validation set, used for early stopping and
  for choosing the threshold.

The script is `scripts/make_splits.py`; same annotations and same seed give
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
    [Theory-driven baseline], [56.5 / 59.2], [53.1 / 53.6], [−3.4], [−5.6],
    [Temporal CNN], [56.3 / 59.1], [50.5 / 51.9], [−5.8], [−7.2],
    [Cross-Modal Fusion, late], [72.2 / 80.4], [56.1 / 57.3], [−16.1], [−23.1],
    [#h(1em) video only], [73.2 / 80.2], [49.9 / 51.0], [−23.3], [−29.2],
    [#h(1em) audio only], [58.8 / 62.5], [54.9 / 58.3], [−3.9], [−4.2],
    [VideoMAE + HuBERT, LoRA], [66.3 / 73.1], [56.1 / 60.6], [−10.2], [−12.5],
    [#h(1em) video only], [65.4 / 71.1], [51.6 / 53.6], [−13.8], [−17.5],
    [#h(1em) audio only], [62.8 / 67.0], [56.1 / 59.5], [−6.7], [−7.4],
    [MARLIN + ConvNeXt#super[‡]], [58.9 / 58.9], [54.8 / 55.4], [−4.1], [−3.5],
    [Majority class], [52.4–53.3], [52.1–52.6], [—], [—],
    table.hline(),
  ),
  caption: [The same models on the official and the subject-independent protocol. Baseline,
    Temporal CNN and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT and MARLIN + ConvNeXt: seed 0.
    #super[‡]Recomputed from the run's per-clip predictions against the `dataset.csv` labels (@sec:results).
    Its subject-split predictions, like those of Cross-Modal Fusion, cover 800 of the 886 holdout
    clips (152–165 per fold instead of 177–178).],
) <tab:protocol>

#figure(
  image("../../figures/fig1-protocol-gap.pdf", width: 100%),
  caption: [Every model on both protocols. Filled: official 3-fold; open: subject-independent 5-fold;
    the numbers on the right give the change in accuracy and AUC. The drop is concentrated in the
    visual channel: DINOv2 alone loses 23 accuracy and 29 AUC points and falls below the majority
    floor (grey band), while the acoustic branches lose 3–7. MARLIN + ConvNeXt has the smallest gap
    of the deep models.],
) <fig:protocol-gap>

On unseen speakers every model falls to within a few points of the majority floor (@fig:protocol-gap). The drop is
concentrated in the visual channel: DINOv2 alone loses 23 accuracy points and falls to chance, and
the LoRA-adapted VideoMAE loses 14, whereas WavLM, HuBERT and the 17 acoustic features lose 3–7. The
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

MARLIN + ConvNeXt has the smallest protocol gap of the deep models (−4.1 ACC, −3.5 AUC), but also
the lowest official score of the three models built on pre-trained encoders, it was run without
single-stream variants, and its subject-split predictions, like those of Cross-Modal Fusion, miss 86 holdout clips. Whether its face-specific
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
inside the training folds. The reserved test set of 24 subjects (155 downloadable clips) exists for this reason.
The configuration to be scored on it is frozen — Cross-Modal Fusion, late variant, seeds 0, 1 and
2, retrained on the same five folds (`cross-modal-fusion/test.ipynb`) — and the set is opened
exactly once; it was not opened for this report. With 24 subjects, a subject-level binomial
interval around 60% accuracy is about ±19.6 points, enough to confirm the order of magnitude of the
five-fold estimate but not to rank models.

== MUMIN diagnostics
Linear probes on the frozen embeddings recover the 41 MUMIN annotations better than chance: mean
AUC 0.665 against a 0.567 noise band for DINOv2, and 0.585 against 0.539 for WavLM, with 25 of 26
frequent cues above their band in at least one modality. The encoders therefore _see_ the annotated
behaviours. Yet a logistic regression trained on the 41 ground-truth annotations themselves reaches
only 51.1% ACC / 52.5% AUC on the subject split, against 58.84% reported in the paper on the official
folds. The behaviours exist and are encoded; they do not predict deception on new people.

== What the embedding knows: identity or scene? <app:probes>
The subject split removes the person but not the recording session: an episode is one studio, one
lighting setup, one day, and 82–94% of the holdout clips still come from an episode present in
training (@tab:leak). To see which of the two the encoder carries, we fit linear probes
(multinomial logistic regression on standardised inputs, five folds) to the time-averaged DINOv2
vector of the 1,299 clips that have a cached embedding, and compare each probe with the same
classifier on permuted labels.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right),
    table.hline(),
    table.header([Probe], [Folds], [Accuracy], [Chance], [Classes], [Clips]),
    table.hline(stroke: 0.5pt),
    [Speaker], [random], [96.3%], [6.8%], [145], [1,299],
    [Episode], [random], [88.0%], [1.7%], [59], [1,299],
    [Episode], [speaker held out], [7.4%], [1.7%], [59], [1,299],
    [Episode of Lee Mack], [random], [87.1%], [1.8%], [45], [186],
    [Episode of David Mitchell], [random], [81.4%], [4.0%], [42], [172],
    table.hline(),
  ),
  caption: [Linear probes on the frozen time-averaged DINOv2 vector. Chance is the mean accuracy of
    the same classifier on permuted labels (five repeats) and Classes the number of distinct labels.
    Random folds put clips of one round on both sides of the boundary, as the official protocol does;
    speaker-held-out folds do not.],
) <tab:probes>

Under folds that put clips of one round on both sides — the situation the official protocol creates
— the vector identifies the speaker with 96.3% accuracy over 145 speakers, against 6.8% for permuted
labels, and the episode with 88.0% over 59 (1.7%). Identity is therefore not merely a correlate of
the label in this representation: it is almost perfectly readable. When the speaker is held out,
episode accuracy falls to 7.4% against the same 1.7%: the scene survives the split, at 4.4 times
chance, but an order of magnitude weaker than identity. Within a single speaker the probe still
separates the session at 87% and 81% over 45 and 42 episodes, but there clips of one round are
adjacent frames of the same video, so we read those two as an upper bound.

The official split is therefore not merely leaky but trivially leaky: identity alone is readable.
This does not excuse the subject-independent estimate, which still contains the residual scene
information because the split is not episode-disjoint; it says that an episode-disjoint split would
remove the smaller of the two channels. We have not run it, so whether it would lower the numbers
without reordering them remains an expectation.
