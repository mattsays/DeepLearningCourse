#import "../template.typ": *

// ================================================================ 3
= Dataset and Evaluation Protocol <sec:data>

== DOLOS
The clips come from the British game show _Would I Lie To You?_. In each round a contestant reads a
statement from a card, true or false and known only to them, and defends it under questioning from
the opposing team, who must guess. The truth is revealed at the end of the round, so every clip has
an objective _game ground truth_ rather than an annotator's opinion. The speakers are comedians and
celebrities, the setting is a studio, and the stakes are low.

The paper reports 1,675 clips (899 deception, 776 truth), 213 participants and 84 episodes, with a
median duration of 5 s. The annotation files we received differ slightly (@tab:stats); in
particular, 5 of their clips are longer than the 19 s maximum stated by the paper (up to 48 s),
most likely annotation errors. The clips
are not distributed as video: they are cut from YouTube episodes. Of the 1,678 annotated clips,
1,452 could still be downloaded with their media; all download failures come from 9 episodes that
are no longer available. Structurally, the corpus consists of *149 episode–label rounds* (a
median of 11 clips per round), is skewed by gender (*79.1% male* vs. *20.9% female* clips,
with deception rates of 56.4% and 43.6%, respectively), and includes two recurring team captains
(_Lee Mack_ and _David Mitchell_) who contribute 468 clips (28% of the dataset, deceptive 69.0%
of the time).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    table.hline(),
    table.header([], [Paper], [Local annotations]),
    table.hline(stroke: 0.5pt),
    [Clips], [1,675], [1,678 rows (1,677 unique)],
    [Deception / truth], [899 / 776], [901 / 777 (53.7% / 46.3%)],
    [Subjects], [213], [180 (after alias merging)],
    [Episodes], [84], [75 distinct videos],
    [Male / female clips], [1,324 / 351], [1,327 / 351 (79.1% / 20.9%)],
    [Duration], [2–19 s, median 5 s], [2–48 s, median 5 s (5 clips > 19 s)],
    [Clips downloadable today], [—], [1,452],
    table.hline(),
  ),
  caption: [DOLOS statistics: as published and as measured on the local annotation files.],
) <tab:stats>

== Preprocessing
A shared script cuts the annotated time range out of each episode into one `.mp4` per clip, and
extracts its audio track as 16 kHz mono WAV with `ffmpeg`. Everything after that is specific to each
architecture, because each one consumes a different representation: face crops at different rates
and resolutions, raw waveforms, spectrograms or landmark coordinates. @tab:inputs summarises the
inputs; @sec:arch gives the details.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.hline(),
    table.header([Model], [Visual input], [Audio input], [Face detector]),
    table.hline(stroke: 0.5pt),
    [Theory-driven baseline], [—], [16 kHz waveform → 17 statistics], [—],
    [Temporal CNN], [478 landmarks at 10 fps → 15 regions], [8 eGeMAPS descriptors at 10 fps], [MediaPipe],
    [Cross-Modal Fusion], [16 face crops, 224²], [whole waveform, 16 kHz], [MTCNN],
    [VideoMAE + HuBERT], [16 face crops, 224² (of 32 cached)], [waveform, ≤ 10 s, 16 kHz], [YuNet + SFace],
    [MARLIN + ConvNeXt], [16 face crops, 224²], [3-channel log-mel ($S, Delta, Delta^2$)], [MTCNN],
    table.hline(),
  ),
  caption: [Inputs of each model. Frames are always sampled uniformly over the whole clip.],
) <tab:inputs>

== Official protocol
We follow the evaluation of the dataset paper and of earlier deception benchmarks: the three
train/test folds distributed with DOLOS. Each fold places about two thirds of the clips in training
and one third in test (1,099/549, 1,095/553 and 1,100/548 clips in the paper's files), and results
are averaged over the three test folds. The folds do not cover the whole corpus: each one
re-divides the same 1,646 of the 1,677 distinct annotated clips, and the remaining 31, mostly clips
with broken timestamps, belong to no fold. The 1,650 of @tab:coverage is therefore the sum of the
three test files, not a count of distinct clips, and it is the paper's denominator for @tab:pecl.

Because some clips can no longer be downloaded, and because each architecture needs its own
features, every model is scored on a subset of each test fold: from 77% to 86% of the paper's test
clips (@tab:coverage). Four models are scored on essentially all 1,427 downloadable clips; only
Cross-Modal Fusion, which needs a cached embedding for both modalities, loses 153 more. Our numbers and the paper's are therefore
related but not identical evaluations, and our models are not scored on exactly the same clips as
each other.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Test clips scored], [Fold 1], [Fold 2], [Fold 3], [Total]),
    table.hline(stroke: 0.5pt),
    [Paper's files], [549], [553], [548], [1,650],
    [Downloadable clips], [483], [483], [461], [1,427],
    table.hline(stroke: 0.3pt),
    [Theory-driven baseline], [483], [483], [461], [1,427],
    [Temporal CNN], [482], [483], [461], [1,426],
    [Cross-Modal Fusion], [429], [432], [413], [1,274],
    [VideoMAE + HuBERT, LoRA], [482], [483], [461], [1,426],
    [MARLIN + ConvNeXt], [482], [483], [461], [1,426],
    table.hline(),
  ),
  caption: [Test clips actually scored on each official fold. Cross-Modal Fusion needs a cached
    embedding for every modality. The fold-1 test file lists one clip twice: the baseline and Cross-Modal Fusion score both
    entries, the other models score it once.],
) <tab:coverage>

== Metrics and evaluation rules
We report the three metrics of the paper:
- *accuracy (ACC)*, the share of correct decisions;
- *F1* on the `deception` class, the harmonic mean of precision and recall for lies;
- *AUC*, the area under the ROC curve, which does not depend on the decision threshold and is
  therefore the most stable comparison on test folds of ≈450 clips.

We add *balanced accuracy* (the mean of the two per-class recalls) where available, and the
_majority baseline_, always predicting the most frequent class on the same clips, as the floor
every model must beat: 52.4–53.3% depending on the subset.

The official folds have no validation set, and the models handle this in two ways (@tab:rules).
Cross-Modal Fusion and the theory-driven baseline train for a fixed number of epochs and classify at
the paper's operating point, threshold 0.5. The Temporal CNN, VideoMAE + HuBERT and
MARLIN + ConvNeXt carve a
stratified 15% validation set out of each training fold, at the clip level, and use it for early
stopping and to choose the threshold. Neither rule ever looks at the test fold. The second rule
trains on less data but tunes more; we keep both and state which one each number uses.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.hline(),
    table.header([Model], [Validation], [Stopping], [Threshold]),
    table.hline(stroke: 0.5pt),
    [Theory-driven baseline], [none], [fixed 60 epochs], [0.5],
    [Cross-Modal Fusion], [none], [fixed 30 epochs], [0.5],
    [Temporal CNN], [15% of train, clip-level], [best val AUC], [max. val balanced ACC],
    [VideoMAE + HuBERT, LoRA], [15% of train, clip-level], [best val AUC], [max. val balanced ACC],
    [MARLIN + ConvNeXt], [15% of train, clip-level], [best val loss], [max. val balanced ACC],
    table.hline(),
  ),
  caption: [How each model is trained and thresholded on the official folds.],
) <tab:rules>
