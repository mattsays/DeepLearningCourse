// Multimodal Deception Detection on DOLOS — course report.
// Compile: typst compile report/report.typ   (or paste into typst.app together with refs.bib)
// Every red TODO box marks a number or a passage that is not measured yet. Do not fill a TODO
// with a remembered number: take it from the run's results file.

// ---------------------------------------------------------------- helpers
#let todo(body) = box(
  fill: rgb("#fde8e8"),
  stroke: 0.5pt + rgb("#c0392b"),
  inset: (x: 3pt, y: 1.5pt),
  radius: 2pt,
  text(fill: rgb("#c0392b"), size: 0.85em)[*TODO:* #body],
)
#let tbd = todo[run]

// ---------------------------------------------------------------- document setup
#set document(title: "Multimodal Deception Detection on the DOLOS Dataset")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: ("New Computer Modern", "Linux Libertine"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#set figure(gap: 0.7em)
#show figure.caption: set text(size: 0.9em)
#set table(stroke: none, inset: (x: 5pt, y: 3pt))
#show table.cell.where(y: 0): set text(weight: "bold")
#show table: set text(size: 9pt)
#show table: set par(justify: false)
#set math.equation(numbering: "(1)")

// ---------------------------------------------------------------- title
#align(center)[
  #text(size: 16pt, weight: "bold")[Multimodal Deception Detection on the DOLOS Dataset: \ A Comparative Study of Audio-Visual Architectures]
  #v(0.8em)
  #grid(
    columns: 5,
    column-gutter: 1.2em,
    [#todo[Sicoli Mattia - 3126999071]], [#todo[Scuratti Matteo - 3126999078]], [#todo[Tondi Gabriele - 3126999073]], [#todo[Moliterno Fabio - 3126999076]], [#todo[Mariotti Lorenzo - 3126999231]], [#todo[Abdallah Yehia - 3126999]],
  )
  #v(0.3em)
  #text(size: 9.5pt)[Big Data and Deep Learning]
]

#v(1em)

#block(inset: (x: 2em))[
  #text(weight: "bold")[Abstract.]
  Automated deception detection asks a model to tell a lie from a truthful statement using only the
  face and voice of the speaker. We study it on DOLOS, a 1,675-clip audio-visual corpus from a
  television game show, and compare four deep architectures against a theory-driven acoustic
  baseline under the official 3-fold protocol of the dataset paper. The architectures are a temporal
  CNN over facial-landmark and prosodic features, a cross-modal fusion head over frozen DINOv2 and
  WavLM embeddings, VideoMAE and HuBERT adapted with LoRA, and a MARLIN plus ConvNeXt recurrent
  model. The cross-modal fusion model is the best: it reaches 72.2% accuracy and 80.4% AUC while
  training only 165k parameters, above the published PECL result of 64.75%. Ablations show that the
  visual stream carries almost all of this performance. #todo[one sentence on the subject-independent
  check of Appendix A, if the group decides to mention it here.]
]

// ================================================================ 1
= Introduction

Deception has no single physical signature. Unlike speech or action recognition, where the target
is explicit, a lie may leave only fleeting traces: a hesitation, a shift in pitch, a facial
expression that does not match the words. Human observers are barely above chance at the task
@depaulo2003cues, and cues are spread across channels, which motivates a multimodal approach.

We treat deception detection as binary classification. Given a raw video clip of 2–19 s in which a
person answers a question, the model predicts `truth` (0) or `deception` (1). It may use only
signals that can be computed from the video and audio at inference time; human behavioural
annotations are never an input.

Our contributions are:
- a controlled comparison of four audio-visual architectures, from a temporal CNN trained from
  scratch to large self-supervised encoders adapted with LoRA, together with a 609-parameter
  baseline built from psychological theory, all on the official DOLOS protocol;
- a cross-modal fusion model on frozen DINOv2 and WavLM that outperforms the published PECL result
  while training about 30 times fewer parameters;
- unimodal ablations for every architecture, which show which channel carries the signal.

@sec:related reviews prior work, @sec:data describes the dataset and protocol,
@sec:baseline the theory-driven baseline and @sec:arch the four architectures and their results.
@sec:best analyses the best model, and @sec:discussion discusses limitations.

// ================================================================ 2
= Related Work <sec:related>

== Behavioural cues to deception
Psychology offers two accounts of why lies may be observable. The _cognitive load_ account
@vrij2008cognitive holds that building and maintaining a lie costs more mental effort than recalling
the truth, which slows speech and adds pauses. The _arousal_ account links concealed emotion to
changes in respiration and vocal-fold tension, and so to pitch and voice quality. Meta-analyses find
these cues real but weak @depaulo2003cues. The MUMIN scheme @allwood2007mumin codes facial and
gestural behaviour by hand, and earlier multimodal datasets — Real-Life Trials
@perezrosas2015trial, Bag-of-Lies @gupta2019bagoflies and Box of Lies @soldner2019boxoflies —
use such annotations as features.

== DOLOS and PECL
DOLOS @guo2023dolos is the largest of these in-the-wild datasets and introduces PECL
(Parameter-Efficient Crossmodal Learning). PECL keeps the first four layers of an ImageNet ViT
@dosovitskiy2021vit and of wav2vec 2.0 @baevski2020wav2vec and trains two kinds of modules:
- a _UT-Adapter_ beside each attention and MLP block: 768→128 projection, a 1-D temporal
  convolution (kernel 3) and a 128→768 projection;
- a _PAVF_ (plug-in audio-visual fusion) module between each pair of visual and audio layers. It
  cross-attends 64 face frames and 64 audio tokens.

In total it trains 5.06M of 71.08M parameters (7.12%). @tab:pecl reports its results.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Model (paper, 3-fold)], [ACC], [F1], [AUC]),
    table.hline(stroke: 0.5pt),
    [ViT + UT-Adapter (visual)], [61.44], [69.42], [58.89],
    [W2V2 + UT-Adapter (audio)], [59.19], [73.46], [52.54],
    [Concatenation (last layer)], [61.62], [70.20], [60.50],
    [PECL (PAVF)], [64.75], [71.20], [62.71],
    [PECL + multi-task (25 MUMIN labels)], [66.84], [73.35], [64.58],
    [MUMIN + MLP (hand annotations)], [58.84], [—], [—],
    table.hline(),
  ),
  caption: [Results reported by #cite(<guo2023dolos>, form: "prose") on the official 3-fold protocol
    (quoted, not reproduced).],
) <tab:pecl>

== Self-supervised encoders and parameter-efficient adaptation
DOLOS is too small to fine-tune large encoders end to end, so every architecture we study builds on
self-supervised pre-training:
- DINOv2 @oquab2024dinov2 for images;
- WavLM @chen2022wavlm and HuBERT @hsu2021hubert for speech;
- VideoMAE @tong2022videomae for video, and MARLIN @cai2023marlin, its facial counterpart;
- ConvNeXt @liu2022convnext for spectrograms.

Two strategies limit overfitting: keeping the encoder frozen and training only a head, or
low-rank adaptation (LoRA) @hu2022lora, which adds a trainable low-rank update to frozen weights.

// ================================================================ 3
= Dataset and Evaluation Protocol <sec:data>

== DOLOS
The clips come from the British game show _Would I Lie To You?_. In each round a contestant reads a
statement from a card, true or false and known only to them, and defends it under questioning. The
truth is revealed at the end of the round, so every clip has an objective _game ground truth_ rather
than an annotator's opinion.

The paper reports 1,675 clips (899 deception, 776 truth), 213 participants and 84 episodes, with a
median duration of 5 s. The annotation files we received differ slightly (@tab:stats). Of these
clips, 1,452 could still be downloaded with their media. #todo[final number of clips used by each
architecture; the Cross-Modal Fusion caches cover ≈1,158.]

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
    table.hline(),
  ),
  caption: [DOLOS statistics: as published, and as measured on the local annotation files
    (`lie-detector/dataset/README.md`).],
) <tab:stats>

== Preprocessing
Every architecture starts from the same clip files: the annotated time range is cut from the source
episode, and the 16 kHz mono audio track is extracted. Visual pipelines detect and crop the face with
MTCNN @zhang2016mtcnn or MediaPipe @lugaresi2019mediapipe; the frame rate and number of frames are
specific to each architecture (@sec:arch). #todo[one line per architecture on frames sampled, if
not stated in @sec:arch.]

== Official protocol
We follow the evaluation of the dataset paper and of earlier deception benchmarks: the three
train/test folds distributed with DOLOS, each with about 1,100 training and 550 test clips. Results
are averaged over the three test folds. Using the same folds makes our numbers directly comparable
with @tab:pecl.

== Metrics
We report the three metrics of the paper:
- *accuracy (ACC)*;
- *F1* on the `deception` class;
- *AUC*, which is threshold-free and therefore the most stable comparison on test folds of ≈550
  clips.

The _majority baseline_ — always predicting the most frequent class — is the floor: 53.7% on the
whole corpus. The official folds have no validation set, so every architecture follows the same
rule. #todo[agree on one rule for all four architectures: fixed epoch budget + threshold 0.5
(Cross-Modal Fusion, theory-driven baseline) or an inner validation split carved from train
(VideoMAE + HuBERT). Selecting the epoch on the test fold is not allowed.]

// ================================================================ 4
= Theory-Driven Mathematical Baseline <sec:baseline>

Before introducing networks with millions of parameters, we set a transparent reference: a
classifier that sees only acoustic measurements chosen because psychology predicts they should
differ between lies and truths.

== Theory
The baseline encodes the two accounts of @sec:related as measurable quantities:
- _cognitive load_ appears in the timing of speech: pauses, hesitations and speaking rate;
- _arousal_ appears in the voice source: pitch level and range, micro-instability, harmonic clarity.

One caveat: Vrij's paradigm works best when an interviewer raises the load on purpose. In DOLOS the
classifier observes short answers passively, without the question.

== Features
Each clip becomes a 17-dimensional vector, computed on CPU in milliseconds:
- *Energy and pauses* (9 features). The waveform is framed at 25 ms with a 10 ms hop. Frames more
  than 25 dB below the clip's 90th-percentile energy are silent, and silent runs of at least 200 ms
  count as pauses. The features are duration, RMS mean and standard deviation, voiced energy range,
  voiced ratio, pause count, pause ratio, mean pause length and onsets per second.
- *Pitch and voice quality* (8 features, Praat via Parselmouth @jadoul2018parselmouth, 75–500 Hz).
  The features are $F_0$ mean and standard deviation, $F_0$ range in semitones
  ($12 log_2 (F_0 \/ 100)$, comparable across sexes), local jitter, local shimmer,
  harmonics-to-noise ratio, and intensity mean and standard deviation.

== Classifier
Features are standardised with statistics from the training fold only. A two-layer MLP follows:
Dropout → Linear(17, 32) → GELU → Dropout → Linear(32, 1). This gives *609 trainable parameters*,
trained with class-weighted binary cross-entropy. Its results are in @tab:main.

// ================================================================ 5
= Deep Architectures <sec:arch>

All four architectures share one template: an encoder per modality turns the clip into a sequence
or a vector, a fusion step combines the modalities, and a head outputs one deception logit.
@fig:overview contrasts them. They differ in what is pre-trained, what is trained, and where time is
modelled.

#figure(
  rect(width: 100%, height: 5cm, stroke: 0.5pt + gray)[#align(center + horizon)[#todo[four-panel
    overview figure: Temporal CNN · Cross-Modal Fusion · VideoMAE + HuBERT (LoRA) · MARLIN +
    ConvNeXt]]],
  caption: [The four architectures. Grey = frozen, colour = trained.],
) <fig:overview>

== Time-Batching Temporal CNN (deep baseline)
*Why.* It is the simplest learned model that still sees _time_: no pre-trained encoder, only
interpretable per-frame descriptors. It tells us how far hand-designed signals go once a network
can learn short temporal patterns (e.g. "the jaw tightens over 0.3 s").

*How.* Each frame is described by 53 features:
- facial geometry from MediaPipe FaceLandmarker, pooled over 15 facial regions;
- a subset of the eGeMAPS prosodic descriptors @eyben2016gemaps, aligned to the video frames.

Four 1-D convolutions (kernels 5, 3, 3, 3; 64 channels; batch norm; ReLU) slide over time. A masked
mean over the valid frames lets clips of different length share a batch, which is where the name
comes from. A linear head outputs the logit. During training only, an auxiliary head regresses the
41 MUMIN annotations (loss weight 0.3); it is unused at inference.

*Size.* 57,322 trainable parameters (54,657 without the auxiliary head). #todo[confirm the hidden
size used in the reported run.]

== Cross-Modal Fusion
*Why.* With ≈1,100 training clips per fold, fine-tuning any large encoder risks overfitting. This
architecture takes PECL's parameter-efficiency idea to its limit: both encoders stay *completely
frozen*, their embeddings are computed once, and only a small fusion head is trained.

*How.*
- Video: DINOv2-small embeds 16 uniformly sampled face crops (384-d `[CLS]` tokens).
- Audio: WavLM-base-plus, chosen for its paralinguistic pre-training, gives 64 temporal tokens of
  768 dimensions.
- Both streams are projected to $d = 128$, averaged over time, concatenated and classified by a
  small MLP (@sec:best).

*Size.* 164,737 trainable parameters, about 30 times fewer than PECL.

== VideoMAE + HuBERT with LoRA
*Why.* Frozen per-frame encoders cannot adapt to the domain and see no motion inside the encoder.
VideoMAE is a _video_ Transformer, pre-trained to reconstruct masked space-time tubes, so it models
facial motion natively. HuBERT learns speech units from raw audio. LoRA adapts both to deception
cues while keeping the pre-trained weights intact.

*How.*
- Video: VideoMAE-Base on 16 face crops of the speaker.
- Audio: HuBERT-Base on the raw waveform.
- LoRA with rank $r = 8$ on the attention query and value projections of each encoder: 297k
  trainable parameters per branch.
- Late fusion: the two branches' logits are standardised and combined as $s = w z_v + (1 - w) z_a$,
  with $w$ chosen on validation.

*Size.* 594,434 trainable parameters in total.

== MARLIN + ConvNeXt
*Why.* Deceptive cues are often local in time, for example a brief micro-expression. This
architecture pairs a _facial_ video encoder with a convolutional audio encoder and adds an explicit
sequence model, so the decision can rest on a few salient instants rather than on a clip-wide
average.

*How.*
- Video: MARLIN, a masked autoencoder pre-trained on facial video, encodes MTCNN face crops into
  $T_v = 8$ temporal steps of 768 dimensions.
- Audio: ConvNeXt-Tiny reads a 3-channel log-mel spectrogram (static, $Delta$, $Delta^2$). Pooling over
  frequency only keeps the time axis.
- Fusion: the audio is linearly interpolated onto the video time grid, both streams are projected to
  128 dimensions and concatenated.
- Sequence head: a 2-layer bidirectional GRU (or a dilated TCN) followed by temporal attention
  pooling.

*Size.* #todo[trainable parameters; frozen vs fine-tuned stages of the run reported in @tab:main.]

== Results <sec:results>

@tab:main compares all models on the official 3-fold protocol. @tab:unimodal isolates the
contribution of each modality.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.hline(),
    table.header([Model], [ACC], [F1], [AUC], [Trainable], [Train time]),
    table.hline(stroke: 0.5pt),
    [Majority class (whole corpus)], [53.7], [—], [—], [—], [—],
    [PECL @guo2023dolos (reported)], [64.75], [71.20], [62.71], [5.06M], [—],
    [Theory-driven baseline (3 seeds)], [55.1 ± 2.1], [#tbd], [58.8], [609], [CPU],
    table.hline(stroke: 0.3pt),
    [Time-Batching Temporal CNN], [#tbd], [#tbd], [#tbd], [57,322], [#tbd],
    [*Cross-Modal Fusion* (3 seeds)], [*72.2 ± 3.0*], [*73.1*], [*80.4*], [164,737], [1.3 min],
    [VideoMAE + HuBERT, LoRA (seed 0)], [66.3 ± 1.5], [65.9], [73.1], [594,434], [54.3 min],
    [MARLIN + ConvNeXt], [#tbd], [#tbd], [#tbd], [#tbd], [#tbd],
    table.hline(),
  ),
  caption: [Official 3-fold protocol, mean ± standard deviation over the test folds. Training time
    is summed over the three folds (T4 GPU) and excludes feature extraction.],
) <tab:main>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Architecture], [Audio only], [Video only], [Fused]),
    table.hline(stroke: 0.5pt),
    [Time-Batching Temporal CNN], [#tbd], [#tbd], [#tbd],
    [Cross-Modal Fusion], [58.8 / 62.5], [73.2 / 80.2], [72.2 / 80.4],
    [VideoMAE + HuBERT, LoRA], [62.8 / 67.0], [65.4 / 71.1], [66.3 / 73.1],
    [MARLIN + ConvNeXt], [#tbd], [#tbd], [#tbd],
    table.hline(),
  ),
  caption: [Unimodal ablations, ACC / AUC on the official protocol.],
) <tab:unimodal>

#todo[One paragraph of reading. Draft: every architecture beats the majority floor and the
theory-driven baseline. The two architectures with a strong visual encoder are the best. In both, the
video branch alone matches or beats the fused model, so the gain from fusion is small.]

// ================================================================ 6
= Best Model: Cross-Modal Fusion <sec:best>

== Architecture
Let $V in RR^(16 times 384)$ be the DINOv2 embeddings of 16 face crops and $A in RR^(64 times 768)$
the WavLM tokens. The late-fusion head computes
$ v = W_v dot.c "mean"_t (V), quad a = W_a dot.c "mean"_t (A), quad
  hat(y) = "MLP"("LayerNorm"([v; a])) $
with $W_v in RR^(128 times 384)$ and $W_a in RR^(128 times 768)$. The MLP is
Dropout(0.3) → Linear(256, 64) → GELU → Dropout(0.3) → Linear(64, 1).

Training uses AdamW (learning rate $10^(-3)$, weight decay 0.05), batch size 16 and 30 epochs. The
loss is binary cross-entropy weighted by the training-fold prevalence. The decision threshold is 0.5.
Because the encoders are frozen, one fold trains in under a minute.

#figure(
  rect(width: 100%, height: 4cm, stroke: 0.5pt + gray)[#align(center + horizon)[#todo[detailed
    diagram of the fusion head]]],
  caption: [Cross-Modal Fusion. Only the projections and the MLP are trained.],
) <fig:fusion>

== Ablations
#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.hline(),
    table.header([Variant], [ACC], [F1], [AUC], [Trainable]),
    table.hline(stroke: 0.5pt),
    [Late fusion (mean pooling)], [72.2 ± 3.0], [73.1], [80.4], [164,737],
    [Video only (DINOv2)], [73.2 ± 1.7], [74.8], [80.2], [57,857],
    [Audio only (WavLM)], [58.8 ± 2.0], [61.6], [62.5], [107,009],
    [Attention pooling], [#tbd], [#tbd], [#tbd], [#tbd],
    [Frame-difference features (delta)], [#tbd], [#tbd], [#tbd], [#tbd],
    [Congruence only ($1 - cos(v, a)$)], [#tbd], [#tbd], [#tbd], [147,714],
    table.hline(),
  ),
  caption: [Cross-Modal Fusion variants on the official protocol (3 seeds).],
) <tab:ablation>

The visual branch alone reaches the performance of the fused model (73.2 vs 72.2 ACC, 80.2 vs 80.4
AUC); audio alone is 14 points lower.

#todo[complete after the attention, delta and congruence runs
on the official folds.]

== Comparison with PECL
The model outperforms PECL by 7.5 ACC and 17.7 AUC points with 30 times fewer trainable parameters.
The two designs differ on three points:
- *Frozen encoders vs adapters.* PECL trains 5.06M parameters inside the encoders; we train only
  165k in the head.
- *Encoders.* We use DINOv2 and WavLM (last layer) instead of an ImageNet ViT and wav2vec 2.0 (first
  four layers). DINOv2 in particular is a much stronger visual descriptor.
- *Temporal modelling.* PECL aligns 64 frames with 64 audio tokens and fuses them layer by layer; we
  pool 16 frames and 64 tokens over time and fuse once.

PECL's multi-task supervision with MUMIN labels is not used: annotations are never part of our
models.

== Analysis
#todo[Per-gender accuracy (male clips are 79% of the data). Error analysis: which clips fail, and
does performance depend on clip length? Why does the visual channel dominate? Connect to
@sec:discussion.]

// ================================================================ 7
= Discussion and Limitations <sec:discussion>

*What the visual stream recognises.* All the evidence points to the face. The best model gets its
accuracy from the video branch, and the frozen DINOv2 encoder was trained for generic visual
similarity, not for expressions. On the official folds, test clips share speakers and recording
sessions with training: 95–99% of the test clips in each fold have their speaker in the training
set. Part of the accuracy may therefore reflect recognising the person and the round rather than
the lie. Appendix A tests this directly with a subject-independent split.

*Small and imbalanced data.* About 1,100 training clips per fold, 79% male clips, and two recurring
team captains who account for 28% of the corpus. Standard deviations across folds and seeds reach
several points, so differences of 1–2 points are not meaningful.

*Clip coverage.* Not every original clip could still be downloaded, and architectures with cached
features run on slightly different subsets. The comparison with the paper is therefore approximate.
#todo[exact count per architecture.]

*External validity.* A game show is a low-stakes, performed setting, and its contestants are
comedians. Detecting deception in court or at a border is a different problem.

*Ethics.* A deception detector at 60–70% accuracy wrongly accuses many truthful people. These models
are research tools and must not be used to make decisions about individuals. The DOLOS annotations
are licensed for academic use only, and the footage is not ours to redistribute.

// ================================================================ 8
= Conclusion

#todo[Three take-aways. Draft: (1) frozen self-supervised encoders with a 165k-parameter head
outperform the published PECL baseline on DOLOS; (2) the visual channel carries almost all of the
performance, and multimodal fusion adds little; (3) a 609-parameter theory-driven baseline is
weaker but stable, and sets a floor that every model beats. Future work: subject-independent
evaluation as standard (Appendix A), and models of facial dynamics at higher frame rates.]

#bibliography("refs.bib", style: "ieee")

// ================================================================ appendix
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

= Subject-Independent Evaluation <app:subject>

== Motivation: speakers on both sides of the split
The official folds are drawn at random over _clips_, so the same person, often within the same
round, appears in both training and test. @tab:leak measures this on our copy of the protocol.
The 1,678 clips also form only 151 (episode, label) groups, with a median of 10 clips per group.
All clips of a group share the speaker, clothing, set and label.

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
  otherwise they would make up ≈70% of a test fold.
+ 27 subjects (182 clips) are held out as a test set that is opened only once, at the end.
+ The remaining 151 subjects (1,021 clips, 47.7% deception) are split into 5 folds. The assignment
  is optimised so that every fold has 47.3–48.0% deception and the same share of female clips.
+ Inside each fold, 15% of the training subjects form a validation set, used for early stopping and
  for choosing the threshold.

The script is `lie-detector/scripts/make_splits.py`; same annotations and same seed give
byte-identical files.

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
    [Time-Batching Temporal CNN], [#tbd], [#tbd], [#tbd], [#tbd],
    [MARLIN + ConvNeXt], [#tbd], [#tbd], [#tbd], [#tbd],
    [Majority (working set)], [—], [52.4 / —], [—], [—],
    table.hline(),
  ),
  caption: [The same models on the official and the subject-independent protocol. Cross-Modal
    Fusion and the baseline: 3 seeds; VideoMAE + HuBERT: seed 0.],
) <tab:protocol>

On unseen speakers, every model falls to within a few points of the majority floor. The drop is
concentrated in the visual channel: DINOv2 alone loses 23 ACC points and falls to chance, whereas
WavLM and the 17 acoustic features lose about 4. The acoustic representations carry no face,
clothing or set, so they have little identity to exploit. The DOLOS paper's own cross-dataset
results (57.54% on Bag-of-Lies, 56.17% on Real-Life Trials, both with unseen subjects) fall in the
same range as our subject-independent estimates.

#todo[Virgin test set: report it once, for the final configuration, or state that it was not opened.]

== MUMIN diagnostics
Linear probes on the frozen embeddings recover the 41 MUMIN annotations better than chance: mean
AUC 0.665 against a 0.567 noise band for DINOv2, and 0.585 against 0.539 for WavLM. The encoders
therefore _see_ the annotated behaviours. Yet a classifier trained on the 41 ground-truth
annotations reaches only 51.1% ACC / 52.5% AUC on the subject split, against 58.84% in the paper on
the official folds. The behaviours exist; they do not predict deception on new people.

#todo[optional: A.5 purity- and episode-aware folds (`lie-detector/docs/splits-v2.md`).]

= Implementation Details <app:impl>

#todo[Hyperparameter table per architecture (optimiser, learning rate, epochs, batch size,
augmentation), hardware (Colab / Kaggle T4), software versions, seeds.]

= Author Contributions

#todo[One line per author.]
