// Multimodal Deception Detection on DOLOS — course report.
// Compile: typst compile report/report.typ   (or paste into typst.app together with refs.bib)
// Every red TODO box marks a number or a passage that is not measured yet. Do not fill a TODO
// with a remembered number: take it from the run's results file.
//
// Sources of the numbers (paths relative to lie-detector/notebooks/architectures/):
//   Cross-Modal Fusion   cross-modal-fusion/results/{late,video,audio}-{official,subject}/
//   VideoMAE + HuBERT    lore/results/fusion-video-lora-r8-official+audio-lora-r8-official/
//                        lore/results/fusion-video-lora-r8+audio-lora-r8/  and lore/docs/report.md
//   Theory baseline      load/train_load.ipynb (cell outputs) and load/README.md
//   Temporal CNN         teo/runs/cnn_baseline/best.pt (validation metadata only)
//   MARLIN + ConvNeXt    fab/train_marlin_kaggle.ipynb (code only, no outputs committed)

// ---------------------------------------------------------------- helpers
#let todo(body) = box(
  fill: rgb("#fde8e8"),
  stroke: 0.5pt + rgb("#c0392b"),
  inset: (x: 3pt, y: 1.5pt),
  radius: 2pt,
  text(fill: rgb("#c0392b"), size: 0.85em)[*TODO:* #body],
)
#let notrun = text(fill: luma(110), style: "italic")[not run]

// diagram blocks: frozen = grey, trained = blue, parameter-free = white
#let frozen-fill = luma(225)
#let train-fill = rgb("#d6e6f7")
#let blk(body, fill: white) = box(
  fill: fill,
  stroke: 0.5pt + luma(90),
  inset: (x: 4pt, y: 3pt),
  radius: 2pt,
  text(size: 7.5pt, body),
)
#let arr = text(size: 8pt, fill: luma(80))[→]

// ---------------------------------------------------------------- document setup
#set document(title: "Multimodal Deception Detection on the DOLOS Dataset")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
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
    columns: 3,
    column-gutter: 2.5em,
    row-gutter: 0.6em,
    [Sicoli Mattia \ #text(size: 8.5pt)[3126999071]],
    [Scuratti Matteo \ #text(size: 8.5pt)[3126999078]],
    [Tondi Gabriele \ #text(size: 8.5pt)[3126999073]],
    [Moliterno Fabio \ #text(size: 8.5pt)[3126999076]],
    [Mariotti Lorenzo \ #text(size: 8.5pt)[3126999231]],
    [Abdallah Yehia \ #text(size: 8.5pt)[3126999085]],
  )
  #v(0.3em)
  #text(size: 9.5pt)[Big Data and Deep Learning]
]

#v(1em)

#block(inset: (x: 2em))[
  #text(weight: "bold")[Abstract.]
  Automated deception detection asks a model to tell a lie from a truthful statement using only the
  face and voice of the speaker. We study it on DOLOS, a 1,675-clip audio-visual corpus recorded
  from a television game show, and compare four deep architectures against a 609-parameter acoustic
  baseline derived from the psychology of deception, under the official 3-fold protocol of the
  dataset paper. The architectures span the main design choices of the field: a temporal CNN
  trained from scratch on facial-landmark and acoustic descriptors, a cross-modal fusion head on
  frozen DINOv2 and WavLM embeddings, VideoMAE and HuBERT adapted with LoRA, and a temporal
  convolutional model on frozen MARLIN and ConvNeXt features. The cross-modal fusion model is the
  best: it reaches 72.2% accuracy and 80.4% AUC while training only 165k parameters, 7.5 accuracy
  points above the published PECL result, and LoRA adaptation reaches 66.3% / 73.1%. Unimodal
  ablations show that the visual stream carries almost all of this performance. In an appendix we
  re-evaluate the same models on a split where no speaker appears in both training and test:
  accuracy then falls to about 56%, which indicates that much of the in-dataset performance comes
  from recognising speakers rather than lies.
]

// ================================================================ 1
= Introduction

Deception has no single physical signature. Unlike speech or action recognition, where the target
is explicit, a lie may leave only fleeting traces: a hesitation, a shift in pitch, a facial
expression that does not match the words. Human observers are barely above chance at the task
@depaulo2003cues, and the cues that do exist are spread across channels and are individually weak.
Both facts motivate a multimodal, data-driven approach, and both make it hard: the signal is small,
and the datasets that contain it are small too.

We treat deception detection as binary classification. Given a raw video clip of 2–19 s in which a
person answers a question, the model predicts `truth` (0) or `deception` (1). It may use only
signals that can be computed from the video and audio at inference time. Human behavioural
annotations, such as the MUMIN codes shipped with the dataset, are never an input; at most they are
used as an auxiliary training target.

The question we ask is practical: given a dataset of about 1,700 clips, _which way of using
pre-trained knowledge works best_? We compare four answers that cover the spectrum, from no
pre-training at all to large self-supervised encoders that are adapted to the task, and we anchor
them to a baseline that uses no learned representation but only theory-motivated acoustic
statistics.

Our contributions are:
- a controlled comparison of four audio-visual architectures, from a temporal CNN trained from
  scratch to self-supervised video and speech Transformers adapted with LoRA, together with a
  609-parameter baseline built from psychological theory, all on the official DOLOS protocol;
- a cross-modal fusion model on frozen DINOv2 and WavLM that outperforms the published PECL result
  by 7.5 accuracy and 17.7 AUC points while training about 30 times fewer parameters;
- unimodal ablations and per-group analyses showing that the visual channel carries almost all the
  performance, together with evidence (Appendix A) that this performance depends on seeing the
  same speakers at training and test time.

@sec:related reviews prior work, @sec:data describes the dataset and protocol,
@sec:baseline the theory-driven baseline and @sec:arch the four architectures and their results.
@sec:best analyses the best model in depth, @sec:discussion discusses limitations and
@sec:conclusion concludes.

// ================================================================ 2
= Related Work <sec:related>

== Behavioural cues to deception
Psychology offers two accounts of why lies may be observable. The _cognitive load_ account
@vrij2008cognitive holds that building and maintaining a lie costs more mental effort than recalling
the truth: the liar must invent content, keep it consistent and monitor the listener, which slows
speech and adds pauses. The _arousal_ account links concealed emotion to changes in respiration and
vocal-fold tension, and so to pitch, voice quality and facial tension. A large meta-analysis finds
these cues real but weak, with small effect sizes and large variation between individuals
@depaulo2003cues.

Early automatic systems therefore relied on hand-coded behaviour. The MUMIN scheme
@allwood2007mumin annotates facial expressions, head movements, gaze and gestures, and multimodal
deception datasets such as Real-Life Trials @perezrosas2015trial, Bag-of-Lies @gupta2019bagoflies and
Box of Lies @soldner2019boxoflies use such annotations as features, often together with transcripts.
These annotations are expensive and cannot be produced at inference time, which is why our models
learn from raw video and audio instead.

== DOLOS and PECL
DOLOS @guo2023dolos is the largest of these in-the-wild datasets and introduces PECL
(Parameter-Efficient Crossmodal Learning) as its reference model. PECL keeps the first four layers
of an ImageNet ViT @dosovitskiy2021vit and of wav2vec 2.0 @baevski2020wav2vec, on the argument that
deception cues are low-level, and trains two kinds of modules:
- a _UT-Adapter_ beside each attention and MLP block: a 768→128 projection, a 1-D temporal
  convolution (kernel 3) and a 128→768 projection, which gives both encoders local temporal context;
- a _PAVF_ (plug-in audio-visual fusion) module between each pair of visual and audio layers, which
  learns a 64×64 correlation between 64 face frames and 64 audio tokens and cross-attends them in
  both directions.

In total PECL trains 5.06M of 71.08M parameters (7.12%). An optional multi-task variant also
predicts 25 MUMIN labels. @tab:pecl reports its results on the official folds. The paper also
cross-tests PECL on datasets with unseen subjects, where accuracy drops to 57.54% (Bag-of-Lies) and
56.17% (Real-Life Trials); we return to this in Appendix A.

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
DOLOS is too small to train or fully fine-tune large encoders, so every architecture we study except
one builds on self-supervised pre-training:
- *DINOv2* @oquab2024dinov2, a ViT trained by self-distillation on 142M curated images, whose
  `[CLS]` token is a strong general-purpose image descriptor;
- *WavLM* @chen2022wavlm and *HuBERT* @hsu2021hubert, speech Transformers trained by masked
  prediction of discrete units; WavLM adds denoising and speaker-mixture training, which makes it
  stronger on paralinguistic tasks such as speaker and emotion recognition;
- *VideoMAE* @tong2022videomae, a video Transformer trained to reconstruct 90% masked space-time
  tubelets, and *MARLIN* @cai2023marlin, the same idea applied to facial video with facial-region
  guided masking;
- *ConvNeXt* @liu2022convnext, a modernised CNN that we apply to log-mel spectrograms.

Two strategies keep the number of trained parameters small. The first keeps the encoder _frozen_,
pre-computes its embeddings once and trains only a head. The second, _low-rank adaptation_ (LoRA)
@hu2022lora, adds to a frozen weight matrix $W_0$ a trainable update $Delta W = B A$ of rank $r$,
so that the encoder itself adapts to the task with a fraction of a percent of its parameters.

// ================================================================ 3
= Dataset and Evaluation Protocol <sec:data>

== DOLOS
The clips come from the British game show _Would I Lie To You?_. In each round a contestant reads a
statement from a card, true or false and known only to them, and defends it under questioning from
the opposing team, who must guess. The truth is revealed at the end of the round, so every clip has
an objective _game ground truth_ rather than an annotator's opinion. The speakers are comedians and
celebrities, the setting is a studio, and the stakes are low.

The paper reports 1,675 clips (899 deception, 776 truth), 213 participants and 84 episodes, with a
median duration of 5 s. The annotation files we received differ slightly (@tab:stats). The clips
are not distributed as video: they are cut from YouTube episodes. Of the 1,678 annotated clips,
1,452 could still be downloaded with their media; all download failures come from 9 episodes that
are no longer available. Two properties of the corpus matter for the rest of the paper: it is
dominated by male speakers (79%), and two recurring team captains contribute 468 clips (28%).

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
    [Clips downloadable today], [—], [1,452],
    table.hline(),
  ),
  caption: [DOLOS statistics: as published, and as measured on the local annotation files
    (`lie-detector/dataset/README.md`).],
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
    [MARLIN + ConvNeXt], [16 face crops, 224²], [3-channel log-mel, 128 bands], [MTCNN],
    table.hline(),
  ),
  caption: [Inputs of each model. Frames are always sampled uniformly over the whole clip.],
) <tab:inputs>

== Official protocol
We follow the evaluation of the dataset paper and of earlier deception benchmarks: the three
train/test folds distributed with DOLOS. Each fold places about two thirds of the clips in training
and one third in test (1,099/549, 1,095/553 and 1,100/548 clips in the paper's files), and results
are averaged over the three test folds. Using the same folds keeps our numbers comparable with
@tab:pecl.

Because some clips can no longer be downloaded, and because each architecture needs its own
features, every model is scored on a slightly different subset of each test fold (@tab:coverage).
The differences are a few percent of the test set, but they mean that the comparison with the paper
is approximate.

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
    [Theory-driven baseline], [381], [—], [—], [1,132],
    [Cross-Modal Fusion], [429], [432], [413], [1,274],
    [VideoMAE + HuBERT, LoRA], [482], [483], [461], [1,426],
    table.hline(),
  ),
  caption: [Test clips actually scored on each official fold. Cross-Modal Fusion needs a cached
    embedding for every modality; the baseline reuses a feature cache built for a different clip
    list. Per-fold counts of the baseline after fold 1 were not logged.],
) <tab:coverage>

== Metrics and evaluation rules
We report the three metrics of the paper:
- *accuracy (ACC)*, the share of correct decisions;
- *F1* on the `deception` class, the harmonic mean of precision and recall for lies;
- *AUC*, the area under the ROC curve, which does not depend on the decision threshold and is
  therefore the most stable comparison on test folds of ≈450 clips.

We add *balanced accuracy* (the mean of the two per-class recalls) where available, and the
_majority baseline_ — always predicting the most frequent class on the same clips — as the floor
every model must beat: 52.4–53.3% depending on the subset.

The official folds have no validation set, and the models handle this in two ways (@tab:rules).
Cross-Modal Fusion and the theory-driven baseline train for a fixed number of epochs and classify at
the paper's operating point, threshold 0.5. VideoMAE + HuBERT and MARLIN + ConvNeXt carve a
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
    [VideoMAE + HuBERT, LoRA], [15% of train, clip-level], [best val AUC], [max. val balanced ACC],
    [MARLIN + ConvNeXt], [15% of train, clip-level], [best val loss], [max. val balanced ACC],
    table.hline(),
  ),
  caption: [How each model is trained and thresholded on the official folds.],
) <tab:rules>

// ================================================================ 4
= Theory-Driven Mathematical Baseline <sec:baseline>

Before introducing networks with millions of parameters, we set a transparent reference: a
classifier that sees only acoustic measurements chosen because psychology predicts they should
differ between lies and truths. It answers a simple question — how far does _theory alone_ go? — and
it gives every deep model a floor that is harder to beat than the majority class.

== Theory
The baseline encodes the two accounts of @sec:related as measurable quantities:
- _cognitive load_ appears in the timing of speech: longer and more frequent pauses, a lower
  speaking rate, hesitations;
- _arousal_ appears in the voice source: a raised and more variable pitch, more cycle-to-cycle
  instability (jitter, shimmer) and a less clear, breathier or tenser voice (lower
  harmonics-to-noise ratio).

One caveat: Vrij's paradigm works best when an interviewer raises the load on purpose, for example
with unexpected questions. In DOLOS the classifier observes short answers passively, without the
question, so the theory predicts at most a weak signal.

== Features
Each clip becomes a 17-dimensional vector, computed on CPU in milliseconds per clip.

*Energy and pauses (9 features).* The waveform is split into 25 ms frames with a 10 ms hop, and the
energy of each frame is expressed in dB. A frame is _voiced_ if its energy is within 25 dB of the
clip's 90th-percentile energy; the relative threshold makes the rule independent of recording level.
Silent runs of at least 200 ms count as _pauses_, so that the short silences of stop consonants are
ignored. From this envelope we compute the duration, the mean and standard deviation of the RMS
amplitude, the energy range of voiced frames (95th minus 5th percentile, in dB), the voiced ratio,
the number of pauses, the share of time spent in pauses, the mean pause length and the number of
voice onsets per second, a proxy for the syllable rate.

*Pitch and voice quality (8 features).* Using Praat through Parselmouth @jadoul2018parselmouth, the
fundamental frequency $F_0$ is tracked every 10 ms between 75 and 500 Hz. We keep the mean and
standard deviation of $F_0$ over voiced frames and its range in semitones,
$12 log_2 (F_0 \/ 100)$, which makes male and female voices comparable. From Praat's periodic point
process we compute local jitter (period instability) and local shimmer (amplitude instability) with
the standard settings; the harmonics-to-noise ratio is averaged over the clip; intensity is
summarised by its mean and standard deviation in dB. Clips with fewer than three voiced frames get
missing pitch features, which are imputed with the median.

== Classifier and training
The features are standardised with means and standard deviations computed on the training fold
only, then passed to a two-layer MLP:
$ hat(y) = W_2 dot.c "Dropout"("GELU"(W_1 dot.c "Dropout"(z) + b_1)) + b_2, quad
  W_1 in RR^(32 times 17), W_2 in RR^(1 times 32), $
with dropout 0.3. This gives *609 trainable parameters*. The model is trained with binary
cross-entropy weighted by the class ratio of the training fold, AdamW (learning rate $10^(-3)$,
weight decay $10^(-2)$), batch size 32 and gradient clipping at 1.0, for a fixed 60 epochs on the
official folds, with three seeds. One fold trains in about three seconds. Its results appear with the
others in @tab:main.

// ================================================================ 5
= Deep Architectures <sec:arch>

All four architectures share one template: an encoder per modality turns the clip into a sequence
or a vector, a fusion step combines the modalities, and a head outputs one deception logit. They
differ in three choices: _what is pre-trained_, _what is trained_, and _where time is modelled_
(@fig:overview).

#figure(
  {
    set align(left)
    set par(justify: false)
    grid(
      columns: (auto, 1fr),
      column-gutter: 8pt,
      row-gutter: 7pt,
      align: horizon,
      text(size: 8pt, weight: "bold")[Temporal CNN],
      [#blk[landmarks + LLDs, T×53] #arr #blk(fill: train-fill)[4× Conv1d, 64 ch] #arr #blk[masked mean] #arr #blk(fill: train-fill)[Linear → logit]],
      text(size: 8pt, weight: "bold")[Cross-Modal Fusion],
      [#blk(fill: frozen-fill)[DINOv2 ∥ WavLM] #arr #blk(fill: train-fill)[Linear → 128, ×2] #arr #blk[mean, concat] #arr #blk(fill: train-fill)[LN + MLP → logit]],
      text(size: 8pt, weight: "bold")[VideoMAE + HuBERT],
      [#blk(fill: frozen-fill)[VideoMAE ∥ HuBERT] #blk(fill: train-fill)[\+ LoRA r = 8] #arr #blk[mean] #arr #blk(fill: train-fill)[LN + Linear, ×2] #arr #blk[weighted z-sum]],
      text(size: 8pt, weight: "bold")[MARLIN + ConvNeXt],
      [#blk(fill: frozen-fill)[MARLIN ∥ ConvNeXt] #arr #blk(fill: train-fill)[LN + Linear, fuse] #arr #blk(fill: train-fill)[dilated TCN] #arr #blk[mean] #arr #blk(fill: train-fill)[MLP → logit]],
    )
  },
  caption: [The four architectures. Grey: frozen pre-trained encoder; blue: trained; white:
    parameter-free. "∥" separates the visual and the audio branch.],
) <fig:overview>

== Time-Batching Temporal CNN (deep baseline)
*Why.* It is the simplest learned model that still sees _time_. It uses no pre-trained encoder:
only interpretable per-frame descriptors of the face and voice, on which a small convolutional
network learns short temporal patterns — for example, "the mouth corners move down over 0.3 s while
loudness drops". It tells us how far hand-designed signals go once a network, rather than a fixed
statistic, decides how to combine them over time.

*Input.* Each clip is resampled to a 10 fps grid. For every frame, MediaPipe FaceLandmarker
@lugaresi2019mediapipe returns 478 three-dimensional face landmarks; we group them into 15 regions
(eyes, eyebrows, nose bridge and tip, outer and inner mouth, jaw, chin, cheeks, forehead, irises)
and average the $(x, y, z)$ coordinates of each region, which gives 45 values per frame. When no
face is found, the last valid vector is carried forward. The audio is described by the first eight
frame-level low-level descriptors of the eGeMAPS set @eyben2016gemaps, computed with openSMILE and
linearly interpolated onto the same 100 ms grid. The two streams are truncated to the shorter one and
concatenated into a $T times 53$ matrix, with no normalisation and no maximum length.

*Architecture.* Four 1-D convolutions slide along time: the first has kernel 5 and maps 53 to 64
channels, the next three have kernel 3 and keep 64 channels; each is followed by batch
normalisation and ReLU, and the first two by dropout 0.3. The receptive field is 11 frames, about
1.1 s. Clips have different lengths, so a batch is zero-padded to its longest clip and the network
receives the true length of each sequence. The features are then averaged over the _valid_ frames
only — a masked mean — which lets clips of any length share a batch without the padding diluting the
representation; we call this _time batching_. A linear layer maps the 64-dimensional summary to the
deception logit. An optional second head regresses the 41 MUMIN annotations from the same summary;
it is used only as an auxiliary training loss and is discarded at inference.

*Training.* Binary cross-entropy with positive-class weighting, plus the optional MUMIN
mean-squared-error term; AdamW (learning rate $3 dot 10^(-4)$, weight decay $10^(-4)$), batch size 16,
gradient clipping at 5.0, learning-rate halving on validation-loss plateaus, and early stopping on
validation loss.

*Size.* 57,322 trainable parameters, of which 54,657 are used at inference (the rest is the MUMIN
head).

== Cross-Modal Fusion
*Why.* With about 850 training clips per fold, fine-tuning a large encoder end to end overfits
quickly. This architecture takes the parameter-efficiency idea of PECL to its limit: both encoders
stay *completely frozen*, their embeddings are computed once and cached, and only a small fusion
head is trained. Training then takes seconds, which makes it possible to run several seeds and many
variants, and it isolates the question of _how much the pre-trained representations already
contain_.

*Visual stream.* Sixteen frames are sampled uniformly over the clip. On each frame MTCNN
@zhang2016mtcnn detects the face (minimum size 40 px, detections below confidence 0.90 rejected); the
box is enlarged to a square of 1.5 times its longer side, clamped to the frame, and resized to
224×224. If no face is found the centre square of the frame is used. Each crop goes through
DINOv2-small, and the `[CLS]` token of the last layer is kept, giving a $16 times 384$ matrix per
clip.

*Audio stream.* The whole waveform goes through WavLM-base-plus in a single pass. Its last hidden
state has about 50 vectors per second; adaptive average pooling over contiguous windows reduces it to
64 tokens of dimension 768, the same token count as PECL. Both caches are stored in half precision,
and a cache built with different settings is refused at load time.

*Fusion head.* Each stream is projected to $d = 128$ by a linear layer applied to every token,
averaged over time, and the two 128-dimensional vectors are concatenated. A LayerNorm, dropout 0.3,
a 256→64 linear layer with GELU, a second dropout and a 64→1 layer produce the logit. @sec:best gives
the exact formulation and the training recipe.

*Size.* 164,737 trainable parameters, about 30 times fewer than PECL. The two input projections
account for 147,712 of them.

== VideoMAE + HuBERT with LoRA
*Why.* Frozen image encoders see each frame in isolation and cannot adapt to the domain. VideoMAE is
a _video_ Transformer: it embeds 2×16×16 space-time tubelets, so motion is part of its input, and
its masked-reconstruction pre-training makes it data-efficient. HuBERT plays the same role for raw
speech. LoRA lets both adapt their attention to deception cues while changing only 0.3% of their
parameters, a middle ground between freezing and fine-tuning.

*Face pipeline.* Thirty-two frames are sampled uniformly over the clip. Faces are detected with the
YuNet detector and described with SFace identity embeddings; faces are clustered by identity, and
the speaker is the identity visible in the most frames, which matters because 753 of the 1,452 clips
show more than one person. Frames where the speaker is not visible borrow the nearest frame where
they are. Box centres and sizes are median-filtered over five frames within each camera shot, and a
square crop of 1.6 times the face size is resized to 224×224. A face was found in 99.9% of frames
and the speaker in 96.0%; no clip fell back to a centre crop. At training time the model sees 16 of
the 32 frames — the even or the odd ones at random — with a random resized crop (scale 0.8–1.0) and
horizontal flip shared across frames; at test time it sees the even frames without augmentation.

*Audio pipeline.* Training uses a random 8 s window of the 16 kHz waveform and evaluation the first
10 s; each crop is z-normalised. Batches group clips of similar length to limit padding.

*Encoders and adaptation.* VideoMAE-Base (fine-tuned on Kinetics-400) turns 16 frames into
8×14×14 = 1,568 tokens, which are averaged and normalised by the checkpoint's own LayerNorm.
HuBERT-Base (LibriSpeech 960 h) produces about 50 vectors per second, which are averaged over the
real frames only. LoRA with rank $r = 8$, scaling $alpha = 16$ and dropout 0.1 is applied to the
query and value projections of all 12 attention layers of each encoder: 24 adapter matrices and
294,912 parameters per branch. Each branch ends with dropout 0.3 and a 768→1 linear head. Before
every fold a check verifies that every adapter receives a non-zero gradient and actually changes the
output.

*Late fusion.* The two branches are trained separately. Their logits are standardised with the
mean and standard deviation of their own validation logits, giving $z_v$ and $z_a$, and combined as
$ s = w z_v + (1 - w) z_a, quad w in {0, 0.05, ..., 1}, $
with $w$ chosen by validation AUC. The decision threshold is then chosen on the fused validation
scores. The chosen weights on the official folds were 0.50, 0.80 and 0.45: the video branch
contributes at least half.

*Training.* Binary cross-entropy with positive-class weighting; AdamW with learning rate
$2 dot 10^(-4)$ for the adapters and $10^(-3)$ for the head, weight decay 0.01; one epoch of linear
warm-up followed by a cosine decay; effective batch 16 (4×4 accumulated for video, 8×2 for audio);
at most 10 epochs for video and 15 for audio, with early stopping on validation AUC. Training uses
fp16 mixed precision, and gradient checkpointing keeps the video branch under 0.8 GB of GPU memory.

*Size.* 297,217 trainable parameters per branch (adapters, final LayerNorm and head), 594,434 in
total — 0.34% of VideoMAE and 0.31% of HuBERT.

== MARLIN + ConvNeXt
*Why.* Deceptive cues are often local in time, such as a brief micro-expression or a vocal break.
This architecture pairs a video encoder pre-trained specifically on _faces_ with a convolutional
audio encoder, keeps a time axis in both, and adds an explicit sequence model, so that the decision
can rest on a few salient instants rather than on a clip-wide average.

*Visual stream.* Sixteen frames are sampled uniformly and the face is cropped with MTCNN (detections
below confidence 0.85 rejected, square of 1.5 times the face size, 224×224). The frozen MARLIN
ViT-Base encoder, pre-trained on YouTube Faces, turns the 16 frames into 1,568 tubelet tokens;
because each tubelet spans two frames, they form an 8×14×14 grid, and averaging over the spatial grid
gives a sequence of 8 vectors of dimension 768.

*Audio stream.* A 128-band mel spectrogram (1,024-sample window, 512-sample hop at 16 kHz) is
converted to dB and stacked with its first and second temporal derivatives into a three-channel
image, so that an ImageNet-pretrained CNN receives level, velocity and acceleration of the spectrum
in its three input channels. A frozen ConvNeXt-Tiny encodes it; averaging over frequency only keeps
about one 768-dimensional vector per second, which is linearly interpolated onto the 8 video steps.

*Sequence head.* Each stream passes through LayerNorm, a 768→32 linear layer, GELU and dropout; the
two are concatenated and mixed by a 64→64 linear layer. A temporal convolutional network of three
causal layers (kernel 3, dilations 1, 2 and 4, so that the last step sees the whole sequence) with a
residual connection models the sequence, and the result is averaged over time and classified by
LayerNorm, dropout 0.3 and a 64→64→1 MLP. The code also supports a two-layer bidirectional GRU and
additive attention pooling in place of the TCN and the mean; the official configuration uses the TCN
with mean pooling.

*Training.* Binary cross-entropy with positive-class weighting; AdamW (learning rate
$3 dot 10^(-4)$, weight decay 0.01) with cosine annealing over at most 35 epochs, batch size 16,
gradient clipping at 1.0, early stopping on validation loss with patience 7, threshold chosen on
validation.

*Size.* Both encoders are frozen; the head has 97,857 trainable parameters (52,288 in the two input
projections, 37,056 in the TCN).

== Results <sec:results>

@tab:main compares all models on the official 3-fold protocol, @tab:folds breaks the scores down by
fold, and @tab:unimodal isolates the contribution of each modality.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([Model], [ACC], [F1], [AUC], [Bal. ACC], [Trainable], [Train time]),
    table.hline(stroke: 0.5pt),
    [Majority class (same clips)], [52.4–53.3], [—], [—], [50.0], [—], [—],
    [PECL @guo2023dolos (reported)], [64.75], [71.20], [62.71], [—], [5.06M], [—],
    [Theory-driven baseline], [55.1 ± 2.2], [58.0], [58.8], [55.0], [609], [≈10 s],
    table.hline(stroke: 0.3pt),
    [Temporal CNN#super[†]], [62.9], [—], [64.8], [—], [57,322], [CPU],
    [*Cross-Modal Fusion*], [*72.2 ± 3.0*], [*73.1*], [*80.4*], [*72.2*], [164,737], [≈25 s],
    [VideoMAE + HuBERT, LoRA], [66.3 ± 1.5], [65.9], [73.1], [66.6], [594,434], [54 min],
    [MARLIN + ConvNeXt], [#notrun], [#notrun], [#notrun], [#notrun], [97,857], [—],
    table.hline(),
  ),
  caption: [Official 3-fold protocol: mean ± standard deviation over test folds (and seeds, for
    the 3-seed models). Baseline and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT: seed 0. Training
    time is for one seed over the three folds on a T4 GPU, excluding feature extraction.
    #super[†]Not on the official folds: validation set (213 clips) of a single random 85/15
    clip-level split; included for orientation only.],
) <tab:main>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Fold], [Cross-Modal Fusion], [VideoMAE + HuBERT], [Theory-driven baseline]),
    table.hline(stroke: 0.5pt),
    [1], [69.0 / 77.1], [65.2 / 72.9], [52.3 / 55.0],
    [2], [73.4 / 81.3], [67.9 / 76.0], [56.9 / 61.3],
    [3], [74.2 / 82.9], [65.7 / 70.3], [56.0 / 60.0],
    table.hline(),
  ),
  caption: [ACC / AUC per official fold (mean over seeds where several were run).],
) <tab:folds>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Architecture], [Audio only], [Video only], [Fused]),
    table.hline(stroke: 0.5pt),
    [Cross-Modal Fusion (3 seeds)], [58.8 / 62.5], [*73.2 / 80.2*], [72.2 / 80.4],
    [VideoMAE + HuBERT, LoRA], [62.8 / 67.0], [65.4 / 71.1], [66.3 / 73.1],
    [Temporal CNN], [#notrun], [#notrun], [62.9 / 64.8#super[†]],
    [MARLIN + ConvNeXt], [#notrun], [#notrun], [#notrun],
    table.hline(),
  ),
  caption: [Unimodal ablations, ACC / AUC on the official protocol. #super[†]Random-split
    validation, see @tab:main.],
) <tab:unimodal>

*Reading.* Every measured model beats the majority floor, and every deep model beats the
theory-driven baseline by a wide margin: 17 accuracy and 22 AUC points for Cross-Modal Fusion. The
ranking follows the strength of the _visual_ representation. The two models built on pre-trained
visual Transformers are the best, and both outperform PECL; the frozen DINOv2 features are better
than the LoRA-adapted VideoMAE, although VideoMAE sees motion and is adapted to the task.

Fusion adds little. In Cross-Modal Fusion the video branch alone matches the fused model (73.2 vs
72.2 ACC, 80.2 vs 80.4 AUC), and the audio branch alone is 14 points lower. In VideoMAE + HuBERT the
fused model gains one accuracy point over video alone, and an untuned 50/50 fusion is slightly
better still (67.1 / 73.7). The audio branches are consistently weaker but not useless: WavLM and
HuBERT reach 59–63% accuracy, between the theory-driven baseline and the visual models.

The results are stable across folds. Cross-Modal Fusion is the best model on every fold, and its
spread across folds (69.0–74.2) is comparable to its spread across seeds (70.5–73.8, @tab:seeds).
Fold 1 is the hardest for all three models.

The Temporal CNN reaches 62.9% accuracy and 64.8% AUC, but on a validation set of a random clip-level
split, which is not comparable with the official test folds. MARLIN + ConvNeXt is fully implemented,
with feature extraction and a three-fold training loop, but its official run has not been completed.

#todo[MARLIN + ConvNeXt and Temporal CNN: run on the official folds and replace the "not run" and
#super[†] entries in @tab:main and @tab:unimodal.]

// ================================================================ 6
= Best Model: Cross-Modal Fusion <sec:best>

== Architecture
Let $V in RR^(16 times 384)$ be the DINOv2 `[CLS]` embeddings of the 16 face crops of a clip and
$A in RR^(64 times 768)$ its WavLM tokens. The model projects every token, averages over time,
and classifies the concatenation:
$ v = 1/16 sum_(t=1)^(16) (W_v V_t + b_v), quad a = 1/64 sum_(t=1)^(64) (W_a A_t + b_a), $
$ hat(y) = W_2 dot.c "Dropout"("GELU"(W_1 dot.c "Dropout"("LN"([v; a])) + b_1)) + b_2, $
with $W_v in RR^(128 times 384)$, $W_a in RR^(128 times 768)$, $W_1 in RR^(64 times 256)$,
$W_2 in RR^(1 times 64)$ and dropout 0.3. The probability of deception is $sigma(hat(y))$. The only
normalisation is the LayerNorm after the concatenation: features enter the projections as they come
out of the encoders. @fig:fusion shows the data flow, and @tab:params the parameter budget.

#figure(
  {
    set align(center)
    grid(
      columns: 9,
      column-gutter: 3pt,
      row-gutter: 6pt,
      align: horizon,
      blk(fill: frozen-fill)[DINOv2 \ 16 × 384], arr, blk(fill: train-fill)[Linear \ 384→128], arr, blk[mean over t],
      grid.cell(rowspan: 2, arr),
      grid.cell(rowspan: 2, blk[concat \ 256]),
      grid.cell(rowspan: 2, arr),
      grid.cell(rowspan: 2, blk(fill: train-fill)[LN → Dropout → Linear 256→64 \ → GELU → Dropout → Linear 64→1 \ → sigmoid]),
      blk(fill: frozen-fill)[WavLM \ 64 × 768], arr, blk(fill: train-fill)[Linear \ 768→128], arr, blk[mean over t],
    )
  },
  caption: [Cross-Modal Fusion. Grey: frozen, cached encoders; blue: trained; white:
    parameter-free.],
) <fig:fusion>

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.hline(),
    table.header([Component], [Parameters]),
    table.hline(stroke: 0.5pt),
    [Video projection (384→128)], [49,280],
    [Audio projection (768→128)], [98,432],
    [LayerNorm (256)], [512],
    [Linear 256→64], [16,448],
    [Linear 64→1], [65],
    table.hline(stroke: 0.3pt),
    [*Total trained*], [*164,737*],
    [Frozen encoders (DINOv2-small + WavLM-base-plus)], [≈116M],
    table.hline(),
  ),
  caption: [Parameter budget of Cross-Modal Fusion. 90% of the trained parameters are in the input
    projections.],
) <tab:params>

*Training.* AdamW with learning rate $10^(-3)$ and weight decay 0.05, batch size 16, gradient
clipping at 1.0, no learning-rate schedule and no augmentation. The loss is binary cross-entropy
with the positive class weighted by the negative/positive ratio of the training fold (0.88–0.94 on
the official folds). On the official folds the model trains for a fixed 30 epochs, the final weights
are used, and the threshold is 0.5. Each fold takes about 8 s on a T4 GPU, because the encoders never
run during training. Three seeds (0, 1, 2) control initialisation and shuffling; reported means and
standard deviations are over the nine seed × fold runs.

== Ablations and stability
@tab:ablation compares the fused model with its two single-stream versions, which replace the
concatenation with one projected stream and a 128→64→1 head.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.hline(),
    table.header([Variant], [ACC], [F1], [AUC], [Bal. ACC], [Trainable]),
    table.hline(stroke: 0.5pt),
    [Late fusion (video + audio)], [72.2 ± 3.0], [73.1], [80.4], [72.2], [164,737],
    [Video only (DINOv2)], [73.2 ± 1.7], [74.8], [80.2], [73.0], [57,857],
    [Audio only (WavLM)], [58.8 ± 2.0], [61.6], [62.5], [58.7], [107,009],
    table.hline(),
  ),
  caption: [Cross-Modal Fusion variants on the official protocol, 3 seeds × 3 folds.],
) <tab:ablation>

The visual branch alone matches the fused model; the difference (+1.0 ACC, −0.2 AUC) is well within
one standard deviation. With a third of the parameters, video-only is also the more stable of the two
(±1.7 vs ±3.0). Audio contributes a real but much weaker signal, and late fusion does not combine the
two any better than video alone.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Seed], [Late fusion], [Video only], [Audio only]),
    table.hline(stroke: 0.5pt),
    [0], [73.8 / 81.8], [72.8 / 79.5], [59.3 / 62.7],
    [1], [70.5 / 79.8], [73.7 / 80.8], [59.1 / 61.9],
    [2], [72.2 / 79.6], [73.0 / 80.5], [58.1 / 62.9],
    table.hline(),
  ),
  caption: [ACC / AUC per seed, averaged over the three official folds.],
) <tab:seeds>

Across seeds the AUC of the fused model stays between 79.6 and 81.8 (@tab:seeds), so the ranking of
the variants does not depend on initialisation. Summed over the nine runs, the fused model makes
1,313 true negatives, 505 false positives, 559 false negatives and 1,445 true positives: precision
74.1%, recall on lies 72.1% and specificity 72.2%. Errors are therefore balanced between the two
classes, as expected from a threshold of 0.5 combined with class weighting.

The code implements a wider design space (@tab:variants): attention pooling instead of the mean,
frame-difference ("delta") features that describe facial motion, a _congruence_ model that decides
only from the disagreement $1 - cos(v, a)$ between the two streams, and a transcript-based text
branch. These variants were explored only on the subject-independent split of Appendix A and were
not run on the official folds.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, right),
    table.hline(),
    table.header([Variant], [Change with respect to late fusion], [Trainable]),
    table.hline(stroke: 0.5pt),
    [Video / audio / late attention], [additive attention pooling over time instead of the mean], [66,178 / 115,330 / 173,058],
    [Video delta / late delta], [adds the mean of $|x_t - x_(t-1)|$ over frame embeddings], [115,585 / 222,465],
    [Congruence], [logit from $1 - cos(v, a)$ only], [147,714],
    [Text / audio + text], [Whisper transcript embedded by MiniLM], [57,857 / 164,737],
    table.hline(),
  ),
  caption: [Implemented Cross-Modal Fusion variants not evaluated on the official folds.],
) <tab:variants>

== Comparison with PECL
On the same official protocol the model outperforms PECL by 7.5 ACC and 17.7 AUC points, and the
PECL multi-task variant by 5.4 and 15.8, while training 30 times fewer parameters. The two designs
differ on three points:
- *Frozen encoders vs adapters.* PECL trains 5.06M parameters inside the first four layers of its
  encoders; we train only 165k in a head on top of frozen encoders.
- *Encoders.* We use the last layer of DINOv2 and WavLM instead of the first four layers of an
  ImageNet ViT and wav2vec 2.0. DINOv2 in particular is a much stronger visual descriptor.
- *Temporal modelling.* PECL aligns 64 frames with 64 audio tokens and fuses them layer by layer; we
  average 16 frames and 64 tokens over time and fuse once.

PECL's multi-task supervision with MUMIN labels is not used: annotations are never part of our
models. The comparison has two caveats. We score 1,274 test clips instead of 1,650, and the
reference implementation selects its reported epoch on the test fold#footnote[Re-scored in the same
way — best test epoch, threshold 0.5 — VideoMAE + HuBERT reaches 68.0 ± 1.8 ACC and 72.9 AUC. We do
not use this selection rule anywhere else.], which, if anything, favours PECL.

== Analysis
*By gender.* Female speakers are 20% of the test clips. Using the seed-averaged predictions of the
fused model (pooled ACC 73.4%, AUC 81.3%), accuracy is 70.3% (AUC 79.3) on the 256 clips of women
and 74.1% (AUC 81.9) on the 1,016 clips of men. VideoMAE + HuBERT shows the same gap (64.9% vs
66.6%), and so does the theory-driven baseline (50.9% vs 56.2%), which suggests that the cause is
the smaller amount of female training data rather than a specific architecture.

*By clip length.* Splitting the test clips into three equal groups by duration (2–4 s, 4–5 s,
5–48 s), the fused model reaches 71.2%, 71.7% and 77.1% accuracy (AUC 79.0, 80.4, 84.2). Longer
clips help the video branch (78.3% on the longest third) but show no consistent trend for audio
(55.9%, 63.4%, 58.5%). With a fixed budget of 16 frames, a longer clip spreads the frames over more
of the answer and over more camera shots, so the visual summary covers more of the scene.

*What the visual embedding contains.* Linear probes trained on the frozen, time-averaged embeddings
recover the hand-annotated MUMIN behaviours well above a prevalence-matched noise level: 24 of 26
frequent cues from DINOv2 (mean AUC 0.665 against 0.567 for noise) and 18 from WavLM (0.585 against
0.539). The best-read visual cues are static configurations — touching the head or body (0.759),
mouth corners down (0.756), facing the interlocutor (0.753) — while fast facial actions such as
eyebrow raising (0.549), nodding and frowning are barely readable at 16 frames per clip. The best
audio cues are vocalisations: filled pauses such as "ah"/"mmh" (0.818) and laughter (0.737). The
embeddings therefore describe who is in the frame and how they are positioned far better than how
their face moves, which is consistent with video dominating the score and with the discussion in
@sec:discussion.

// ================================================================ 7
= Discussion and Limitations <sec:discussion>

*What the visual stream recognises.* All the evidence points to the face. The best model gets its
accuracy from the video branch, and the frozen DINOv2 encoder was trained for generic visual
similarity, not for expressions; its probes read posture and position well but fast facial movement
poorly. On the official folds, test clips share speakers and recording sessions with training: in
each fold 95–99% of the test clips have their speaker in the training set, and every test clip comes
from an episode that is also in training. A clip of a given speaker in a given round shares the
label, clothing, lighting and studio with the other clips of that round. Part of the accuracy may
therefore reflect recognising the person and the round rather than the lie. Two observations support
this reading: VideoMAE + HuBERT ranks clips of the _same_ speaker only moderately well (within-speaker
AUC 0.63 for audio and 0.66 for video, against 0.67 and 0.71 over all test clips), and Appendix A shows that all models
lose most of their advantage when speakers are disjoint. The official protocol measures performance
on speakers the model has already seen, not generalisation to new people.

*Small and imbalanced data.* Each fold trains on 750–870 clips. 79% of the clips show men, and two
recurring team captains account for 28% of the corpus and lie in 69% of their clips. Standard
deviations across folds and seeds reach 2–3 points, so differences of 1–2 points between models are
not meaningful.

*Heterogeneous evaluation rules and coverage.* The models were scored on 1,132–1,426 of the 1,650
test clips of the paper (@tab:coverage), and two different stopping and threshold rules were used
(@tab:rules). Since AUC is threshold-free, the AUC ranking is the more reliable comparison. The
fixed-epoch rule gives Cross-Modal Fusion no tuning advantage, so its lead is unlikely to be an
artefact of the rule.

*Temporal resolution.* Sixteen frames per clip is about 3 frames per second at the median duration
of 5 s, 5.3 fps on the shortest clips and 1.8 fps on the longest, while facial micro-expressions last
40–600 ms. WavLM's 250–450 frames per clip are compressed four to seven times into 64 tokens. None
of the models can see the fastest facial cues, and our frozen model averages away the order of
frames entirely.

*External validity.* A game show is a low-stakes, performed setting, and its contestants are
practised comedians who lie for entertainment. Detecting deception in a courtroom, an interview or at
a border is a different problem, with different base rates and different consequences of errors.

*Ethics.* A deception detector at 70% accuracy wrongly flags almost three truthful people in ten.
These models are research tools: they must not be used to make decisions about individuals, and
their apparent accuracy on a benchmark does not transfer to real interrogations. The DOLOS
annotations are licensed for academic use only, and the footage is not ours to redistribute.

// ================================================================ 8
= Conclusion <sec:conclusion>

We compared four audio-visual architectures for deception detection on DOLOS, against a
theory-driven acoustic baseline, on the official 3-fold protocol. Three conclusions stand out.

First, _strong frozen representations beat adaptation on this dataset_. A 165k-parameter head on
frozen DINOv2 and WavLM embeddings reaches 72.2% accuracy and 80.4% AUC, above the published PECL
model and above VideoMAE and HuBERT adapted with LoRA (66.3% / 73.1%), at a training cost of
seconds rather than an hour.

Second, _the visual channel carries the performance and fusion adds little_. Video alone matches the
fused model in both architectures where we could measure it; audio alone is 4–14 points lower. The
psychologically motivated acoustic features reach only 55.1% accuracy, above chance but far below
any learned visual model.

Third, _benchmark accuracy is not yet evidence of lie detection_. The probes show that the visual
embeddings encode posture and identity far better than fast facial motion, and the subject-disjoint
evaluation of Appendix A brings every model back to 50–56% accuracy. We therefore recommend reporting
a speaker-independent split alongside the official one.

Future work follows from these limits: completing the official runs of MARLIN + ConvNeXt and the
Temporal CNN; denser frame sampling and motion-sensitive features to reach the timescale of
micro-expressions; true cross-modal interaction (cross-attention or joint self-attention) instead of
late fusion; and training objectives that discourage speaker identification, such as
speaker-adversarial losses or per-speaker normalisation.

#bibliography("refs.bib", style: "ieee")

// ================================================================ appendix
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1")

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

= Implementation Details <app:impl>

@tab:hparams lists the training settings of every model as found in the code. Feature extraction
and the Cross-Modal Fusion, baseline and MARLIN heads ran on Google Colab or Kaggle with an NVIDIA T4
GPU; VideoMAE + HuBERT ran on Kaggle with two T4 GPUs, one branch per GPU (torch 2.11,
transformers 5.23, peft 0.21.3); the Temporal CNN and the acoustic features run on CPU. All numbers in
this report come from the saved result files of each run, which record the dataset and split hashes.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.hline(),
    table.header([Model], [Optimiser, LR], [Weight decay], [Batch], [Epochs], [Other]),
    table.hline(stroke: 0.5pt),
    [Theory baseline], [AdamW, 1e-3], [0.01], [32], [60 (fixed)], [dropout 0.3, clip 1.0, 3 seeds],
    [Temporal CNN], [AdamW, 3e-4], [1e-4], [16], [early stop (val loss)], [plateau LR ×0.5, clip 5.0],
    [Cross-Modal Fusion], [AdamW, 1e-3], [0.05], [16], [30 (fixed)], [dropout 0.3, clip 1.0, 3 seeds],
    [VideoMAE + HuBERT], [AdamW, 2e-4 (LoRA), 1e-3 (head)], [0.01], [16 (accum.)], [≤ 10 video, ≤ 15 audio], [warm-up + cosine, fp16, r 8, α 16],
    [MARLIN + ConvNeXt], [AdamW, 3e-4], [0.01], [16], [≤ 35, patience 7], [cosine, clip 1.0, dropout 0.3],
    table.hline(),
  ),
  caption: [Training hyperparameters. All models use binary cross-entropy with the positive class
    weighted by the negative/positive ratio of the training data.],
) <tab:hparams>

= Author Contributions

#todo[One line per author: who built which architecture, the data pipeline, the splits and the
report.]
