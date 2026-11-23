#import "../template.typ": *

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
