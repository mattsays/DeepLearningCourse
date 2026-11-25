#import "../template.typ": *

// ================================================================ 6
= Best Official-Split Model: Cross-Modal Fusion <sec:best>

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
(±1.7 vs ±3.0). Audio contributes a real but much weaker signal, and this late-fusion head does not
combine the two any better than video alone.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Seed], [Late fusion], [Video only], [Audio only]),
    table.hline(stroke: 0.5pt),
    [0], [73.8 / 81.8], [72.8 / 79.5], [59.3 / 62.7],
    [1], [70.5 / 79.8], [73.6 / 80.8], [59.1 / 61.9],
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
only from the disagreement $1 - cos(v, a)$ between the two streams. These variants were tried
during development on the subject-independent split of Appendix A, but those exploratory runs were
not kept as result files and none was run on the official folds, so we report no numbers for them.

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
    table.hline(),
  ),
  caption: [Implemented Cross-Modal Fusion variants not evaluated on the official folds.],
) <tab:variants>

== Comparison with PECL
On the official folds the model scores 7.4 ACC and 17.7 AUC points above the PECL result reported
by #cite(<guo2023dolos>, form: "prose"), and 5.3 and 15.8 above its multi-task variant, while
training 30 times fewer _trainable_ parameters. This is a higher reported score under a related but
non-identical evaluation, not a head-to-head result: we did not re-run PECL, and the two numbers are
computed on different sets of test clips (caveats below). The two designs differ on three points:
- *Frozen encoders vs adapters.* PECL trains 5.06M parameters inside the first four layers of its
  encoders; we train only 165k in a head on top of frozen encoders.
- *Encoders.* We use the last layer of DINOv2 and WavLM instead of the first four layers of an
  ImageNet ViT and wav2vec 2.0. DINOv2 in particular is a much stronger visual descriptor.
- *Temporal modelling.* PECL aligns 64 frames with 64 audio tokens and fuses them layer by layer; we
  average 16 frames and 64 tokens over time and fuse once.

PECL's multi-task supervision with MUMIN labels is not used: annotations are never part of our
models. The comparison has three caveats. First, we score 1,274 test clips instead of the 1,650
of the paper's test files, and the missing clips are not a random sample: they are the ones that
could not be downloaded or had no cached embedding. Second, each of our own models is scored on a
different subset (@tab:coverage), so even our internal ranking is not on identical clips. Third,
the reference implementation selects its reported epoch on the test fold#footnote[Re-scored in the
same way — best test epoch, threshold 0.5 — VideoMAE + HuBERT reaches 68.0 ± 1.8 ACC and 72.9 AUC.
We do not use this selection rule anywhere else.], which, if anything, favours PECL. A direct
comparison would require re-running PECL on exactly our clips.

== Analysis
*By gender.* Female speakers are 20% of the test clips. Using the seed-averaged predictions of the
fused model (pooled ACC 73.3%, AUC 81.3%), accuracy is 70.3% (AUC 79.3) on the 256 clips of women
and 74.1% (AUC 81.9) on the 1,016 clips of men. VideoMAE + HuBERT shows the same gap (64.9% vs
66.6%), so the gap is not specific to one architecture; the acoustic baseline, on the other hand,
shows none (57.6% on 290 female and 56.5% on 1,135 male clips). The smaller amount of female
training data is one plausible explanation for the gap of the two visual models, but our
experiments do not isolate it: female test clips also have a different base rate (46.9% deception
against 53.8% for male clips).

*By clip length.* Splitting the test clips into three equal groups by duration (2–4 s, 4–5 s,
5–48 s), the fused model reaches 71.2%, 71.7% and 77.1% accuracy (AUC 79.0, 80.4, 84.2). Longer
clips help the video branch (78.3% on the longest third) but show no consistent trend for audio
(55.9%, 63.4%, 58.5%). One possible reason is that, with a fixed budget of 16 frames, a longer clip
spreads the frames over more of the answer and over more camera shots, so the visual summary covers
more of the scene; we have not tested this directly.

*What the visual embedding contains.* Linear probes trained on the frozen, time-averaged embeddings
(seed 0, on the subject-independent folds of Appendix A)
recover the hand-annotated MUMIN behaviours well above a prevalence-matched noise level: 24 of 26
frequent cues from DINOv2 (mean AUC 0.665 against 0.567 for noise) and 18 from WavLM (0.585 against
0.539). The best-read visual cues are static configurations — touching the head or body (0.759),
mouth corners down (0.756), facing the interlocutor (0.753) — while fast facial actions such as
eyebrow raising (0.549), nodding and frowning are barely readable at 16 frames per clip. The best
audio cues are vocalisations: speech disturbances other than "ah"/"mmh" fillers (0.818) and laughter (0.737). The
visual embeddings therefore encode static configuration — posture, position, framing — far better
than facial motion. This is consistent with, though not proof of, the reading in @sec:discussion
that the video branch partly recognises the person and the scene.
