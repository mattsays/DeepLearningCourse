#import "../template.typ": *

// ================================================================ 2
= Related Work <sec:related>

== Behavioural cues to deception
Psychology offers two accounts of why lies may be observable. The _cognitive load_ account
@vrij2008cognitive holds that building and maintaining a lie costs more mental effort than recalling
the truth: the liar must invent content, keep it consistent and monitor the listener, which slows
speech and adds pauses. Crucially, this effect is strongest when an interviewer actively _elicits_
load (for instance through unexpected questions); passive observation of short answers is far
weaker. The _arousal_ account links concealed emotion to changes in respiration and vocal-fold
tension, and so to pitch, voice quality and facial tension. Large meta-analyses find these passive
cues real but weak, with small effect sizes, large variation between individuals, and human
accuracy barely above chance @depaulo2003cues.

Early automatic systems therefore relied on hand-coded behaviour. The MUMIN scheme
@allwood2007mumin annotates facial expressions, head movements, gaze and gestures, and multimodal
deception datasets such as Real-Life Trials @perezrosas2015trial, Bag-of-Lies @gupta2019bagoflies and
Box of Lies @soldner2019boxoflies use such annotations as features, often together with transcripts.
These annotations are expensive and cannot be produced at inference time, which is why our models
learn from raw video and audio instead—using either low-level geometric and acoustic descriptors
(MediaPipe facial landmarks @lugaresi2019mediapipe, openSMILE eGeMAPS @eyben2016gemaps, and
Praat/Parselmouth phonetics @jadoul2018parselmouth) or pre-trained neural representations.

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
predicts 25 MUMIN labels. @tab:pecl reports its results on the official folds. Like earlier
deception corpora, the official DOLOS folds split randomly at the clip level, placing the same
speakers on both sides of the split; when the paper cross-tests PECL on external datasets with
unseen subjects, accuracy drops to 57.54% (Bag-of-Lies) and 56.17% (Real-Life Trials). We measure
this protocol effect directly in Appendix A.

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

== Self-supervised encoders, graph/temporal models, and adaptation
DOLOS is too small to train or fully fine-tune large backbones from scratch. Our study therefore
contrasts compact models trained from scratch—a 1-D temporal CNN and a Graph Attention Network
(`DualHeadGAT`, @velickovic2018gat) over sliding-window landmark and acoustic graphs—against
architectures built on self-supervised pre-training:
- *DINOv2* @oquab2024dinov2, a ViT trained by self-distillation on 142M curated images, whose
  `[CLS]` token is a strong general-purpose image descriptor;
- *WavLM* @chen2022wavlm and *HuBERT* @hsu2021hubert, speech Transformers trained by masked
  prediction of discrete units; WavLM adds denoising and speaker-mixture training, which makes it
  stronger on paralinguistic tasks such as speaker and emotion recognition;
- *VideoMAE* @tong2022videomae, a video Transformer trained to reconstruct 90% masked space-time
  tubelets, and *MARLIN* @cai2023marlin, the same idea applied to facial video with facial-region
  guided masking;
- *ConvNeXt* @liu2022convnext, a modernised CNN that we pair with 3-channel log-mel spectrograms
  and 3-D video convolutions (`Swin3D` / `X3D`) or MARLIN.

Two strategies keep the number of trained parameters small on large backbones. The first keeps the
encoder _frozen_, pre-computes its embeddings once and trains only a lightweight fusion head. The
second, _low-rank adaptation_ (LoRA) @hu2022lora, adds to a frozen weight matrix $W_0$ a trainable
update $Delta W = B A$ of rank $r$, so that the encoder itself adapts to the task with a fraction
of a percent of its parameters. @tab:adaptation contrasts these regimes with from-scratch training
and with PECL's internal adapters.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, left),
    table.hline(),
    table.header([Regime], [Trainable component], [Trainable params], [Main trade-off]),
    table.hline(stroke: 0.5pt),
    [Theory-driven], [MLP on 17 acoustic stats], [609], [No visual identity leak; limited to passive acoustics],
    [From scratch], [1-D CNN or GAT on landmarks + LLDs], [18k–57k], [Strong task inductive bias; no pre-trained priors],
    [Frozen + head], [Linear projections + fusion MLP], [58k–165k], [Fast, low overfitting; cannot adapt internal layers],
    [Adapters (PECL)], [UT-Adapter + PAVF inside 4 layers], [5.06M], [Mid-level cross-attention; higher capacity on ~1.7k clips],
    [LoRA], [Rank-$r$ Q/V updates + linear heads], [594k], [Adapts deep video/audio Transformers; 0.3% of backbone],
    table.hline(),
  ),
  caption: [Parameter-efficiency regimes compared in this study and in #cite(<guo2023dolos>, form: "prose").],
) <tab:adaptation>
