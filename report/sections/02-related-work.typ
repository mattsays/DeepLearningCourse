#import "../template.typ": *

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
