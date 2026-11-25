#import "../template.typ": *

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
  Automated deception detection is a binary classification task that asks a model to distinguish
  `truth` from `deception` in short video clips using only audio-visual signals
  computable at inference time, without manual behavioural annotations. We study this problem on
  DOLOS, a 1,675-clip corpus recorded from the television game show _Would I Lie To You?_, and
  compare four deep paradigms against a 609-parameter acoustic baseline grounded in Cognitive Load
  and Vocal Arousal theories (17 hand-crafted features) under the official 3-fold clip-level protocol
  of the dataset paper. The four paradigms span the main design choices of the field: a from-scratch
  temporal CNN on facial-landmark and acoustic descriptors, a cross-modal fusion head on frozen
  DINOv2 and WavLM embeddings, VideoMAE and HuBERT adapted with LoRA, and a temporal convolutional
  head on frozen MARLIN and ConvNeXt features. The cross-modal fusion model achieves the strongest
  in-protocol performance, reaching 72.2% accuracy and 80.4% AUC while training a 165k-parameter
  head on frozen encoders (PECL trains 5.06M). This is 7.4 accuracy points above the PECL result
  reported by the dataset paper, under a related but non-identical evaluation: we score a subset of
  the official test clips and do not re-run PECL. LoRA adaptation reaches 66.3% accuracy and 73.1%
  AUC, MARLIN + ConvNeXt 58.9% and 58.9%, and the from-scratch temporal CNN 56.3% and 59.1%.
  Unimodal ablations show that, on the official split, video alone comes within about a point of
  the fused model in every architecture where we measured it, while the pre-trained audio branches
  trail by 3.5–13.3 points. In Appendix A, we re-evaluate the models on a strict subject-independent 5-fold
  split where no speaker crosses training, validation, and test boundaries: the development
  estimates drop to 50–56% accuracy, driven by a collapse in the visual
  channel, while the acoustic stream keeps a modest signal whose AUC, unlike its accuracy, stays
  above chance under a speaker-cluster bootstrap. This indicates that much of
  the clip-level performance reflects speaker and round recognition rather than generalisable
  deception cues.
]
