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
  `truth` from `deception` in short video clips (2–19 s) using only audio-visual signals computable
  at inference time, without manual behavioural annotations. We study this problem on DOLOS, a
  1,675-clip corpus recorded from the television game show _Would I Lie To You?_, and compare four
  deep paradigms against a 609-parameter acoustic baseline grounded in Cognitive Load and Vocal
  Arousal theories (17 hand-crafted features) under the official 3-fold clip-level protocol of the
  dataset paper. The four paradigms span the main design choices of the field: from-scratch
  temporal CNN and graph attention (`DualHeadGAT`) models on facial-landmark and acoustic
  descriptors, a cross-modal fusion head on frozen DINOv2 and WavLM embeddings, VideoMAE and HuBERT
  adapted with LoRA, and sequential modelling on frozen Swin3D/MARLIN and ConvNeXt features. The
  cross-modal fusion model achieves the strongest in-protocol performance, reaching 72.2% accuracy
  and 80.4% AUC while training only 165k parameters (~30× fewer than the published PECL baseline,
  which it outperforms by 7.5 accuracy points), while LoRA adaptation reaches 66.3% accuracy and
  73.1% AUC. Unimodal ablations show that the visual stream carries almost all of this performance
  on the official split. In Appendix A, we re-evaluate the models on a strict subject-independent
  5-fold split where no speaker crosses training, validation, and test boundaries: accuracy drops
  to ~56%, driven by a collapse in the visual channel while the acoustic stream retains modest
  above-chance signal, demonstrating that much of the clip-level performance reflects speaker and
  round recognition rather than generalisable deception cues.
]
