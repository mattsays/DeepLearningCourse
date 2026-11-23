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
