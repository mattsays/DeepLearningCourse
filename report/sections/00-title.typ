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
  Automated deception detection asks a model to tell a truthful statement from a lie using only the
  audio and video of a short clip, without manual behavioural annotations. We study it on DOLOS, a
  1,675-clip corpus from the game show _Would I Lie To You?_, and compare four deep architectures
  with a 609-parameter acoustic baseline built on cognitive-load and vocal-arousal theory, under the
  official 3-fold protocol of the dataset paper. The architectures span the main design choices of
  the field: a temporal CNN trained from scratch on facial landmarks and acoustic descriptors, a
  cross-modal fusion head on frozen DINOv2 and WavLM embeddings, VideoMAE and HuBERT adapted with
  LoRA, and a temporal convolutional head on frozen MARLIN and ConvNeXt features. The cross-modal
  fusion model scores highest, with 72.2% accuracy and 80.4% AUC while training only 165k
  parameters; this is 7.4 accuracy points above the PECL result reported by the dataset paper,
  under a related but non-identical evaluation. LoRA adaptation reaches 66.3% / 73.1%,
  MARLIN + ConvNeXt 58.9% / 58.9%, the temporal CNN 56.3% / 59.1% and the acoustic baseline
  56.5% / 59.2%. Unimodal ablations show that the visual stream carries most of the performance and
  that late fusion adds little over the best single modality.
]
