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
  Automated deception detection asks a model to separate truth from deception in short video clips
  using only signals computable at inference time, without behavioural annotations. We study it on
  DOLOS (1,675 clips, median 5 s) and compare four audio-visual paradigms against a 609-parameter
  acoustic baseline: a from-scratch temporal CNN on facial landmarks and acoustic descriptors, a
  fusion head on frozen DINOv2 and WavLM embeddings, VideoMAE and HuBERT adapted with LoRA, and a
  convolutional head on frozen MARLIN and ConvNeXt features. On the official 3-fold protocol the
  frozen-embedding fusion head reaches 72.2% accuracy and 80.4% AUC, the best AUC of the four
  paradigms, but video alone comes within a point of it in every architecture we measured, so the
  second stream adds no measurable accuracy. On a speaker-independent 5-fold split every model falls
  back to 50–56% and the drop is concentrated in the visual channel: much of the 72% is speaker and
  round recognition, not a behavioural cue. We therefore recommend reporting a speaker-independent
  split beside the official one, and read the official numbers as a ceiling rather than a capability.
]
