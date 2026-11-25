#import "../template.typ": *

// ================================================================ 1
= Introduction

Automated deception detection sits at the intersection of multimodal machine learning and human
behavioural analysis, and it has no single physical signature. Unlike standard speech or action
recognition, where the target signal is explicit, a lie may leave only fleeting traces hidden in
the interplay between channels: a speaker can maintain a composed facial expression while betraying
slight tension, hesitation, or pitch instability in their voice. Human observers are barely above
chance at the task @bond2006accuracy, and the cues that do exist are spread across modalities and
are individually weak @depaulo2003cues. Both facts motivate a multimodal, data-driven approach, and both make it
hard: the signal is small, and the datasets that contain it are small too.

We treat deception detection as binary classification. Given a raw video clip of a few seconds
(median 5 s) in which a person answers a question, the model predicts `truth` (0) or `deception` (1). It may use only
signals that can be computed directly from the video and audio streams at inference time. Human
behavioural annotations, such as the MUMIN codes shipped with the dataset, are never an input.

The question we ask is practical: given a dataset of about 1,700 clips, _which way of using
pre-trained knowledge works best_? Rather than relying on a single model family, we design a
comparative study across four distinct architectural paradigms, a from-scratch temporal CNN on
facial landmarks and acoustic descriptors, a cross-modal fusion head on frozen DINOv2 and WavLM
embeddings, parameter-efficient LoRA adaptation of VideoMAE and HuBERT, and temporal convolutional
modelling over frozen MARLIN and ConvNeXt features, anchored to a 609-parameter baseline built
solely from theory-motivated acoustic statistics.

Our contributions are:
- a comparison of four audio-visual paradigms spanning the full pre-training spectrum, together
  with a 609-parameter baseline grounded in psychological theory, evaluated under the official
  3-fold DOLOS protocol used by prior work;
- a cross-modal fusion model on frozen DINOv2 and WavLM whose official-split score is 7.4 accuracy
  and 17.7 AUC points above the PECL result reported by the dataset paper, under a related but
  non-identical evaluation (a subset of the official test clips; PECL is not re-run), while
  training about 30 times fewer _trainable_ parameters (165k against 5.06M; the frozen encoders
  hold ≈116M more);
- unimodal ablations and per-group analyses (by gender and clip length) showing that the visual
  channel carries most of the official-split performance and that late fusion adds little over
  the best single modality.

@sec:related reviews prior work, @sec:data describes the dataset and protocol,
@sec:baseline the theory-driven baseline and @sec:arch the four architectures and their results.
@sec:best analyses the best model in depth, @sec:discussion discusses limitations and
@sec:conclusion concludes. Appendix A reports a complementary evaluation on an alternative split,
and Appendix B the implementation details.
