#import "../template.typ": *

// ================================================================ 1
= Introduction

Automated deception detection sits at the intersection of multimodal machine learning and human
behavioural analysis, and it has no single physical signature. Unlike standard speech or action
recognition, where the target signal is explicit, a lie may leave only fleeting traces hidden in
the interplay between channels: a speaker can maintain a composed facial expression while betraying
slight tension, hesitation, or pitch instability in their voice. Human observers are barely above
chance at the task @depaulo2003cues, and the cues that do exist are spread across modalities and
are individually weak. Both facts motivate a multimodal, data-driven approach, and both make it
hard: the signal is small, and the datasets that contain it are small too.

We treat deception detection as binary classification. Given a raw video clip of 2–19 s in which a
person answers a question, the model predicts `truth` (0) or `deception` (1). It may use only
signals that can be computed directly from the video and audio streams at inference time. Human
behavioural annotations, such as the MUMIN codes shipped with the dataset, are never an input; at
most they are used as an auxiliary training target.

The question we ask is practical: given a dataset of about 1,700 clips, _which way of using
pre-trained knowledge works best_? Rather than relying on a single model family, we design a
controlled comparative study across four distinct architectural paradigms—from-scratch temporal CNN
and graph attention (`DualHeadGAT`) models on facial landmarks and acoustic descriptors, a
cross-modal fusion head on frozen DINOv2 and WavLM embeddings, parameter-efficient LoRA adaptation
of VideoMAE and HuBERT, and spatio-temporal convolutional and sequential modelling over frozen
Swin3D/MARLIN and ConvNeXt features—anchored to a 609-parameter baseline built solely from
theory-motivated acoustic statistics.

Our contributions are:
- a controlled comparison of four audio-visual paradigms spanning the full pre-training spectrum
  (including both temporal convolutional and graph attention models trained from scratch), together
  with a 609-parameter baseline grounded in psychological theory, evaluated under the official
  3-fold DOLOS protocol for direct comparability with prior work;
- a cross-modal fusion model on frozen DINOv2 and WavLM that outperforms the published PECL result
  by 7.5 accuracy and 17.7 AUC points while training about 30 times fewer parameters;
- unimodal ablations, per-group analyses, and a strict subject-independent 5-fold re-evaluation
  (Appendix A) showing that the visual channel carries almost all of the official-split
  performance and that this gain depends heavily on seeing the same speakers at training and test
  time.

@sec:related reviews prior work, @sec:data describes the dataset and protocol,
@sec:baseline the theory-driven baseline and @sec:arch the four architectures and their results.
@sec:best analyses the best model in depth, @sec:discussion discusses limitations and
@sec:conclusion concludes.
