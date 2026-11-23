#import "../template.typ": *

// ================================================================ 1
= Introduction

Deception has no single physical signature. Unlike speech or action recognition, where the target
is explicit, a lie may leave only fleeting traces: a hesitation, a shift in pitch, a facial
expression that does not match the words. Human observers are barely above chance at the task
@depaulo2003cues, and the cues that do exist are spread across channels and are individually weak.
Both facts motivate a multimodal, data-driven approach, and both make it hard: the signal is small,
and the datasets that contain it are small too.

We treat deception detection as binary classification. Given a raw video clip of 2–19 s in which a
person answers a question, the model predicts `truth` (0) or `deception` (1). It may use only
signals that can be computed from the video and audio at inference time. Human behavioural
annotations, such as the MUMIN codes shipped with the dataset, are never an input; at most they are
used as an auxiliary training target.

The question we ask is practical: given a dataset of about 1,700 clips, _which way of using
pre-trained knowledge works best_? We compare four answers that cover the spectrum, from no
pre-training at all to large self-supervised encoders that are adapted to the task, and we anchor
them to a baseline that uses no learned representation but only theory-motivated acoustic
statistics.

Our contributions are:
- a controlled comparison of four audio-visual architectures, from a temporal CNN trained from
  scratch to self-supervised video and speech Transformers adapted with LoRA, together with a
  609-parameter baseline built from psychological theory, all on the official DOLOS protocol;
- a cross-modal fusion model on frozen DINOv2 and WavLM that outperforms the published PECL result
  by 7.5 accuracy and 17.7 AUC points while training about 30 times fewer parameters;
- unimodal ablations and per-group analyses showing that the visual channel carries almost all the
  performance, together with evidence (Appendix A) that this performance depends on seeing the
  same speakers at training and test time.

@sec:related reviews prior work, @sec:data describes the dataset and protocol,
@sec:baseline the theory-driven baseline and @sec:arch the four architectures and their results.
@sec:best analyses the best model in depth, @sec:discussion discusses limitations and
@sec:conclusion concludes.
