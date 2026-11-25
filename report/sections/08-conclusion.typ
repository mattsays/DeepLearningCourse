#import "../template.typ": *

// ================================================================ 8
= Conclusion <sec:conclusion>

We compared four audio-visual architectures for deception detection on DOLOS, against a
theory-driven acoustic baseline, on the official 3-fold protocol. Three conclusions stand out.

First, _on the official split, a light head on strong frozen representations scored highest_. A
165k-parameter head on frozen DINOv2 and WavLM embeddings reaches 72.2% accuracy and 80.4% AUC,
above the PECL result reported by the dataset paper (under a related but non-identical evaluation),
above our VideoMAE + HuBERT pipeline adapted with LoRA (66.3% / 73.1%) and above MARLIN + ConvNeXt
(58.9% / 58.9%), at a training cost of seconds rather than an hour. The pipelines differ in more
than the adaptation strategy, so this is an observation about these pipelines, not evidence that
freezing beats adapting in general.

Second, _the visual channel carries most of the performance, and our late-fusion heads did not
improve on it_. Video alone comes within about a point of the fused model in all three
architectures where we measured it; audio alone is 3.5–13.3 points lower for the two pre-trained
models, and close to it only for the from-scratch Temporal CNN, whose two streams are equally weak.

Third, _general-purpose pre-trained visual encoders make the difference_. The psychologically
motivated acoustic features reach 56.5% accuracy with 609 parameters: above the majority class,
level with the Temporal CNN, which learns from landmarks and acoustic descriptors without
pre-training, and within 2.4 points of the face-specific MARLIN + ConvNeXt. Only the two pipelines
built on DINOv2 and VideoMAE move clearly beyond this floor, to 66–72%.

Appendix A complements these results with an evaluation of the same models on an alternative,
speaker-disjoint split, which we report as an open question for the design of future benchmarks.

Future work follows from the limits of @sec:discussion: re-running PECL and every model on one
identical set of clips; more seeds and single-stream variants for VideoMAE + HuBERT and
MARLIN + ConvNeXt; denser frame sampling and motion-sensitive features to reach the timescale of
micro-expressions; true cross-modal interaction (cross-attention or joint self-attention) instead
of late fusion; and evaluating the same models on additional splits and datasets.
