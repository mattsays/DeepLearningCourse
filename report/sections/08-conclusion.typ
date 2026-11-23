#import "../template.typ": *

// ================================================================ 8
= Conclusion <sec:conclusion>

We compared four audio-visual architectures for deception detection on DOLOS, against a
theory-driven acoustic baseline, on the official 3-fold protocol. Three conclusions stand out.

First, _strong frozen representations beat adaptation on this dataset_. A 165k-parameter head on
frozen DINOv2 and WavLM embeddings reaches 72.2% accuracy and 80.4% AUC, above the published PECL
model and above VideoMAE and HuBERT adapted with LoRA (66.3% / 73.1%), at a training cost of
seconds rather than an hour.

Second, _the visual channel carries the performance and fusion adds little_. Video alone matches the
fused model in both architectures where we could measure it; audio alone is 4–14 points lower. The
psychologically motivated acoustic features reach only 55.1% accuracy, above chance but far below
any learned visual model.

Third, _benchmark accuracy is not yet evidence of lie detection_. The probes show that the visual
embeddings encode posture and identity far better than fast facial motion, and the subject-disjoint
evaluation of Appendix A brings every model back to 50–56% accuracy. We therefore recommend reporting
a speaker-independent split alongside the official one.

Future work follows from these limits: completing the official runs of MARLIN + ConvNeXt and the
Temporal CNN; denser frame sampling and motion-sensitive features to reach the timescale of
micro-expressions; true cross-modal interaction (cross-attention or joint self-attention) instead of
late fusion; and training objectives that discourage speaker identification, such as
speaker-adversarial losses or per-speaker normalisation.
