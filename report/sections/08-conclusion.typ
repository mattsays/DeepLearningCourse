#import "../template.typ": *

// ================================================================ 8
= Conclusion <sec:conclusion>

We compared four audio-visual architectures for deception detection on DOLOS, against a
theory-driven acoustic baseline, on the official 3-fold protocol. Three conclusions stand out.

First, _on the official split, a light head on strong frozen representations scored highest_. A
165k-parameter head on frozen DINOv2 and WavLM embeddings reaches 72.2% accuracy and 80.4% AUC,
above the PECL result reported by the dataset paper (under a related but non-identical evaluation),
above our VideoMAE + HuBERT pipeline adapted with LoRA (66.3% / 73.1%) and above MARLIN + ConvNeXt
(59.1% / 60.3%), at a training cost of seconds rather than an hour. The pipelines differ in more
than the adaptation strategy, so this is an observation about these pipelines, not evidence that
freezing beats adapting in general.

Second, _the visual channel carries the official-split performance, and our late-fusion heads did
not improve on it_. Video alone matches the fused model in both architectures where we could
measure it; audio alone is 4–14 points lower. The psychologically motivated acoustic features reach
only 55.1% accuracy, above chance but below every learned model on the official folds.

Third, _benchmark accuracy is not yet evidence of lie detection_. The probes show that the visual
embeddings encode static posture and framing far better than fast facial motion, and the
subject-disjoint evaluation of Appendix A brings every model back to 50–56% accuracy. We therefore
recommend reporting a speaker-independent split alongside the official one.

Future work follows from these limits: an official run of the Temporal CNN; re-running PECL and
every model on one identical set of clips; nested, subject-disjoint model selection and an
episode-disjoint control; controls for identity reliance, such as speaker-identification probes and
face-blurred inputs; denser frame sampling and motion-sensitive features to reach the timescale of
micro-expressions; true cross-modal interaction (cross-attention or joint self-attention) instead of
late fusion; and training objectives that discourage speaker identification, such as
speaker-adversarial losses or per-speaker normalisation.
