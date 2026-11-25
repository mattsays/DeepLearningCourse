#import "../template.typ": *

// ================================================================ 8
= Conclusion <sec:conclusion>

We compared four audio-visual architectures for deception detection on DOLOS, against a
theory-driven acoustic baseline, on the official 3-fold protocol. Three conclusions stand out.

First, _on the official split, a light head on strong frozen representations reached the best AUC
of the four paradigms_. A
165k-parameter head on frozen DINOv2 and WavLM embeddings reaches 72.2% accuracy and 80.4% AUC,
above the PECL result reported by the dataset paper (under a related but non-identical evaluation),
above our VideoMAE + HuBERT pipeline adapted with LoRA (66.3% / 73.1%) and above MARLIN + ConvNeXt
(58.9% / 58.9%), at a training cost of seconds rather than an hour. The pipelines differ in more
than the adaptation strategy, so this is an observation about these pipelines, not evidence that
freezing beats adapting in general.

Second, _the visual channel carries the official-split performance, and our late-fusion heads did
not improve on it_. Video alone comes within about a point of the fused model in all three
architectures where we measured it; audio alone is 3.5–13.3 points lower for the two pre-trained models,
and close to it only for the from-scratch Temporal CNN, whose two streams are equally weak. The psychologically motivated acoustic features reach
56.5% accuracy: above chance, level with the Temporal CNN, within 2.4 points of MARLIN + ConvNeXt, and far below the
two models built on general-purpose pre-trained visual encoders.

Third, _benchmark accuracy is not yet evidence of lie detection_. The probes show that the visual
embeddings encode static posture and framing far better than fast facial motion, and the same vectors
name the speaker with 96.3% accuracy (@tab:probes); on the subject-disjoint evaluation of Appendix A
every model then falls back to chance level (49.9–56.1% accuracy). We therefore
recommend reporting a speaker-independent split alongside the official one.

Future work follows from these limits: re-running PECL and
every model on one identical set of clips; nested, subject-disjoint model selection and an
episode-disjoint control; controls for identity reliance, such as speaker-identification probes and
face-blurred inputs; denser frame sampling and motion-sensitive features to reach the timescale of
micro-expressions; true cross-modal interaction (cross-attention or joint self-attention) instead of
late fusion; and training objectives that discourage speaker identification, such as
speaker-adversarial losses or per-speaker normalisation.
