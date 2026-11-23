#import "../template.typ": *

// ================================================================ 4
= Theory-Driven Mathematical Baseline <sec:baseline>

Before introducing networks with millions of parameters, we set a transparent reference: a
classifier that sees only acoustic measurements chosen because psychology predicts they should
differ between lies and truths. It answers a simple question — how far does _theory alone_ go? — and
it gives every deep model a floor that is harder to beat than the majority class.

== Theory
The baseline encodes the two accounts of @sec:related as measurable quantities:
- _cognitive load_ appears in the timing of speech: longer and more frequent pauses, a lower
  speaking rate, hesitations;
- _arousal_ appears in the voice source: a raised and more variable pitch, more cycle-to-cycle
  instability (jitter, shimmer) and a less clear, breathier or tenser voice (lower
  harmonics-to-noise ratio).

One caveat: Vrij's paradigm works best when an interviewer raises the load on purpose, for example
with unexpected questions. In DOLOS the classifier observes short answers passively, without the
question, so the theory predicts at most a weak signal.

== Features
Each clip becomes a 17-dimensional vector, computed on CPU in milliseconds per clip.

*Energy and pauses (9 features).* The waveform is split into 25 ms frames with a 10 ms hop, and the
energy of each frame is expressed in dB. A frame is _voiced_ if its energy is within 25 dB of the
clip's 90th-percentile energy; the relative threshold makes the rule independent of recording level.
Silent runs of at least 200 ms count as _pauses_, so that the short silences of stop consonants are
ignored. From this envelope we compute the duration, the mean and standard deviation of the RMS
amplitude, the energy range of voiced frames (95th minus 5th percentile, in dB), the voiced ratio,
the number of pauses, the share of time spent in pauses, the mean pause length and the number of
voice onsets per second, a proxy for the syllable rate.

*Pitch and voice quality (8 features).* Using Praat through Parselmouth @jadoul2018parselmouth, the
fundamental frequency $F_0$ is tracked every 10 ms between 75 and 500 Hz. We keep the mean and
standard deviation of $F_0$ over voiced frames and its range in semitones,
$12 log_2 (F_0 \/ 100)$, which makes male and female voices comparable. From Praat's periodic point
process we compute local jitter (period instability) and local shimmer (amplitude instability) with
the standard settings; the harmonics-to-noise ratio is averaged over the clip; intensity is
summarised by its mean and standard deviation in dB. Clips with fewer than three voiced frames get
missing pitch features, which are imputed with the median.

== Classifier and training
The features are standardised with means and standard deviations computed on the training fold
only, then passed to a two-layer MLP:
$ hat(y) = W_2 dot.c "Dropout"("GELU"(W_1 dot.c "Dropout"(z) + b_1)) + b_2, quad
  W_1 in RR^(32 times 17), W_2 in RR^(1 times 32), $
with dropout 0.3. This gives *609 trainable parameters*. The model is trained with binary
cross-entropy weighted by the class ratio of the training fold, AdamW (learning rate $10^(-3)$,
weight decay $10^(-2)$), batch size 32 and gradient clipping at 1.0, for a fixed 60 epochs on the
official folds, with three seeds. One fold trains in about three seconds. Its results appear with the
others in @tab:main.
