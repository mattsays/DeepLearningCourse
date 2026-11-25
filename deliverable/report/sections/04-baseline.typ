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
ignored. From the waveform and this envelope we compute the duration, the root-mean-square amplitude, the
square root of the standard deviation of the instantaneous power, the energy range of voiced frames (95th minus 5th percentile, in dB), the voiced ratio,
the number of pauses, the share of time spent in pauses, the mean pause length and the number of
voice onsets per second, a proxy for the syllable rate.

*Pitch and voice quality (8 features).* Using Praat through Parselmouth @jadoul2018parselmouth, the
fundamental frequency $F_0$ is tracked every 10 ms between 75 and 500 Hz. We keep the mean and
standard deviation of $F_0$ over voiced frames and its range in semitones,
$12 log_2 (F_0 \/ 100)$, which makes male and female voices comparable. From Praat's periodic point
process we compute local jitter (period instability) and local shimmer (amplitude instability) with
the standard settings; the harmonics-to-noise ratio is averaged over the clip; intensity is
summarised by its mean and standard deviation in dB. Clips with fewer than three voiced frames get
missing pitch features, which are imputed with the median. @tab:load-feats summarises the 17 features
and the psychological account each group targets.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.hline(),
    table.header([Family], [Features (17 total)], [Tool], [Target account]),
    table.hline(stroke: 0.5pt),
    [Energy & pauses (9)],
    [`duration_s`, `rms_mean`, `rms_std`, `energy_range_db`, `voiced_ratio`, `pause_count`, `pause_ratio`, `pause_mean_ms`, `onsets_per_s`],
    [NumPy],
    [Cognitive load (speech timing, hesitations)],
    [Pitch & voice quality (8)],
    [`f0_mean`, `f0_std`, `f0_range_st`, `jitter_local`, `shimmer_local`, `hnr_mean`, `intensity_mean_db`, `intensity_std_db`],
    [Praat],
    [Vocal arousal (pitch, cycle instability, HNR)],
    table.hline(),
  ),
  caption: [The 17 hand-crafted acoustic features used by the `LoadHead` baseline.],
) <tab:load-feats>

== Classifier and training
The features are standardised with means and standard deviations computed on the training fold
only, then passed to a two-layer MLP:
$ hat(y) = W_2 dot.c "Dropout"("GELU"(W_1 dot.c "Dropout"(z) + b_1)) + b_2, quad
  W_1 in RR^(32 times 17), W_2 in RR^(1 times 32), $
with dropout 0.3. This gives *609 trainable parameters* ($17 times 32 + 32$ in the first layer and
$32 times 1 + 1$ in the output layer). The model is trained with binary cross-entropy weighted by
the class ratio of the training fold, AdamW (learning rate $10^(-3)$, weight decay $10^(-2)$),
batch size 32 and gradient clipping at 1.0, for a fixed 60 epochs on the official 3-fold split,
with three seeds; the threshold is the paper's 0.5. As for the fusion head, 60 is the budget of the
five-fold development split (Appendix A), where early stopping stops at 18.1 epochs on average, so the number
was not read off official-fold scores. One fold trains in about four seconds on CPU.
The features are computed for every downloadable clip (no extraction failed, and Praat left no
missing values), so the baseline is scored on all 1,427 downloadable official test clips. On the
official split, `LoadHead` reaches *56.5 ± 1.7% ACC* and *59.2% AUC* (per-seed AUC: 59.6, 59.2,
58.9), establishing the non-neural floor in @tab:main.
