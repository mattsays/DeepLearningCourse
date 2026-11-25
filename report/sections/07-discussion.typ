#import "../template.typ": *

// ================================================================ 7
= Discussion and Limitations <sec:discussion>

*Small and imbalanced data.* Each official fold trains on about 800–970 clips, depending on whether
the model carves out a validation set. 79% of the clips show men, and two recurring team captains
account for 28% of the corpus and lie in 69% of their clips. Standard deviations across folds and
seeds reach 2–3 points, so differences of 1–2 points between models are not meaningful.
VideoMAE + HuBERT and MARLIN + ConvNeXt were run with a single seed, so their spread reflects the
folds only.

*Heterogeneous evaluation rules and coverage.* The models of @tab:coverage were scored on
1,274–1,427 of the 1,650 test clips of the paper, and two different stopping and threshold rules
were used (@tab:rules). Since AUC is threshold-free, the AUC ranking is the more reliable
comparison. The fixed-epoch rule gives Cross-Modal Fusion no tuning advantage, so its lead is
unlikely to be an artefact of the rule. The front ends differ as well — face detector, crop, number
of frames and audio window (@tab:inputs) — so the ranking compares complete pipelines rather than
isolated architectural choices. All results in the main text use the official folds; Appendix A
adds a complementary evaluation of the same models on an alternative, speaker-disjoint split.

*Temporal resolution.* Sixteen frames per clip is about 3 frames per second at the median duration
of 5 s, 5.3 fps on a 3 s clip (5th percentile of duration) and 1.8 fps on a 9 s clip (95th
percentile), while facial micro-expressions last 40–600 ms. WavLM's 250–450 frames for a median to
95th-percentile clip are compressed four to seven times into 64 tokens. None of the models can see
the fastest facial cues, and our frozen model averages away the order of frames entirely. The
probes of @sec:best agree: the frozen DINOv2 embeddings read static configuration — posture,
position, framing — well, and fast facial actions such as eyebrow raising or nodding poorly. The
HuBERT branch hears at most the first 10 s of a clip at evaluation, while its video counterpart
spans the whole clip. Both late-fusion models combine the modalities only after averaging each over
time, so they cannot test whether the _timing_ of facial and vocal cues relative to each other
carries signal.

*External validity.* A game show is a low-stakes, performed setting, and its contestants are
practised comedians who lie for entertainment. Detecting deception in a courtroom, an interview or at
a border is a different problem, with different base rates and different consequences of errors.

*Ethics.* A deception detector at 70% accuracy wrongly flags almost three truthful people in ten.
These models are research tools: they must not be used to make decisions about individuals, and
their apparent accuracy on a benchmark does not transfer to real interrogations. The DOLOS
annotations are licensed for academic use only, and the footage is not ours to redistribute.
