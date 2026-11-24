#import "../template.typ": *

// ================================================================ 7
= Discussion and Limitations <sec:discussion>

*What the visual stream recognises.* The evidence points to static visual identity and context
rather than to facial behaviour: the face crops keep hair, framing, head pose and some background,
not only the face. The best model gets its accuracy from the video branch, and the frozen DINOv2 encoder was trained for generic visual
similarity, not for expressions; its probes read posture and position well but fast facial movement
poorly. On the official folds, test clips share speakers and recording sessions with training: in
each fold 95–99% of the test clips have their speaker in the training set, and every test clip comes
from an episode that is also in training. A clip of a given speaker in a given round shares the
label, clothing, lighting and studio with the other clips of that round. Part of the accuracy may
therefore reflect recognising the person and the round rather than the lie. Two observations support
this reading: VideoMAE + HuBERT ranks clips of the _same_ speaker only moderately well (within-speaker
AUC 0.63 for audio and 0.66 for video, against 0.67 and 0.71 over all test clips), and Appendix A shows that all models
lose most of their advantage when speakers are disjoint.

*Small and imbalanced data.* Each fold trains on 750–870 clips. 79% of the clips show men, and two
recurring team captains account for 28% of the corpus and lie in 69% of their clips. Standard
deviations across folds and seeds reach 2–3 points, so differences of 1–2 points between models are
not meaningful.

*Heterogeneous evaluation rules and coverage.* The models of @tab:coverage were scored on
1,132–1,426 of the 1,650 test clips of the paper, and two different stopping and threshold rules
were used (@tab:rules). Since AUC is threshold-free, the AUC ranking is the more reliable
comparison. The fixed-epoch rule gives Cross-Modal Fusion no tuning advantage, so its lead is
unlikely to be an artefact of the rule. The front ends differ as well — face detector, crop, number
of frames and audio window (@tab:inputs) — so the ranking compares complete pipelines rather than
isolated architectural choices.

*Temporal resolution.* Sixteen frames per clip is about 3 frames per second at the median duration
of 5 s, 5.3 fps on the shortest clips and 1.8 fps on the longest, while facial micro-expressions last
40–600 ms. WavLM's 250–450 frames per clip are compressed four to seven times into 64 tokens. None
of the models can see the fastest facial cues, and our frozen model averages away the order of
frames entirely. The HuBERT branch hears at most the first 10 s of a clip at evaluation, while its
video counterpart spans the whole clip. Both late-fusion models combine the modalities only after
averaging each over time, so they cannot test whether the _timing_ of facial and vocal cues
relative to each other carries signal.

*External validity.* A game show is a low-stakes, performed setting, and its contestants are
practised comedians who lie for entertainment. Detecting deception in a courtroom, an interview or at
a border is a different problem, with different base rates and different consequences of errors.

*Ethics.* A deception detector at 70% accuracy wrongly flags almost three truthful people in ten.
These models are research tools: they must not be used to make decisions about individuals, and
their apparent accuracy on a benchmark does not transfer to real interrogations. The DOLOS
annotations are licensed for academic use only, and the footage is not ours to redistribute.
