#import "../../template.typ": *

== VideoMAE + HuBERT with LoRA
*Why.* Frozen image encoders see each frame in isolation and cannot adapt to the domain. VideoMAE is
a _video_ Transformer: it embeds 2×16×16 space-time tubelets, so motion is part of its input, and
its masked-reconstruction pre-training makes it data-efficient. HuBERT plays the same role for raw
speech. LoRA lets both adapt their attention to deception cues while changing only 0.3% of their
parameters, a middle ground between freezing and fine-tuning.

*Face pipeline.* Thirty-two frames are sampled uniformly over the clip. Faces are detected with the
YuNet detector and described with SFace identity embeddings; faces are clustered by identity, and
the speaker is the identity visible in the most frames, which matters because 753 of the 1,452 clips
show more than one person. Frames where the speaker is not visible borrow the nearest frame where
they are. Box centres and sizes are median-filtered over five frames within each camera shot, and a
square crop of 1.6 times the face size is resized to 224×224. A face was found in 99.9% of frames
and the speaker in 96.0%; no clip fell back to a centre crop. At training time the model sees 16 of
the 32 frames — the even or the odd ones at random — with a random resized crop (scale 0.8–1.0) and
horizontal flip shared across frames; at test time it sees the even frames without augmentation.

*Audio pipeline.* Training uses a random 8 s window of the 16 kHz waveform and evaluation the first
10 s; each crop is z-normalised. Batches group clips of similar length to limit padding. Because of
this window, the audio branch never hears the end of clips longer than 10 s (up to 48 s in the local
annotations), whereas the video frames always span the whole clip; the two branches therefore do
not see exactly the same part of a long answer.

*Encoders and adaptation.* VideoMAE-Base (fine-tuned on Kinetics-400) turns 16 frames into
8×14×14 = 1,568 tokens, which are averaged and normalised by the checkpoint's own LayerNorm.
HuBERT-Base (LibriSpeech 960 h) produces about 50 vectors per second, which are averaged over the
real frames only. LoRA with rank $r = 8$, scaling $alpha = 16$ and dropout 0.1 is applied to the
query and value projections of all 12 attention layers of each encoder: 24 adapter matrices and
294,912 parameters per branch. Each branch ends with dropout 0.3 and a 768→1 linear head. Before
every fold a check verifies that every adapter receives a non-zero gradient and actually changes the
output.

*Late fusion.* The two branches are trained separately. Their logits are standardised with the
mean and standard deviation of their own validation logits, giving $z_v$ and $z_a$, and combined as
$ s = w z_v + (1 - w) z_a, quad w in {0, 0.05, ..., 1}, $
with $w$ chosen by validation AUC. The decision threshold is then chosen on the fused validation
scores. The chosen weights on the official folds were 0.50, 0.80 and 0.45: the video branch
contributes at least half.

*Training.* Binary cross-entropy with positive-class weighting; AdamW with learning rate
$2 dot 10^(-4)$ for the adapters and $10^(-3)$ for the head, weight decay 0.01; one epoch of linear
warm-up followed by a cosine decay; effective batch 16 (4×4 accumulated for video, 8×2 for audio);
at most 10 epochs for video and 15 for audio, with early stopping on validation AUC. Training uses
fp16 mixed precision, and gradient checkpointing keeps the video branch under 0.8 GB of GPU memory.

*Size.* 297,217 trainable parameters per branch (adapters, final LayerNorm and head), 594,434 in
total — 0.34% of VideoMAE and 0.31% of HuBERT.
