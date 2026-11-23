#import "../../template.typ": *

== Time-Batching Temporal CNN (deep baseline)
*Why.* It is the simplest learned model that still sees _time_. It uses no pre-trained encoder:
only interpretable per-frame descriptors of the face and voice, on which a small convolutional
network learns short temporal patterns — for example, "the mouth corners move down over 0.3 s while
loudness drops". It tells us how far hand-designed signals go once a network, rather than a fixed
statistic, decides how to combine them over time.

*Input.* Each clip is resampled to a 10 fps grid. For every frame, MediaPipe FaceLandmarker
@lugaresi2019mediapipe returns 478 three-dimensional face landmarks; we group them into 15 regions
(eyes, eyebrows, nose bridge and tip, outer and inner mouth, jaw, chin, cheeks, forehead, irises)
and average the $(x, y, z)$ coordinates of each region, which gives 45 values per frame. When no
face is found, the last valid vector is carried forward. The audio is described by the first eight
frame-level low-level descriptors of the eGeMAPS set @eyben2016gemaps, computed with openSMILE and
linearly interpolated onto the same 100 ms grid. The two streams are truncated to the shorter one and
concatenated into a $T times 53$ matrix, with no normalisation and no maximum length.

*Architecture.* Four 1-D convolutions slide along time: the first has kernel 5 and maps 53 to 64
channels, the next three have kernel 3 and keep 64 channels; each is followed by batch
normalisation and ReLU, and the first two by dropout 0.3. The receptive field is 11 frames, about
1.1 s. Clips have different lengths, so a batch is zero-padded to its longest clip and the network
receives the true length of each sequence. The features are then averaged over the _valid_ frames
only — a masked mean — which lets clips of any length share a batch without the padding diluting the
representation; we call this _time batching_. A linear layer maps the 64-dimensional summary to the
deception logit. An optional second head regresses the 41 MUMIN annotations from the same summary;
it is used only as an auxiliary training loss and is discarded at inference.

*Training.* Binary cross-entropy with positive-class weighting, plus the optional MUMIN
mean-squared-error term; AdamW (learning rate $3 dot 10^(-4)$, weight decay $10^(-4)$), batch size 16,
gradient clipping at 5.0, learning-rate halving on validation-loss plateaus, and early stopping on
validation loss.

*Size.* 57,322 trainable parameters, of which 54,657 are used at inference (the rest is the MUMIN
head).
