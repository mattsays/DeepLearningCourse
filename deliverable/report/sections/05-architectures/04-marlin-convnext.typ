#import "../../template.typ": *

== MARLIN + ConvNeXt
*Why.* Deceptive cues are often local in time, such as a brief micro-expression or a vocal break.
This architecture pairs a video encoder pre-trained specifically on _faces_ with a convolutional
audio encoder, keeps a time axis in both, and adds an explicit sequence model, so that the decision
can rest on a few salient instants rather than on a clip-wide average.

*Visual stream.* Sixteen frames are sampled uniformly and the face is cropped with MTCNN (detections
below confidence 0.85 rejected, square of 1.5 times the face size, 224×224). The frozen MARLIN
ViT-Base encoder, pre-trained on YouTube Faces, turns the 16 frames into 1,568 tubelet tokens;
because each tubelet spans two frames, they form an 8×14×14 grid, and averaging over the spatial grid
gives a sequence of 8 vectors of dimension 768.

*Audio stream.* A 128-band mel spectrogram (1,024-sample window, 512-sample hop at 16 kHz) is
converted to dB and stacked with its first and second temporal derivatives into a three-channel
image, so that an ImageNet-pretrained CNN receives level, velocity and acceleration of the spectrum
in its three input channels. A frozen ConvNeXt-Tiny encodes it; averaging over frequency only keeps
about one 768-dimensional vector per second, which is linearly interpolated onto the 8 video steps.

*Sequence head.* Each stream passes through LayerNorm, a 768→32 linear layer, GELU and dropout; the
two are concatenated and mixed by a 64→64 linear layer. A temporal convolutional network of three
causal layers (kernel 3, dilations 1, 2 and 4, so that the last step sees the whole sequence) with a
residual connection models the sequence, and the result is averaged over time and classified by
LayerNorm, dropout 0.3 and a 64→64→1 MLP. The code also supports a two-layer bidirectional GRU and
additive attention pooling in place of the TCN and the mean; the official configuration uses the TCN
with mean pooling.

*Training.* Binary cross-entropy with positive-class weighting; AdamW (learning rate
$3 dot 10^(-4)$, weight decay 0.01) with cosine annealing over at most 35 epochs, batch size 16,
gradient clipping at 1.0, early stopping on validation loss with patience 7, threshold chosen on
validation.

*Size.* Both encoders are frozen; the head has 97,857 trainable parameters (52,288 in the two input
projections, 37,056 in the TCN).
