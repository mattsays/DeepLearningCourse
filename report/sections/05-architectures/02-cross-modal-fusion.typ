#import "../../template.typ": *

// ================================================================ 5.2
== Cross-Modal Fusion
*Why.* With fewer than a thousand training clips per fold, fine-tuning a large encoder end to end
overfits quickly. This architecture takes the parameter-efficiency idea of PECL @guo2023dolos to its
limit: both encoders stay *completely frozen*, their embeddings are computed once and cached, and
only a small fusion head is trained (@tab:adaptation). Training then takes seconds, which allows
several seeds and many variants, and it isolates the question of _how much the pre-trained
representations already contain_. Unlike PECL, nothing inside either backbone changes, and the two
streams meet once, after their last layer.

*What the head receives.* The encoders run once, offline, and cache one array per clip in half
precision: $16 times 384$ values for the visual stream and $64 times 768$ for the audio, 55,296 floats
from which the head must produce one logit. A cache whose manifest records other settings — frame
count, crop margin, token count, encoder revision — is refused at load time, so a comparison cannot
silently mix two extractions.

*Visual stream.* Sixteen frames are sampled uniformly over the clip. MTCNN @zhang2016mtcnn detects
the face on a 640-pixel copy of each frame (minimum face 40 px, detections below confidence 0.90
discarded); the box is mapped back, enlarged to a square of 1.5 times its longer side, clamped to the
frame and resized to 224×224, with the centre square as fallback, so every clip yields exactly 16
crops. DINOv2-small @oquab2024dinov2 keeps the last-layer `[CLS]` token of each crop.

*Audio stream.* The whole 16 kHz waveform (stereo averaged to mono) goes through WavLM-base-plus
@chen2022wavlm in one pass. Its last hidden state, 50 vectors per second, is reduced by adaptive
average pooling to 64 tokens of dimension 768, PECL's token count; a token therefore spans about
80 ms at the median 5 s clip and more in a longer one.

*Fusion head.* Each stream is projected to $d = 128$ by a linear layer applied to every token and
averaged over time; the two vectors are concatenated (@fig:fusion) and pass through LayerNorm,
dropout 0.3, a 256→64 linear layer with GELU, a second dropout and a 64→1 layer. The tokens are not
normalised before projection, so the only LayerNorm sits after the concatenation. @sec:best gives
the formulation and the training recipe.

*What the design gives up.* Averaging before fusion makes the head exactly invariant to the order
of frames and tokens: it cannot tell the beginning of an answer from its end, and it cannot relate
what the face and the voice do _at the same instant_ — the mismatch the literature treats as a
central cue. The two streams do not share a clock anyway: 16 frames against 64 tokens, so no index
pairs a frame with its sound, and a co-attention layer would have to learn the correspondence from
content. Sampling adds a second loss: one frame every $approx 310$ ms at the median clip, while
micro-expressions last 40–600 ms, so the fastest cues can fall between two samples. Pooling the
audio to 16 tokens or re-extracting the video at 32–64 frames would address this; neither was done
for the official runs, so the temporal caveat of @sec:discussion applies in full.

*Size and cost.* 164,737 trainable parameters, 30 times fewer _trainable_ parameters than PECL
(5.06M); @tab:params breaks the budget down. The frozen encoders add ≈116M parameters; they run once
to build the caches, and that extraction is not counted in the training times we report. Capacity
is capped by the data: a larger head drives the training loss to zero within a few epochs while
validation stops following it. Time, in contrast, is cheap — about 8 s per fold, 25 s for the three
official folds — which buys the repeated seeds needed to tell a one-point difference from noise.

*Controls.* The implementation also contains the variants of @tab:variants, none run on the official
folds. What was run there is a video-only and an audio-only variant with the same classifier; they
test whether fusing the two channels adds anything over the better one, and @tab:unimodal shows that
on the official folds this late-fusion head does not.
