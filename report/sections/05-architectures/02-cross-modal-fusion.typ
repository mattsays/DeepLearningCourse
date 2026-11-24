#import "../../template.typ": *

// ================================================================ 5.2
== Cross-Modal Fusion
*Why.* With fewer than a thousand training clips per fold, fine-tuning a large encoder end to end
overfits quickly. This architecture takes the parameter-efficiency idea of PECL @guo2023dolos to its
limit: both encoders stay *completely frozen*, their embeddings are computed once and cached, and
only a small fusion head is trained (@tab:adaptation). Training then takes seconds, which makes it
possible to run several seeds and many variants, and it isolates the question of _how much the
pre-trained representations already contain_. The design is PECL-inspired rather than PECL: nothing
inside either backbone changes, and the two streams meet once, after their last layer.

*What learns.* The trainable part is a probe with a fusion step, and it has five decisions to make:
project each token, normalise it, handle time, fuse the two streams, and pool before the classifier.
The configuration on the board answers all five as cheaply as they can be answered: one linear
projection per stream, no normalisation of the tokens themselves — so the two streams arrive on their
own scales and the only LayerNorm sits after they are concatenated — and the mean over time taken
_before_ the streams meet. That last choice is not neutral. The head is exactly invariant to the
order of the 16 frames and of the 64 audio tokens, so it cannot tell the beginning of an answer from
its end, and the pooling that describes each modality is the same operation that discards its
temporal structure. @sec:best gives the formulation and the training recipe.

*What the head receives.* The encoders run once, offline, and write one array per clip into a shared
cache in half precision: $16 times 384$ values for the visual stream and $64 times 768$ for the
audio. Per clip the trainable network therefore receives $16 times 384 + 64 times 768 = 55,296$
floats and has to return a single logit, so every design choice above is a choice about _reading_
two fixed representations rather than about computing them. A cache whose manifest records other
settings — frame count, crop margin, token count, encoder revision — is refused at load time, which
is what keeps a comparison from silently mixing two extractions.

*Visual stream.* Sixteen frames are sampled uniformly over the clip. On each frame MTCNN
@zhang2016mtcnn detects the face on a 640-pixel copy of the frame, with a minimum face size of 40 px,
and the box is mapped back to the original resolution; a detection whose confidence is below 0.90 is
discarded. The box is then enlarged to a square of 1.5 times its longer side, clamped to the frame
and resized to 224×224. If no face is found the centre square of the frame is used, so every clip
yields exactly 16 crops. Each crop goes through DINOv2-small @oquab2024dinov2 and the `[CLS]` token
of the last layer is kept, giving a $16 times 384$ matrix per clip.

*Audio stream.* The whole waveform goes through WavLM-base-plus @chen2022wavlm in a single pass; the
files are 16 kHz and stereo is averaged to mono. Its last hidden state has 50 vectors per second, and
adaptive average pooling over contiguous windows reduces it to 64 tokens of dimension 768, the same
token count as PECL. The pooling adapts to the clip, so a token spans about 80 ms at the median
5-second clip and proportionally more in a longer one: the tokens are ordered, but their duration is
not fixed.

*Fusion head.* Each stream is projected to $d = 128$ by a linear layer applied to every token,
averaged over time, and the two 128-dimensional vectors are concatenated (@fig:fusion). A LayerNorm,
dropout 0.3, a 256→64 linear layer with GELU, a second dropout and a 64→1 layer produce the logit.

*The alignment problem.* The two streams do not share a clock. PECL can compare frame $t$ with audio
token $t$ because both branches hand it 64 steps: an expression at index 12 of the face and a prosodic
rise at index 12 of the voice are, by construction, the same moment. Our cache holds 16 frames against
64 tokens, four to one, so a single index names two moments a quarter of a clip apart and nothing in
the data says which token belongs with which frame. A co-attention layer could learn that
correspondence from content alone; the configuration on the board does not try. It averages each
stream over time before the two meet, which removes the problem by removing the time axis — and the
price is exact. The model can represent _that_ the face moved quickly and _that_ the voice tightened,
but never _together_, so the cue the literature treats as central, a mismatch between what the face
and the voice do at the same instant, lies outside its hypothesis space.

The sampling makes the loss worse. Sixteen frames over a median clip of 5 s is one sample every
$approx 310$ ms, while facial micro-expressions last 40–600 ms: the fastest cue can begin and end
between two samples, so it is not merely mistimed, it is absent from the input. Two remedies are on
the table and neither was applied to the official runs. Pooling the audio down to 16 tokens restores
the 1:1 index and throws away acoustic time resolution; re-extracting the video at 32 or 64 frames
restores the resolution and costs another pass over the corpus. Until one of them is taken, the
caveat on temporal resolution in @sec:discussion applies to this model in full.

*The parameter budget.* Capacity is capped by the data rather than by the hardware. With fewer than a
thousand training clips per fold a larger head cannot be identified: the training loss reaches zero
within a few epochs while validation stops following it, which is why every frozen-feature model here
sits in the $10^5$ range (@tab:adaptation). Time is the resource that is abundant. With the encoders
idle a fold trains in about 8 s — 25 s for the three official folds — so a full grid of folds and
seeds costs less than a single experiment on a trainable encoder. That asymmetry is the practical
argument for the design: it buys repetition, and repetition is what separates a one-point difference
from noise (@sec:discussion).

*Size.* 164,737 trainable parameters, 30 times fewer _trainable_ parameters than PECL (5.06M);
@tab:params breaks the budget down. The frozen encoders add ≈116M parameters; they run once to build
the caches, and that extraction is not counted in the training times we report.

*The design space.* The implementation contains more than the configuration that reached the board;
@tab:variants prices the alternatives, and none was run on the official folds. What was run there is
the pair of controls: a video-only and an audio-only variant that keep the same classifier and switch
one stream off. They answer the only question that can demote this architecture — _does fusing two
channels add anything over the better single one?_ — and @tab:unimodal reports the answer: on the
official folds, this late-fusion head does not.
