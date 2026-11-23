#import "../../template.typ": *

== Cross-Modal Fusion
*Why.* With about 850 training clips per fold, fine-tuning a large encoder end to end overfits
quickly. This architecture takes the parameter-efficiency idea of PECL to its limit: both encoders
stay *completely frozen*, their embeddings are computed once and cached, and only a small fusion
head is trained. Training then takes seconds, which makes it possible to run several seeds and many
variants, and it isolates the question of _how much the pre-trained representations already
contain_.

*Visual stream.* Sixteen frames are sampled uniformly over the clip. On each frame MTCNN
@zhang2016mtcnn detects the face (minimum size 40 px, detections below confidence 0.90 rejected); the
box is enlarged to a square of 1.5 times its longer side, clamped to the frame, and resized to
224×224. If no face is found the centre square of the frame is used. Each crop goes through
DINOv2-small, and the `[CLS]` token of the last layer is kept, giving a $16 times 384$ matrix per
clip.

*Audio stream.* The whole waveform goes through WavLM-base-plus in a single pass. Its last hidden
state has about 50 vectors per second; adaptive average pooling over contiguous windows reduces it to
64 tokens of dimension 768, the same token count as PECL. Both caches are stored in half precision,
and a cache built with different settings is refused at load time.

*Fusion head.* Each stream is projected to $d = 128$ by a linear layer applied to every token,
averaged over time, and the two 128-dimensional vectors are concatenated. A LayerNorm, dropout 0.3,
a 256→64 linear layer with GELU, a second dropout and a 64→1 layer produce the logit. @sec:best gives
the exact formulation and the training recipe.

*Size.* 164,737 trainable parameters, about 30 times fewer than PECL. The two input projections
account for 147,712 of them.
