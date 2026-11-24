# References — `tia`, frozen backbones + cross-modal fusion

The literature this direction stands on. One PDF per row, named `<topic>-<arXiv id>.pdf`; the
one file that predates this index keeps its original name (`2303.12745v2.pdf`).

The PDFs are in this folder, committed (eleven files, 81 MB). `./fetch_references.sh --check`
compares each one with the server's byte count and says what is missing; run it without
`--check` to re-download whatever got lost. It exists because the link was slow the day they
were fetched, and because a paper that silently truncates is worse than a missing one.

Every row answers *what in this direction would be unsupported without it*. A paper that only
sounds related does not belong here: the report cites these, and a citation nobody can trace to a
design decision is decoration.

## The reference architecture

| File | Paper | Why it is here |
|---|---|---|
| [`2303.12745v2.pdf`](2303.12745v2.pdf) | Guo et al., *Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning*, ICCV 2023 (arXiv:2303.12745) | The dataset and the architecture this direction is a variant of. Owns the numbers we compare against (Tab. 2: 64.75 / 66.84 ACC) and the cross-dataset rows (Tab. 3) that make the protocol effect arguable. Its `PAVF` fusion is what [`model.py`](../model.py) does not implement; see [`docs/trainable-module.md`](../docs/trainable-module.md) §Fidelity to PECL. |

## The frozen backbones

| File | Paper | Why it is here |
|---|---|---|
| [`dinov2-2304.07193.pdf`](dinov2-2304.07193.pdf) | Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (arXiv:2304.07193) | The video stream: one CLS vector per face crop, no fine-tuning. Supports the premise that a frozen all-purpose feature is enough to describe a face — and, by being a strong *identity* encoder, it is also the explanation of the leak in [`docs/protocol-effect.md`](../docs/protocol-effect.md). |
| [`wavlm-2110.13900.pdf`](wavlm-2110.13900.pdf) | Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing* (arXiv:2110.13900) | The audio stream. Its pre-training includes denoising and speech-separation objectives, which is the argument for expecting paralinguistic information (stress, hesitation) in the representation, not just words. |
| [`vjepa2-2506.09985.pdf`](vjepa2-2506.09985.pdf) | Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (arXiv:2506.09985) | The video-native alternative cache — `extract_vjepa2.ipynb`, `(32, 1024)` per clip. It is the experiment that separates "a face encoder at 3 fps" from "a model that sees motion". Not on the board: one backbone swap, reported as a one-variable comparison. |

The supervised face encoder (HSEmotion, EfficientNet-B0 on AffectNet) has no paper in this
folder: the weights come from the [face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition)
repository (Apache-2.0), and the Ekman-style hypothesis it tests is stated in
[`README.md`](../README.md#other-video-backbones).

## The fusion modules

Each of these is a rung of the ladder in [`docs/trainable-module.md`](../docs/trainable-module.md#experiment-ladder),
not a citation added for coverage.

| File | Paper | Why it is here |
|---|---|---|
| [`mult-1906.00295.pdf`](mult-1906.00295.pdf) | Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019 (arXiv:1906.00295) | Cross-modal attention across streams that are *not* index-aligned — which is our case, 16 frames against 64 audio tokens. The paper the alignment problem in §"16 frames vs 64 audio tokens" comes from. |
| [`perceiver-2103.03206.pdf`](perceiver-2103.03206.pdf) | Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021 (arXiv:2103.03206) | The latent-bottleneck fusion: a few learned queries compress 80 tokens instead of attending over all of them. It is the cheap way to let the two modalities interact without a 80×80 attention map. |
| [`gmu-1702.01992.pdf`](gmu-1702.01992.pdf) | Arevalo et al., *Gated Multimodal Units for Information Fusion* (arXiv:1702.01992) | The gated, per-token alternative to concatenation: one scalar per dimension deciding how much of each channel to keep. The baseline against which "the gate learns something" can fail. |
| [`adapters-1902.00751.pdf`](adapters-1902.00751.pdf) | Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019 (arXiv:1902.00751) | PECL's adapters (`768→256→768` inside every encoder layer) are this. It is what this direction deliberately does *not* do, and the reason the report calls ours **PECL-inspired** rather than PECL. |
| [`lora-2106.09685.pdf`](lora-2106.09685.pdf) | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv:2106.09685) | The other way to spend a parameter budget on a frozen backbone: adapt inside the encoder instead of fusing outside it. It is the axis on which `tia` and `lore` are comparable. |

## Method and protocol

| File | Paper | Why it is here |
|---|---|---|
| [`multimodal-ml-survey-1705.09406.pdf`](multimodal-ml-survey-1705.09406.pdf) | Baltrušaitis, Ahuja, Morency, *Multimodal Machine Learning: A Survey and Taxonomy*, TPAMI 2019 (arXiv:1705.09406) | The vocabulary the report uses for fusion (early / late / tensor) and for the five challenges; it is what makes "late fusion is a control, not a candidate" a statement about the literature rather than a preference. |
| [`shortcut-learning-2004.07780.pdf`](shortcut-learning-2004.07780.pdf) | Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020 (arXiv:2004.07780) | The mechanism behind the protocol effect: a model takes the easiest feature that correlates with the label, and on a clip-level split the easiest feature is the person. This is why the board reads the subject-independent split and why [`docs/protocol-effect.md`](../docs/protocol-effect.md) measures the leak instead of only asserting it. |

## Adding one

1. Download the arXiv PDF, name it `<topic>-<id>.pdf`, add a row with the claim it supports and
   the file in this direction that makes that claim.
2. Cite it where the claim is made (`docs/`, `README.md`), not only here — this file is the
   index, not the argument.
3. If a paper is quoted for a *number*, the number and its table go in the document that quotes
   it, and this row only points there.
