# Plan, blockers and kill criteria

## Status

The pipeline is written end to end: mouth ROI → 25 fps → AV-HuBERT → `(64, 768)` cache →
probe → five folds. It has **not been run**, and three things have to be true before it can be.

| Blocker | Severity | What it means |
|---|---|---|
| AV-HuBERT licence | **decision for the group** | non-commercial research only; explicitly forbids biometric processing and surveillance |
| fairseq on a modern Python | **session risk** | the official code targets Python 3.8; the bundled fork may not build on a current Colab image |
| simplified mouth alignment | quality risk | no mean-face stabilisation, so head rotation stays in the crop |

## The licence is not a formality

The AV-HuBERT licence grants use "solely for your non-commercial research purposes" and then,
in the restrictions:

> "You will not … use … the Software Products … for (iii) purposes of surveillance, including
> any research or development relating to surveillance, (iv) **biometric processing**"

Deception detection from face and voice is behavioural biometrics in most readings of that
term, and the output is a judgement about a person. The project is coursework at a university,
which is the kind of use the grant contemplates, but this is exactly the sentence to read
before the group commits. Two honest ways forward:

1. **Use it and say so** in the report, with the licence cited, the use described as academic
   and non-commercial, and no redistribution of weights or derived features.
2. **Drop it** and run the V-JEPA 2 backend, which is MIT.

Either is defensible; silently using it is not. The MJEPA path, for the record, would have
been MIT.

## What runs today

Nothing has been executed. The parts that can be checked without a GPU or the weights are
checked: the probe variants on both widths, `pool_avhubert` on a 475-frame sequence, the cache
writer, the training loop on synthetic data (both protocols), and the notebook passes
`nbformat`. See the notes in the folder README.

## The ladder

| # | Step | Cost | Passes if |
|---|---|---|---|
| 0 | Environment cell + 3 clips + frame grid | 15 min | fairseq imports, the crop sits on the mouth |
| 1 | Full extraction, both passes | ~45–75 min T4 (estimate; §7 prints the real rate) | ≥90% of clips cached in both folders |
| 2 | `audio` only | 5 min/fold | near `tia`'s ~59 AUC (plumbing: same speech, different encoder) |
| 3 | `video` only | 5 min/fold | **off chance** — the whole point |
| 4 | `unified` vs `audio` | 2 runs | fusion beats the audio stream |
| 5 | `congruence` | 1 run | > 55 AUC: the mouth/voice relation carries signal |

Step 3 is the one that matters. If a speech-conditioned mouth encoder is at chance on DOLOS,
then the face is a weak channel for this task and no visual backbone will fix it.

## Kill criteria

- **fairseq will not import after a session of trying** → run V-JEPA 2 from
  `cross-modal-fusion/extract_vjepa2.ipynb` and report `avh` as blocked-by-dependency.
- **The frame grid shows the crop off the mouth on more than a few clips** → fix the box
  before encoding; a drifting crop is worse than a face crop.
- **`video`-only at chance** → the hypothesis fails. Report it next to `tia`'s video row, which
  was also at chance with a completely different encoder. Two encoders failing the same way is
  a stronger finding than one.

## Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Python/fairseq | the fork predates current torch and Python | environment cell fails fast with instructions; V-JEPA 2 as the fallback |
| 25 fps preprocessing | ~200k dlib landmark predictions for the protocol | detect faces once per second, landmarks every frame; resumable cache |
| Mouth crop drift | laughter, head turns, hand gestures | re-detect every second; frame-grid check before the full run |
| No stabilisation | the checkpoint expects mean-face warped crops | recorded in the manifest; if results are flat, implement the warp before blaming the model |
| Two unimodal passes | the checkpoint was trained with both modalities, then used one at a time | modality dropout makes each pass in-distribution; if a stream looks degenerate, check the fused pass before blaming the crop |
| fbank frontend | `python_speech_features` is not torchaudio's Kaldi fbank, and the per-frame layer norm is unusual | copied verbatim from `hubert_dataset.py`; a different frontend is a silently different input |

## If the licence is rejected

`cross-modal-fusion/extract_vjepa2.ipynb` produces a V-JEPA 2 video cache for the `tia`
direction, and `tia`'s probe already accepts a different video width. That path is MIT, runs
today, and answers the weaker question — *does a video-native encoder beat frame-wise
features?* — without any licence question.
