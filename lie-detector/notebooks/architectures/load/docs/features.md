# The 17 features, one by one

Computed by [`features.py`](../features.py) from the 16 kHz mono wav that the `tia` pipeline already
extracted. One vector per clip, `float32`, written to `cached_features/acoustic-load/<clip_id>.npy`.

Nothing here needs a GPU, a backbone, or a download: the whole cache is ~80 KB and builds in minutes.

## Energy family — the cognitive-load theory

From the energy envelope alone (25 ms window, 10 ms hop). A frame is *silent* when it sits more than
25 dB below the clip's 90th percentile; a silence run of at least 200 ms is a pause.

| Feature | What it measures | Expected under load |
|---|---|---|
| `duration_s` | clip length | control, not a cue |
| `rms_mean` | overall loudness | — |
| `rms_std` | loudness variation | higher |
| `energy_range_db` | dynamic range over voiced frames | higher |
| `voiced_ratio` | fraction of frames above the silence floor | lower |
| `pause_count` | silences ≥ 200 ms | **higher** |
| `pause_ratio` | share of the clip spent in silence | **higher** |
| `pause_mean_ms` | mean silence length | **higher** |
| `onsets_per_s` | silent→voiced transitions per second | **lower** (slower speech) |

## Praat family — the vocal-arousal theory

Through `parselmouth`, which wraps Praat. Pitch is tracked between 75 and 500 Hz.

| Feature | What it measures | Expected under arousal |
|---|---|---|
| `f0_mean` | mean fundamental frequency | **higher** |
| `f0_std` | pitch variability | **higher** |
| `f0_range_st` | pitch range in semitones | **wider** |
| `jitter_local` | cycle-to-cycle period perturbation | **higher** |
| `shimmer_local` | cycle-to-cycle amplitude perturbation | **higher** |
| `hnr_mean` | harmonic-to-noise ratio | **lower** (noisier voice) |
| `intensity_mean_db` | mean intensity over voiced frames | higher |
| `intensity_std_db` | intensity variation | higher |

## Two things that are deliberately missing

**Response latency** — the gap between the end of the question and the start of the answer. It is one
of the strongest cues in the cognitive-load literature, and it is **unmeasurable here**: DOLOS clips
start at the speaker, and the question is not in the file. This is the single biggest reason the
direction cannot test the theory properly.

**Filled hesitations** ("um", "uh") — a lexical cue, so it belongs to the text stream, not to the
acoustics. They are visible once the Whisper transcripts exist.

## Implementation notes

- The features have wildly different units (5 s, 150 Hz, 0.005 jitter). The model standardises them
  with a scaler fitted on the **training fold only** — see `LoadHead.fit_scaler`.
- Praat returns `-300 dB` or `NaN` for silent frames and occasionally fails on a clip. Those values
  are masked out of the intensity statistics, and any residual `NaN` in a saved vector is imputed
  with the column median over the working set — feature marginals only, never labels.
- `FEATURE_NAMES` fixes the order. A cache built with a different list is a different cache: the
  notebook does not check this, so if you add a feature, delete `cached_features/acoustic-load/`
  first.

## Reproducing the vector

```python
from features import FEATURE_NAMES, acoustic_features

vector = acoustic_features(clips[clip_id].audio)      # (17,)
print(dict(zip(FEATURE_NAMES, vector.round(4))))
```

`python features.py` runs a self-test on a synthetic tone and prints all 17 values, so the extractor
can be checked without touching the dataset.
