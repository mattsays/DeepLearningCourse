"""Theory-driven acoustic features for the `load` direction.

The idea is the opposite of `tia`: instead of asking a frozen foundation model what it sees,
compute a small set of features that a specific psychological theory says should differ between
truth and deception, and give those to the classifier.

Two families, from two literatures:

**Cognitive load** (Vrij's cognitive approach). Lying is more demanding than telling the truth, so
a liar speaks more slowly, pauses more and fills the gaps. Observable in the energy envelope:
pause structure, speech rate, energy variance.

**Vocal arousal** (voice-stress research). Concealed emotion leaks into the voice as tension: higher
and more variable pitch, more jitter and shimmer, lower harmonic-to-noise ratio. Observable with
Praat's analysis, which is why `parselmouth` is the one extra dependency.

Both families are cheap: they read the 16 kHz wav that `tia` already extracted, and produce one
17-dimensional vector per clip in a few milliseconds. Nothing here touches a GPU.

The empirical status is honest in `docs/theory.md`: effect sizes for behavioural cues to deception
are small (DePaulo et al., 2003), and Vrij's approach works better when the interviewer *elicits*
cues than when an observer measures them passively. This direction measures passively, because
that is what a classifier can do.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

try:  # Praat bindings; absent locally, installed in the notebook
    import parselmouth
    from parselmouth.praat import call

    HAVE_PRAAT = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_PRAAT = False

#: Order is fixed: a cache built with a different list is a different cache.
ENERGY_FEATURES = [
    "duration_s",
    "rms_mean",
    "rms_std",
    "energy_range_db",
    "voiced_ratio",
    "pause_count",
    "pause_ratio",
    "pause_mean_ms",
    "onsets_per_s",
]
PRAAT_FEATURES = [
    "f0_mean",
    "f0_std",
    "f0_range_st",
    "jitter_local",
    "shimmer_local",
    "hnr_mean",
    "intensity_mean_db",
    "intensity_std_db",
]
FEATURE_NAMES = ENERGY_FEATURES + PRAAT_FEATURES

FRAME_MS = 25          # analysis window for the energy envelope
HOP_MS = 10            # 10 ms hop, the same as Praat's default
SILENCE_DB = 25.0      # a frame this far below the clip's loud level counts as silence
MIN_PAUSE_MS = 200     # below this it is a closure, not a pause


def _envelope(speech: np.ndarray, sample_rate: int) -> np.ndarray:
    """Frame energy in dB, one value per HOP_MS."""
    win = max(1, int(sample_rate * FRAME_MS / 1000))
    hop = max(1, int(sample_rate * HOP_MS / 1000))
    if len(speech) < win:
        speech = np.pad(speech, (0, win - len(speech)))
    frames = np.lib.stride_tricks.sliding_window_view(speech, win)[::hop]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of every run of True, end exclusive."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def energy_features(speech: np.ndarray, sample_rate: int) -> dict:
    """Pause structure and speech rate, from the energy envelope alone."""
    db = _envelope(speech, sample_rate)
    loud = np.percentile(db, 90)
    voiced = db > (loud - SILENCE_DB)

    silent_runs = _runs(~voiced)
    pauses_ms = [(end - start) * HOP_MS for start, end in silent_runs
                 if (end - start) * HOP_MS >= MIN_PAUSE_MS]

    # an onset is a silent-to-voiced transition: a syllable or a word boundary
    onsets = sum(1 for start, _ in _runs(voiced) if start > 0)
    duration = len(speech) / sample_rate

    return {
        "duration_s": duration,
        "rms_mean": float(np.mean(speech ** 2) ** 0.5),
        "rms_std": float(np.std(speech ** 2) ** 0.5),
        # over voiced frames only: digital silence would otherwise dominate the range
        "energy_range_db": float(np.percentile(db[voiced], 95) - np.percentile(db[voiced], 5))
        if voiced.any() else 0.0,
        "voiced_ratio": float(voiced.mean()),
        "pause_count": len(pauses_ms),
        "pause_ratio": float(sum(pauses_ms) / 1000.0 / max(duration, 1e-6)),
        "pause_mean_ms": float(np.mean(pauses_ms)) if pauses_ms else 0.0,
        "onsets_per_s": float(onsets / max(duration, 1e-6)),
    }


def praat_features(speech: np.ndarray, sample_rate: int) -> dict:
    """Pitch, perturbation and harmonics, through Praat."""
    nan = {name: float("nan") for name in PRAAT_FEATURES}
    if not HAVE_PRAAT:
        return nan

    sound = parselmouth.Sound(speech.astype(np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if f0.size < 3:
        return nan

    semitones = 12.0 * np.log2(f0 / 100.0)          # relative to a fixed reference: comparable
    point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
    intensity = call(sound, "To Intensity", 75.0, 0.01)   # (minimum pitch, time step)

    # Praat returns -300 dB (or NaN) for silent frames: keep only the voiced ones
    levels = np.asarray(intensity.values, dtype=float)
    levels = levels[np.isfinite(levels) & (levels > 0)]

    def safe(value):
        return float(value) if np.isfinite(value) else float("nan")

    return {
        "f0_mean": float(f0.mean()),
        "f0_std": float(f0.std()),
        "f0_range_st": float(semitones.max() - semitones.min()),
        "jitter_local": safe(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)),
        "shimmer_local": safe(call([sound, point_process], "Get shimmer (local)",
                                   0, 0, 0.0001, 0.02, 1.3, 1.6)),
        "hnr_mean": safe(call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0).values.mean()),
        "intensity_mean_db": safe(levels.mean()) if levels.size else float("nan"),
        "intensity_std_db": safe(levels.std()) if levels.size else float("nan"),
    }


def acoustic_features(wav_path) -> np.ndarray:
    """One 17-dimensional vector for one clip, in `FEATURE_NAMES` order."""
    speech, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sample_rate != 16000:
        raise RuntimeError(f"{wav_path}: {sample_rate} Hz, expected 16000")

    values = {**energy_features(speech, sample_rate), **praat_features(speech, sample_rate)}
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)


if __name__ == "__main__":
    import tempfile

    # a synthetic clip: 3 s of a 150 Hz tone with a 0.5 s gap in the middle
    sample_rate = 16000
    t = np.arange(int(1.25 * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))
    speech = np.concatenate([tone, np.zeros(int(0.5 * sample_rate)), tone]).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        sf.write(handle.name, speech, sample_rate)
        vector = acoustic_features(handle.name)

    assert vector.shape == (len(FEATURE_NAMES),), vector.shape
    for name, value in zip(FEATURE_NAMES, vector):
        print(f"{name:20s} {value:10.4f}")
    print(f"\npraat available: {HAVE_PRAAT}")
