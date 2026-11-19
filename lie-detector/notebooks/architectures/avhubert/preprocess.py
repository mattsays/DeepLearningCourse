"""Mouth crops and filterbank features for the `avh` direction.

One owner of the two preprocessing decisions, imported by `extract_features.ipynb` (for the
visual check) and by `scripts/extract_avhubert.py` (for the extraction run). They are not
free functions of convenience: if the notebook and the extraction script cropped differently,
the cache would not describe the frames you looked at.

Everything here is deterministic. No augmentation, no random crop: a cached feature array has
to be reproducible from the clip.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

FPS = 25
CROP = 96
FBANK_BINS = 26
FBANK_STACK = 4
FBANK_DIM = FBANK_BINS * FBANK_STACK


def load_landmarks(model_path):
    """dlib's HOG detector and the 68-point predictor, read from one file."""
    import dlib

    return dlib.get_frontal_face_detector(), dlib.shape_predictor(str(model_path))


def sample_25fps(path, fps: int = FPS) -> list:
    """Decode sequentially and keep every `source_fps / fps`-th frame.

    Sequential `read()` rather than seeking: seeking lands on keyframes and silently returns
    the wrong picture when the GOP is long.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or float(fps)
    step = source_fps / fps
    frames, index, take = [], 0, 0.0
    ok, frame = capture.read()
    while ok:
        if index >= round(take):
            frames.append(frame)
            take += step
        ok, frame = capture.read()
        index += 1
    capture.release()
    return frames


def mouth_crop(frame, detector, predictor, box=None, size: int = CROP):
    """96×96 grayscale mouth ROI, following dlib's 68 landmarks.

    `box` is the last detected face: detecting on every frame is the expensive part of the
    pass (tens of milliseconds), while the landmark prediction is about a millisecond, so the
    caller re-detects once per second and reuses the box in between.

    The official `align_mouth.py` warps each crop onto a mean face; this does not. The crop
    follows the head, which keeps rotation in the data — recorded as `alignment` in the
    manifest, and the reason the notebook draws a grid of frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if box is None:
        faces = detector(gray, 0)
        box = faces[0] if faces else None
    if box is None:                                   # no face: centre crop, same shape
        side = min(gray.shape)
        top = (gray.shape[0] - side) // 2
        left = (gray.shape[1] - side) // 2
        return cv2.resize(gray[top:top + side, left:left + side], (size, size)), None

    points = predictor(gray, box)
    mouth = np.array([(points.part(i).x, points.part(i).y) for i in range(48, 68)])
    x0, y0 = mouth.min(axis=0)
    x1, y1 = mouth.max(axis=0)
    width, height = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(width * 1.6, height * 2.2)             # chin and both corners
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, gray.shape[1])), int(min(cy + side / 2, gray.shape[0]))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        patch = gray
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA), box


def video_tensor(path, detector, predictor, size: int = CROP) -> torch.Tensor:
    """Clip -> `(1, T, 1, size, size)` at 25 fps, float in `[0, 1]`."""
    crops, box = [], None
    for position, frame in enumerate(sample_25fps(path)):
        if position % FPS == 0:                       # re-detect once per second
            box = None
        crop, box = mouth_crop(frame, detector, predictor, box, size)
        crops.append(crop)
    array = np.stack(crops).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(1).unsqueeze(0).contiguous()


def audio_features(path, sample_rate: int = 16000) -> torch.Tensor:
    """16 kHz wav -> `(T, 104)` at 25 fps, as `hubert_dataset.py` builds it.

    `logfbank` (26 bins, 10 ms hop), four frames stacked, then a per-frame layer norm over the
    104 bins. The normalisation is unusual — it is over the feature axis, not over time — and
    it is copied from the official dataset, not chosen here.
    """
    import soundfile as sf
    from python_speech_features import logfbank

    waveform, rate = sf.read(str(path), dtype="int16")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, found {rate}")
    features = logfbank(waveform, samplerate=sample_rate).astype(np.float32)   # (T, 26)
    remainder = len(features) % FBANK_STACK
    if remainder:
        features = np.concatenate(
            [features, np.zeros((FBANK_STACK - remainder, FBANK_BINS), dtype=np.float32)]
        )
    features = features.reshape(-1, FBANK_DIM)
    return torch.nn.functional.layer_norm(torch.from_numpy(features), (FBANK_DIM,))


def audio_tensor(path, frames: int, sample_rate: int = 16000) -> torch.Tensor:
    """`(1, frames, 104)`: the filterbank trimmed or zero-padded to the video's frame count."""
    features = audio_features(path, sample_rate)
    if features.shape[0] >= frames:
        features = features[:frames]
    else:
        padding = torch.zeros(frames - features.shape[0], features.shape[1])
        features = torch.cat([features, padding], dim=0)
    return features.unsqueeze(0).contiguous()
