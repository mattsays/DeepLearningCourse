"""
extract_features.py
Extracts per-frame visual (MediaPipe Tasks FaceLandmarker, 15 pooled regions)
and audio (openSMILE eGeMAPS subset) features from a paired clips/audio
directory layout and saves a (T, 53) numpy array per clip.

Expected input layout:
    <input_folder>/
    ├── clips/      # video files (.mp4, .avi, .mov, .mkv, .webm)
    └── audio/      # matching audio files (.wav, .flac, .ogg, .mp3, .m4a)
Each clip is paired with audio by basename (stem).

Notes:
  - A fresh FaceLandmarker is created per video, because MediaPipe's internal
    temporal cache cannot be reset between files (causes 'Input timestamp must
    be monotonically increasing' errors otherwise).
  - Already-processed clips (existing <stem>_feats.npy) are skipped, so the
    script is safe to re-run after a failure.
"""

import argparse
import gc
import os
import tempfile
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import opensmile
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from scipy.interpolate import interp1d

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"

LANDMARK_REGIONS = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 2],
    "nose_tip": [19, 94, 2, 98, 327, 326, 97],
    "mouth_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409],
    "mouth_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415],
    "jaw_left": [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
    "jaw_right": [58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284],
    "chin": [199, 175, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 18, 200, 421, 313],
    "left_cheek": [116, 123, 147, 187, 207, 206, 216, 212, 202, 204, 194, 32, 211, 210, 214, 192],
    "right_cheek": [345, 352, 376, 411, 427, 426, 436, 432, 422, 424, 418, 262, 431, 430, 434, 416],
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378],
    "iris_center": [468, 473],
}


def _ensure_model(dest_dir: str) -> str:
    """Download the face_landmarker model if not already present."""
    dest = os.path.join(dest_dir, MODEL_FILENAME)
    if not os.path.exists(dest):
        print(f"Downloading {MODEL_FILENAME} from {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, dest)
        print(f"Saved model to {dest}")
    return dest


def _build_landmarker(model_path: str) -> vision.FaceLandmarker:
    """Create a fresh FaceLandmarker with explicit CPU delegate."""
    base_options = python.BaseOptions(
        model_asset_path=model_path,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _safe_mean(coords: np.ndarray, indices: list[int]) -> np.ndarray:
    valid = [i for i in indices if i < coords.shape[0]]
    if not valid:
        return np.zeros(coords.shape[1], dtype=np.float32)
    return coords[valid].mean(axis=0).astype(np.float32)


def extract_landmark_regions(
    video_path: str,
    landmarker: vision.FaceLandmarker,
    target_fps: float = 10.0,
    max_frames: int | None = None,
) -> np.ndarray:
    """
    Returns [T_video, 15*3] pooled facial-region coordinates.
    Timestamps increment deterministically (1/target_fps per sampled frame),
    starting from 0 — correct because the landmarker is fresh per video.
    """
    cap = cv2.VideoCapture(video_path)
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    if fps_orig <= 0:
        fps_orig = 30.0
    step = max(1, int(round(fps_orig / target_fps)))
    ts_step_ms = max(1, int(round(1000.0 / target_fps)))

    region_names = list(LANDMARK_REGIONS.keys())
    num_regions = len(region_names)
    landmark_dim = 3
    frames_out = []
    last_valid = np.zeros(num_regions * landmark_dim, dtype=np.float32)

    ts_ms = 0
    frame_idx = 0
    sampled = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            if max_frames is not None and sampled >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            ts_ms += ts_step_ms

            if result.face_landmarks and len(result.face_landmarks) > 0:
                face = result.face_landmarks[0]
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face], dtype=np.float32
                )
                pooled = np.concatenate([
                    _safe_mean(coords, LANDMARK_REGIONS[name])
                    for name in region_names
                ])
                last_valid = pooled
            frames_out.append(last_valid.copy())
            sampled += 1
            del frame, rgb, mp_image, result
        else:
            del frame
        frame_idx += 1

    cap.release()
    del cap
    gc.collect()

    if not frames_out:
        return np.zeros((0, num_regions * landmark_dim), dtype=np.float32)
    return np.stack(frames_out, axis=0)


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio
    g = gcd(int(sr), 16000)
    up = 16000 // g
    down = int(sr) // g
    return resample_poly(audio, up, down).astype(np.float32)


def extract_audio_features(audio_path: str, target_fps: float = 10.0,
                           n_features: int = 8) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return np.zeros((0, n_features), dtype=np.float32)

    audio = _resample_to_16k(audio, sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    try:
        sf.write(wav_path, audio, 16000, subtype="PCM_16")
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            sampling_rate=16000,
        )
        df = smile.process_file(wav_path)
        if df.empty:
            return np.zeros((0, n_features), dtype=np.float32)

        feat_cols = df.columns[:n_features].tolist()
        values = df[feat_cols].to_numpy(dtype=np.float32)

        # openSMILE LLD output uses a MultiIndex of (start, end) timedeltas.
        idx = df.index
        if isinstance(idx, pd.MultiIndex):
            if "start" in (idx.names or []):
                start_index = idx.get_level_values("start")
            else:
                start_index = idx.get_level_values(0)
        else:
            start_index = idx
        times = pd.to_timedelta(start_index).total_seconds().to_numpy(dtype=np.float32)

        t_max = times[-1] if len(times) > 1 else 0.0
        if t_max <= 0:
            return np.zeros((1, n_features), dtype=np.float32)
        grid = np.arange(0, t_max, 1.0 / target_fps, dtype=np.float32)
        interp = interp1d(times, values, axis=0,
                          kind="linear", bounds_error=False,
                          fill_value=(values[0], values[-1]))
        return interp(grid).astype(np.float32)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def extract_video_features(
    video_path: str,
    audio_path: str,
    landmarker: vision.FaceLandmarker,
    target_fps: float = 10.0,
    n_audio: int = 8,
    max_frames: int | None = None,
) -> np.ndarray:
    lm = extract_landmark_regions(video_path, landmarker, target_fps, max_frames)
    au = extract_audio_features(audio_path, target_fps, n_audio)

    if lm.shape[0] == 0 and au.shape[0] == 0:
        return np.zeros((0, 45 + n_audio), dtype=np.float32)
    if lm.shape[0] == 0:
        lm = np.zeros((au.shape[0], 45), dtype=np.float32)
    if au.shape[0] == 0:
        au = np.zeros((lm.shape[0], n_audio), dtype=np.float32)

    T = min(lm.shape[0], au.shape[0])
    if T == 0:
        return np.zeros((0, 45 + n_audio), dtype=np.float32)
    return np.concatenate([lm[:T], au[:T]], axis=-1)


def _find_audio(audio_dir: str, stem: str) -> str | None:
    for ext in AUDIO_EXTS:
        candidate = os.path.join(audio_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Tasks + openSMILE features."
    )
    parser.add_argument("--input", required=True,
                        help="Folder containing 'clips/' and 'audio/' subfolders.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write {video_id}_feats.npy files.")
    parser.add_argument("--model_dir", default="./models",
                        help="Directory to cache face_landmarker.task.")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target sampling fps (default: 10).")
    parser.add_argument("--n_audio", type=int, default=8,
                        help="Number of openSMILE LLD features to keep (default: 8).")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Diagnostic: cap sampled frames per video.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess clips even if _feats.npy already exists.")
    args = parser.parse_args()

    clips_dir = os.path.join(args.input, "clips")
    audio_dir = os.path.join(args.input, "audio")
    if not os.path.isdir(clips_dir):
        raise SystemExit(f"'clips' subfolder not found in {args.input}")
    if not os.path.isdir(audio_dir):
        raise SystemExit(f"'audio' subfolder not found in {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = _ensure_model(args.model_dir)

    clip_files = sorted(
        f for f in os.listdir(clips_dir) if f.lower().endswith(VIDEO_EXTS)
    )
    if not clip_files:
        print(f"No video files found in {clips_dir}")
        return

    # Skip clips that already have a cached feature file
    if not args.overwrite:
        before = len(clip_files)
        clip_files = [
            f for f in clip_files
            if not os.path.exists(
                os.path.join(args.output_dir,
                             os.path.splitext(f)[0] + "_feats.npy")
            )
        ]
        skipped_existing = before - len(clip_files)
        if skipped_existing:
            print(f"Skipping {skipped_existing} already-processed clip(s).")
        if not clip_files:
            print("Nothing to do.")
            return

    print(f"Processing {len(clip_files)} clip(s). "
          f"Landmarker will be recreated per video.")

    missing_audio = []
    processed, skipped = 0, 0

    for i, clip_name in enumerate(clip_files, 1):
        stem = os.path.splitext(clip_name)[0]
        clip_path = os.path.join(clips_dir, clip_name)
        audio_path = _find_audio(audio_dir, stem)

        if audio_path is None:
            missing_audio.append(stem)
            skipped += 1
            print(f"[{i}/{len(clip_files)}] [skip] {stem}: no matching audio")
            continue

        out_path = os.path.join(args.output_dir, f"{stem}_feats.npy")
        try:
            # Fresh landmarker per video: avoids stale temporal-cache state.
            landmarker = _build_landmarker(model_path)
            try:
                feats = extract_video_features(
                    clip_path, audio_path, landmarker,
                    target_fps=args.fps,
                    n_audio=args.n_audio,
                    max_frames=args.max_frames,
                )
            finally:
                landmarker.close()

            np.save(out_path, feats)
            print(f"[{i}/{len(clip_files)}] [ok] {stem}: shape={feats.shape}")
            processed += 1
            del feats
            gc.collect()
        except Exception as e:
            skipped += 1
            print(f"[{i}/{len(clip_files)}] [fail] {stem}: {e}")

    print(f"\nDone: {processed} processed, {skipped} skipped.")
    if missing_audio:
        print(f"Missing audio for {len(missing_audio)} clips: {missing_audio[:10]}"
              + (" ..." if len(missing_audio) > 10 else ""))


if __name__ == "__main__":
    main()