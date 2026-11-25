#!/usr/bin/env python3
"""Offline feature pre-extraction script for Direction 3 with MARLIN Facial Video Backbone.

Extracts:
1. Facial video features: MARLIN (Masked Autoencoder for facial video Representation LearnINg, CVPR 2023)
   via `marlin-pytorch` (`marlin_vit_base_ytf` or `marlin_vit_small_ytf`).
   Faces are detected and cropped with MTCNN (matching `cross-modal-fusion` protocol: 16 frames,
   margin=0.25, square crop, resize to 224x224).
   MARLIN outputs (1, 1568, 768) spatiotemporal tokens -> spatial pooling across (14, 14) patches -> (8, 768).
2. Acoustic features: 3-channel Log-Mel spectrogram (S, delta, delta^2) + ConvNeXt-Tiny (8, 768).
3. Multimodal descriptor: Concatenated (8, 1536), saved into `DATA/cached_features/<ENCODER>/<clip_id>.npy`.

Usage:
------
    # Full dataset extraction on Colab GPU:
    python scripts/extract_features_marlin.py --data-dir /content/drive/MyDrive/dolos --encoder marlin_convnext

    # Quick smoke test on 10 clips:
    python scripts/extract_features_marlin.py --data-dir /content/drive/MyDrive/dolos --limit 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Add repo root to import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract MARLIN + ConvNeXt features for Direction 3")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/dolos"),
        help="Path to shared DOLOS dataset directory (contains dataset.csv, clips/, audio/)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="marlin_convnext",
        help="Cache subfolder name under cached_features/",
    )
    parser.add_argument(
        "--marlin-model",
        type=str,
        default="marlin_vit_base_ytf",
        choices=["marlin_vit_small_ytf", "marlin_vit_base_ytf", "marlin_vit_large_ytf"],
        help="Pretrained MARLIN checkpoint from GitHub release",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled video frames per clip (MARLIN expects 16)",
    )
    parser.add_argument(
        "--face-margin",
        type=float,
        default=0.25,
        help="Margin around detected face bounding box (fraction of box side)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to process (for debugging)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing cache files",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Video & Face Cropping Helpers (aligned with cross-modal-fusion protocol)
# --------------------------------------------------------------------------- #

def sample_video_frames(video_path: Path, num_frames: int = 16) -> list[np.ndarray]:
    """Sample `num_frames` frames evenly spread over the video clip in BGR."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0
        while cap.grab():
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))

    if total_frames <= 0:
        total_frames = num_frames

    targets = np.linspace(0, max(total_frames - 1, 1), num_frames).round().astype(int)
    wanted = set(targets.tolist())

    frames, index = [], 0
    max_target = int(targets.max())
    while index <= max_target:
        if not cap.grab():
            break
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
        index += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


def centre_box(frame: np.ndarray) -> tuple[float, float, float, float]:
    """Centred square of the whole frame when no face is detected."""
    h, w = frame.shape[:2]
    side = min(h, w)
    return ((w - side) / 2.0, (h - side) / 2.0, (w + side) / 2.0, (h + side) / 2.0)


def square_crop(frame: np.ndarray, box: tuple[float, float, float, float], size: int = 224, margin: float = 0.25) -> np.ndarray:
    """Crop square face region with margin and resize to (size, size)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = min(max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin), float(min(h, w)))
    side = max(side, 8.0)
    half = side / 2.0

    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    left, top = int(round(cx - half)), int(round(cy - half))

    crop = frame[max(top, 0):min(top + int(round(side)), h),
                 max(left, 0):min(left + int(round(side)), w)]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def load_audio_waveform(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio waveform as a 1D float32 numpy array at target_sr."""
    import torchaudio
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy().astype(np.float32)
    except Exception:
        import scipy.io.wavfile as wav
        sr, data = wav.read(str(audio_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0)
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
    return data


# --------------------------------------------------------------------------- #
# Feature Extractor Class: MARLIN (Faces) + ConvNeXt (Audio)
# --------------------------------------------------------------------------- #

class MarlinFeatureExtractor:
    def __init__(self, marlin_model_name: str = "marlin_vit_base_ytf", face_margin: float = 0.25, device: str = "cuda"):
        import torch
        import torchaudio.transforms as T_audio
        import timm
        from marlin_pytorch import Marlin

        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.face_margin = face_margin
        print(f"Loading MARLIN and Audio models on {self.device}...")

        # 1. Face Detector (MTCNN)
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(keep_all=False, device=self.device, min_face_size=40, factor=0.709, post_process=False)
            print("MTCNN face detector initialized successfully.")
        except ImportError:
            print("Warning: facenet-pytorch not installed. Using centered crop fallback.", file=sys.stderr)
            self.detector = None

        # 2. MARLIN Facial Video Backbone
        self.marlin = Marlin.from_online(marlin_model_name).to(self.device)
        self.marlin.eval()
        self.marlin_dim = 768 if "base" in marlin_model_name else (384 if "small" in marlin_model_name else 1024)
        print(f"MARLIN ({marlin_model_name}) loaded: embedding dimension = {self.marlin_dim}")

        # 3. Audio Backbone (3-channel Mel spectrogram + ConvNeXt-Tiny)
        self.mel_spec = T_audio.MelSpectrogram(
            sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
        ).to(self.device)
        self.amp_to_db = T_audio.AmplitudeToDB().to(self.device)
        self.compute_deltas = T_audio.ComputeDeltas().to(self.device)

        self.audio_model = timm.create_model(
            "convnext_tiny.fb_in22k",
            pretrained=True,
            in_chans=3,
            num_classes=0,
            global_pool="",
        ).to(self.device)
        self.audio_model.eval()
        self.audio_dim = 768

        self.total_dim = self.marlin_dim + self.audio_dim
        print(f"Models ready! Output feature dimension per step: {self.total_dim} ({self.marlin_dim} MARLIN + {self.audio_dim} Audio)")

    def detect_face_box(self, frame_bgr: np.ndarray) -> tuple[float, float, float, float] | None:
        """Detect dominant face bounding box on downscaled frame for speed."""
        if self.detector is None:
            return None
        h, w = frame_bgr.shape[:2]
        detect_side = 640
        scale = detect_side / max(h, w)
        small = frame_bgr if scale >= 1.0 else cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        rgb_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        boxes, probs = self.detector.detect(rgb_img)
        if boxes is None:
            return None
        score = float(np.asarray(probs).reshape(-1)[0]) if probs is not None else 0.0
        if score < 0.85:
            return None
        box = np.asarray(boxes, dtype=np.float64).reshape(-1)[:4] / scale
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    def extract_clip(self, video_path: Path | None, audio_path: Path | None) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        has_video = video_path is not None and video_path.exists()
        has_audio = audio_path is not None and audio_path.exists()

        if not has_video and not has_audio:
            # Impute zeros for missing media clips to satisfy CachedFeatures protocol
            return np.zeros((8, self.total_dim), dtype=np.float32)

        # 1. Process Video with MARLIN
        if has_video:
            raw_frames = sample_video_frames(video_path, num_frames=16)
            face_crops = []
            for frame in raw_frames:
                box = self.detect_face_box(frame)
                if box is None:
                    box = centre_box(frame)
                    crop_bgr = square_crop(frame, box, size=224, margin=0.0)
                else:
                    crop_bgr = square_crop(frame, box, size=224, margin=self.face_margin)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                face_crops.append(crop_rgb)

            # Stack into (T=16, H=224, W=224, C=3) -> permute to (1, C=3, T=16, H=224, W=224)
            crops_arr = np.stack(face_crops, axis=0)  # (16, 224, 224, 3)
            v_tensor = torch.from_numpy(crops_arr).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
            v_tensor = v_tensor.to(self.device)

            with torch.no_grad():
                # MARLIN extract_features with keep_seq=True -> (1, 1568, marlin_dim)
                raw_tokens = self.marlin.extract_features(v_tensor, keep_seq=True)
                # Reshape to 3D grid: 1568 = (nt=8, nh=14, nw=14)
                tokens_grid = raw_tokens.view(1, 8, 14, 14, self.marlin_dim)
                # Spatial pooling across (nh, nw) -> (1, T_v=8, marlin_dim)
                v_seq = tokens_grid.mean(dim=(2, 3))
        else:
            v_seq = torch.zeros((1, 8, self.marlin_dim), device=self.device)

        # 2. Process Audio with ConvNeXt
        if has_audio:
            aud_arr = load_audio_waveform(audio_path, target_sr=16000)
            aud_tensor = torch.from_numpy(aud_arr).unsqueeze(0).to(self.device)

            with torch.no_grad():
                spec = self.mel_spec(aud_tensor)       # (1, n_mels=128, T_audio)
                spec_db = self.amp_to_db(spec)
                delta1 = self.compute_deltas(spec_db)
                delta2 = self.compute_deltas(delta1)
                spec_3ch = torch.stack([spec_db, delta1, delta2], dim=1)  # (1, 3, 128, T_audio)

                feat = self.audio_model.forward_features(spec_3ch)        # (1, 768, H_freq, W_time)
                a_seq = feat.mean(dim=2).permute(0, 2, 1)                 # (1, W_time, 768)

                # Temporal alignment to T_v=8
                a_aligned = F.interpolate(
                    a_seq.permute(0, 2, 1),
                    size=8,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)  # (1, 8, 768)
        else:
            a_aligned = torch.zeros((1, 8, self.audio_dim), device=self.device)

        # 3. Concatenate MARLIN Face Sequence + Audio Sequence
        with torch.no_grad():
            fused = torch.cat([v_seq, a_aligned], dim=-1)
            output = fused.squeeze(0).cpu().numpy().astype(np.float32)

        return output


# --------------------------------------------------------------------------- #
# Main Preprocessing Pipeline
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    data_dir = args.data_dir
    csv_path = data_dir / "dataset.csv"

    if not csv_path.exists():
        print(f"Error: dataset table not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    table = pd.read_csv(csv_path, dtype=str)
    print(f"Loaded {len(table)} rows from {csv_path}")

    # Output cache folder
    cache_dir = data_dir / "cached_features" / args.encoder
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target feature cache: {cache_dir}")

    # Filter to clips needing cache
    valid_clips = []
    for _, row in table.iterrows():
        clip_id = str(row["clip_id"])
        out_file = cache_dir / f"{clip_id}.npy"
        if args.force or not out_file.exists():
            v_path = data_dir / str(row["video_path"]) if pd.notna(row.get("video_path")) else None
            a_path = data_dir / str(row["audio_path"]) if pd.notna(row.get("audio_path")) else None
            valid_clips.append((clip_id, v_path, a_path, out_file))

    print(f"Clips to process: {len(valid_clips)} (skipped {len(table) - len(valid_clips)} already cached)")

    if args.limit:
        valid_clips = valid_clips[:args.limit]
        print(f"Applying limit: processing first {len(valid_clips)} clips.")

    if not valid_clips:
        print("All clips are already cached! Nothing to do.")
        return

    # Initialize Extractor
    extractor = MarlinFeatureExtractor(
        marlin_model_name=args.marlin_model,
        face_margin=args.face_margin,
        device=args.device
    )

    start_time = time.time()
    success_count = 0
    error_count = 0

    pbar = tqdm(valid_clips, desc=f"Extracting MARLIN ({args.marlin_model}) Features")
    for clip_id, v_path, a_path, out_file in pbar:
        try:
            feat_arr = extractor.extract_clip(v_path, a_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_file, feat_arr)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\n[Warning] Failed on {clip_id}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    avg_speed = elapsed / max(success_count, 1)
    print("\nExtraction Summary:")
    print(f"  Successfully extracted: {success_count} clips")
    print(f"  Errors: {error_count} clips")
    print(f"  Total time: {elapsed:.1f} s ({avg_speed:.2f} s/clip)")
    print(f"  Cache directory: {cache_dir}")


if __name__ == "__main__":
    main()
