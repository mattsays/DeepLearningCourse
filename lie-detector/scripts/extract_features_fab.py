#!/usr/bin/env python3
"""Offline feature pre-extraction script for Direction 3 (FAB).

Extracts spatio-temporal video features (Swin3D / X3D) and acoustic features
(3-channel Log-Mel spectrogram + ConvNeXt-Tiny) and saves aligned temporal
sequences to `DATA/cached_features/<ENCODER>/<clip_id>.npy`.

Why pre-extract offline:
-------------------------
Decoding 1,675 clips across Google Drive's network filesystem during every
training epoch takes 20-40 minutes per epoch and hits Colab timeouts.
With pre-extracted sequence features (~400 MB total), `CachedFeatures`
preloads the entire dataset directly into RAM at startup, reducing training
time to 2-3 seconds per epoch!

Output representation per clip:
-------------------------------
Shape: (T_v=8, 1536) float32 array
- channels [0:768]    : Swin3D spatio-temporal video descriptors
- channels [768:1536] : ConvNeXt acoustic descriptors (aligned via 1D interpolation)

Usage:
------
    # Full dataset run on Colab GPU:
    python scripts/extract_features_fab.py --data-dir /content/drive/MyDrive/dolos

    # Quick smoke test on 10 clips:
    python scripts/extract_features_fab.py --data-dir /content/drive/MyDrive/dolos --limit 10

    # Overwrite existing cache:
    python scripts/extract_features_fab.py --data-dir /content/drive/MyDrive/dolos --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser(description="Pre-extract features for Direction 3 (FAB)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/dolos"),
        help="Path to shared DOLOS dataset directory (contains dataset.csv, clips/, audio/)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="swin3d_convnext_3ch",
        help="Cache subfolder name under cached_features/",
    )
    parser.add_argument(
        "--video-backbone",
        type=str,
        default="swin3d_t",
        choices=["swin3d_t", "x3d_m"],
        help="3D video backbone model",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled video frames per clip",
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
# Media Loading Helpers (robust with fallback)
# --------------------------------------------------------------------------- #

def sample_video_frames(video_path: Path, num_frames: int = 16) -> np.ndarray:
    """Sample `num_frames` uniformly from video and return (3, T, 224, 224) float32 normalized."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
        current_idx = 0
        target_pos = 0

        while cap.isOpened() and target_pos < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx == indices[target_pos]:
                # Convert BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                frames.append(resized)
                target_pos += 1
                # Skip duplicate target indices if video is very short
                while target_pos < len(indices) and indices[target_pos] == current_idx:
                    frames.append(resized)
                    target_pos += 1
            current_idx += 1
    else:
        # Fallback: read all available frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(resized)
        if len(frames) > 0:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be decoded from {video_path}")

    # Pad if video has fewer frames than requested
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Convert to array (T, H, W, C) -> (C, T, H, W) normalized
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = np.transpose(arr, (3, 0, 1, 2))  # (C, T, H, W)
    return tensor


def load_audio_waveform(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio waveform as mono float32 at target sample rate."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            import torchaudio.transforms as T
            wav = T.Resample(sr, target_sr)(wav)
        return wav.squeeze(0).numpy()
    except Exception:
        # Fallback using scipy.io.wavfile
        from scipy.io import wavfile
        sr, data = wavfile.read(str(audio_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
        return data


# --------------------------------------------------------------------------- #
# Feature Extractor Class
# --------------------------------------------------------------------------- #

class FabFeatureExtractor:
    def __init__(self, video_backbone_name: str = "swin3d_t", device: str = "cuda"):
        import torch
        import torchaudio.transforms as T_audio
        import torchvision.models.video as video_models
        import timm

        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"Loading extractor models on {self.device}...")

        # 1. Video Backbone
        if video_backbone_name == "swin3d_t":
            self.video_model = video_models.swin3d_t(weights="DEFAULT").to(self.device)
            self.video_model.head = torch.nn.Identity()
            self.video_model.avgpool = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Video backbone {video_backbone_name} not yet implemented.")
        self.video_model.eval()

        # 2. Audio Preprocessing Transforms
        self.mel_spec = T_audio.MelSpectrogram(
            sample_rate=16000, n_fft=1024, win_length=1024, hop_length=512, n_mels=128
        ).to(self.device)
        self.amp_to_db = T_audio.AmplitudeToDB().to(self.device)
        self.compute_deltas = T_audio.ComputeDeltas().to(self.device)

        # 3. Audio Backbone (ConvNeXt-Tiny with 3 input channels)
        self.audio_model = timm.create_model(
            "convnext_tiny.fb_in22k",
            pretrained=True,
            in_chans=3,
            num_classes=0,
            global_pool="",
        ).to(self.device)
        self.audio_model.eval()

        print("Models successfully loaded and ready.")

    def extract_clip(self, video_path: Path, audio_path: Path) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        # 1. Process Video
        vid_arr = sample_video_frames(video_path, num_frames=16)
        vid_tensor = torch.from_numpy(vid_arr).unsqueeze(0).to(self.device)  # (1, 3, 16, 224, 224)

        with torch.no_grad():
            # Swin3D feature stages preserving time axis
            x = self.video_model.patch_embed(vid_tensor)
            x = self.video_model.pos_drop(x)
            x = self.video_model.features(x)
            x = self.video_model.norm(x)
            # x shape: (1, T_out=8, H_out=7, W_out=7, C=768)
            # Spatial pooling across H and W only
            v_seq = x.mean(dim=[2, 3])  # (1, T_v=8, 768)

        # 2. Process Audio
        aud_arr = load_audio_waveform(audio_path, target_sr=16000)
        aud_tensor = torch.from_numpy(aud_arr).unsqueeze(0).to(self.device)  # (1, Samples)

        with torch.no_grad():
            spec = self.mel_spec(aud_tensor)       # (1, n_mels=128, T_audio)
            spec_db = self.amp_to_db(spec)
            delta1 = self.compute_deltas(spec_db)
            delta2 = self.compute_deltas(delta1)
            # Stack into 3-channel pseudo-RGB spectrogram
            spec_3ch = torch.stack([spec_db, delta1, delta2], dim=1)  # (1, 3, 128, T_audio)

            # Pass into ConvNeXt-Tiny
            feat = self.audio_model.forward_features(spec_3ch)  # (1, 768, H_freq, W_time)
            # Pool frequency axis only, preserve time axis
            a_seq = feat.mean(dim=2).permute(0, 2, 1)  # (1, W_time, 768)

            # 3. Align temporal lengths (interpolate audio time to match video time T_v=8)
            a_aligned = F.interpolate(
                a_seq.permute(0, 2, 1),
                size=v_seq.size(1),
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # (1, T_v=8, 768)

            # 4. Concatenate multimodal descriptors: (1, 8, 1536)
            fused = torch.cat([v_seq, a_aligned], dim=-1)
            output = fused.squeeze(0).cpu().numpy().astype(np.float32)

        return output


# --------------------------------------------------------------------------- #
# Main Execution Pipeline
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

    # Filter to available clips with both video and audio
    valid_clips = []
    for _, row in table.iterrows():
        clip_id = str(row["clip_id"])
        v_path = data_dir / str(row["video_path"]) if pd.notna(row.get("video_path")) else None
        a_path = data_dir / str(row["audio_path"]) if pd.notna(row.get("audio_path")) else None

        if v_path and a_path and v_path.exists() and a_path.exists():
            out_file = cache_dir / f"{clip_id}.npy"
            if args.force or not out_file.exists():
                valid_clips.append((clip_id, v_path, a_path, out_file))

    print(f"Clips to process: {len(valid_clips)} (skipped {len(table) - len(valid_clips)} already cached or missing media)")

    if args.limit:
        valid_clips = valid_clips[:args.limit]
        print(f"Applying limit: processing first {len(valid_clips)} clips.")

    if not valid_clips:
        print("All clips are already cached! Nothing to do.")
        return

    # Initialize Extractor
    extractor = FabFeatureExtractor(video_backbone_name=args.video_backbone, device=args.device)

    start_time = time.time()
    success_count = 0
    error_count = 0

    pbar = tqdm(valid_clips, desc="Extracting Direction 3 Features")
    for clip_id, v_path, a_path, out_file in pbar:
        try:
            feat_arr = extractor.extract_clip(v_path, a_path)
            # Ensure parent directory exists and save atomic numpy array
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
