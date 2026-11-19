#!/usr/bin/env python
"""Extract AV-HuBERT features for the `avh` direction.

Runs **inside the Python 3.8 environment** the notebook provisions, because fairseq 0.12.2 is
the only release that ships a wheel for the checkpoint's task and it stops at cp38. Keeping the
extraction in a subprocess also keeps the main Colab kernel free of a 2022-era dependency.

    python scripts/extract_avhubert.py --data <shared folder> \
        --checkpoint <base_vox_iter5.pt> --landmarks <shape_predictor_68.dat> [options]

Writes one `.npy` per clip per pass into `<data>/cached_features/avhubert-{video,audio}/`, with a
`manifest.json` that records everything that changes the content of an array. Resumable: an
existing array is skipped, `--force` rebuilds.

Both passes use the same checkpoint with `model.modalities` switched; AV-HuBERT was trained with
modality dropout, so a mouth-only and an audio-only forward are both in-distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# the direction owns the preprocessing and the encoder wrapper; import them, never copy them
REPO = Path(__file__).resolve().parents[1]
DIRECTION = REPO / "notebooks" / "architectures" / "avhubert"
sys.path.insert(0, str(DIRECTION))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

import model as M                                           # noqa: E402
import preprocess as P                                      # noqa: E402

TOKENS = 64


def clip_ids_from_splits(data: Path) -> list[str]:
    """Union of the five subject folds and the reserved test set, stable order.

    Read straight from the frozen JSONs so the extraction cannot disagree with the protocol.
    """
    ids = []
    splits = data / "splits"
    for path in sorted(splits.glob("subject_fold*.json")) + [splits / "subject_test.json"]:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for block in payload.values():
            if isinstance(block, dict) and "clips" in block:
                ids += block["clips"]
        if "clips" in payload:
            ids += payload["clips"]
    return list(dict.fromkeys(ids))


def read_dataset(data: Path) -> dict[str, dict]:
    """`clip_id -> {video, audio, status}` from `dataset.csv`, paths resolved."""
    rows = {}
    with open(data / "dataset.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            video = Path(row["video_path"]) if row.get("video_path") else None
            audio = Path(row["audio_path"]) if row.get("audio_path") else None
            rows[row["clip_id"]] = {
                "video": (data / video) if video else None,
                "audio": (data / audio) if audio else None,
                "status": row.get("status", ""),
            }
    return rows


class Cache:
    """One folder of `<clip_id>.npy` plus the manifest that says how it was built."""

    def __init__(self, directory: Path, meta: dict, force: bool = False):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.json"
        for stale in self.dir.glob(".*.tmp.npy"):
            stale.unlink()
        previous = json.loads(self.manifest.read_text()) if self.manifest.exists() else None
        if force and previous:
            for path in self.dir.glob("*.npy"):
                path.unlink()
            previous = None
        if previous and previous.get("settings") != meta:
            raise RuntimeError(
                f"{self.dir.name}: settings differ from the ones this folder was built with. "
                "Re-run with --force to rebuild."
            )
        self.meta = previous or {"settings": meta, "count": 0, "failures": {}}
        if not previous:
            self.meta["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.save()

    def save(self) -> None:
        self.manifest.write_text(json.dumps(self.meta, indent=2))

    def has(self, clip_id: str) -> bool:
        return (self.dir / f"{clip_id}.npy").exists()

    def write(self, clip_id: str, array: np.ndarray) -> None:
        temporary = self.dir / f".{clip_id}.tmp.npy"
        np.save(temporary, array.astype(np.float16))
        temporary.rename(self.dir / f"{clip_id}.npy")

    def pending(self, clip_ids: list[str]) -> list[str]:
        return [clip_id for clip_id in clip_ids if not self.has(clip_id)]


def self_test(args, dataset, ids) -> int:
    """Preprocessing only: one clip through crop and filterbank, no encoder, no cache."""
    detector, predictor = P.load_landmarks(args.landmarks)
    for clip_id in ids[:1]:
        video = P.video_tensor(dataset[clip_id]["video"], detector, predictor)
        audio = P.audio_tensor(dataset[clip_id]["audio"], video.shape[1])
        print(f"{clip_id}: video {tuple(video.shape)} | audio {tuple(audio.shape)}")
        print(f"  video range {video.min():.2f}..{video.max():.2f} | "
              f"audio std {audio.std():.3f}")
        if args.dump_grid:
            import cv2
            frames = (video[0, ::5, 0].numpy() * 255).astype(np.uint8)
            columns = 4
            rows = (len(frames) + columns - 1) // columns
            grid = np.zeros((rows * P.CROP, columns * P.CROP), np.uint8)
            for index, frame in enumerate(frames[:rows * columns]):
                row, column = divmod(index, columns)
                grid[row * P.CROP:(row + 1) * P.CROP,
                     column * P.CROP:(column + 1) * P.CROP] = frame
            cv2.imwrite(str(args.dump_grid), grid)
            print("  wrote", args.dump_grid)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="shared folder with clips/, dataset.csv, splits/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True,
                        help="shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N clips")
    parser.add_argument("--force", action="store_true", help="rebuild both caches")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--self-test", action="store_true", help="preprocessing only, no encoder")
    parser.add_argument("--dump-grid", type=Path, default=None,
                        help="with --self-test: write a PNG grid of mouth crops")
    args = parser.parse_args(argv)

    data = args.data.resolve()
    dataset = read_dataset(data)
    ids = [clip_id for clip_id in clip_ids_from_splits(data) if clip_id in dataset]
    ids = [clip_id for clip_id in ids
           if dataset[clip_id]["video"] and dataset[clip_id]["video"].exists()
           and dataset[clip_id]["audio"] and dataset[clip_id]["audio"].exists()]
    print(f"{len(ids)} clips with video and audio in {data}")

    if args.self_test:
        return self_test(args, dataset, ids)

    device = args.device if torch.cuda.is_available() else "cpu"
    detector, predictor = P.load_landmarks(args.landmarks)
    encoder, width = M.load_avhubert(args.checkpoint, device=device,
                                     modalities=("video", "audio"))
    print(f"AV-HuBERT loaded | width {width} | device {device}")

    common = {"checkpoint": args.checkpoint.name, "weights": "pretrained",
              "fps": P.FPS, "tokens": args.tokens, "width": width,
              "alignment": "dlib68_mouth_bbox_no_meanface", "dtype": "float16"}
    cache_v = Cache(data / "cached_features" / "avhubert-video",
                    {**common, "encoder": "avhubert-video", "modality": "video",
                     "crop": P.CROP}, force=args.force)
    cache_a = Cache(data / "cached_features" / "avhubert-audio",
                    {**common, "encoder": "avhubert-audio", "modality": "audio",
                     "fbank": [P.FBANK_BINS, P.FBANK_STACK]}, force=args.force)

    pending = set(cache_v.pending(ids)) | set(cache_a.pending(ids))
    todo = [clip_id for clip_id in ids if clip_id in pending]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} of {len(ids)} clips to encode")

    started = time.perf_counter()
    try:
        for position, clip_id in enumerate(todo, 1):
            try:
                video = P.video_tensor(dataset[clip_id]["video"], detector, predictor)
                audio = P.audio_tensor(dataset[clip_id]["audio"], video.shape[1])

                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=device == "cuda"):
                    encoder.modalities = ["video"]
                    with torch.no_grad():
                        features_v = M.encode_avhubert(encoder, video=video.to(device))
                    cache_v.write(clip_id, M.pool_avhubert(features_v, args.tokens)[0].numpy())

                    encoder.modalities = ["audio"]
                    with torch.no_grad():
                        features_a = M.encode_avhubert(encoder, audio=audio.to(device))
                    cache_a.write(clip_id, M.pool_avhubert(features_a, args.tokens)[0].numpy())

                cache_v.meta["failures"].pop(clip_id, None)
                cache_a.meta["failures"].pop(clip_id, None)
            except Exception as error:                  # one clip, never the run
                message = f"{type(error).__name__}: {str(error)[:180]}"
                cache_v.meta["failures"][clip_id] = message
                cache_a.meta["failures"][clip_id] = message
                print(f"  {clip_id}: {message[:100]}", flush=True)

            if position % 10 == 0 or position == len(todo):
                rate = position / (time.perf_counter() - started)
                cache_v.save()
                cache_a.save()
                print(f"  {position}/{len(todo)}  {rate:.2f} clips/s  "
                      f"eta {(len(todo) - position) / rate / 60:.0f} min", flush=True)
    except KeyboardInterrupt:
        print("interrupted: what is written stays, rerun to continue")

    for cache in (cache_v, cache_a):
        cache.meta["count"] = len(list(cache.dir.glob("*.npy")))
        cache.meta["dataset_fingerprint"] = _fingerprint(data / "dataset.csv")
        cache.save()
        print(f"{cache.dir.name}: {cache.meta['count']} arrays, "
              f"{len(cache.meta['failures'])} failures")
    return 0


def _fingerprint(path: Path) -> str:
    import hashlib

    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    sys.exit(main())
