"""Face-frame cache for the `lore` direction.

For every clip:

1. ``frames`` (32) frame indices spread uniformly over the whole clip, decoded with OpenCV;
2. faces detected with OpenCV's YuNet (``cv2.FaceDetectorYN``);
3. the speaker selected by **identity**: every face is embedded with OpenCV's SFace recogniser
   (``cv2.FaceRecognizerSF``), faces are clustered by cosine similarity, and the identity seen in
   the most frames is the speaker. Clips often open or cut to another contestant (a reaction
   shot); those slots are filled with the nearest frame that shows the speaker, so the model never
   sees someone else's face. Without the SFace model the fallback is an overlap tracker (start from
   the largest face, follow the overlapping box, re-anchor on the largest face after a cut);
4. boxes median-smoothed over time, a square crop of ``scale`` x the face size, padded with black
   at the frame border, resized to ``size`` x ``size``;
5. the crops stored as ONE sprite JPEG per clip (rows of 8 frames, row-major) plus a small JSON
   with the statistics. ``index.csv`` collects them.

Why a sprite: one small file per clip, quick to copy to Colab and to decode, and a human can open
it to see exactly what the video model sees.

``mode="full"`` skips detection and stores a centre square crop of the whole frame instead
(the standard VideoMAE preprocessing) — the full-frame ablation.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
SPRITE_COLS = 8


@dataclass
class FaceCacheConfig:
    frames: int = 32          # frames stored per clip
    size: int = 224           # side of each stored frame (VideoMAE input size)
    scale: float = 1.6        # crop side = scale x face size (face + hair + chin + a bit of neck)
    min_crop: int = 64        # never crop smaller than this, in source pixels
    score: float = 0.6        # YuNet confidence threshold
    det_max_side: int = 640   # frames are downscaled to this before detection
    smooth: int = 5           # median filter window over the track, in sampled frames
    cut_iou: float = 0.3      # consecutive boxes overlapping less than this = camera cut (no smoothing across)
    min_iou: float = 0.1      # overlap tracker only: below this we assume a camera cut
    identity: bool = True     # pick the speaker by face identity (SFace); False = overlap tracker
    id_threshold: float = 0.363   # SFace cosine similarity for "same person" (OpenCV's reference value)
    jpeg_quality: int = 90
    mode: str = "face"        # "face" | "full"

    def cache_name(self) -> str:
        prefix = "faces-yunet" if self.mode == "face" else "fullframe"
        return f"{prefix}-t{self.frames}-s{self.size}"


# --------------------------------------------------------------------------- #
# sprite helpers (also used by the dataset)
# --------------------------------------------------------------------------- #
def make_sprite(frames: list[np.ndarray]) -> np.ndarray:
    """``T`` frames of ``(S, S, 3)`` -> one ``(T/8*S, 8*S, 3)`` image, row-major."""
    if len(frames) % SPRITE_COLS:
        raise ValueError(f"number of frames must be a multiple of {SPRITE_COLS}")
    rows = [np.concatenate(frames[i:i + SPRITE_COLS], axis=1) for i in range(0, len(frames), SPRITE_COLS)]
    return np.concatenate(rows, axis=0)


def split_sprite(sprite: np.ndarray, frames: int, size: int) -> np.ndarray:
    """Inverse of :func:`make_sprite`: ``(rows*S, 8*S, 3)`` -> ``(T, S, S, 3)``."""
    rows = frames // SPRITE_COLS
    if sprite.shape[0] != rows * size or sprite.shape[1] != SPRITE_COLS * size:
        raise ValueError(f"sprite shape {sprite.shape} does not match {frames} frames of {size}px")
    grid = sprite.reshape(rows, size, SPRITE_COLS, size, 3).transpose(0, 2, 1, 3, 4)
    return grid.reshape(frames, size, size, 3)


def read_image(path: Path) -> np.ndarray:
    """``cv2.imread`` that also works with non-ASCII Windows paths. Returns BGR uint8."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    Path(path).write_bytes(buffer.tobytes())


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _ensure_model(path: Path | str, url: str) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    if path.stat().st_size < 100_000:
        raise RuntimeError(f"{path} is a Git LFS pointer, not the ONNX model — download it by hand")
    return path


def ensure_yunet(path: Path | str) -> Path:
    """Download the YuNet face detector once (~230 KB, opencv_zoo)."""
    return _ensure_model(path, YUNET_URL)


def ensure_sface(path: Path | str) -> Path:
    """Download the SFace face recogniser once (~37 MB, opencv_zoo)."""
    return _ensure_model(path, SFACE_URL)


def sample_indices(n_frames: int, count: int) -> np.ndarray:
    return np.linspace(0, max(n_frames - 1, 0), count).round().astype(int)


def _grab(path: Path, wanted: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    """Decode sequentially, keep only ``wanted`` frames. Returns (frames, frames actually read)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    targets = set(int(i) for i in wanted)
    last = max(targets)
    out: dict[int, np.ndarray] = {}
    index = 0
    while index <= last:
        if not cap.grab():
            break
        if index in targets:
            ok, frame = cap.retrieve()
            if ok:
                out[index] = frame
        index += 1
    cap.release()
    return out, index


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def read_frames(path: Path, count: int) -> tuple[list[np.ndarray], dict]:
    """``count`` BGR frames sampled uniformly over the clip, plus decoding info."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    wanted = sample_indices(n_frames, count)
    frames, _ = _grab(path, wanted)
    if len(frames) < len(set(wanted.tolist())):
        # the container reported more frames than it holds: count them and sample again
        n_frames = _count_frames(path)
        if n_frames == 0:
            raise RuntimeError(f"no decodable frame in {path}")
        wanted = sample_indices(n_frames, count)
        frames, _ = _grab(path, wanted)
    available = sorted(frames)
    if not available:
        raise RuntimeError(f"no decodable frame in {path}")
    picked = [frames[i] if i in frames else frames[min(available, key=lambda a: abs(a - i))]
              for i in wanted.tolist()]
    height, width = picked[0].shape[:2]
    return picked, {"n_frames": n_frames, "fps": round(fps, 3), "width": width, "height": height,
                    "indices": wanted.tolist()}


# --------------------------------------------------------------------------- #
# detection and tracking
# --------------------------------------------------------------------------- #
class FaceDetector:
    """YuNet wrapper returning ``(k, 15)`` rows in source pixels: x, y, w, h, 5 landmarks (x, y), score."""

    def __init__(self, model_path: Path | str, score: float = 0.6, max_side: int = 640):
        self.net = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, 0.3, 50)
        self.max_side = max_side

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        image = frame
        if scale < 1.0:
            image = cv2.resize(frame, (round(width * scale), round(height * scale)),
                               interpolation=cv2.INTER_AREA)
        self.net.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.net.detect(image)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), dtype=np.float32)
        rows = faces[:, :15].astype(np.float32).copy()
        rows[:, :14] /= scale
        return rows


class FaceIdentifier:
    """SFace wrapper: one L2-normalised 128-d identity embedding per detected face."""

    def __init__(self, model_path: Path | str):
        self.net = cv2.FaceRecognizerSF.create(str(model_path), "")

    def __call__(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray:
        aligned = self.net.alignCrop(frame, row.astype(np.float32))
        feature = self.net.feature(aligned).reshape(-1).astype(np.float32)
        return feature / (np.linalg.norm(feature) + 1e-8)


def select_speaker(frames: list[np.ndarray], detections: list[np.ndarray],
                   identifier: FaceIdentifier, threshold: float):
    """Pick the speaker by identity. Returns ``(src, boxes, stats)`` or ``None`` if no face at all.

    ``src[i]`` is the sampled frame whose pixels fill slot ``i`` (itself when the speaker is
    visible, otherwise the nearest frame where they are) and ``boxes[i]`` the speaker's box there.
    """
    items = [(i, j) for i, dets in enumerate(detections) for j in range(len(dets))]
    if not items:
        return None
    features = np.stack([identifier(frames[i], detections[i][j]) for i, j in items])

    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    for index, feature in enumerate(features):          # greedy clustering, frame order
        if centroids:
            similarity = np.array([centroid @ feature for centroid in centroids])
            best = int(similarity.argmax())
            if similarity[best] >= threshold:
                members[best].append(index)
                mean = features[members[best]].mean(axis=0)
                centroids[best] = mean / (np.linalg.norm(mean) + 1e-8)
                continue
        centroids.append(feature.copy())
        members.append([index])

    def frames_of(cluster: int) -> set:
        return {items[m][0] for m in members[cluster]}

    def area_of(cluster: int) -> float:
        return float(np.median([detections[items[m][0]][items[m][1]][2] * detections[items[m][0]][items[m][1]][3]
                                for m in members[cluster]]))

    speaker = max(range(len(members)), key=lambda c: (len(frames_of(c)), area_of(c)))
    chosen: list = [None] * len(frames)
    for index in members[speaker]:
        i, j = items[index]
        similarity = float(features[index] @ centroids[speaker])
        if chosen[i] is None or similarity > chosen[i][1]:
            chosen[i] = (detections[i][j][:4], similarity)
    present = [i for i in range(len(frames)) if chosen[i] is not None]
    src = [i if chosen[i] is not None else min(present, key=lambda p: abs(p - i)) for i in range(len(frames))]
    boxes = [chosen[s][0] for s in src]
    stats = {"identities": len(members), "speaker_rate": round(len(present) / len(frames), 3),
             "held": len(frames) - len(present)}
    return src, boxes, stats


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def track_speaker(detections: list[np.ndarray], min_iou: float = 0.1) -> tuple[list, int]:
    """One box per frame (or None). Returns (boxes, number of re-anchors after a cut)."""
    chosen: list = [None] * len(detections)
    previous = None
    resets = 0
    for i, dets in enumerate(detections):
        if len(dets) == 0:
            continue
        if previous is not None:
            overlaps = np.array([_iou(previous, box) for box in dets[:, :4]])
            best = int(overlaps.argmax())
            if overlaps[best] >= min_iou:
                chosen[i] = dets[best, :4]
                previous = chosen[i]
                continue
            resets += 1
        largest = int((dets[:, 2] * dets[:, 3]).argmax())
        chosen[i] = dets[largest, :4]
        previous = chosen[i]
    return chosen, resets


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))], dtype=np.float32)


def smooth_track(chosen: list, window: int, cut_iou: float = 0.3) -> np.ndarray | None:
    """Fill gaps from the nearest detected frame, median-smooth within each shot.

    The track is split wherever two consecutive boxes overlap less than ``cut_iou`` (a camera cut)
    and each piece is smoothed on its own: smoothing across a cut would move a short shot's box to
    where the face was in the neighbouring shot. Returns ``(T, 3)``: cx, cy, side.
    """
    valid = [i for i, box in enumerate(chosen) if box is not None]
    if not valid:
        return None
    boxes = np.stack([chosen[min(valid, key=lambda v: abs(v - i))] for i in range(len(chosen))])
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2
    side = np.maximum(boxes[:, 2], boxes[:, 3])
    cuts = [0] + [i for i in range(1, len(boxes)) if _iou(boxes[i - 1], boxes[i]) < cut_iou] + [len(boxes)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        for values in (cx, cy, side):
            values[start:end] = _median_filter(values[start:end], window)
    return np.stack([cx, cy, side], axis=1)


def crop_square(frame: np.ndarray, cx: float, cy: float, side: float, out: int) -> np.ndarray:
    side = max(int(round(side)), 2)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    height, width = frame.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - width), max(0, y0 + side - height)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad_l, y0 + pad_t
    patch = frame[y0:y0 + side, x0:x0 + side]
    interpolation = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    return cv2.resize(patch, (out, out), interpolation=interpolation)


def process_clip(video: Path, detector: FaceDetector | None, cfg: FaceCacheConfig,
                 identifier: FaceIdentifier | None = None) -> tuple[np.ndarray, dict]:
    """Sprite (BGR uint8) + statistics for one clip."""
    frames, info = read_frames(video, cfg.frames)
    height, width = info["height"], info["width"]
    centre = (width / 2, height / 2, min(height, width))
    stats: dict = dict(info)
    src = list(range(len(frames)))                 # which sampled frame fills each slot

    if cfg.mode == "full":
        track = np.tile(np.array(centre, dtype=np.float32), (len(frames), 1))
        stats.update(det_rate=None, multi_rate=None, fallback=0, face_px=None)
    else:
        detections = [detector(frame) for frame in frames]
        if identifier is not None and cfg.identity:
            picked = select_speaker(frames, detections, identifier, cfg.id_threshold)
            if picked is None:
                chosen: list = [None] * len(frames)
            else:
                src, chosen, id_stats = picked
                stats.update(id_stats)
        else:
            chosen, resets = track_speaker(detections, cfg.min_iou)
            stats["resets"] = int(resets)
        track = smooth_track(chosen, cfg.smooth, cfg.cut_iou)
        fallback = track is None
        if fallback:
            src = list(range(len(frames)))
            track = np.tile(np.array(centre, dtype=np.float32), (len(frames), 1))
        else:
            track[:, 2] = np.maximum(track[:, 2] * cfg.scale, cfg.min_crop)
        found = [box for box in chosen if box is not None]
        stats.update(
            det_rate=round(float(np.mean([len(d) > 0 for d in detections])), 3),
            multi_rate=round(float(np.mean([len(d) > 1 for d in detections])), 3),
            fallback=int(fallback),
            face_px=round(float(np.median([max(b[2], b[3]) for b in found])), 1) if found else None,
        )

    crops = [crop_square(frames[s], cx, cy, side, cfg.size) for s, (cx, cy, side) in zip(src, track)]
    stats["crop_px"] = round(float(np.median(track[:, 2])), 1)
    return make_sprite(crops), stats


# --------------------------------------------------------------------------- #
# whole-dataset build (multiprocessing, resumable)
# --------------------------------------------------------------------------- #
_DETECTOR: FaceDetector | None = None
_IDENTIFIER: FaceIdentifier | None = None


def _init_worker(detector_path: str | None, identifier_path: str | None, score: float, max_side: int) -> None:
    global _DETECTOR, _IDENTIFIER
    cv2.setNumThreads(1)
    _DETECTOR = FaceDetector(detector_path, score, max_side) if detector_path else None
    _IDENTIFIER = FaceIdentifier(identifier_path) if identifier_path else None


def _work(task: tuple) -> dict:
    clip_id, video, out_dir, cfg_dict = task
    cfg = FaceCacheConfig(**cfg_dict)
    out_dir = Path(out_dir)
    start = time.time()
    try:
        sprite, stats = process_clip(Path(video), _DETECTOR, cfg, _IDENTIFIER)
        write_jpeg(out_dir / f"{clip_id}.jpg", sprite, cfg.jpeg_quality)
        stats.update(clip_id=clip_id, status="ok", error="")
    except Exception as exc:  # one broken clip must not stop the build
        stats = {"clip_id": clip_id, "status": "error", "error": repr(exc)[:300]}
    stats["seconds"] = round(time.time() - start, 2)
    (out_dir / "meta" / f"{clip_id}.json").write_text(json.dumps(stats), encoding="utf-8")
    return stats


def build_cache(clip_ids: list[str], clips_dir: Path, out_dir: Path, cfg: FaceCacheConfig,
                model_path: Path | None, sface_path: Path | None = None, workers: int = 4,
                overwrite: bool = False):
    """Build (or resume) the cache for ``clip_ids``. Returns the index as a DataFrame."""
    from multiprocessing import Pool

    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    config_file = out_dir / "cache_config.json"
    if config_file.exists() and not overwrite:
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        if previous != asdict(cfg):
            raise RuntimeError(f"{out_dir} was built with {previous}; use another --out-dir or --overwrite")
    config_file.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    def done(clip_id: str) -> bool:
        meta = out_dir / "meta" / f"{clip_id}.json"
        return (out_dir / f"{clip_id}.jpg").exists() and meta.exists() and \
            json.loads(meta.read_text(encoding="utf-8")).get("status") == "ok"

    todo = [c for c in clip_ids if overwrite or not done(c)]
    print(f"{len(clip_ids)} clips, {len(clip_ids) - len(todo)} already cached, {len(todo)} to process")
    tasks = [(c, str(Path(clips_dir) / f"{c}.mp4"), str(out_dir), asdict(cfg)) for c in todo]
    if tasks:
        detector_path = str(model_path) if cfg.mode == "face" else None
        identifier_path = str(sface_path) if (cfg.mode == "face" and cfg.identity and sface_path) else None
        if cfg.mode == "face" and cfg.identity and not identifier_path:
            raise ValueError("identity=True needs the SFace model (sface_path)")
        init_args = (detector_path, identifier_path, cfg.score, cfg.det_max_side)
        if workers <= 1:
            _init_worker(*init_args)
            for task in tqdm(tasks, desc="faces"):
                _work(task)
        else:
            with Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
                for _ in tqdm(pool.imap_unordered(_work, tasks, chunksize=2), total=len(tasks), desc="faces"):
                    pass
    return write_index(out_dir, clip_ids)


def write_index(out_dir: Path, clip_ids: list[str]):
    import pandas as pd

    rows = []
    for clip_id in clip_ids:
        meta = Path(out_dir) / "meta" / f"{clip_id}.json"
        rows.append(json.loads(meta.read_text(encoding="utf-8")) if meta.exists()
                    else {"clip_id": clip_id, "status": "missing"})
    index = pd.DataFrame(rows)
    if "indices" in index:
        index["indices"] = index["indices"].map(
            lambda v: ";".join(map(str, v)) if isinstance(v, list) else "")
    front = ["clip_id", "status", "speaker_rate", "held", "identities", "det_rate", "multi_rate", "fallback",
             "resets", "face_px", "crop_px", "n_frames", "fps", "width", "height", "seconds", "error", "indices"]
    index = index[[c for c in front if c in index] + [c for c in index if c not in front]]
    index.sort_values("clip_id").to_csv(Path(out_dir) / "index.csv", index=False)
    return index


def summarize(index) -> str:
    ok = index[index.status == "ok"]
    lines = [f"clips: {len(index)}  ok: {len(ok)}  errors: {int((index.status == 'error').sum())}  "
             f"missing: {int((index.status == 'missing').sum())}"]
    if "det_rate" in ok and ok.det_rate.notna().any():
        det = ok.det_rate.astype(float)
        lines.append(f"frames with a face: mean {det.mean() * 100:.1f}%  |  clips with <50%: "
                     f"{int((det < 0.5).sum())}  |  no face at all (centre-crop fallback): "
                     f"{int(ok.fallback.astype(int).sum())}")
        lines.append(f"frames with >1 face: mean {ok.multi_rate.astype(float).mean() * 100:.1f}%")
        if "speaker_rate" in ok and ok.speaker_rate.notna().any():
            rate = ok.speaker_rate.astype(float)
            held = ok.held.fillna(0).astype(int)
            lines.append(f"speaker visible: mean {rate.mean() * 100:.1f}% of slots  |  clips with slots "
                         f"filled from the nearest speaker frame: {int((held > 0).sum())}  |  clips with "
                         f">1 identity: {int((ok.identities.fillna(0).astype(int) > 1).sum())}")
        if "resets" in ok and ok.resets.notna().any():
            lines.append(f"clips with a camera-cut re-anchor: {int((ok.resets.fillna(0).astype(int) > 0).sum())}")
        face = ok.face_px.dropna().astype(float)
        if len(face):
            lines.append("face size px (median per clip): p10 %.0f  median %.0f  p90 %.0f"
                         % tuple(np.percentile(face, [10, 50, 90])))
    lines.append("crop side px (median per clip): p10 %.0f  median %.0f  p90 %.0f"
                 % tuple(np.percentile(ok.crop_px.astype(float), [10, 50, 90])))
    lines.append(f"processing: {ok.seconds.astype(float).sum() / 60:.1f} CPU-min total")
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_cache(out_dir: Path, zip_path: Path) -> Path:
    """One uncompressed zip (JPEGs do not compress) holding ``<cache name>/...`` for Drive."""
    out_dir, zip_path = Path(out_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(out_dir.glob("*.jpg")) + [out_dir / "index.csv", out_dir / "cache_config.json"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            archive.write(file, arcname=f"{out_dir.name}/{file.name}")
    return zip_path
