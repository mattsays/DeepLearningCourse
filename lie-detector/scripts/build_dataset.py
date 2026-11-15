#!/usr/bin/env python3
"""Build the project dataset from the DOLOS annotations.

For every clip listed in ``dataset/DOLOS/dolos_timestamps.csv`` this script:

1. assigns a stable **clip id** derived from the DOLOS file name (see "Clip ids"),
2. downloads only the annotated time range from YouTube (video + audio track),
3. extracts a 16 kHz mono ``.wav`` next to it,
4. validates the result with ``ffprobe`` (duration, presence of an audio stream),
5. appends one row to a canonical dataset table (label, subject, folds, 41 MUMIN features).

Clip ids
--------
The id is the lower-case slug of the ``file_name`` column shipped by DOLOS, so it
can always be recomputed from the dataset files::

    AN_WILTY_EP15_truth1      -> an_wilty_ep15_truth1
    LS_WILTY_EP73_ true1      -> ls_wilty_ep73_true1
    BRI_WILTY_EP61_lie_22     -> bri_wilty_ep61_lie_22

Rows are processed in file order, so the mapping is deterministic. The single
duplicated name in DOLOS (``YW_WILTY_EP46_truth1``, two different time ranges)
becomes ``yw_wilty_ep46_truth1`` and ``yw_wilty_ep46_truth1_2``.

Usage
-----
    python scripts/build_dataset.py --dry-run           # plan only, writes metadata
    python scripts/build_dataset.py --limit 20          # smoke test
    python scripts/build_dataset.py --workers 4         # full run
    python scripts/build_dataset.py --episodes 15,16 --force

Requirements: ``yt-dlp``, ``ffmpeg``/``ffprobe`` on PATH, ``pandas`` (+ ``openpyxl``
for the subject/gender columns; without it they are left empty).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import threading

try:
    from tqdm import tqdm
except ImportError:                      # keep the script usable without tqdm
    def tqdm(iterable=None, **kwargs):
        return iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # lie-detector/
DEFAULT_DATASET = ROOT / "dataset" / "DOLOS"

METADATA_FIELDS = [
    "clip_id", "source_name", "yt_id", "start_s", "end_s", "duration_expected",
    "duration_actual", "label", "y", "gender", "subject",
    "test_fold", "protocols", "video_path", "audio_path", "status", "error",
]


# --------------------------------------------------------------------------- #
# MUMIN annotations (41 binary features)
# --------------------------------------------------------------------------- #
MUMIN_FEATURES = [
    "smile", "laughter", "scowl", "frowning", "raising", "frequent_blinking",
    "exaggerated_opening_eyes", "exaggerated_closing", "exaggerated_opening_mouth",
    "closed_awkward_pauses", "corners_up", "corners_down", "protruded", "retracted",
    "move_upwards", "move_downwards", "towards_interlocutor", "up", "down", "sideways",
    "nod", "move_forward", "move_backward", "tilt", "side_turn", "shake", "head_up",
    "rubbing", "scratching", "touching_head_body", "covering_face_mouth",
    "crossing_fingers", "palm_up", "open", "folded", "shrugs",
    "non_ah_mmh_disturbances", "word_phrase_repetitions", "silent_pauses",
    "loudness", "tension",
]
MUMIN_COLUMNS = [f"mumin_{name}" for name in MUMIN_FEATURES]


def load_mumin(dataset_dir: Path, clips: list["Clip"]) -> dict[str, dict[str, int]]:
    """Attach the 41 binary MUMIN features to each clip.

    ``train_dolos.xlsx`` is row-aligned with ``Dolos.xlsx`` (same order, verified),
    so the key is built on the workbook and joined by ``(name, start, end)``.
    Nine clips in episodes 72-76 disagree on the timestamps between the workbook and
    ``dolos_timestamps.csv``; those fall back to a name match. The one duplicated
    name resolves correctly through the timestamps.
    """
    import pandas as pd

    values_path = dataset_dir / "train_dolos.xlsx"
    book_path = dataset_dir / "Dolos.xlsx"
    if not values_path.exists() or not book_path.exists():
        print("[warn] MUMIN files missing: the mumin_* columns will be empty", file=sys.stderr)
        return {}

    values = pd.read_excel(values_path, header=2).iloc[:, 3:]
    values = values.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    book = pd.read_excel(book_path, header=2)
    book.columns = [str(c).replace("\u200b", "").replace("\xa0", " ").strip() for c in book.columns]
    book = book.rename(columns={"File name of the video clip": "file",
                                "Start time in the episode": "start",
                                "End time in the episode": "end"})
    if len(book) != len(values):
        print(f"[warn] MUMIN rows {len(values)} != workbook rows {len(book)}, skipping", file=sys.stderr)
        return {}

    by_key: dict[tuple, list[int]] = {}
    name_counts: dict[str, int] = {}
    by_name: dict[str, list[int]] = {}
    for (_, row), features in zip(book.iterrows(), values.itertuples(index=False)):
        key = (str(row["file"]).strip(), to_seconds_xlsx(row["start"]), to_seconds_xlsx(row["end"]))
        vector = [int(v) for v in features]
        by_key[key] = vector
        name = key[0]
        name_counts[name] = name_counts.get(name, 0) + 1
        by_name[name] = vector

    dense = {name: vector for name, vector in by_name.items() if name_counts[name] == 1}

    out: dict[str, dict[str, int]] = {}
    fallback = 0
    for clip in clips:
        vector = by_key.get((clip.source_name, clip.start_s, clip.end_s))
        if vector is None:
            vector = dense.get(clip.source_name)
            fallback += 1
        if vector is None:
            vector = [0] * len(MUMIN_FEATURES)
        out[clip.clip_id] = dict(zip(MUMIN_COLUMNS, vector))
    if fallback:
        print(f"[info] {fallback} clips matched MUMIN by name instead of timestamps")
    return out


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def to_seconds_xlsx(value) -> int | None:
    """Decode a time cell of Dolos.xlsx (mixed types, see dataset/README §10)."""
    import datetime as dt

    if isinstance(value, dt.time):
        return value.hour * 60 + value.minute
    if isinstance(value, dt.timedelta):
        return int(value.total_seconds()) // 60
    if isinstance(value, float):
        minutes = int(value)
        return minutes * 60 + round((value - minutes) * 100)
    if isinstance(value, str):
        match = re.match(r"\s*(\d+):(\d+)", value)
        return int(match.group(1)) * 60 + int(match.group(2)) if match else None
    return None


def to_seconds(value: str) -> int:
    """Parse DOLOS ``mm:ss`` timestamps. A bare number (``6``) means ``m:00``."""
    text = str(value).strip()
    if re.fullmatch(r"\d{1,3}", text):
        return int(text) * 60
    hours, minutes, seconds = (text.split(":") + ["0", "0"])[:3]
    return int(hours) * 60 + int(minutes) + (int(seconds) if seconds else 0)


# Free-text participant names contain the same person under several spellings.
# ``Lee``/``David`` are the two recurring team captains, always present in the
# episodes they appear in: no episode contains both the short and the long form,
# so they are the same person (verified on the shipped DOLOS.xlsx).
ALIASES = {
    "Lee": "Lee Mack",
    "David": "David Mitchell",
}


def canonical_subject(name: str, known: list[str]) -> str:
    """Merge a free-text name into its canonical form.

    Two rules, both conservative:

    1. an explicit alias in ``ALIASES``;
    2. a single-token name that is a prefix of exactly one longer name
       (``Gabby`` -> ``Gabby Logan``). Ambiguous cases such as ``David`` (four
       candidates) are left alone unless listed in ``ALIASES``.

    Over-merging two different people is harmless for the splits (they end up in
    the same fold); under-merging is not, because the same face would appear in
    both training and evaluation.
    """
    clean = str(name).strip()
    if clean in ALIASES:
        return ALIASES[clean]
    if " " not in clean:
        candidates = {n for n in known if n.lower().startswith(clean.lower() + " ")}
        if len(candidates) == 1:
            return candidates.pop()
    return clean


def episode_of(clip_id: str) -> int | None:
    """Episode number encoded in a clip id, when present."""
    match = re.search(r"_ep(\d+)", clip_id)
    return int(match.group(1)) if match else None


def slugify(name: str) -> str:
    """Stable clip id: lower-case, only [a-z0-9_], no leading/trailing underscore."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "clip"


@dataclass
class Clip:
    clip_id: str
    source_name: str
    yt_id: str
    start_s: int
    end_s: int
    label: str
    y: int
    gender: str = ""
    subject: str = ""
    test_fold: str = ""
    protocols: str = ""

    @property
    def duration(self) -> int:
        return self.end_s - self.start_s


def load_clips(dataset_dir: Path) -> list[Clip]:
    """Read dolos_timestamps.csv, enrich it with Dolos.xlsx, and assign ids."""
    import pandas as pd

    stamps = pd.read_csv(dataset_dir / "dolos_timestamps.csv", encoding="utf-8-sig", dtype=str)

    # metadata from the primary workbook: subject name, gender, free-text details
    extra: dict[str, tuple[str, str]] = {}
    try:
        book = pd.read_excel(dataset_dir / "Dolos.xlsx", header=2)
        book.columns = [str(c).replace("\u200b", "").replace("\xa0", " ").strip() for c in book.columns]
        book = book.rename(columns={
            "File name of the video clip": "file",
            "Participants name": "subject",
            "Participants gender": "gender",
        })
        known = {str(v).strip() for v in book["subject"].dropna()}
        for _, row in book.iterrows():
            key = str(row["file"]).strip()
            subject = canonical_subject(row.get("subject", ""), list(known))
            extra.setdefault(key, (subject, str(row.get("gender", "")).strip().lower()))
    except Exception as exc:  # missing openpyxl, unreadable sheet, …
        print(f"[warn] subject/gender unavailable from Dolos.xlsx: {exc}", file=sys.stderr)

    # protocol membership, so the manifest is the single source of truth
    protocols: dict[str, set[str]] = {}
    folds: dict[str, set[int]] = {}
    proto_dir = dataset_dir / "Training_Protocols"
    for path in sorted(proto_dir.glob("*.csv")):
        name = path.stem                       # train_fold1, test_fold2, short, …
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            key = line.split(",")[0].strip()
            protocols.setdefault(key, set()).add(name)
            if name.startswith("test_fold"):
                folds.setdefault(key, set()).add(int(name[-1]))

    clips: list[Clip] = []
    seen: dict[str, int] = {}
    for _, row in stamps.iterrows():
        source = str(row["file_name"]).strip()
        base = slugify(source)
        seen[base] = seen.get(base, 0) + 1
        clip_id = base if seen[base] == 1 else f"{base}_{seen[base]}"

        label_raw = str(row["label"]).strip().lower()
        label = "truth" if label_raw.startswith("t") else "deception"
        subject, gender = extra.get(source, ("", ""))

        clips.append(Clip(
            clip_id=clip_id,
            source_name=source,
            yt_id=str(row["YT_Video_ID"]).strip(),
            start_s=to_seconds(row["start_time"]),
            end_s=to_seconds(row["end_time"]),
            label=label,
            y=0 if label == "truth" else 1,
            gender=gender,
            subject=subject,
            test_fold=",".join(str(f) for f in sorted(folds.get(source, set()))),
            protocols=";".join(sorted(protocols.get(source, set()))),
        ))
    return clips


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #
def ytdlp_prefix() -> list[str] | None:
    """Invoke yt-dlp as a module when installed in this interpreter, else from PATH."""
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        if shutil.which("yt-dlp"):
            return ["yt-dlp"]
        return None


def run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    # ``errors="replace"`` matters: ffmpeg and yt-dlp occasionally emit bytes that are
    # not valid UTF-8, and a strict decode would kill the whole run on one clip.
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=timeout)


def probe(path: Path) -> dict:
    """Return {'duration': float|None, 'has_audio': bool, 'has_video': bool}."""
    proc = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(path),
    ])
    if proc.returncode != 0:
        return {"duration": None, "has_audio": False, "has_video": False, "error": proc.stderr.strip()[:200]}
    data = json.loads(proc.stdout or "{}")
    kinds = {s.get("codec_type") for s in data.get("streams", [])}
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        duration = None
    return {"duration": duration, "has_audio": "audio" in kinds, "has_video": "video" in kinds, "error": ""}


def hhmmss(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def download(clip: Clip, out_dir: Path, fmt: str, retries: int) -> tuple[str, str]:
    """Download one clip. Returns (status, error_message)."""
    target = out_dir / f"{clip.clip_id}.mp4"
    stem = f"{clip.clip_id}.part"
    url = f"https://www.youtube.com/watch?v={clip.yt_id}"
    for stale in out_dir.glob(f"{stem}.*"):        # leftovers from an interrupted run
        stale.unlink(missing_ok=True)

    cmd = ytdlp_prefix() + [
        "--quiet", "--no-warnings", "--no-playlist",
        "--retries", str(retries), "--fragment-retries", str(retries),
        "--socket-timeout", "20",
        "--download-sections", f"*{hhmmss(clip.start_s)}-{hhmmss(clip.end_s)}",
        "--force-keyframes-at-cuts",          # exact boundaries, costs a re-encode
        "--merge-output-format", "mp4",
        "-f", fmt,
        "-o", str(out_dir / f"{stem}.%(ext)s"),
        url,
    ]
    try:
        proc = run(cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return "error", "timeout"

    produced = sorted(out_dir.glob(f"{stem}.*"))
    if proc.returncode != 0 or not produced:
        lines = [l for l in (proc.stderr + proc.stdout).splitlines() if l.strip()]
        message = next((l for l in reversed(lines) if "ERROR" in l), lines[-1] if lines else "yt-dlp failed")
        return "error", message[:300]

    produced[0].rename(target)
    info = probe(target)
    if info["error"]:
        target.unlink(missing_ok=True)
        return "error", f"ffprobe: {info['error']}"
    if not info["has_video"]:
        target.unlink(missing_ok=True)
        return "error", "no video stream"
    if abs((info["duration"] or 0) - clip.duration) > 2.0:
        return "duration_mismatch", f"expected {clip.duration}s, got {info['duration']}"
    if not info["has_audio"]:
        return "no_audio", "clip has no audio stream"
    return "ok", ""


def extract_audio(video: Path, audio: Path) -> tuple[bool, str]:
    audio.parent.mkdir(parents=True, exist_ok=True)
    proc = run([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio),
    ], timeout=180)
    if proc.returncode != 0:
        return False, (proc.stderr or "ffmpeg failed").strip().splitlines()[-1][:200]
    return True, ""


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed" / "clips")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "processed" / "audio")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "processed" / "dataset.csv")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="stop after N clips (smoke test)")
    parser.add_argument("--episodes", default=None, help="comma-separated episode numbers, e.g. 15,16")
    parser.add_argument("--force", action="store_true", help="re-download clips that already exist")
    parser.add_argument("--no-audio", action="store_true", help="skip the 16 kHz wav extraction")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--quiet", action="store_true",
                        help="do not print a line per clip (the progress bar stays)")
    parser.add_argument("--dry-run", action="store_true", help="write the table, download nothing")
    parser.add_argument(
        "--format", default="bv*[height<=720]+ba/b[height<=720]/bv*+ba/b",
        help="yt-dlp format selector (default: best video <=720p + best audio)",
    )
    args = parser.parse_args()

    if not args.dry_run:
        if ytdlp_prefix() is None:
            print("[error] yt-dlp not installed (pip install yt-dlp)", file=sys.stderr)
            return 2
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                print(f"[error] '{tool}' not found on PATH", file=sys.stderr)
                return 2

    clips = load_clips(args.dataset_dir)
    all_clips = clips
    print(f"annotations: {len(clips)} clips from {args.dataset_dir}")

    if args.episodes:
        wanted = {int(e) for e in args.episodes.split(",") if e.strip()}
        clips = [c for c in clips if episode_of(c.clip_id) in wanted]
        print(f"episode filter: {len(clips)} clips")

    if args.limit:
        clips = clips[: args.limit]
        print(f"limit: {len(clips)} clips")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    data_root = args.csv.parent          # paths in the CSV are relative to this folder

    rows: dict[str, dict] = {}
    if args.csv.exists():
        with args.csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["clip_id"]] = row

    lock = threading.Lock()
    counts = {"ok": 0, "skipped": 0, "error": 0, "duration_mismatch": 0, "no_audio": 0,
              "invalid_range": 0, "planned": 0}
    todo: list[Clip] = []

    for clip in clips:
        video = args.out_dir / f"{clip.clip_id}.mp4"
        if clip.duration <= 0:
            # broken annotations: end <= start (7 rows in the shipped CSV)
            rows[clip.clip_id] = {
                "clip_id": clip.clip_id, "source_name": clip.source_name, "yt_id": clip.yt_id,
                "start_s": clip.start_s, "end_s": clip.end_s, "duration_expected": clip.duration,
                "duration_actual": "", "label": clip.label, "y": clip.y,
                "gender": clip.gender, "subject": clip.subject,
                "test_fold": clip.test_fold, "protocols": clip.protocols,
                "video_path": "", "audio_path": "", "status": "invalid_range",
                "error": f"end {clip.end_s}s <= start {clip.start_s}s",
            }
            counts["invalid_range"] = counts.get("invalid_range", 0) + 1
            continue
        if video.exists() and not args.force:
            rows.setdefault(clip.clip_id, {}).update(
                clip_id=clip.clip_id, source_name=clip.source_name, yt_id=clip.yt_id,
                start_s=clip.start_s, end_s=clip.end_s, duration_expected=clip.duration,
                label=clip.label, y=clip.y, gender=clip.gender, subject=clip.subject,
                test_fold=clip.test_fold, protocols=clip.protocols,
                video_path=str(video.relative_to(data_root)), status="ok",
            )
            counts["skipped"] += 1
            continue
        todo.append(clip)

    print(f"to download: {len(todo)} | already present: {counts['skipped']}"
          f" | invalid annotations: {counts.get('invalid_range', 0)}")

    def process(clip: Clip) -> dict:
        try:
            return _process(clip)
        except Exception as exc:                      # noqa: BLE001 - one clip must not kill the run
            return {
                "clip_id": clip.clip_id, "source_name": clip.source_name, "yt_id": clip.yt_id,
                "start_s": clip.start_s, "end_s": clip.end_s, "duration_expected": clip.duration,
                "duration_actual": "", "label": clip.label, "y": clip.y,
                "gender": clip.gender, "subject": clip.subject,
                "test_fold": clip.test_fold, "protocols": clip.protocols,
                "video_path": "", "audio_path": "", "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }

    def _process(clip: Clip) -> dict:
        status, error = download(clip, args.out_dir, args.format, args.retries)
        video = args.out_dir / f"{clip.clip_id}.mp4"
        audio = args.audio_dir / f"{clip.clip_id}.wav"
        if status == "ok" and not args.no_audio:
            ok, error = extract_audio(video, audio)
            if not ok:
                status, error = "no_audio", error
        info = probe(video) if video.exists() else {"duration": None}
        return {
            "clip_id": clip.clip_id, "source_name": clip.source_name, "yt_id": clip.yt_id,
            "start_s": clip.start_s, "end_s": clip.end_s, "duration_expected": clip.duration,
            "duration_actual": f"{info['duration']:.2f}" if info["duration"] else "",
            "label": clip.label, "y": clip.y, "gender": clip.gender, "subject": clip.subject,
            "test_fold": clip.test_fold, "protocols": clip.protocols,
            "video_path": str(video.relative_to(data_root)) if video.exists() else "",
            "audio_path": str(audio.relative_to(data_root)) if audio.exists() else "",
            "status": status, "error": error,
        }

    if todo and not args.dry_run:
        bar = tqdm(total=len(todo), unit="clip", desc="download", dynamic_ncols=True,
                   mininterval=1.0, file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, clip): clip for clip in todo}
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    rows[row["clip_id"]] = row
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                bar.update(1)
                bar.set_postfix(ok=counts.get("ok", 0),
                                err=counts.get("error", 0) + counts.get("no_audio", 0)
                                + counts.get("duration_mismatch", 0),
                                refresh=False)
                if row["status"] != "ok":
                    bar.write(f"{row['clip_id']}: {row['status']} — {row['error'][:80]}")
                elif not args.quiet:
                    bar.write(f"{row['clip_id']}: ok")
        bar.close()
    elif args.dry_run:
        counts["planned"] = len(todo)
        for clip in todo:
            rows[clip.clip_id] = {
                "clip_id": clip.clip_id, "source_name": clip.source_name, "yt_id": clip.yt_id,
                "start_s": clip.start_s, "end_s": clip.end_s, "duration_expected": clip.duration,
                "duration_actual": "", "label": clip.label, "y": clip.y,
                "gender": clip.gender, "subject": clip.subject,
                "test_fold": clip.test_fold, "protocols": clip.protocols,
                "video_path": "", "audio_path": "", "status": "planned", "error": "",
            }

    mumin = load_mumin(args.dataset_dir, all_clips)
    empty_mumin = {column: "" for column in MUMIN_COLUMNS}
    for clip in all_clips:
        if clip.clip_id in rows:
            rows[clip.clip_id].update(mumin.get(clip.clip_id, empty_mumin))

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS + MUMIN_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for clip in all_clips:
            if clip.clip_id in rows:
                writer.writerow(rows[clip.clip_id])

    total = {k: v for k, v in counts.items() if v}
    print(f"\ndataset table: {args.csv}")
    print("status:", ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
    if counts.get("ok", 0) == 0 and counts.get("error", 0) == 0:
        return 2                                     # nothing downloaded and nothing failed: setup problem
    return 0                                          # a completed pass, even with some failed clips


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
