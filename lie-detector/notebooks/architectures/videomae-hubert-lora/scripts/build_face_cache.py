"""Build the face-frame cache of the `lore` direction (run once, where the clips are).

    python scripts/build_face_cache.py                  # face crops, every usable clip
    python scripts/build_face_cache.py --mode full      # full-frame cache (ablation)
    python scripts/build_face_cache.py --limit 20 --out-dir <dir> --no-zip   # quick look

Writes ``<data>/cached_features/<cache name>/`` (sprites, meta/, index.csv, cache_config.json) and,
unless ``--no-zip``, ``inputs/<cache name>.zip`` (git-ignored) to upload to Google Drive.
Resumable: re-running skips the clips already cached. Preprocessing is label-free, so every
usable clip is cached (the final, once-only test evaluation needs them too).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import LORE_ROOT, Paths, use_team_lib  # noqa: E402
from lore.faces import (SFACE_FILE, YUNET_FILE, FaceCacheConfig, build_cache,  # noqa: E402
                        ensure_sface, ensure_yunet, file_sha256, summarize, zip_cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="shared DOLOS folder (default: <repo>/dolos)")
    parser.add_argument("--lib-dir", type=Path, default=None, help="lie-detector/ (default: <repo>/lie-detector)")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data>/cached_features/<cache name>")
    parser.add_argument("--mode", choices=["face", "full"], default="face")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--scale", type=float, default=1.6)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips (inspection)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--no-identity", action="store_true",
                        help="follow faces by box overlap instead of picking the speaker by identity")
    parser.add_argument("--yunet", type=Path, default=LORE_ROOT / "assets" / YUNET_FILE)
    parser.add_argument("--sface", type=Path, default=LORE_ROOT / "assets" / SFACE_FILE)
    args = parser.parse_args(argv)

    cfg = FaceCacheConfig(frames=args.frames, size=args.size, scale=args.scale, mode=args.mode,
                          identity=not args.no_identity)
    paths = Paths.resolve(data=args.data_dir, lib=args.lib_dir, face_cache=cfg.cache_name())
    out_dir = args.out_dir or paths.faces

    use_team_lib(paths.lib)
    from lib.dataset import Clips

    clips = Clips.load(paths.data)
    print(clips.describe_source())
    clip_ids = sorted(clips.available())
    if args.limit:
        clip_ids = clip_ids[: args.limit]

    model = ensure_yunet(args.yunet) if cfg.mode == "face" else None
    sface = ensure_sface(args.sface) if (cfg.mode == "face" and cfg.identity) else None
    index = build_cache(clip_ids, paths.data / "clips", out_dir, cfg, model, sface,
                        workers=args.workers, overwrite=args.overwrite)
    print(f"\ncache: {out_dir}")
    print(summarize(index))

    failed = index[index.status != "ok"]
    if len(failed):
        print(f"\n{len(failed)} clips not cached, first: {failed.clip_id.head(5).tolist()}")
        print("re-run the same command to retry them")
        return 1
    if not args.no_zip:
        target = zip_cache(out_dir, LORE_ROOT / "inputs" / f"{out_dir.name}.zip")
        size_mb = target.stat().st_size / 1e6
        print(f"\nzip for Drive: {target}  ({size_mb:.0f} MB, sha256 {file_sha256(target)[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
