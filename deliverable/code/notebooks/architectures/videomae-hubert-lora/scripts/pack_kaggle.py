"""Build the two zips to upload as PRIVATE Kaggle datasets for ``train_lore.ipynb``.

    python scripts/pack_kaggle.py                 # -> kaggle_upload/{lore-data,lore-code}.zip

* ``lore-data.zip``  dolos/dataset.csv, dolos/splits/, dolos/audio/, dolos/cached_features/<face cache>/
  (the face sprites come from the local cache folder, or from ``inputs/<face cache>.zip``).
* ``lore-code.zip``  lore/{lore,scripts,tests} — re-pack and upload a new dataset version
  whenever the code changes.

The DOLOS terms forbid redistribution: keep both Kaggle datasets **private** and never share them
outside the group (``lie-detector/dataset/README.md`` §12).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.config import DEFAULT_FACE_CACHE, LORE_ROOT, Paths  # noqa: E402

CODE_DIRS = ("lore", "scripts", "tests")             # assets/ (face detector) is not needed to train


def add_tree(z: zipfile.ZipFile, root: Path, arc_prefix: str) -> int:
    n = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            z.write(path, f"{arc_prefix}/{path.relative_to(root).as_posix()}")
            n += 1
    return n


def pack_data(paths: Paths, face_cache: str, out: Path) -> None:
    faces_dir = paths.faces.parent / face_cache
    faces_zip = LORE_ROOT / "inputs" / f"{face_cache}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:       # wav/jpg barely compress
        z.write(paths.data / "dataset.csv", "dolos/dataset.csv")
        n = add_tree(z, paths.data / "splits", "dolos/splits")
        n += add_tree(z, paths.audio, "dolos/audio")
        if (faces_dir / "cache_config.json").exists():
            n += add_tree(z, faces_dir, f"dolos/cached_features/{face_cache}")
        elif faces_zip.exists():
            with zipfile.ZipFile(faces_zip) as src:
                for name in src.namelist():
                    if not name.endswith("/"):
                        z.writestr(f"dolos/cached_features/{name}", src.read(name))
                        n += 1
        else:
            raise FileNotFoundError(f"face cache not found: {faces_dir} or {faces_zip}")
    print(f"{out}  ({n + 1} files, {out.stat().st_size / 1e6:.0f} MB)")


def pack_code(out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        n = sum(add_tree(z, LORE_ROOT / d, f"lore/{d}") for d in CODE_DIRS if (LORE_ROOT / d).is_dir())
        n += add_tree(z, LORE_ROOT.parents[2] / "lib", "lib")      # shared lib/, read through LORE_LIB
    print(f"{out}  ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--face-cache", default=DEFAULT_FACE_CACHE)
    p.add_argument("--out-dir", type=Path, default=LORE_ROOT / "kaggle_upload")
    p.add_argument("--code-only", action="store_true", help="re-pack only the code (after a code change)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pack_code(args.out_dir / "lore-code.zip")
    if not args.code_only:
        pack_data(Paths.resolve(face_cache=args.face_cache), args.face_cache, args.out_dir / "lore-data.zip")


if __name__ == "__main__":
    main()
