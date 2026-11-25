"""Build the course submission folder: code + paper + one README, nothing else.

    python tools/build_deliverable.py            # -> deliverable/ and deliverable.zip

Only git-tracked files are copied: the working tree also holds git-ignored DOLOS media, face crops
and run outputs that must never leave the machine. `lie-detector/` keeps its inner layout under
`deliverable/code/`, so every `Path(__file__).parents[n]` lookup in the code still resolves. The
copy is then patched in a few places so it runs on its own, without cloning the team repository.
These patches are listed in PY_PATCHES and NB_PATCHES. The training code and hyperparameters are
not changed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "deliverable"
ZIP = REPO / "deliverable.zip"
README = REPO / "tools" / "deliverable_README.md"

A = "notebooks/architectures/"

# Paths relative to lie-detector/. An entry ending in "/" takes every tracked file below it.
CODE = [
    "lib/__init__.py", "lib/dataset.py", "lib/metrics.py",
    "scripts/build_dataset.py", "scripts/make_splits.py", "scripts/extract_features_marlin.py",
    "notebooks/preprocessing.ipynb",
    # theory-driven baseline
    A + "load/features.py", A + "load/run_load.py", A + "load/results/",
    # Temporal CNN
    A + "teo/cnn-batching/train_official_temporal.py", A + "teo/cnn-batching/train_fold_temporal.py",
    A + "teo/cnn-batching/model_temporal.py",
    A + "teo/features/extract_features.py", A + "teo/features/mask_streams.py",
    A + "teo/results/",
    # Cross-Modal Fusion
    A + "cross-modal-fusion/model.py", A + "cross-modal-fusion/training.py",
    A + "cross-modal-fusion/export.py",
    A + "cross-modal-fusion/extract_features.ipynb", A + "cross-modal-fusion/campaign.ipynb",
    A + "cross-modal-fusion/test.ipynb", A + "cross-modal-fusion/diagnostics.ipynb",
    A + "cross-modal-fusion/results/",
    # VideoMAE + HuBERT with LoRA
    A + "lore/lore/",
    A + "lore/scripts/run_cv.py", A + "lore/scripts/fuse.py", A + "lore/scripts/build_face_cache.py",
    A + "lore/scripts/export_results.py", A + "lore/scripts/cluster_test.py",
    A + "lore/scripts/pack_kaggle.py",
    A + "lore/tests/test_lora_active.py", A + "lore/tests/test_protocol.py",
    A + "lore/train_lore.ipynb", A + "lore/results/",
    # MARLIN + ConvNeXt
    A + "fab/train_marlin_kaggle.ipynb", A + "fab/results/",
]

REPORT = ["report/report.pdf", "report/report.typ", "report/template.typ", "report/refs.bib",
          "report/sections/"]

# ---------------------------------------------------------------- patches of the copy

COLAB_SETUP = '''\
# CODE_ROOT is the submitted `code/` folder (it holds lib/ and scripts/). Upload it to Drive and
# adjust the default below, or set the CODE_ROOT environment variable before running.
import os, sys
CODE_ROOT = os.environ.get("CODE_ROOT", "/content/drive/MyDrive/code")
%cd {CODE_ROOT}
sys.path.insert(0, '.')
'''

KAGGLE_SETUP = '''\
# CODE_ROOT is the submitted `code/` folder (lib/, scripts/, notebooks/). On Kaggle, attach it as a
# dataset: it is copied to /kaggle/working/code because /kaggle/input is read-only. Locally, the
# folder is found by walking up from the notebook. Set the CODE_ROOT environment variable to override.
import os, sys, shutil
from pathlib import Path

CODE_ROOT = os.environ.get("CODE_ROOT")
if CODE_ROOT is None:
    inputs = Path("/kaggle/input")
    hits = sorted(inputs.rglob("lib/dataset.py"), key=lambda p: len(p.parts)) if inputs.exists() else []
    if hits:
        CODE_ROOT = Path("/kaggle/working/code")
        if not CODE_ROOT.exists():
            shutil.copytree(hits[0].parents[1], CODE_ROOT)
    else:
        CODE_ROOT = Path.cwd()
        while not (CODE_ROOT / "lib" / "dataset.py").exists() and CODE_ROOT != CODE_ROOT.parent:
            CODE_ROOT = CODE_ROOT.parent
CODE_ROOT = Path(CODE_ROOT)
%cd {CODE_ROOT}
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
print("Active directory:", Path.cwd())
'''

DRIVE_DATA = 'Path("/content/drive/MyDrive/dolos")'
ENV_DATA = 'Path(os.environ.get("DOLOS_DATA", "/content/drive/MyDrive/dolos"))'

# file (relative to deliverable/code/) -> [(old, new)], each old must occur exactly once
PY_PATCHES = {
    A + "lore/lore/config.py": [
        ('lib = _pick(lib, "LORE_LIB", REPO_ROOT / "lie-detector")',
         'lib = _pick(lib, "LORE_LIB", LORE_ROOT.parents[2])      # the code/ folder that holds lib/'),
    ],
    A + "lore/scripts/pack_kaggle.py": [
        ('        n = sum(add_tree(z, LORE_ROOT / d, f"lore/{d}") for d in CODE_DIRS if (LORE_ROOT / d).is_dir())\n',
         '        n = sum(add_tree(z, LORE_ROOT / d, f"lore/{d}") for d in CODE_DIRS if (LORE_ROOT / d).is_dir())\n'
         '        n += add_tree(z, LORE_ROOT.parents[2] / "lib", "lib")      # shared lib/, read through LORE_LIB\n'),
    ],
}

# notebook -> list of actions:
#   ("cell", marker, new_source)   replace the whole source of the one cell containing marker
#   ("sub", old, new)              replace text inside the one cell containing old
#   ("drop", marker)               delete every cell containing marker
#   ("clear",)                     clear all outputs
NB_PATCHES = {
    "notebooks/preprocessing.ipynb": [
        ("cell", "git clone -q https://github.com/mattsays/DeepLearningCourse", COLAB_SETUP),
        ("sub", f"DRIVE = {DRIVE_DATA}", f"DRIVE = {ENV_DATA}"),
        ("drop", "The v2 protocol"), ("drop", "make_splits_v2.py"), ("drop", "report_splits.py"),
        ("drop", "architecture_template.ipynb"),
        ("sub", "`splits/`, `splits_v2/`, `clips/`", "`splits/`, `clips/`"),
    ],
    A + "cross-modal-fusion/campaign.ipynb": [
        ("cell", "git clone -q https://github.com/mattsays/DeepLearningCourse", COLAB_SETUP),
        ("sub", f"DATA = {DRIVE_DATA}", f"DATA = {ENV_DATA}"),
    ],
    A + "cross-modal-fusion/diagnostics.ipynb": [
        ("cell", "git clone -q https://github.com/mattsays/DeepLearningCourse", COLAB_SETUP),
        ("sub", f"DATA = {DRIVE_DATA}", f"DATA = {ENV_DATA}"),
        # the committed outputs (and the report) come from the face cache, not the body variant
        ("sub", 'VIDEO_CACHE = cache_dir("dinov2-small-body", DATA)',
                'VIDEO_CACHE = cache_dir("dinov2-small", DATA)'),
    ],
    A + "cross-modal-fusion/extract_features.ipynb": [
        ("cell", "git clone -q https://github.com/mattsays/DeepLearningCourse", COLAB_SETUP),
        ("sub", f"DATA = {DRIVE_DATA}", f"DATA = {ENV_DATA}"),
        # build the two caches the reported model uses (DINOv2 face crops + WavLM)
        ("sub", '    #"dinov2-small":    {"kind": "video", "hf_id": "facebook/dinov2-small"},\n'
         '    #"wavlm-base-plus": {"kind": "audio", "hf_id": "microsoft/wavlm-base-plus"},',
         '    "dinov2-small":    {"kind": "video", "hf_id": "facebook/dinov2-small"},\n'
         '    "wavlm-base-plus": {"kind": "audio", "hf_id": "microsoft/wavlm-base-plus"},'),
        ("sub", '    "hsemotion-b0-vgaf": {"kind": "video", "model": "enet_b0_8_best_vgaf"},',
         '    #"hsemotion-b0-vgaf": {"kind": "video", "model": "enet_b0_8_best_vgaf"},'),
        ("clear",),          # one output is a picture of a speaker's face crops
    ],
    A + "cross-modal-fusion/test.ipynb": [
        ("sub",
         'REPO = "/content/DeepLearningCourse"\n'
         'if COLAB:\n'
         '    if not os.path.isdir(f"{REPO}/.git"):\n'
         '        subprocess.run(["git", "clone", "-q",\n'
         '                        "https://github.com/mattsays/DeepLearningCourse", REPO], check=True)\n'
         '    subprocess.run(["git", "pull", "-q"], cwd=f"{REPO}/lie-detector", check=False)\n'
         '    os.chdir(f"{REPO}/lie-detector")\n',
         '# CODE_ROOT: the submitted `code/` folder, uploaded to Drive (or set the CODE_ROOT variable).\n'
         'CODE_ROOT = os.environ.get("CODE_ROOT", "/content/drive/MyDrive/code")\n'
         'if COLAB:\n'
         '    os.chdir(CODE_ROOT)\n'),
        ("sub", f"SHARED = {DRIVE_DATA}", f"SHARED = {ENV_DATA}"),
    ],
    A + "lore/train_lore.ipynb": [
        ("sub",
         "# The team repo, only for lie-detector/lib (splits + metrics, never copied)\n"
         "if not os.path.isdir('/tmp/DeepLearningCourse/.git'):\n"
         "    !git clone -q https://github.com/mattsays/DeepLearningCourse /tmp/DeepLearningCourse\n"
         "!git -C /tmp/DeepLearningCourse pull -q\n",
         "# lib/ (splits + metrics) is inside the same code dataset: scripts/pack_kaggle.py adds it\n"
         "LIB = find_one('lib/dataset.py', 'lib/ inside the code dataset').parent.parent\n"),
        ("sub", "'LORE_LIB': '/tmp/DeepLearningCourse/lie-detector',", "'LORE_LIB': str(LIB),"),
    ],
    A + "fab/train_marlin_kaggle.ipynb": [
        ("cell", "git clone -q -b fab_marlin https://github.com/mattsays/DeepLearningBigData", KAGGLE_SETUP),
    ],
}


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    return [p for p in out.decode("utf-8").split("\0") if p]


def select(files: list[str], prefix: str, wanted: list[str]) -> list[str]:
    picked = []
    for want in wanted:
        full = prefix + want
        hits = [f for f in files if f.startswith(full)] if want.endswith("/") else [f for f in files if f == full]
        hits = [f for f in hits if not f.endswith(".md")]
        if not hits:
            sys.exit(f"nothing tracked matches {full}")
        picked += hits
    return picked


def patch_text(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"{where}: expected one match, found {n}: {old[:70]!r}")
    return text.replace(old, new)


def patch_notebook(path: Path, actions: list[tuple]) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def src(cell):
        return "".join(cell["source"])

    def one(marker):
        hits = [c for c in cells if marker in src(c)]
        if len(hits) != 1:
            sys.exit(f"{path.name}: {len(hits)} cells contain {marker[:60]!r}")
        return hits[0]

    for action in actions:
        kind = action[0]
        if kind == "cell":
            cell = one(action[1])
            cell["source"] = action[2].splitlines(keepends=True)
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
        elif kind == "sub":
            cell = one(action[1])
            cell["source"] = patch_text(src(cell), action[1], action[2], path.name).splitlines(keepends=True)
        elif kind == "drop":
            before = len(cells)
            cells[:] = [c for c in cells if action[1] not in src(c)]
            if len(cells) == before:
                sys.exit(f"{path.name}: no cell to drop for {action[1]!r}")
        elif kind == "clear":
            for cell in cells:
                if cell["cell_type"] == "code":
                    cell["outputs"], cell["execution_count"] = [], None
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    files = tracked()
    plan = [(f, OUT / "code" / f.removeprefix("lie-detector/")) for f in select(files, "lie-detector/", CODE)]
    plan += [(f, OUT / f) for f in select(files, "", REPORT)]

    if OUT.exists():                 # empty it file by file: a shell may sit inside a subfolder
        for p in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
            else:
                p.unlink()
    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / src, dst)
    shutil.copy2(README, OUT / "README.md")

    code = OUT / "code"
    for rel, patches in PY_PATCHES.items():
        p = code / rel
        text = p.read_bytes().decode("utf-8")
        eol = "\r\n" if "\r\n" in text else "\n"
        for old, new in patches:
            text = patch_text(text, old.replace("\n", eol), new.replace("\n", eol), rel)
        p.write_bytes(text.encode("utf-8"))
    for rel, actions in NB_PATCHES.items():
        patch_notebook(code / rel, actions)

    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                z.write(p, Path("deliverable") / p.relative_to(OUT))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(plan) + 1} files, {size / 1e6:.1f} MB -> {OUT.relative_to(REPO)}/, "
          f"{ZIP.stat().st_size / 1e6:.1f} MB -> {ZIP.name}")


if __name__ == "__main__":
    main()
