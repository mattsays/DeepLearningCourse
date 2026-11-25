"""Paths and run configuration for the `lore` direction.

Every path can be overridden by an argument or an environment variable, so the same
code runs on the laptop (``<repo>/dolos``) and on Colab (``/content/drive/MyDrive/dolos``).

    LORE_DATA        the shared DOLOS folder: dataset.csv, splits/, clips/, audio/
    LORE_LIB         the team's ``lie-detector/`` directory (its ``lib/`` package is imported)
    LORE_FACES_ROOT  directory that contains the face caches (default: <data>/cached_features)
    LORE_AUDIO       16 kHz wav directory (default: <data>/audio)
    LORE_RUNS        where fold results are written (default: <this folder>/runs, git-ignored)
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

LORE_ROOT = Path(__file__).resolve().parents[1]      # lie-detector/notebooks/architectures/lore/
REPO_ROOT = LORE_ROOT.parents[3]                      # local checkout of DeepLearningCourse

DEFAULT_FACE_CACHE = "faces-yunet-t32-s224"
VIDEO_BACKBONE = "MCG-NJU/videomae-base-finetuned-kinetics"
AUDIO_BACKBONE = "facebook/hubert-base-ls960"


def _pick(explicit, env: str, default) -> Path:
    if explicit:
        return Path(explicit)
    value = os.environ.get(env)
    return Path(value) if value else Path(default)


@dataclass
class Paths:
    data: Path       # dataset.csv, splits/, clips/, audio/
    lib: Path        # lie-detector/ (contains lib/dataset.py and lib/metrics.py)
    faces: Path      # one face cache: <clip_id>.jpg sprites + index.csv + cache_config.json
    audio: Path      # <clip_id>.wav
    runs: Path       # runs/<run_name>/fold<k>/...

    @classmethod
    def resolve(cls, data=None, lib=None, faces_root=None, face_cache: str = DEFAULT_FACE_CACHE,
                audio=None, runs=None) -> "Paths":
        data = _pick(data, "LORE_DATA", REPO_ROOT / "dolos")
        lib = _pick(lib, "LORE_LIB", LORE_ROOT.parents[2])      # the code/ folder that holds lib/
        faces_root = _pick(faces_root, "LORE_FACES_ROOT", data / "cached_features")
        audio = _pick(audio, "LORE_AUDIO", data / "audio")
        runs = _pick(runs, "LORE_RUNS", LORE_ROOT / "runs")
        return cls(data=data, lib=lib, faces=faces_root / face_cache, audio=audio, runs=runs)

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}


def use_team_lib(lib_dir: Path | str) -> None:
    """Make ``import lib.dataset`` / ``lib.metrics`` resolve to the team's package (never copied)."""
    lib_dir = str(Path(lib_dir).resolve())
    if not (Path(lib_dir) / "lib" / "dataset.py").exists():
        raise FileNotFoundError(f"{lib_dir}/lib/dataset.py not found — point LORE_LIB at lie-detector/")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)


@dataclass
class RunConfig:
    """Everything that defines one cross-validation run of one branch."""

    modality: str                                   # "video" | "audio"
    run_name: str = ""
    protocol: str = "subject"                       # "subject" (the board) | "official" (paper's 3 clip-level folds)
    backbone: str = ""
    # LoRA
    head_only: bool = False                         # ablation: no LoRA, frozen backbone + trained head
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: tuple[str, ...] = ()
    lora_mlp_last_n: int = 0                        # ablation: LoRA also on the MLP of the last N blocks
    use_dora: bool = False                          # ablation: DoRA instead of LoRA
    head_dropout: float = 0.3
    # data
    face_cache: str = DEFAULT_FACE_CACHE
    num_frames: int = 16
    rrc_scale: tuple[float, float] = (0.8, 1.0)
    eval_tta: bool = False                          # video: average the logits of every frame offset
    audio_train_seconds: float = 8.0
    audio_eval_seconds: float = 10.0
    subject_balanced: bool = False                  # ablation: down-weight subjects with many clips
    # optimisation
    epochs: int = 10
    patience: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    max_grad_norm: float = 1.0
    grad_checkpointing: bool = False
    num_workers: int = 2
    seed: int = 0
    # debugging
    debug: bool = False                             # ~200 clips, 2 epochs: the loop end to end in minutes
    limit: int | None = None                        # N clips per split (smoke tests)

    def __post_init__(self):
        if self.modality not in ("video", "audio"):
            raise ValueError(f"modality must be 'video' or 'audio', got {self.modality!r}")
        if self.protocol not in ("subject", "official"):
            raise ValueError(f"protocol must be 'subject' or 'official', got {self.protocol!r}")
        self.lora_targets = tuple(self.lora_targets)
        self.rrc_scale = tuple(self.rrc_scale)
        if not self.run_name:
            self.run_name = auto_run_name(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})


VIDEO_DEFAULTS = dict(backbone=VIDEO_BACKBONE, lora_targets=("query", "value"),
                      epochs=10, patience=3, batch_size=4, grad_accum=4)
AUDIO_DEFAULTS = dict(backbone=AUDIO_BACKBONE, lora_targets=("q_proj", "v_proj"),
                      epochs=15, patience=4, batch_size=8, grad_accum=2, num_workers=0)


def default_config(modality: str, **overrides) -> RunConfig:
    """Defaults of the main configuration, then any override (``None`` values are ignored)."""
    base = dict(VIDEO_DEFAULTS if modality == "video" else AUDIO_DEFAULTS)
    given = {k: v for k, v in overrides.items() if v is not None}
    if (given.get("debug") or given.get("limit")) and "epochs" not in given:
        base["epochs"] = 2
    base.update(given)
    return RunConfig(modality=modality, **base)


def auto_run_name(cfg: RunConfig) -> str:
    """Readable, unique-per-configuration name: ``video-lora-r8``, ``audio-headonly``, ``video-dora-r8-mlp4``..."""
    parts = [cfg.modality]
    if cfg.head_only:
        parts.append("headonly")
    else:
        parts.append(("dora" if cfg.use_dora else "lora") + f"-r{cfg.lora_r}")
        if cfg.lora_mlp_last_n:
            parts.append(f"mlp{cfg.lora_mlp_last_n}")
    if cfg.modality == "video":
        if cfg.backbone and cfg.backbone != VIDEO_BACKBONE:
            parts.append(cfg.backbone.split("/")[-1])
        if cfg.face_cache != DEFAULT_FACE_CACHE:
            parts.append(cfg.face_cache.split("-")[0])
    elif cfg.backbone and cfg.backbone != AUDIO_BACKBONE:
        parts.append(cfg.backbone.split("/")[-1])
    if cfg.subject_balanced:
        parts.append("subjbal")
    if cfg.protocol == "official":
        parts.append("official")
    if cfg.seed:
        parts.append(f"seed{cfg.seed}")
    if cfg.debug or cfg.limit:
        parts.append("debug" if not cfg.limit else f"limit{cfg.limit}")
    return "-".join(parts)
