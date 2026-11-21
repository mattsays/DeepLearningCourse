"""Write one training run to ``results/`` in the layout the report reads.

``train.ipynb`` keeps every fold's metrics in a DataFrame and every clip's probability in
another; without this step they die with the Colab kernel and the numbers in the README
cannot be reproduced from the repository. One call writes both, plus the provenance needed
to say *which* run produced them.

    from export import save_run
    save_run(summary, per_clip, config)

``config`` carries the settings the run was launched with plus the fingerprints measured
from the data and the checkout. :func:`run_config` builds it from the notebook's globals so
the cell stays two lines; :func:`describe` fills the parts that need the dataset objects.

Layout (owned by ``results/README.md``; do not restate it elsewhere)::

    results/<run>/run.json        settings, fingerprints, trainable parameters
    results/<run>/summary.json    mean and spread over folds and seeds
    results/<run>/folds.csv       one row per (seed, fold)
    results/<run>/preds.csv       one row per clip: y_true, y_prob, seed, fold
    results/<run>/board_row.md    the results-board line, or why this run has none

``run_name`` is ``<variant>-<split>``, with ``-nocaptains`` and ``+<protocol>`` appended when
the run is an ablation, e.g. ``late-subject``, ``late-official``, ``late-subject+2``.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FOLD_COLUMNS = ["seed", "fold", "acc", "f1", "auc", "balanced_acc", "precision", "recall",
                "threshold", "n", "tn", "fp", "fn", "tp", "seconds", "epochs"]
SUMMARY_METRICS = ["acc", "f1", "auc", "balanced_acc"]


def _git(*args) -> str | None:
    """Best effort: a run started from a notebook that is not a checkout has no commit."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):        # no git, no sandbox, no problem
        return None


def run_name(variant: str, split: str, protocol: str = "v1", captains: str = "include") -> str:
    """The directory name of a run: which model, on which protocol."""
    name = f"{variant}-{split}"
    if captains != "include":
        name += f"-{captains}"
    if protocol != "v1":
        name += f"+{protocol}"
    return name


def describe(clips, folds) -> dict:
    """The two fingerprints that make a run identifiable: the data and the split files."""
    import hashlib

    manifest = Path(folds.directory) / "manifest.json"
    return {
        "dataset": {"fingerprint": clips.fingerprint,
                    "clips": len(clips),
                    "source": clips.describe_source()},
        "protocol_files": {"directory": str(folds.directory),
                           "manifest_sha256": (hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
                                               if manifest.exists() else None),
                           "verified": bool(getattr(folds, "verified", False))},
    }


def run_config(**settings) -> dict:
    """Settings of the run, plus what the checkout and the machine can be asked for.

    Unknown keys are kept: the notebook passes its own names and they are worth recording
    even if this module does not read them.
    """
    config = dict(settings)
    config.setdefault("created_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    config.setdefault("git_commit", _git("rev-parse", "HEAD"))
    config.setdefault("git_branch", _git("rev-parse", "--abbrev-ref", "HEAD"))
    config.setdefault("git_dirty", bool(_git("status", "--porcelain")))
    config.setdefault("python", platform.python_version())
    return config


def _check(summary: pd.DataFrame, per_clip: pd.DataFrame) -> None:
    missing = [c for c in FOLD_COLUMNS if c not in summary.columns]
    if missing:
        raise KeyError(f"the fold summary is missing {missing}: pass the DataFrame built by "
                       f"train.ipynb's campaign cell, not a hand-made table")
    for column in ("clip_id", "y_true", "y_prob"):
        if column not in per_clip.columns:
            raise KeyError(f"the pooled predictions are missing {column!r}")
    if len(summary) == 0 or len(per_clip) == 0:
        raise ValueError("nothing to save: the summary or the predictions are empty")


def _flatten_folds(summary: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts are a dict per row in ``lib.metrics.evaluate``; one column each here."""
    frame = summary.copy()
    if "confusion" in frame.columns:
        for key in ("tn", "fp", "fn", "tp"):
            frame[key] = frame["confusion"].map(lambda c, k=key: c.get(k) if isinstance(c, dict) else None)
        frame = frame.drop(columns=["confusion"])
    return frame[[c for c in FOLD_COLUMNS if c in frame.columns]]


def _summary_json(summary: pd.DataFrame, config: dict) -> dict:
    out = {"run": config["run"], "n_folds": int(summary.fold.nunique()),
           "n_seeds": int(summary.seed.nunique()),
           "folds": sorted(int(f) for f in summary.fold.unique()),
           "seeds": sorted(int(s) for s in summary.seed.unique())}
    for metric in SUMMARY_METRICS:
        out[metric] = float(summary[metric].mean())
        out[f"{metric}_std"] = float(summary[metric].std())
    if "seconds" in summary:
        out["train_minutes"] = round(float(summary.seconds.sum()) / 60, 1)
    if "epochs" in summary:
        out["epochs_mean"] = float(summary.epochs.mean())
    out["trainable_params"] = config.get("trainable_params")
    out["threshold_range"] = [float(summary.threshold.min()), float(summary.threshold.max())]
    per_seed = []
    for seed, group in summary.groupby("seed"):
        row = {"seed": int(seed), "n_folds": int(len(group))}
        for metric in SUMMARY_METRICS:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        per_seed.append(row)
    out["per_seed"] = per_seed
    return out


def _board_row(summary: pd.DataFrame, config: dict, per_clip: pd.DataFrame) -> str:
    """The one line for the results board, or the reason this split has none."""
    split, protocol = config.get("split"), config.get("protocol", "v1")
    if split == "official":
        return ("This run is the **official 3-fold clip-level split**: a reproduction of the "
                "paper's protocol, not a board row. Report it only as the leaky side of the "
                "protocol effect (see `docs/protocol-effect.md`).\n")
    if protocol != "v1":
        return ("This run is a **protocol ablation** (`+v2`) and is not for the results board: "
                "the board carries `v1` rows only, so the directions stay comparable.\n")

    from lib.metrics import format_row

    board = {m: float(summary[m].mean()) for m in SUMMARY_METRICS if m in summary}
    board["acc_std"] = float(summary.acc.std())
    majority = float((per_clip.y_true.mean()
                      if per_clip.y_true.mean() >= 0.5 else 1 - per_clip.y_true.mean()))
    return format_row(config.get("direction", "tia"), board,
                      owner=config.get("owner", "—"),
                      method=config.get("variant", "—"),
                      params=config.get("trainable_params", "—"),
                      majority=majority,
                      runtime=f"{summary.seconds.mean():.0f} s/fold") + "\n"


def save_run(summary: pd.DataFrame, per_clip: pd.DataFrame, config: dict,
             root: str | Path = "results") -> Path:
    """Write ``<root>/<run>/`` and return the directory.

    Overwrites a previous run of the same name: the file set is the contract, and two
    versions of one run in the tree would make the report quote whichever it read first.
    The git commit in ``run.json`` is what identifies the code that produced it.
    """
    folds = _flatten_folds(summary)
    _check(folds, per_clip)
    config = dict(config)
    config["run"] = config.get("run") or run_name(config["variant"], config["split"],
                                                  config.get("protocol", "v1"),
                                                  config.get("captains", "include"))
    directory = Path(root) / config["run"]
    directory.mkdir(parents=True, exist_ok=True)

    folds = folds.sort_values(["seed", "fold"])
    folds.to_csv(directory / "folds.csv", index=False)

    # `train.ipynb` averages the probabilities over seeds before writing the pooled frame, so
    # `seed` and `fold` are there only when the caller passes the un-averaged `preds`.
    keep = [c for c in ("clip_id", "y_true", "y_prob", "seed", "fold") if c in per_clip.columns]
    pooled = per_clip[keep].sort_values("clip_id")
    pooled.to_csv(directory / "preds.csv", index=False)

    (directory / "summary.json").write_text(
        json.dumps(_summary_json(summary, config), indent=2) + "\n", encoding="utf-8")
    (directory / "run.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "board_row.md").write_text(_board_row(summary, config, per_clip),
                                            encoding="utf-8")
    print(f"saved {directory}: {len(folds)} fold/seed rows, {len(pooled)} clips")
    return directory
