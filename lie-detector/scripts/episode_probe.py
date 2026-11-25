"""Episode probe on the frozen DINOv2 embeddings.

Question the report leaves open: the subject-independent split removes the *person*, but not the
*scene*. An episode is one recording session — same studio, lighting, camera setup, audio chain and
day — and 82-94% of the holdout clips still come from an episode that appears in training. If a
linear probe can name the episode from the embedding, then the encoder carries scene information and
the 56% of Appendix A may still include it.

Three probes, each answering a different question:

  1. episode, subjects held out   can the episode of a *new* person be named? (generalises)
  2. speaker, random folds        how much person identity sits in the same vector? (separability)
  3. within-speaker episode       one person across many episodes: only the scene can carry the signal

Probes 2 and 3 use random folds on purpose: they ask whether the vector *separates* the classes, not
whether a class can be predicted for a person or a scene never seen. Probe 1 uses GroupKFold on the
subject, which is the protocol the report cares about.

Each probe is also run on permuted labels, giving the empirical chance level for the same classifier,
dimensionality and folds.

Run where the feature cache exists (a local copy, or Colab with the Drive folder mounted):

    python scripts/episode_probe.py

Reads:
    data/processed/dataset.csv                      label, subject, episode
    data/processed/cached_features/dinov2-small/    one .npy per clip, 16 x 384

Nothing is trained for the task and nothing is written: the encoders stay frozen and the script only
reads. The numbers it prints are the ones to quote in Appendix A.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.dataset import cache_dir  # noqa: E402

MIN_EPISODES_PER_SPEAKER = 10
MIN_CLIPS_PER_SPEAKER = 40
SEED = 0
FOLDS = 5


def read_meta(dataset_csv: Path) -> dict[str, tuple[str, str]]:
    """clip_id -> (subject, episode). The episode is the ``EP<n>`` / ``ep<n>`` token of the source name."""
    out: dict[str, tuple[str, str]] = {}
    with dataset_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["source_name"]
            match = re.search(r"ep\s*(\d+)", name, re.IGNORECASE)
            episode = f"EP{int(match.group(1))}" if match else name.split("_")[0]
            out[row["clip_id"].lower()] = (row["subject"], episode)
    return out


def read_embeddings(cache: Path, meta: dict[str, tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    """Mean-pool each clip's frame vectors: the same summary the fusion head starts from."""
    vectors, ids = [], []
    for clip_id in meta:
        path = cache / f"{clip_id}.npy"
        if not path.exists():
            continue
        vectors.append(np.load(path).mean(axis=0))
        ids.append(clip_id)
    if not vectors:
        raise SystemExit(f"no cached vectors under {cache}; run extract_features.ipynb first")
    return np.vstack(vectors), ids


def _folds(n: int, groups) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        return list(KFold(n_splits=min(FOLDS, n), shuffle=True, random_state=SEED).split(np.zeros(n)))
    unique = len(set(groups))
    return list(GroupKFold(n_splits=max(2, min(FOLDS, unique))).split(np.zeros(n), groups=groups))


def accuracy(x, labels, groups=None) -> float:
    x = StandardScaler().fit_transform(x)
    correct = total = 0
    for train, test in _folds(len(labels), groups):
        if len(set(labels[train])) < 2:
            continue
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(x[train], labels[train])
        correct += int((model.predict(x[test]) == labels[test]).sum())
        total += len(test)
    return correct / total


def permuted(x, labels, groups=None, repeats=5) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = [accuracy(x, rng.permutation(labels), groups) for _ in range(repeats)]
    return float(np.mean(values)), float(np.std(values))


def report(name: str, x, labels, groups=None) -> None:
    labels = np.asarray(labels)
    accuracy_value = accuracy(x, labels, groups)
    _, counts = np.unique(labels, return_counts=True)
    chance, spread = permuted(x, labels, groups)
    print(
        f"{name:<34} accuracy {accuracy_value:6.3f}   majority {counts.max() / len(labels):6.3f}   "
        f"permuted {chance:6.3f} ± {spread:5.3f}   classes {len(counts):3d}   n {len(labels)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="defaults to data/processed")
    parser.add_argument("--encoder", default="dinov2-small", help="cache folder name")
    parser.add_argument("--synthetic", action="store_true", help="smoke test on random vectors")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or root / "data" / "processed"
    meta = read_meta(data_dir / "dataset.csv")

    if args.synthetic:
        rng = np.random.default_rng(SEED)
        ids = list(rng.choice(list(meta), size=700, replace=False))
        x = rng.normal(size=(len(ids), 384))
        print("synthetic mode: random vectors, every accuracy must sit at chance")
    else:
        x, ids = read_embeddings(cache_dir(args.encoder, data_dir), meta)

    subjects = np.array([meta[i][0] for i in ids])
    episodes = np.array([meta[i][1] for i in ids])
    print(f"clips with vectors: {len(ids)} / {len(meta)}   episodes {len(set(episodes))}   "
          f"subjects {len(set(subjects))}\n")

    report("episode (subjects held out)", x, episodes, subjects)
    report("speaker (random folds)", x, subjects)
    report("episode (random folds)", x, episodes)

    counts: dict[str, set[str]] = {}
    for clip_id in ids:
        subject, episode = meta[clip_id]
        counts.setdefault(subject, set()).add(episode)
    frequent = sorted((s for s, seen in counts.items()
                       if len(seen) >= MIN_EPISODES_PER_SPEAKER
                       and sum(1 for c in ids if meta[c][0] == s) >= MIN_CLIPS_PER_SPEAKER),
                      key=lambda s: -len(counts[s]))
    for speaker in frequent:
        rows = [k for k, clip_id in enumerate(ids) if meta[clip_id][0] == speaker]
        report(f"episode of {speaker[:20]}", x[rows], [meta[ids[k]][1] for k in rows])
        print(f"{'':34} {len(rows)} clips across {len(counts[speaker])} episodes, same person "
              "in all of them: anything above chance is the scene")


if __name__ == "__main__":
    main()
