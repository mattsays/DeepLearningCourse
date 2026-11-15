# Running on Colab

Google Colab is the only GPU available to the team. These are the limits and the habits that make
the difference between a session that produces a result and one that produces a crash.

## The budget

Nobody has a local GPU, so the workflow is built around Colab's limits: a free T4, sessions that die
after a few idle hours, and a 12-hour ceiling.

- **Save everything to Drive**: checkpoints, caches, logs, result JSONs. `/content` is disposable.
- **Pin versions** (`%pip install -q transformers==<x> torch==<y>`) — Colab images change without notice
  and a notebook that breaks the day before the deadline is a lost direction.
- **Small first.** Debug on 200 clips, then scale to the full split.
- **Record the runtime** (GPU type, package versions) in your result row; it is part of reproducibility.
- Stay within roughly 8 GB of GPU memory and a single-digit number of hours per experiment, or the
  experiment will not be reproducible by someone else in the group.

## Setup, per session

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone -q https://github.com/mattsays/DeepLearningCourse /content/DeepLearningCourse
%cd /content/DeepLearningCourse/lie-detector
!git pull -q
import sys; sys.path.insert(0, '.')
%load_ext autoreload
%autoreload 2
```

`%autoreload` is what makes `lib/` editable inside Colab: change a file, re-run the cell, no restart.

## Two things that bite

- **The Drive mount is a network filesystem.** Reading 1,700 small files per epoch from it is not slow,
  it is unusable. `dataset.csv` and feature caches are fine (`CachedFeatures` preloads a cache under
  1 GB into RAM); video decoding is not.
- **A disconnect wipes `/content` but not Drive.** That is why the preprocessing writes straight into the
  shared folder and every run is resumable.

