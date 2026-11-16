# Data pipeline

From the DOLOS spreadsheets to the folder every notebook reads. Written by
[`scripts/build_dataset.py`](../scripts/build_dataset.py) and driven by
[`notebooks/preprocessing.ipynb`](../notebooks/preprocessing.ipynb).
Read this if you are running the preprocessing, changing an artifact, or wondering where a file lives.

## Overview

Goal: **after preprocessing, every member works from the same clips and the same splits**, which is the
only way four people without GPUs can iterate before 26 November.

The work is implemented by two scripts; the notebook only drives them.

```bash
python scripts/build_dataset.py --workers 4     # clips + audio + dataset.csv
python scripts/make_splits.py                   # frozen splits
```

```
data/
└── processed/
    ├── clips/<clip_id>.mp4      # the annotated range, video + audio track
    ├── audio/<clip_id>.wav      # 16 kHz mono
    ├── dataset.csv              # one row per clip: label, subject, paths, 41 MUMIN features
    ├── splits/*.json            # frozen protocol — published with the dataset
    └── <direction>/             # a member's own caches
```

| Artifact | Content |
|---|---|
| `clips/` | one `.mp4` per clip that downloaded successfully |
| `audio/` | 16 kHz mono wav extracted from the clip's audio track |
| `dataset.csv` | label, `y`, gender, subject, both paths, download status, **41 MUMIN columns** |
| `splits/` | the virgin test set and the 5 cross-validation folds |
| `cached_features/<encoder>/` | features produced by one encoding model (see below) |

**Why it stops there.** Any further step — cropping faces, sampling frames differently, computing
features, caching embeddings — already encodes a modelling choice, and this project is precisely about
letting four different choices compete. Each member does that inside their own notebook and caches
whatever they like under `data/processed/<direction>/`. Two members who happen to want the same
intermediate representation can share it privately; the shared pipeline does not pick one.

The one exception is the **shared feature cache**: anything another member may reuse goes to
`cached_features/<encoder>/` in the Drive folder, which is where `cache_dir` and the template write
(`DATA/cached_features/<ENCODER>/<clip_id>.npy`). `data/processed/` is local scratch and is never the
place for something the group depends on.

Rules:

1. **Only the scripts touch the raw data.** Nobody re-downloads or re-decodes clips inside an architecture
   notebook.
2. **Idempotent and resumable** — `build_dataset.py` skips what is already on disk, because Colab sessions
   die mid-run.
3. **Mount Drive first** (`from google.colab import drive; drive.mount('/content/drive')`) and keep
   artifacts there; `/content` is wiped on disconnect.
4. **Same clip set for everybody** — `Clips.load()` only hands out clips whose video exists, and
   `missing_from()` reports a split clip that is absent from `dataset.csv`. Use those, not your own
   filter, or two members train on different subsets and nothing is comparable.

## The canonical dataset

One script turns the annotations into the files every notebook reads:

```
clips/<clip_id>.mp4                   # annotated range only, video + audio track
data/processed/audio/<clip_id>.wav    # 16 kHz mono
data/processed/dataset.csv            # one row per clip: metadata + MUMIN — the source of truth
```

`clip_id` is the lower-case slug of the DOLOS `file_name`, so it can always be recomputed from the
dataset files alone: `AN_WILTY_EP15_truth1` → `an_wilty_ep15_truth1`. The single duplicated name in
DOLOS becomes `yw_wilty_ep46_truth1` and `yw_wilty_ep46_truth1_2`.

`dataset.csv` carries, per clip: the 17 metadata columns above (`clip_id, source_name, yt_id, start_s,
end_s, duration_expected, duration_actual, label, y, gender, subject, test_fold, protocols, video_path,
audio_path, status, error`) plus the **41 binary MUMIN features** as `mumin_*` columns. Paths are relative
to the folder that contains the CSV, so the archive can be extracted anywhere.

`dataset.csv`, `clips/` and `audio/` are published together in the shared Drive folder (see below).

```bash
python scripts/build_dataset.py --dry-run                  # writes dataset.csv, downloads nothing
python scripts/build_dataset.py --limit 20                 # smoke test
python scripts/build_dataset.py --workers 4                # full run
```

**If a run stops before the end.** Just run the same command again: it resumes from what is on Drive.
Every clip ends up with a `status` in `dataset.csv`, so nothing is lost silently:

| status | what it means | what to do |
|---|---|---|
| `ok` | video + audio, usable | nothing |
| `error` | yt-dlp or ffprobe failed | re-run; if they all look the same, lower `--workers` |
| `no_audio` | video fine, no audio track | re-run, or drop the clip |
| `duration_mismatch` | cut is off by more than 2 s | inspect it, usually a YouTube segment shift |
| `invalid_range` | `end <= start` in the annotations | nothing, there are 7 of them |
| `planned` | never attempted | run the download |

A single broken clip can no longer abort the job: exceptions are caught per clip and recorded. The
script exits 0 when the pass completes, whatever the counts; read the `status:` line, not the exit code.

Those statuses are not just diagnostics: the splits are built from the clips marked `ok`
([splits.md](splits.md)), so the dataset you managed to download is the dataset you evaluate on.

The script is idempotent (valid existing clips are skipped), resumable, and marks broken annotations
(`end ≤ start`, 7 rows) as `invalid_range` instead of downloading them. Budget roughly 15 s per clip:
a full run is a couple of hours with 4 workers.

## The shared folder on Drive

One folder, one path, for everybody:

```
dolos/                          # /content/drive/MyDrive/dolos
├── dataset.csv                 # 1 MB
├── clips/                      # ~510 MB, 1,671 files
├── audio/                      # ~270 MB, 1,671 files
├── splits/                     # the frozen protocol
└── cached_features/            # one sub-folder per encoding model
    ├── wav2vec2-base/<clip_id>.npy
    └── vit-b16/<clip_id>.npy
```

The owner shares it, everybody else adds a **shortcut** named `dolos` in their own My Drive, so the path
`/content/drive/MyDrive/dolos` is identical in every notebook. There is no archive to extract: the folder
*is* the dataset.

`cached_features/` is named after the **encoder**, not after the direction: two people using the same model
land in the same folder for free, and a third can see what already exists before recomputing it.

The rule for reading it: **anything read once per epoch must come from RAM or local disk**. `dataset.csv`
and the feature caches are fine over the mount — `CachedFeatures` preloads a cache into RAM when it is
under 1 GB, so features cost nothing per epoch. The 1,671 `mp4`/`wav` files do not: decoding video from the
mount pays a network round-trip per file, so a video direction should copy `clips/` to local disk first.

Since nothing is versioned, the loader prints the identity of the table it is reading:

```
dataset.csv sha256:a1027d7ef46d  1678 clips  2 with media
```

Post it in the chat. Two members with different fingerprints are not training on the same dataset, and
nothing they report is comparable.

