# DOLOS — Audio-Visual Deception Detection Dataset

> Multimodal **deception detection** dataset (face + voice) built from clips of a British TV game show.
> Released by NTU ROSE Lab, published at **ICCV 2023**.
> This folder documents the dataset; the annotations themselves live in `DOLOS/`, which is **not tracked
> by git** (see [`.gitignore`](../../.gitignore)) and contains **no video files**.

---

## 1. What DOLOS is

DOLOS is the largest **non-laboratory** deception detection dataset by number of clips and subjects
(as of 2023, compared against Box of Lies, Bag-of-Lies, Real-Life Trials and MU3D):

| | |
|---|---|
| **Domain** | TV game show (*Would I Lie To You?*, BBC) |
| **Task** | binary classification: `truth` vs `deception` |
| **Modalities** | video (face) + audio (speech) + non-verbal annotations |
| **Subjects** | 213 participants (141 M / 72 F) per the paper |
| **Clips** | 1,675 (2–19 s, median 5 s) |
| **Annotations** | **MUMIN** coding scheme: facial expressions, gaze, gestures, prosody |
| **Year** | 2023 |

Its main advantage over lab-collected datasets (e.g. MU3D) or courtroom data (Real-Life Trials) is that
the deception here is **spontaneous and conversational**: participants have a real incentive to lie
(winning the round) and the ground truth is **certified by the game**, not judged by an annotator.

References:

- Official site / access request: <https://rose1.ntu.edu.sg/dataset/DOLOS/>
- Paper: <https://arxiv.org/abs/2303.12745>
- Official code (PECL + dataloader): [NMS05/Audio-Visual-Deception-Detection-DOLOS-...](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning)
- Download / pre-processing tools: [NMS05/AV-Data-Processing](https://github.com/NMS05/AV-Data-Processing)

Both external repos are available locally as git submodules (see `../vendor/` and [AGENTS.md](../../AGENTS.md)).

---

## 2. Where the label comes from

The format of the game is what makes the dataset usable:

1. Two teams of three; in turn one contestant **reads a statement about their own life** from a card.
2. The statement may be **true or false**; only the speaker knows which.
3. The opposing team asks questions and the speaker must **defend the statement**.
4. At the end of the round the truth is **revealed**: if the opponents guessed right, they win.

Consequence for labelling: if the statement was true, **all** answers of that contestant are `truth`
clips; if it was false, they are `deception` clips. The label is therefore *game ground truth*, not a
subjective judgement — the main reason the dataset is considered reliable.

Extraction filters (from the paper): clear voice with no strong background noise, face fully visible
and unoccluded, relevant content only (no off-topic chat).

---

## 3. What is in `DOLOS/`

```
DOLOS/
├── Dolos.xlsx                     # original annotations (metadata + MUMIN intervals)
├── train_dolos.xlsx               # the same 41 features, binarised to 0/1
├── dolos_timestamps.csv           # minimal metadata + timestamps, for downloading from YouTube
├── dolos_timestamps.txt           # byte-identical copy of the .csv
└── Training_Protocols/
    ├── train_fold1..3.csv         # 3-fold train/test split
    ├── short.csv / long.csv       # duration protocol
    └── male.csv / female.csv      # gender protocol
```

Role of each file, per the official documentation:

| File | Role |
|---|---|
| `Dolos.xlsx` | **Primary source.** URLs, timestamps, label, subject metadata and MUMIN annotations as *lists of time intervals*. All other annotation files derive from it. |
| `train_dolos.xlsx` | **Binarised** version: 1 = feature present in the clip, 0 = absent. Intended for PyTorch DataLoaders / multi-task learning. |
| `dolos_timestamps.csv` | Input for `YT_video_downloader.py`: downloads the clips and saves them as `.mp4`. It must be converted to seconds first (§9.1). |
| `Training_Protocols/*.csv` | Lists of file names (no extension) to be used as splits: each row is `file_name, label, gender`. |

> **No video is included.** You need the `.mp4` files to extract frames and audio. Downloads depend on
> YouTube availability; ROSE Lab does not guarantee that the links stay live.

---

## 4. File formats

### 4.1 `dolos_timestamps.csv` / `.txt`

UTF-8 **with BOM**, 5 columns, **1,678 rows** (one per clip):

| Column | Example | Notes |
|---|---|---|
| `YT_Video_ID` | `UCPIBQDI0QM` | YouTube ID of the episode |
| `file_name` | `AN_WILTY_EP15_truth1` | unique clip name |
| `start_time` | `2:06` | `mm:ss` from the start of the video |
| `end_time` | `2:11` | `mm:ss` |
| `label` | `truth` | `truth` / `deception` / casing variants |

The `.txt` is **byte-identical** to the `.csv`: use only one of them.

### 4.2 `Dolos.xlsx` — original annotations

Headers span **3 rows** (row 1 = macro-group, row 2 = sub-group, row 3 = feature name); data starts at
**row 4**. The first 8 columns are metadata:

| # | Column | Example | Notes |
|---|---|---|---|
| A | `Video URL` | `https://www.youtube.com/watch?v=…` | 136 rows use the `youtu.be/…` form |
| B | `File name of the video clip` | `AN_WILTY_EP15_truth1` | join key with the protocols |
| C | `Label "truth" or "deception"` | `truth` | unnormalised values (see §10) |
| D | `Start time in the episode` | `02:06` | see §10: **mixed storage types** |
| E | `End time in the episode` | `02:11` | idem |
| F | `Name of the game show` | `Would I Lie To You` | |
| G | `Participants name` | `Carol Voderman` | free text |
| H | `Participants gender` | `Female` | `Male` / `Female`, with variants |

Columns **I → AW (41 columns)** are the MUMIN annotations. Each cell contains either:

- `0` → feature absent in the clip;
- a string with one or more intervals, e.g. `[2:10,2:11]` or `[[2:06,2:07],[2:10,2:11]]`
  → the feature is present during those intervals (format `mm:ss`).

The macro-groups (from the merged header cells) are 4: **Visual/Facial** (16 features), **Gaze** (4),
**Gesture** (16), **Vocal** (5).

This file is therefore suited to **fine-grained temporal** work (when a smile appears, not just whether
it does). If you need a fixed vector per clip, use `train_dolos.xlsx`.

### 4.3 `train_dolos.xlsx` — binarised features

Same 3-row header, **1,678 rows**, but:

- no URLs and no timestamps (only `file_name`, `label`, `gender` + 41 features);
- every cell is `0`/`1`;
- **row order is identical** to `Dolos.xlsx` (verified: same `file_name` sequence, same labels, same genders).

Use this file for a "classic" classifier on the MUMIN features (e.g. logistic regression / MLP over the
41 indicators) or as an auxiliary task for multi-task learning.

### 4.4 `Training_Protocols/`

CSV **without header**, 3 columns: `file_name, label, gender`.

| File | Rows | `deception` | `truth` | % deception |
|---|---:|---:|---:|---:|
| `train_fold1.csv` | 1,099 | 586 | 513 | 53.3% |
| `test_fold1.csv` | 549 | 293 | 256 | 53.4% |
| `train_fold2.csv` | 1,095 | 583 | 512 | 53.2% |
| `test_fold2.csv` | 553 | 296 | 257 | 53.5% |
| `train_fold3.csv` | 1,100 | 587 | 513 | 53.4% |
| `test_fold3.csv` | 548 | 292 | 256 | 53.3% |
| `short.csv` | 769 | 405 | 364 | 52.7% |
| `long.csv` | 885 | 478 | 407 | 54.0% |
| `male.csv` | 1,245 | 705 | 540 | 56.6% |
| `female.csv` | 317 | 139 | 178 | 43.8% |

The three protocols:

- **3-fold** — the standard evaluation split. Each fold is a random *clip-level* bipartition of the same
  1,646 examples: `train_fold1 ∪ test_fold1 = train_fold2 ∪ test_fold2 = …`.
- **duration** — `short` = short clips (≈2–4 s), `long` = long clips (≈5–10 s). Used to measure how much
  a model depends on the amount of temporal signal.
- **gender** — `male` / `female`, to study gender bias of the detector.

Coverage notes (measured):

- `short` ∪ `long` = **1,652** clips (2 appear in both);
- `male` ∪ `female` = **1,560** clips (2 in both);
- **25 clips appear in no protocol at all** (mostly those with broken timestamps).

---

## 5. File naming convention

```
<BLOCK>_WILTY_EP<number>_<truth|lie|true|deception><index>
   AN   _WILTY_EP15_    truth                           1
```

- `<BLOCK>`: 2–3 letters identifying a block of episodes (`AN`, `BRI`, `LS`, `SB`, `SJ`, `YW`).
  It is neither the subject name nor the gender: treat it as an opaque identifier.
- `EP<n>`: episode number, **1–76** with a gap (74 is missing) → **75 distinct episodes**.
- the trailing token gives the class (`truth`/`true` vs `lie`/`deception`) plus a progressive index.

Examples:

```
AN_WILTY_EP15_truth1     ep.15, truthful clip, #1
AN_WILTY_EP15_lie4       ep.15, deceptive clip, #4
BRI_WILTY_EP61_lie_22    variant with an underscore in the index
LS_WILTY_EP73_ true1     variant with a leading space (careful!)
```

---

## 6. Actual statistics (measured on these files)

The official numbers and the ones computed from the local files match almost everywhere, but not
exactly: this copy contains a few extra rows and some dirty metadata.

| Metric | Paper / website | These files |
|---|---:|---:|
| Clips | 1,675 | **1,678 rows** (1,677 unique names) |
| `deception` / `truth` | 899 / 776 | **901 / 777** (53.7% / 46.3%) |
| Subjects | 213 (141 M / 72 F) | **196 names** → 131 M / 65 F (194 after alias merge) |
| TV episodes | 84 | **75** distinct YouTube IDs |
| Duration | 2–19 s, median 5 s | **2–48 s**, median **5 s**, mean ≈5 s |
| Clips per gender | 1,324 M / 351 F | **1,327 M / 351 F** (79.1% / 20.9%) |
| MUMIN features | 25 facial + 5 vocal | **41 columns** (36 visual + 5 vocal) |

Useful details:

- **Clips per episode**: 9 to 49, mean 22.4.
- **Clips per subject**: mean 8.6, median 5, max **143** (David Mitchell, a permanent team captain).
- **Subject imbalance**: David Mitchell and Lee Mack dominate the dataset → a model can learn the
  person's identity instead of the behaviour. Relevant for validation (§11).
- **5 clips exceed 19 s** (up to 48 s): most likely annotation errors, the paper states 2–19 s.
- **Duration by gender**: essentially identical mean (5.1 s F vs 5.3 s M), but the longest clip in the
  `female` protocol is **10 s** versus **41 s** in `male`: long clips are almost all male.

---

## 7. The 41 MUMIN annotations

Percentages computed over all 1,678 clips from `train_dolos.xlsx`; `Δ` = (% deception) − (% truth).
This is a **single-feature marginal** statistic: a sanity check, not evidence of signal.

| Feature | Sub-group | % clips | % deception | % truth | Δ |
|---|---|---:|---:|---:|---:|
| Smile | General face | 14.5 | 15.6 | 13.3 | +2.4 |
| Laughter | General face | 5.1 | 5.5 | 4.6 | +0.9 |
| Scowl | General face | 1.7 | 2.0 | 1.4 | +0.6 |
| Frowning | Eyebrows | 11.3 | 10.7 | 12.0 | −1.3 |
| Raising | Eyebrows | 43.7 | 45.6 | 41.6 | +4.0 |
| Frequent Blinking | Eyes | 16.7 | 18.0 | 15.2 | +2.8 |
| Exaggerated Opening (eyes) | Eyes | 4.7 | 5.0 | 4.4 | +0.6 |
| Exaggerated Closing | Eyes | 2.2 | 2.0 | 2.4 | −0.4 |
| Exaggerated Opening (mouth) | Mouth openness | 2.7 | 3.2 | 2.2 | +1.0 |
| Closed (awkward pauses) | Mouth openness | 3.1 | 2.9 | 3.3 | −0.5 |
| Corners up | Mouth lips | 18.5 | 19.9 | 17.0 | +2.9 |
| Corners down | Mouth lips | 7.6 | 7.2 | 8.0 | −0.8 |
| Protruded | Mouth lips | 5.1 | 4.6 | 5.8 | −1.2 |
| Retracted | Mouth lips | 4.2 | 3.6 | 5.0 | −1.5 |
| Move Upwards | Chin | 16.3 | 18.5 | 13.6 | +4.9 |
| Move Downwards | Chin | 13.9 | 15.1 | 12.5 | +2.6 |
| Towards interlocutor | Gaze | 77.9 | 81.7 | 73.6 | +8.1 |
| Up | Gaze | 12.4 | 8.4 | 17.0 | **−8.6** |
| Down | Gaze | 31.3 | 28.7 | 34.4 | −5.6 |
| Sideways | Gaze | 24.7 | 25.5 | 23.8 | +1.7 |
| Nod | Head movements | 25.0 | 27.9 | 21.8 | +6.1 |
| Move Forward | Head movements | 8.4 | 10.0 | 6.6 | +3.4 |
| Move Backward | Head movements | 9.4 | 11.2 | 7.3 | +3.9 |
| Tilt | Head movements | 23.3 | 25.9 | 20.3 | +5.5 |
| Side-turn | Head movements | 22.0 | 22.4 | 21.5 | +0.9 |
| Shake | Head movements | 10.7 | 11.7 | 9.7 | +2.0 |
| Head up | Head movements | 6.7 | 6.0 | 7.5 | −1.5 |
| Rubbing | Hand movements | 0.8 | 0.9 | 0.8 | +0.1 |
| Scratching | Hand movements | 1.8 | 2.2 | 1.3 | +0.9 |
| Touching head/body/etc | Hand movements | 5.3 | 5.2 | 5.4 | −0.2 |
| Covering face/mouth/etc | Hand movements | 2.1 | 2.4 | 1.8 | +0.6 |
| Crossing fingers | Hand movements | 1.4 | 1.4 | 1.3 | +0.2 |
| Palm up | Hand movements | 12.9 | 13.4 | 12.2 | +1.2 |
| Open | Arm movement | 16.7 | 17.0 | 16.5 | +0.5 |
| Folded | Arm movement | 2.2 | 2.3 | 2.1 | +0.3 |
| Shrugs | Shoulder | 2.9 | 2.8 | 3.1 | −0.3 |
| Non-ah-mmh speech disturbances | Fluency | 7.7 | 8.4 | 6.9 | +1.5 |
| Word and phrase repetitions | Fluency | 9.7 | 10.9 | 8.4 | +2.5 |
| Silent pauses | Fluency | 2.6 | 2.3 | 2.8 | −0.5 |
| Loudness | Arousal | 0.8 | 1.0 | 0.5 | +0.5 |
| Tension | Arousal | 1.0 | 0.6 | 1.4 | −0.9 |

Two things worth noting:

1. Features are **very sparse**: **24 of 41** appear in less than 10% of the clips. `train_dolos.xlsx`
   therefore yields a very sparse matrix → prefer regularisation or `class_weight` over a large MLP.
   The paper reports **58.84% ACC** with an MLP trained on exactly these MUMIN features (Table 4):
   that is the baseline to beat if you start from these columns.
2. The strongest marginal effect is **negative on looking up** (`Up`, −8.6 pp) and **positive on looking
   at the interlocutor** (+8.1 pp). Do not over-read this: it is not "liars look up less", it is that
   `deception` turns contain more interaction/conversation. **Not a causal feature**, just a dataset
   correlation.

---

## 8. Loading in Python

```python
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("DOLOS")          # folder containing the dataset files


def to_seconds(v):
    """Convert a 'time' cell of Dolos.xlsx to seconds.

    Excel stored 'mm:ss' as a time-of-day 'hh:mm', therefore:
      - datetime.time  -> hour = minutes, minute = seconds
      - timedelta      -> total minutes (e.g. 25:31 became '1 day, 1:31')
      - float          -> integer part = minutes, decimals = seconds (21.59 -> 21:59)
      - str            -> 'mm:ss', possibly with leading spaces
    """
    if isinstance(v, dt.time):
        return v.hour * 60 + v.minute
    if isinstance(v, dt.timedelta):
        return int(v.total_seconds()) // 60
    if isinstance(v, float):
        m = int(v)
        return m * 60 + round((v - m) * 100)
    if isinstance(v, str):
        m, s = re.match(r"\s*(\d+):(\d+)", v).groups()
        return int(m) * 60 + int(s)
    return np.nan


def load_annotations(path=DATA / "Dolos.xlsx"):
    df = pd.read_excel(path, header=2)          # 3 header rows
    df.columns = (
        df.columns.str.replace("\u200b", "", regex=False)   # zero-width spaces in headers
        .str.replace("\xa0", " ", regex=False)              # non-breaking spaces
        .str.strip()
    )
    df = df.rename(columns={
        "File name of the video clip": "file",
        'Label "truth" or "deception"': "label_raw",
        "Start time in the episode": "start",
        "End time in the episode": "end",
        "Participants name": "subject",
        "Participants gender": "gender",
    })

    df["file"] = df["file"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()

    lab = df["label_raw"].astype(str).str.strip().str.lower()
    df["label"] = np.where(lab.str.startswith("t"), "truth", "deception")  # 'lie' -> deception
    df["y"] = (df["label"] == "deception").astype(int)

    df["start_s"] = df["start"].map(to_seconds)
    df["end_s"] = df["end"].map(to_seconds)
    df["duration"] = df["end_s"] - df["start_s"]
    return df


df = load_annotations()
print(df["y"].value_counts())          # 1: 901, 0: 777
print(df["duration"].describe())
```

Binarised features (mind the duplicated column name, §10):

```python
feat = pd.read_excel(DATA / "train_dolos.xlsx", header=2)
feat.columns = [str(c).replace("\u200b", "").replace("\xa0", " ").strip() for c in feat.columns]
# 41 feature columns but only 40 distinct names: "Exaggerated Opening" appears twice
# (eyes and mouth). Rename by position before using names as keys:
names = list(feat.columns[3:])
names[6] = "Exaggerated_Opening_eyes"
names[8] = "Exaggerated_Opening_mouth"
feat.columns = list(feat.columns[:3]) + names
```

Protocols:

```python
cols = ["file", "label", "gender"]
test = pd.read_csv(DATA / "Training_Protocols/test_fold1.csv", header=None, names=cols)
test["gender"] = test["gender"].str.strip().str.lower()   # 'Male ' -> 'male'
```

Timestamps (mind the BOM):

```python
ts = pd.read_csv(DATA / "dolos_timestamps.csv", encoding="utf-8-sig")  # without it: 'ï»¿YT_Video_ID'
```

---

## 9. Typical pipeline

```
DOLOS/
   │  dolos_timestamps.csv
   ▼
[1] YT_video_downloader.py      → clip .mp4 (+ optional cropping)
   ▼
[2] extract_face_frames.py      → cropped RGB face frames
    video_to_audio.py           → .wav audio
   ▼
[3] audio_visual_dataset.py     → tensors (audio waveform, face frames, label)
   ▼
[4] train_test.py / train_fusion.sh
       ├── audio:  Wav2Vec2  + UT-Adapter
       ├── video:  ViT-B/16  + UT-Adapter
       └── fusion: PAVF (audio-visual)
```

Steps 1–3 live in the `AV-Data-Processing` submodule, step 4 in the `DOLOS-PECL` submodule (the **PECL**
architecture, *Parameter-Efficient Crossmodal Learning*: frozen backbones + temporal adapters +
crossmodal fusion). Both are checked out under `../vendor/`; initialise them with
`git submodule update --init --recursive`.

### 9.1 Preparing the download

`YT_video_downloader.py` expects `yt_id,file_name,start_seconds,end_seconds`, **headerless and numeric**
(`float(start_time)`), while `dolos_timestamps.csv` ships a header, a UTF-8 BOM and `mm:ss` strings.
Convert it once, otherwise the script dies on the first row with `ValueError: could not convert string
to float: '2:06'`.

```python
from pathlib import Path
import pandas as pd

DATA = Path("DOLOS")
ts = pd.read_csv(DATA / "dolos_timestamps.csv", encoding="utf-8-sig", dtype=str)


def mmss(x):
    parts = [int(p) for p in str(x).strip().split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 60


out = pd.DataFrame({
    "yt_id": ts["YT_Video_ID"],
    "file_name": ts["file_name"],
    "start": ts["start_time"].map(mmss),
    "end": ts["end_time"].map(mmss),
})
out.to_csv(DATA / "dolos_timestamps_seconds.csv", index=False, header=False)  # 1,678 rows
```

```bash
python "lie-detector/vendor/AV-Data-Processing/AV Data Processing/YT_video_downloader.py" \
       lie-detector/clips \
       lie-detector/dataset/DOLOS/dolos_timestamps_seconds.csv
#      ^ first arg = output folder (created by the script), second = headerless CSV in seconds
```

Requirements: `yt_dlp`, `ffmpeg-python` and the `ffmpeg` binary. The script writes
`bad_files_<output folder>.txt` with the IDs that could not be downloaded — with time, some of the 75
episodes will no longer be available.

---

## 10. Dataset pitfalls (verified on these files)

All of these break a hastily written pipeline. Handle them once, in a single `dataset.py` loader
("dirty → clean").

**Metadata**

1. **Unnormalised labels**: `truth`, `Truth`, `truth ` (trailing space), `deception`, `Deception`, `lie`.
   → `str.strip().str.lower()`, then map `lie → deception`. Without normalisation you get 6 classes.
2. **Unnormalised gender**: `male`, `Male`, `male `, `female`, `Female`, `female `. Same as above.
3. **Show name** in 2 spellings (`Would I Lie To You` / `Would I Lie to You`).
4. **Free-text subject names** with aliases and spaces: `Lee` vs `Lee Mack`, `David` vs `David Mitchell`,
   `Alex` vs `Alex `. Counting subjects requires manual normalisation, otherwise you get 196 instead of ~194.
5. **Duplicated row**: `YW_WILTY_EP46_truth1` appears twice → 1,678 rows but 1,677 clips.
6. **33 file names outside the pattern**: `BRI_WILTY_EP59_deception_20`, `LS_WILTY_EP73_ true1` (space),
   `..._lie_22` with an underscore. Do not parse the label from the file name: the token disagrees with
   the `Label` column on **6 rows**, and 27 names do not contain the token at all.
7. **YouTube URLs** in two forms (`watch?v=` and `youtu.be/`), 76 distinct URLs for 75 episodes.

**Time**

8. **`Start/End time` columns hold 4 different types**: `datetime.time`, `timedelta`, `float`
   (21.59 = 21:59) and `str` (`"  2:15"`). The `to_seconds` helper in §8 handles all of them.
   The cause is that annotators typed `mm:ss` and Excel read it as `hh:mm`.
9. **7 clips with `end ≤ start`**: `BRI_WILTY_EP61_lie_22`, `BRI_WILTY_EP63_lie_28`, `LS_WILTY_EP72_lie17`,
   `LS_WILTY_EP75_true4`, `LS_WILTY_EP76_true3`, `LS_WILTY_EP76_lie12`, `LS_WILTY_EP76_lie15`.
   Drop them or fix them by hand.
10. **40 CSV rows have timestamps without `:`** (`6`, `8`, `20`): these mean `m:00`, not seconds.
11. **`Dolos.xlsx` and `dolos_timestamps.csv` are not perfectly aligned**: 1,667 of 1,678 clips have the
    same duration; about ten (concentrated in episodes 72–76) differ by a few seconds.
    → **pick one source** and use it everywhere (recommendation: `Dolos.xlsx`, it is the primary one).
12. **5 clips longer than 19 s** (max 48 s), outside the range stated by the paper.

**Features**

13. **`Exaggerated Opening` is duplicated**: two columns with the same name (eyes and mouth opening).
    pandas yields `.1` suffixes, or ambiguous columns if you normalise → rename by **position**.
14. **Zero-width and non-breaking spaces** in the headers (`Smile\u200b`, `Hand Movements\xa0\u200b`):
    `df["Smile"]` fails until you strip them.
15. **10 cells hold the value `9`** instead of `0/1` (rows `YW_WILTY_EP49_lie1` and `SB_WILTY_EP29_lie4`).
    In `train_dolos.xlsx` they were already converted to `1`; if you use `Dolos.xlsx`, decide yourself
    (most likely a "missing" marker).
16. **UTF-8 BOM** in `dolos_timestamps.csv` → `encoding="utf-8-sig"`.

---

## 11. Notes for a lie detection project

These change results more than any architecture choice:

1. **The splits are not speaker-independent.** Verified: `train_fold1` has 182 subjects and `test_fold1`
   159, with **147 in common**; all 75 episodes appear in both. The 3-fold is therefore a random
   *clip-level* split, neither person- nor episode-level. Deception is highly personal (tics, tone,
   gestures): a model can recognise the person and "guess" the class. If you care about generalisation,
   build your own **subject-level** or **episode-level** split and report both numbers.
2. **Residual leakage even in the official split**: 1 clip in fold1 and 2 in fold3 are in train *and*
   test. Clean up before evaluating.
3. **Balance**: 53.7% deception at clip level, but the per-subject distribution is very skewed
   (David Mitchell has 143 clips, the median is 5). Use macro averages, not accuracy, and consider
   resampling per subject.
4. **Gender bias**: male clips are 79% of the dataset and show 56.6% deception versus 43.8% for female
   clips. Reporting accuracy per gender is almost mandatory.
5. **`gender` is a subject attribute**, not a clip attribute: do not feed it as input if you want a
   detector that generalises (and it is a sensitive attribute anyway).
6. **Do not expect high numbers**: 5-second median clips and sparse features make the task hard even for
   multimodal models. The paper's 3-fold baselines are **61.4%** visual, **59.2%** audio, **64.8%** PAVF
   fusion, **66.8%** with multi-task learning. A claimed 90% is almost always subject/episode leakage
   (point 1) or an undisclosed split.
7. **Reproducibility**: the videos live on YouTube and can disappear. Store a hash/version of the
   downloaded clips and record which IDs were available, otherwise results are not replicable.

---

## 12. Licence and citation

**Terms of use (from the ROSE Lab website):**

- released **for academic research only**, non-commercial, to educational/research institutes;
- ROSE Lab **does not guarantee** YouTube link availability and **does not own the copyright** of the
  show's videos;
- **prohibited** without explicit permission: redistribution, creating derived datasets, any commercial use;
- access requires accepting the *Release Agreement* (request form on the website).

The files under `DOLOS/` are derived annotations: check the terms before redistributing or publishing
them (including derived artifacts such as extracted features). They are kept out of git for this reason —
do not commit them.

**Citation:**

```bibtex
@inproceedings{guo2023dolos,
  title     = {Audio-Visual Deception Detection: DOLOS Dataset and
               Parameter-Efficient Crossmodal Learning},
  author    = {Guo, Xiaobao and Selvaraj, Nithish Muthuchamy and Yu, Zitong
               and Kong, Adams Wai-Kin and Shen, Bingquan and Kot, Alex},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

---

*This README was generated by analysing the files in `DOLOS/` (1,678 rows, 41 features, 10 protocol
files). Statistics were recomputed from the data rather than copied from the paper; where the two
diverge, it is stated explicitly.*
