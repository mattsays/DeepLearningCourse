# AGENTS.md

Operating notes for AI agents (and humans) working in this repository.

---

## 1. What this repository is

A Deep Learning course project, split in two independent parts:

| Path | Content |
|---|---|
| `presentation/` | HTML slide deck + the `pmaker.html` tool that renders and exports it to PDF |
| `lie-detector/` | Multimodal **deception detection** project built on the DOLOS dataset — project README: [`lie-detector/README.md`](lie-detector/README.md) |

Everything under `lie-detector/` belongs to the same goal: train and evaluate a detector that
classifies a video clip as `truth` or `deception`.

---

## 2. Hard rules

1. **Language: English.** All documentation, code, comments, commit messages, file names, agent
   summaries and dataset notes are written in English. No exceptions — do not mirror the user's
   language into repository files.
2. **Never commit the dataset or any video.** `lie-detector/dataset/DOLOS/` is git-ignored on purpose
   (see `.gitignore`). It contains ROSE Lab annotations released *for academic use only* and
   `youtube`-hosted footage whose copyright is not ours. Redistribution and derived datasets are
   prohibited without explicit permission. If a task seems to require committing data, stop and ask.
3. **Numbers must be measured, not remembered.** Any statistic quoted in a README or report is
   recomputed from the files in `lie-detector/dataset/DOLOS/`. If a number comes from the paper or the
   website, label it as such and say when the local data diverges.
4. **Do not invent dataset facts.** If something cannot be verified from the files, write that it is
   unverified instead of filling the gap.
5. **Keep third-party code in submodules**, never copy-paste it into the tree. Submodules live under
   `lie-detector/vendor/` (tooling) and `lie-detector/baselines/` (reference implementations) — see §4.

---

## 3. Repository layout

```
DeepLearningCourse/
├── AGENTS.md                      # this file
├── .gitignore                     # .venv, dist, DOLOS and every generated artifact
├── presentation/                  # slide deck + pmaker tool (not related to lie-detector)
└── lie-detector/
    ├── README.md                  # project README: goal, layout, preprocessing, evaluation, plan
    ├── AGENTS.md                  # project-level rules (data, notebooks, shared files, results)
    ├── dataset/
    │   ├── README.md              # dataset documentation — the reference for everything DOLOS
    │   └── DOLOS/                 # DATASET, NOT TRACKED, NO VIDEOS (annotations only)
    ├── notebooks/
    │   ├── preprocessing.ipynb    # SHARED: splits + download + publish to Drive
    │   ├── architecture_template.ipynb   # skeleton to copy per direction
    │   └── architectures/         # one sub-folder per research direction
    ├── scripts/
    │   ├── build_dataset.py       # clips + dataset.csv (project README §3.1)
    │   └── make_splits.py         # frozen splits      (project README §3.2)
    ├── lib/
    │   ├── dataset.py             # splits + loaders imported by notebooks (§3.3)
    │   └── metrics.py             # ACC / F1 / AUC, identical for every direction
    ├── data/processed/
    │   ├── clips/                 # downloaded .mp4 — NOT TRACKED
    │   ├── audio/                 # 16 kHz wav — NOT TRACKED
    │   ├── dataset.csv            # clip table + MUMIN — NOT TRACKED
    │   └── splits/                # frozen protocol — published on Drive, NOT TRACKED
    ├── vendor/                    # submodule: third-party tooling
    │   └── AV-Data-Processing/    # YouTube downloader + face/audio extraction
    └── baselines/                 # submodule: reference implementations
        └── DOLOS-PECL/            # official training code (PECL architecture)
```

---

## 4. Submodules

Two official NTU/NMS05 repositories are registered as git submodules — one as tooling, one as the
reference implementation. Both are useful for the full pipeline and are pinned to `main` at the commits
recorded in the index.

| Path | Upstream | Purpose |
|---|---|---|
| `lie-detector/vendor/AV-Data-Processing` | [NMS05/AV-Data-Processing](https://github.com/NMS05/AV-Data-Processing) | `YT_video_downloader.py`, `extract_face_frames.py`, `video_to_audio.py` |
| `lie-detector/baselines/DOLOS-PECL` | [NMS05/Audio-Visual-Deception-Detection-DOLOS-…](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning) | `scripts/` — dataloader, Wav2Vec2/ViT + UT-Adapter models, PAVF fusion, `train_fusion.sh` |

After cloning this repository, initialise them with:

```bash
git submodule update --init --recursive
```

Do not edit files inside a submodule: changes there are lost and break pinning. If a patch is genuinely
needed, copy the file into our tree (with attribution) or open a PR upstream.

---

## 5. The DOLOS dataset

**Documentation:** [`lie-detector/dataset/README.md`](lie-detector/dataset/README.md) — file formats,
column schemas, naming convention, measured statistics, per-feature table, pitfalls. Read it before
writing any loader; it is the single source of truth for this project.

**What is on disk** (local only, ~1 MB of spreadsheets):

| File | Content |
|---|---|
| `DOLOS/Dolos.xlsx` | primary annotations: URL, timestamps, label, subject, 41 MUMIN features as time intervals |
| `DOLOS/train_dolos.xlsx` | the same 41 features binarised to 0/1, row-aligned with the above — merged into `dataset.csv` as `mumin_*` |
| `DOLOS/dolos_timestamps.csv` | `YT_Video_ID, file_name, start_time, end_time, label` — input for the downloader |
| `DOLOS/Training_Protocols/*.csv` | `train_fold1..3`, `test_fold1..3`, `short`, `long`, `male`, `female` |

**What is *not* on disk:** any `.mp4`. Videos are fetched with `scripts/build_dataset.py` (preferred — it
cuts, extracts audio and writes `data/processed/dataset.csv`, see project README §3.1), and availability
is not guaranteed. The upstream script below is kept for reference only.

⚠️ `YT_video_downloader.py` does **not** accept `dolos_timestamps.csv` as-is. It expects
`yt_id,file_name,start_seconds,end_seconds` with **no header**, while the shipped CSV has a header
(plus a UTF-8 BOM) and `mm:ss` times — `float("2:06")` raises and the run dies on the first row.
The conversion script that produces the headerless seconds CSV is in
[`lie-detector/dataset/README.md` §9.1](lie-detector/dataset/README.md); run it once, then:

```bash
python "lie-detector/vendor/AV-Data-Processing/AV Data Processing/YT_video_downloader.py" \
       lie-detector/clips lie-detector/dataset/DOLOS/dolos_timestamps_seconds.csv
#                 ^ output folder (created by the script)   ^ CSV in seconds, headerless
```

It requires `yt_dlp`, `ffmpeg-python` and an `ffmpeg` binary on `PATH`, and writes
`bad_files_<output folder>.txt` listing every ID that failed (deleted/geo-blocked videos).

### Facts that are easy to get wrong

| Topic | Reality |
|---|---|
| Size | 1,678 rows, 1,677 unique clip names (one duplicated row) |
| Labels | 901 `deception` / 777 `truth`, stored with inconsistent casing and `lie` as a synonym |
| Timestamps | Excel stored `mm:ss` as `hh:mm` — cells mix `time`, `timedelta`, `float` and `str` |
| Features | 41 columns, but only 40 distinct names (`Exaggerated Opening` ×2) |
| Official stats | paper: 1,675 clips / 213 subjects / 84 episodes; local files differ (75 episodes, 196 names) |

The full list of 16 verified pitfalls is §10 of the dataset README. Do not write a loader without it.

### Evaluation caveat

The official 3-fold split is **clip-level**, not speaker-level: `train_fold1` and `test_fold1` share 147
subjects out of 182/159 and all 75 episodes, plus 1–2 clips leak across the boundary. A model can learn
the person instead of the behaviour. If a task asks for generalisation claims, build a subject-level or
episode-level split and report both. Population: 79% male clips, 53.7% deception.

### Access and licence

Annotations are for academic, non-commercial use; redistribution and derived datasets require permission
from NTU ROSE Lab. Access is granted through the request form at
<https://rose1.ntu.edu.sg/dataset/DOLOS/>. Keep the local copy inside this workspace only. Cite the ICCV
2023 paper (BibTeX in the dataset README).

---

## 6. Environment

- Python: `.venv/bin/python` at the repository root. It has `numpy`, `pandas`, `torch`, `scikit-learn`,
  `matplotlib`, plus `yt-dlp` and `openpyxl` (needed by `scripts/build_dataset.py` and to read
  `Dolos.xlsx`). `ffmpeg`/`ffprobe` are on `PATH` via Homebrew.
- The `presentation/` tool is pure HTML/JS: open `presentation/pmaker.html` in a browser, keep the
  bundled `presentation/vendor/` folder beside it for PDF export. There is no build step and no server.
- Paths in docs and scripts are written relative to the repository root. Do not hardcode absolute
  paths in committed files.

---

## 7. Working conventions

1. **Check before you write.** Read the file (or run the query) that the task depends on; do not
   extrapolate from names or from an earlier summary.
2. **Verify, then report.** Any claim about counts, distributions or file contents is produced by
   running code in this session, not from memory.
3. **Keep the dataset out of git.** `git add -A` will silently skip it — that is intended. Never
   `git add -f` inside `lie-detector/dataset/DOLOS/`.
4. **Self-contained deliverables.** A new document states where its inputs live, which file it analyses
   and how to reproduce its numbers.
5. **Small, focused diffs.** Do not reformat files you are not otherwise changing.
