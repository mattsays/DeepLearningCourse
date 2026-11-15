# AGENTS.md

Operating notes for AI agents (and humans) working in this repository.

## 1. What this repository is

A Deep Learning course project in two independent parts:

| Path | Content |
|---|---|
| `presentation/` | HTML slide deck + the `pmaker.html` tool that renders and exports it to PDF |
| `lie-detector/` | Multimodal **deception detection** project on the DOLOS dataset |

Everything under `lie-detector/` serves one goal: train and evaluate a detector that classifies a video
clip as `truth` or `deception`. Start from [`lie-detector/README.md`](lie-detector/README.md); the rules
for that subtree are in [`lie-detector/AGENTS.md`](lie-detector/AGENTS.md).

## 2. Hard rules

1. **Language: English.** All documentation, code, comments, commit messages, file names, agent
   summaries and dataset notes are written in English. No exceptions — do not mirror the user's
   language into repository files.
2. **Never commit the dataset or any video.** `lie-detector/dataset/DOLOS/` is git-ignored on purpose:
   it holds ROSE Lab annotations released *for academic use only*, and the footage is hosted on YouTube
   under someone else's copyright. Redistribution and derived datasets are prohibited without explicit
   permission. If a task seems to require committing data, stop and ask.
3. **One fact, one owner.** Every number, path and rule is stated in exactly one document; the others
   link to it. §3 says which document owns what. If you find the same fact stated twice, delete one
   copy instead of updating both.
4. **Numbers must be measured, not remembered.** Any statistic quoted in a README or report is
   recomputed from the files in `lie-detector/dataset/DOLOS/`. If a number comes from the paper or the
   website, label it as such and say when the local data diverges.
5. **Do not invent dataset facts.** If something cannot be verified from the files, write that it is
   unverified instead of filling the gap.
6. **Keep third-party code in submodules**, never copy-paste it into the tree. Submodules live under
   `lie-detector/vendor/` (tooling) and `lie-detector/baselines/` (reference implementation) — see §5.

## 3. Where the truth lives

Read only the row your task needs. Each entry is the single owner of what it describes.

| If the task is… | Read | Section |
|---|---|---|
| what the project is, how to run it | [`lie-detector/README.md`](lie-detector/README.md) | all |
| rules for the lie-detector subtree | [`lie-detector/AGENTS.md`](lie-detector/AGENTS.md) | all |
| DOLOS files, columns, statistics, pitfalls | [`lie-detector/dataset/README.md`](lie-detector/dataset/README.md) | §4, §6, §8, §10 |
| clips, audio, `dataset.csv`, Drive folder | [`lie-detector/docs/data.md`](lie-detector/docs/data.md) | all |
| splits, seeds, leakage, protocol files | [`lie-detector/docs/splits.md`](lie-detector/docs/splits.md) | all |
| metrics, baselines, how to report | [`lie-detector/docs/evaluation.md`](lie-detector/docs/evaluation.md) | all |
| notebook contract, `lib/` API | [`lie-detector/docs/notebooks.md`](lie-detector/docs/notebooks.md) | all |
| Colab limits and session setup | [`lie-detector/docs/colab.md`](lie-detector/docs/colab.md) | all |
| research paradigms (report material) | [`lie-detector/docs/architectures.md`](lie-detector/docs/architectures.md) | all |
| slides and PDF export | [`presentation/README.md`](presentation/README.md) | all |

## 4. Layout

```
DeepLearningCourse/
├── AGENTS.md                      # this file: rules and routing
├── .gitignore                     # .venv, dist, DOLOS and every generated artifact
├── presentation/                  # slide deck + pmaker tool (not related to lie-detector)
└── lie-detector/                  # deception detection: README.md, AGENTS.md, docs/, notebooks/, scripts/, lib/
```

The subtree has its own map in [`lie-detector/README.md`](lie-detector/README.md) §Layout — this file
does not repeat it.

`lie-detector/dataset/DOLOS/` holds the annotations only (~1 MB of spreadsheets, no video). How to get
access, and under which licence, is §12 of the dataset README; what is inside each file is §3–§4.

## 5. Submodules

Two official NTU/NMS05 repositories are registered as git submodules — one as tooling, one as the
reference implementation. Both are pinned to `main` at the commits recorded in the index.

| Path | Upstream | Purpose |
|---|---|---|
| `lie-detector/vendor/AV-Data-Processing` | [NMS05/AV-Data-Processing](https://github.com/NMS05/AV-Data-Processing) | `YT_video_downloader.py`, `extract_face_frames.py`, `video_to_audio.py` |
| `lie-detector/baselines/DOLOS-PECL` | [NMS05/Audio-Visual-Deception-Detection-DOLOS-…](https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning) | `scripts/` — dataloader, Wav2Vec2/ViT + UT-Adapter models, PAVF fusion, `train_fusion.sh` |

After cloning this repository, initialise them with:

```bash
git submodule update --init --recursive
```

Do not edit files inside a submodule: changes there are lost and break pinning. If a patch is genuinely
needed, copy the file into our tree (with attribution) or open a PR upstream. Their code is not ours —
when you search the repository, a hit under `vendor/` or `baselines/` describes upstream behaviour, not
this project's.

## 6. Environment

- Python: `.venv/bin/python` at the repository root. It has `numpy`, `pandas`, `torch`, `scikit-learn`,
  `matplotlib`, plus `yt-dlp` and `openpyxl` (needed by `scripts/build_dataset.py` and to read
  `Dolos.xlsx`). `ffmpeg`/`ffprobe` are on `PATH` via Homebrew.
- The `presentation/` tool is pure HTML/JS: open `presentation/pmaker.html` in a browser, keep the
  bundled `presentation/vendor/` folder beside it for PDF export. There is no build step and no server.
- Training runs on Google Colab, not locally: limits and session setup are in
  [`lie-detector/docs/colab.md`](lie-detector/docs/colab.md).
- Paths in docs and scripts are written relative to the repository root. Do not hardcode absolute
  paths in committed files.

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

## 8. Current state

Nothing generated is in the repository: `data/processed/` does not exist and the clips, `dataset.csv`
and the splits live only in the shared Drive folder, so `python -m lib.dataset` fails until it is
mounted or copied — expected, not a broken checkout. Everything that changes weekly (which notebooks
and board rows exist) is in the README's `## State`.
