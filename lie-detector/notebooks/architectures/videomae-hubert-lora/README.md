# `lore` — VideoMAE + HuBERT with LoRA, late fusion

Direction `lore` of the [lie-detector project](../../../README.md#directions): **Paradigm 2** in
[`docs/architectures.md`](../../../docs/architectures.md). Two frozen self-supervised Transformers,
VideoMAE-Base on 16 face crops of the speaker and HuBERT-Base on the raw voice. Each is adapted with
LoRA (r = 8 on attention Q/V, 297 k trainable parameters per branch), and the two are combined by a late
fusion whose weight is chosen on `val`. The full write-up is [`docs/report.md`](docs/report.md).

**Status: run, seed 0, both protocols.** Subject-level 5 folds (the board) and the paper's official
3 folds (reported separately).

## Results

Subject-independent 5-fold, holdout, threshold chosen on `val`
(source: [`results/fusion-video-lora-r8+audio-lora-r8/comparison.txt`](results/fusion-video-lora-r8+audio-lora-r8/comparison.txt)):

| Model | ACC | F1 | AUC | Balanced ACC | Trainable |
|---|---:|---:|---:|---:|---:|
| video only | 51.6 ± 4.6 | 41.7 | 53.6 | 51.5 | 297,217 |
| audio only | 56.1 ± 5.5 | 56.0 | 59.5 | 56.2 | 297,217 |
| **fused, w tuned on val** | **56.1 ± 7.3** | 55.1 | **60.6** | 56.2 | 594,434 |
| fused, w = 0.5 | 55.4 ± 7.7 | 60.5 | 59.7 | 56.1 | 594,434 |
| *majority class (pooled)* | 52.1 | — | — | — | — |

Paper's clip-level 3 folds, same models
(source: [`results/fusion-video-lora-r8-official+audio-lora-r8-official/comparison.txt`](results/fusion-video-lora-r8-official+audio-lora-r8-official/comparison.txt)).
This protocol is not comparable with the board:

| Model | ACC | AUC | Paper-style ACC (best test epoch, thr 0.5) |
|---|---:|---:|---:|
| video only | 65.4 ± 0.9 | 71.1 | 66.9 ± 2.0 |
| audio only | 62.8 ± 0.9 | 67.0 | 64.4 ± 1.9 |
| fused, w tuned on val | 66.3 ± 1.5 | 73.1 | 68.0 ± 1.8 |
| *majority class (pooled)* | 53.3 | — | — |

Training cost on Kaggle (Tesla T4): 98.8 min for video and 15.7 min for audio over the 5 subject folds.

## What it means

**The voice carries the signal, the face does not.** On unseen speakers the audio branch is the only
one above chance, in AUC only: pooled AUC 58.7 (speaker-bootstrap CI over 0.5: +2.8 to +14.7),
while its 56.1 % accuracy is not significantly above the 52.1 % majority once speakers are
resampled ([`docs/report.md`](docs/report.md) §6.1). Video sits at 51.6 % with
53.6 AUC, and the tuned fusion weight gives video w = 0 in 3 of 5 folds.

**The protocol is most of the paper's number.** The same video model goes from 65.4 % on the paper's
clip-level folds to 51.6 % on unseen speakers (−13.8 points, −17.5 AUC). Audio loses 6.7 points. On
the paper's protocol, with the paper's selection rule, the fusion reaches 68.0 % against the paper's
64.8 % (PAVF) and 66.8 % (multi-task) — on a 13 % smaller copy of the data. Most of what video learns
there is who is speaking, not whether they lie. The analysis (within-speaker AUC, fold variance,
thresholds, gender) is in [`docs/report.md`](docs/report.md) §6–§7.

## Files

| Path | What it is |
|---|---|
| [`train_lore.ipynb`](train_lore.ipynb) | the notebook that produced the results (Kaggle, 2 × T4): pilot → folds 2–5 → fusion → official split |
| [`train_lore_colab.ipynb`](train_lore_colab.ipynb) | same pipeline on Colab (one GPU, Drive for results) |
| [`docs/report.md`](docs/report.md) | the full report: method, protocols, results, analysis, where every number comes from |
| [`lore/`](lore/) | the package: `config`, `faces` (face cache), `data`, `models` (LoRA + activity check), `train`, `fusion`, `pilot` |
| [`scripts/`](scripts/) | `build_face_cache.py`, `run_cv.py`, `fuse.py`, `check_pilot.py`, `pack_kaggle.py`, `export_results.py` |
| [`tests/`](tests/) | protocol, resume, LoRA-activity, pilot and paper-style tests |
| [`results/`](results/) | the committed results: one folder per run, see below |

### `results/` and what is not in git

`results/<run>/` holds `run.json` (configuration, library versions, GPU, hashes of `dataset.csv`, the
splits and the face cache), `summary.json` and, per fold, `metrics.json`, `log.csv` and the predictions.
Fusion folders add `fusion_report.json`, `comparison.txt` and `board_row.md`.

The prediction CSVs keep `clip_id`, the score and the fold, but **not** `y_true`, `subject` or
`gender`. Those are DOLOS annotations, which may not be redistributed. Join them back from
`dataset.csv` on `clip_id`. `scripts/export_results.py` builds `results/` from `runs/`.

Git-ignored, local only (see the root `.gitignore`):

| Folder | Content |
|---|---|
| `runs/` | the full run outputs: labelled predictions and `trainable.pt` (LoRA + head weights, 1.2 MB per fold) |
| `inputs/` | `faces-yunet-t32-s224.zip`, the face cache (~435 MB, derived from the clips) |
| `assets/` | YuNet and SFace ONNX models, downloaded automatically by `lore/faces.py` |
| `kaggle_upload/` | the private Kaggle dataset zips made by `pack_kaggle.py` |
| `kaggle_results/` | the raw Kaggle output zips |

## How to run

From this folder, with the shared DOLOS folder at `<repo>/dolos` or `LORE_DATA` pointing at it (every
path is in [`lore/config.py`](lore/config.py)):

```bash
python scripts/build_face_cache.py                       # once: face cache -> inputs/
python scripts/run_cv.py --modality video                # 5 subject-level folds
python scripts/run_cv.py --modality audio
python scripts/fuse.py --video-run video-lora-r8 --audio-run audio-lora-r8
python scripts/run_cv.py --modality video --protocol official   # paper's split, reported apart
python scripts/run_cv.py --modality audio --protocol official
python scripts/fuse.py --video-run video-lora-r8-official --audio-run audio-lora-r8-official
python scripts/export_results.py                         # runs/ -> results/ (label-free)
```

On a GPU, use the notebooks: `train_lore.ipynb` on Kaggle, or `train_lore_colab.ipynb` on Colab. The
setup steps are in their first cell. Each run writes the split hashes into `run.json`, and folds resume
from per-epoch checkpoints.
