# Multimodal Deception Detection on DOLOS

Course project for *Big Data and Deep Learning*. A model receives a short game-show clip from the
DOLOS dataset (*Would I Lie To You?*) and decides `truth` or `deception`. It uses only audio and video
that can be computed from the clip, never human annotations. We compare four deep architectures
with a theory-driven acoustic baseline.

**Team:** Sicoli Mattia, Scuratti Matteo, Tondi Gabriele, Moliterno Fabio, Mariotti Lorenzo, Abdallah Yehia.

**Paper:** [`report/report.pdf`](report/report.pdf). The Typst source is `report/report.typ` and
`report/sections/`; rebuild the PDF with `typst compile report/report.typ`.

## Results

Official 3-fold clip-level protocol of the DOLOS paper. ACC is the mean ± standard deviation over
test folds (and over seeds where there are three). The full tables, the per-fold breakdown and the
subject-independent protocol are in the paper (§5.5 and Appendix A).

| Model | ACC | F1 | AUC | Trainable params | Seeds |
|---|---:|---:|---:|---:|---:|
| Majority class (same clips) | 52.4–53.3 | 69.6 | 50.0 | 0 | — |
| PECL (reported by the DOLOS paper) | 64.75 | 71.20 | 62.71 | 5.06M | — |
| Theory-driven acoustic baseline | 56.5 ± 1.7 | 60.2 | 59.2 | 609 | 3 |
| Temporal CNN (landmarks + eGeMAPS) | 56.3 ± 2.9 | 56.1 | 59.1 | 57,322 | 3 |
| **Cross-Modal Fusion (frozen DINOv2 + WavLM)** | **72.2 ± 3.0** | **73.1** | **80.4** | 164,737 | 3 |
| VideoMAE + HuBERT with LoRA | 66.3 ± 1.5 | 65.9 | 73.1 | 594,434 | 1 |
| MARLIN + ConvNeXt + TCN | 58.9 ± 0.9 | 59.8 | 58.9 | 97,857 | 1 |

On the subject-independent 5-fold split (no speaker shared between training and evaluation),
every model drops to 50–56% accuracy (paper, Appendix A).

## Layout

```
deliverable/
├── README.md
├── report/                          the paper (PDF + Typst source)
└── code/
    ├── lib/                         dataset.py (clip table, frozen splits, hash checks), metrics.py
    ├── scripts/
    │   ├── build_dataset.py         download the clips, cut them, extract 16 kHz audio, write dataset.csv
    │   ├── make_splits.py           write the frozen split files (official 3-fold + subject 5-fold + test)
    │   └── extract_features_marlin.py   MARLIN + ConvNeXt feature cache (MARLIN + ConvNeXt model)
    └── notebooks/
        ├── preprocessing.ipynb      runs build_dataset.py and make_splits.py on Colab
        └── architectures/
            ├── load/                theory-driven baseline
            ├── teo/                 Temporal CNN
            ├── cross-modal-fusion/  Cross-Modal Fusion (best model)
            ├── lore/                VideoMAE + HuBERT with LoRA
            └── fab/                 MARLIN + ConvNeXt
```

Each architecture folder has a `results/` subfolder with the files the paper's numbers come from.
`folds.csv` holds one row per seed × fold, `preds.csv` holds per-clip predictions, and
`summary.json` is the aggregate. Where a `run.json` exists, it records the dataset and split
hashes, the settings and the library versions. lore's prediction files have their label columns
removed; the other directions keep a `y_true` column.

## Data

The dataset is **not included**: the DOLOS annotations are released by NTU ROSE Lab for academic use
only, and the videos are YouTube episodes under third-party copyright. To reproduce the runs:

1. Request DOLOS from <https://rose1.ntu.edu.sg/dataset/DOLOS/>. Put its annotation files
   (`dolos_timestamps.csv`, `Dolos.xlsx`, `train_dolos.xlsx`, `Training_Protocols/`) in
   `code/dataset/DOLOS/`.
2. Build the data folder (called `dolos/` below), from `code/`, either with
   `notebooks/preprocessing.ipynb` on Colab or locally:
   ```bash
   python scripts/build_dataset.py --out-dir ../dolos/clips --audio-dir ../dolos/audio --csv ../dolos/dataset.csv
   python scripts/make_splits.py --out-dir ../dolos/splits --available-from ../dolos/dataset.csv \
       --test-fraction 0.15 --folds 5 --val-fraction 0.15
   ```
   `build_dataset.py` needs `yt-dlp`, `openpyxl`, `ffmpeg` and `ffprobe`. Some episodes are no longer
   online: 1,452 of the 1,678 annotated clips could be downloaded for this report.
3. The results were produced with `dataset.csv` sha256 prefix `8fa198b1fe71`. Every `run.json` records
   this fingerprint, and `lib/dataset.py` verifies the split hashes when it loads them.

The resulting folder holds `dataset.csv`, `clips/`, `audio/`, `splits/` and, after feature
extraction, `cached_features/<encoder>/`.

## Running the models

Training ran on Google Colab or Kaggle (NVIDIA T4) for the pre-trained models, and on CPU for the
baseline and the Temporal CNN. The Colab notebooks read two settings, each an environment variable
with a default you can edit in the first cells:

| Setting | Meaning | Default |
|---|---|---|
| `CODE_ROOT` | the `code/` folder | `/content/drive/MyDrive/code` |
| `DOLOS_DATA` | the data folder | `/content/drive/MyDrive/dolos` |

All commands below are run from the architecture's folder unless stated otherwise.

### Theory-driven baseline (`load/`)
17 hand-crafted acoustic features (pauses, energy, pitch, voice quality) and a 609-parameter MLP.
Requires `numpy`, `pandas`, `torch`, `scikit-learn`, `soundfile` and `praat-parselmouth`.
```bash
python run_load.py --data <dolos>        # both protocols, 3 seeds -> results/load-{official,subject}/
```

### Temporal CNN (`teo/`)
MediaPipe face landmarks (15 regions) and 8 openSMILE eGeMAPS descriptors at 10 fps, fed to a
from-scratch 1-D CNN. Requires `mediapipe`, `opensmile`, `opencv-python`, `soundfile`, `scipy`, `torch`,
`scikit-learn` and `pandas`. The MediaPipe model file is downloaded automatically.
```bash
python features/extract_features.py --input <dolos> --output_dir <dolos>/cached_features/teo-landmarks-egemaps
python cnn-batching/train_official_temporal.py --protocol official --data <dolos> --cache <dolos>/cached_features/teo-landmarks-egemaps
python cnn-batching/train_official_temporal.py --protocol subject  --data <dolos> --cache <dolos>/cached_features/teo-landmarks-egemaps
# single-stream ablation: writes the sibling caches <cache>-video and <cache>-audio
python features/mask_streams.py --cache <dolos>/cached_features/teo-landmarks-egemaps
python cnn-batching/train_official_temporal.py --protocol official --data <dolos> \
    --cache <dolos>/cached_features/teo-landmarks-egemaps-video --out results/temporal-official-video
python cnn-batching/train_official_temporal.py --protocol official --data <dolos> \
    --cache <dolos>/cached_features/teo-landmarks-egemaps-audio --out results/temporal-official-audio
```

### Cross-Modal Fusion (`cross-modal-fusion/`) — best model
Frozen DINOv2-small (16 MTCNN face crops) and WavLM-base-plus (64 pooled tokens) feed a 165k-parameter
fusion head. `model.py` is the head, `training.py` the training loop and the run driver, `export.py`
the result writer. Run on Colab in this order:
1. `extract_features.ipynb` builds the `dinov2-small` and `wavlm-base-plus` caches (pins
   `transformers==4.46.3` and `facenet-pytorch==2.6.0`).
2. `campaign.ipynb` trains every reported run: late / video / audio × official / subject, seeds
   0–2. Output goes to `results/`.
3. `diagnostics.ipynb` runs the MUMIN linear probes and the annotation-only classifier (paper §6,
   Appendix A).
4. `test.ipynb` runs the closed subject-independent test set. It is **deliberately not opened**
   (`OPEN_TEST_SET = False`): the paper reports development estimates only.

### VideoMAE + HuBERT with LoRA (`lore/`)
VideoMAE-base (speaker-centred face crops, YuNet + SFace) and HuBERT-base, each adapted with LoRA
(r = 8, α = 16), then late fusion of the two branches. Requires `transformers==5.23.0`,
`peft==0.21.3`, `torch` 2.11, `opencv-python` (YuNet and SFace models download automatically),
`scikit-learn` and `pandas`. Paths are set by environment variables:

| Variable | Default |
|---|---|
| `LORE_DATA` | `deliverable/dolos` |
| `LORE_LIB` | `code/` |
| `LORE_FACES_ROOT` | `<data>/cached_features` |
| `LORE_RUNS` | `lore/runs` |

```bash
python scripts/build_face_cache.py                               # face sprites for every usable clip
python -m pytest tests/                                          # LoRA-active and protocol checks
python scripts/run_cv.py --modality audio                        # subject 5-fold
python scripts/run_cv.py --modality video --grad-checkpointing
python scripts/fuse.py --video-run video-lora-r8 --audio-run audio-lora-r8
python scripts/run_cv.py --modality audio --protocol official    # official 3-fold
python scripts/run_cv.py --modality video --protocol official --grad-checkpointing
python scripts/fuse.py --video-run video-lora-r8-official --audio-run audio-lora-r8-official
python scripts/export_results.py                                 # runs/ -> results/ (labels removed)
python scripts/cluster_test.py --dataset <dolos>/dataset.csv     # speaker-cluster bootstrap (Appendix A)
```
The reported runs used `train_lore.ipynb` on Kaggle with two T4 GPUs. That notebook does the same
steps; its first cell explains how `scripts/pack_kaggle.py` packs the code (`lib/` included) and the
data as two private Kaggle datasets.

### MARLIN + ConvNeXt (`fab/`)
A frozen MARLIN ViT-Base on MTCNN face crops and a frozen ConvNeXt-Tiny on 3-channel log-mel
spectrograms feed a dilated TCN head with mean pooling. Run `train_marlin_kaggle.ipynb` on Kaggle
(GPU) with the `code/` folder and the data folder attached as datasets. The notebook installs
`facenet-pytorch`, `marlin-pytorch` and `timm`, extracts the features with
`scripts/extract_features_marlin.py --encoder marlin_convnext --marlin-model marlin_vit_base_ytf`, and
trains on both protocols.

> **Note.** The committed `fab/results/*/summary.json` and `folds.csv` do not match the
> run's own `preds.csv`. The official `preds.csv` also labels 17 truthful clips of episode 73 as
> lies. The paper therefore recomputes every MARLIN + ConvNeXt number from `preds.csv` against the
> labels in `dataset.csv` (paper §5.5).

## Third-party code and models

The pre-trained models are downloaded from their official sources at run time, and no third-party
code is vendored:

| Source | Used for |
|---|---|
| Hugging Face | DINOv2, WavLM, VideoMAE, HuBERT |
| `marlin-pytorch` | MARLIN |
| `timm` / torchvision | ConvNeXt |
| `facenet-pytorch` | MTCNN |
| OpenCV Zoo | YuNet, SFace |
| MediaPipe | FaceLandmarker |
| openSMILE | eGeMAPS |
| Parselmouth | Praat |

The DOLOS dataset and the PECL baseline are by Guo et al., *Audio-Visual Deception Detection: DOLOS
Dataset and Parameter-Efficient Crossmodal Learning*, ICCV 2023. Their reference code is at
<https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning>,
and their download tools are at <https://github.com/NMS05/AV-Data-Processing>. Neither is needed to
run this code.
