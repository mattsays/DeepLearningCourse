# Parameter-Efficient Audio-Visual Deception Detection on DOLOS: What Survives a Speaker-Independent Split

*Direction `lore` of the DOLOS lie-detector project (Paradigm 2 in
[`lie-detector/docs/architectures.md`](../../../../docs/architectures.md)).
Paths are relative to this direction folder, `lie-detector/notebooks/architectures/videomae-hubert-lora/`. The code is in
`lore/`, `scripts/` and `tests/`; the results are in [`results/`](../results/) (label-free export of the
git-ignored `runs/`, see the [README](../README.md)).
Every number below was measured from those files or from the local data copy. Numbers taken from the DOLOS
paper are marked **[paper]**. Appendix B lists where each number comes from.*

---

## Abstract

We adapt two large self-supervised Transformers to deception detection on DOLOS: VideoMAE-Base, applied
to the speaker's face, and HuBERT-Base, applied to the raw voice. Both are adapted with Low-Rank
Adaptation (LoRA). Only 297 k parameters per branch are trained (0.3 %), and the two branches are
combined by a late fusion whose weight is tuned on validation. We evaluate the same models under two
protocols.

**Speaker-independent protocol.** This is our protocol: 5-fold cross-validation, with no person on both
sides of a split.
- Fused model: **56.1 ± 7.3 % accuracy** and 60.6 AUC, against a 52.1 % majority baseline.
- Audio branch alone: 56.1 %, 4.0 points above the 52.1 % majority — not significant once speakers
  are resampled (gain CI −4.7 to +9.1, p = 0.21); its AUC of 58.7 is (CI +2.8 to +14.7 over 0.5,
  p = 0.001). See §6.1, *Is it above chance?*
- Video branch: at chance (51.6 %, AUC 53.6).

**The paper's clip-level 3-fold split.** The same speakers and episodes appear in train and test.
- Fused model: **66.3 ± 1.5 %**. Video reaches 65.4 % and audio 62.8 %.
- With the paper's own model selection (best test epoch at threshold 0.5) we reach **68.0 %**, above the
  64.8 % PAVF and 66.8 % multi-task results **[paper]**. This is on a 13 % smaller copy of the data, so
  the numbers are only indicatively comparable.

The gap between the protocols is **13.8 accuracy points for video** and 6.7 points for audio. Most of
what the video branch learns on the clip-level split is therefore the identity of the speaker or the
episode, not deception. Audio keeps a small but statistically detectable signal on unseen speakers.

---

## 1. Introduction

DOLOS is a set of short clips (median 5 s) of contestants in a British TV panel show, each telling a
true or a false story. Each clip is labelled `truth` (0) or `deception` (1). A fold contains about 1 k
training clips, which is far too few to train a video or speech Transformer from scratch.

We ask two questions:

1. **How far do parameter-efficient adaptations of general-purpose foundation models go on this task?**
   We freeze all pretrained weights and train only LoRA adapters and a linear head.
2. **How much of the reported performance on DOLOS survives when the test speakers are unseen?** The
   paper's official folds are clip-level: 95–100 % of the test clips come from a person who is also in
   training (§3.2). A model can then score well by recognising *who* is talking.

Our contributions:
- a face-centred video pipeline that keeps the speaker's identity across cuts (§4.1);
- two LoRA branches with a built-in check that the adapters actually affect the forward pass (§4.3);
- a resumable training loop for free GPUs;
- a side-by-side evaluation of the **same** models under a speaker-independent and a clip-level protocol.

---

## 2. Related work (brief)

**The DOLOS paper** (Guo et al., ICCV 2023) introduced the dataset and PECL.
- PECL is a parameter-efficient crossmodal learner: Wav2Vec2 and ViT with *uniform temporal adapters*,
  plus a *plug-in audio-visual fusion* (PAVF) module.
- Its baselines on the 3 official folds are **61.4 %** visual, **59.2 %** audio, **64.8 %** PAVF fusion
  and **66.8 %** with multi-task learning **[paper]**, quoted from
  [`lie-detector/dataset/README.md`](../../../../dataset/README.md) §11.
- The reference code is the `lie-detector/baselines/DOLOS-PECL` submodule
  ([`scripts/train_test.py`](../../../../baselines/DOLOS-PECL/scripts/train_test.py)). It
  evaluates on the test fold after every epoch and keeps the epoch with the best test accuracy at
  threshold 0.5.

**Our approach** differs from PECL in three ways:
- stronger, video-native and speech-native backbones (VideoMAE instead of a per-frame ViT, HuBERT
  instead of Wav2Vec2);
- LoRA instead of temporal adapters;
- late instead of mid-level fusion.

---

## 3. Data and evaluation protocols

### 3.1 Data

The annotations come from ROSE Lab; the clips were downloaded from YouTube with the team's
`preprocessing.ipynb`.

| | count |
|---|---|
| annotated clips | 1,678 rows |
| downloaded (`status = ok`) | **1,452** |
| failed | 219 (+ 7 invalid time ranges) |

All 219 failures come from **9 YouTube episodes that no longer exist**:
- 8 are blocked by a copyright claim from Banijay x Merzigo;
- 1 is on an account that YouTube terminated.

They cannot be recovered, and ROSE Lab distributes no footage.

### 3.2 Two protocols

**Subject-level (ours; this is what goes on the results board).**
- The team's frozen split ([`lie-detector/docs/splits.md`](../../../../docs/splits.md)) has 5 folds,
  each with `train` / `val` / `holdout`. No subject crosses a boundary.
- The two most frequent contestants are always in train.
- A reserved test set of 27 subjects is never touched.
- Folds have about 1,013 train, 107 val and 177 holdout clips.
- Pooled over the 5 holdouts: **886 clips from 135 speakers**.

**Official clip-level (the paper's).**
- `official_fold{1,2,3}.json`: the paper's `train_fold`/`test_fold` CSVs, reduced to the downloaded clips.
- About 944 train / 483 test clips per fold, against about 1,099 / 549 in the paper.
- The paper has no validation set. We set aside a fixed, label-stratified **15 % of each fold's train**
  for early stopping, the threshold and the fusion weight (seed 10000 + k, shared by both branches).

The official split leaks identity, measured on our copy:

| official fold | test clips | test speakers also in train | test clips from a training speaker | test clips from a training episode | clips on both sides |
|---|---|---|---|---|---|
| 1 | 482 | 120 / 131 | 95.0 % | 100 % | 1 |
| 2 | 483 | 122 / 130 | 96.9 % | 100 % | 0 |
| 3 | 461 | 126 / 127 | 99.6 % | 100 % | 1 |

In addition, `test_fold1` lists one clip twice (`ls_wilty_ep14_lie11`), and the fold-2 and fold-3
*train* lists repeat one or two clips. We keep the paper's lists as they are; fusion de-duplicates by
clip.

### 3.3 Metrics

We report:
- accuracy (ACC), F1 on `deception`, AUC and balanced accuracy, all computed by the team's
  `lib.metrics.evaluate`;
- the majority-class baseline on the same clips;
- per-gender accuracy.

On the subject-level protocol, the decision threshold is **chosen on val**, never set to 0.5 and never
chosen on the holdout. The headline figure is the mean ± sample std over folds.

---

## 4. Method

```
              ┌──────────────────────── VIDEO BRANCH ─────────────────────────┐
 clip.mp4 ──► face cache (YuNet + SFace) ──► 16×224×224 face crops
              │                       VideoMAE-B (frozen) + LoRA(Q,V)
              │                                    │ 1568 tokens × 768
              │                         mean ─► fc_norm ─► dropout ─► Linear ─► logit_v
              └────────────────────────────────────────────────────────────────┘
              ┌──────────────────────── AUDIO BRANCH ─────────────────────────┐
 clip.wav ──► 16 kHz mono, z-normalised, 8 s random window (train) / first 10 s (eval)
              │                       HuBERT-B (frozen) + LoRA(q_proj,v_proj)
              │                                    │ ~50 frames/s × 768
              │                    masked mean ─► LayerNorm ─► dropout ─► Linear ─► logit_a
              └────────────────────────────────────────────────────────────────┘
                       z-score with val stats     z-score with val stats
                                   └──── w·z_v + (1−w)·z_a ──┘   (w chosen on val)
                                                │
                                   sigmoid ─► threshold (chosen on val) ─► truth / deception
```

| | Video branch | Audio branch |
|---|---|---|
| Input | 16 face crops (224×224) spread over the clip | raw 16 kHz waveform (8 s train / 10 s eval) |
| Backbone | `MCG-NJU/videomae-base-finetuned-kinetics` | `facebook/hubert-base-ls960` |
| Adaptation | LoRA r = 8, α = 16, on attention Q and V, all 12 layers | same, on `q_proj`, `v_proj` |
| Head | mean pool → `fc_norm` → dropout 0.3 → linear | masked mean pool → LayerNorm → dropout 0.3 → linear |
| Trainable | 297,217 of 86.5 M (0.34 %) | 297,217 of 94.7 M (0.31 %) |

### 4.1 Video input: a speaker-centred face cache (`lore/faces.py`)

DOLOS clips are wide TV shots, often with several people and cuts to other contestants. We give
VideoMAE **only the speaker's face**, computed once, offline:

1. 32 frames sampled uniformly over the clip.
2. Faces detected with YuNet (confidence ≥ 0.6).
3. Each face embedded with SFace (128-d) and greedily clustered (cosine ≥ 0.363). The identity seen in
   the most frames is the speaker.
4. Frames where the speaker is not visible are filled with the nearest frame where they are, so the
   model never sees another person.
5. Box centre and size are median-filtered (window 5) within each shot. A new shot starts wherever two
   consecutive boxes overlap less than IoU 0.3.
6. Each frame is cropped to a square of 1.6 × the face size and resized to 224².
7. The 32 crops are stored as one sprite JPEG per clip.

On the 1,452 clips:
- 99.9 % of the frames contain a face;
- the speaker is visible in 96.0 % of them;
- 39.2 % of the clips needed at least one slot filled from a neighbouring frame;
- no clip fell back to a centre crop.

At train time, 16 of the 32 frames are used: a random choice of odd or even frames gives temporal
jitter. One `RandomResizedCrop(0.8–1.0)` and one flip are applied identically to all 16 frames, so the
motion stays coherent. Pixels are normalised with ImageNet mean and std.

### 4.2 Audio input (`lore/data.py`)

- All wavs are read once into RAM as int16 PCM.
- Training uses a random **8 s** window; evaluation uses the **first 10 s** (71.5 % of the clips fit
  entirely).
- Each waveform is z-normalised, as HuBERT expects.
- Batches are zero-padded and bucketed by length.
- The masked mean pool ignores padded frames.

### 4.3 LoRA and an activity check (`lore/models.py`)

LoRA replaces each adapted frozen weight `W` with `W + (α/r)·B·A`. With `B = 0` at initialisation, the
model starts exactly at the pretrained one. Q and V in 12 layers give 294,912 adapter parameters;
adding the norm (1,536) and the head (769) gives 297,217.

**Why the check exists.** In transformers ≤ 5.5, VideoMAE computes Q and V through
`F.linear(self.query.weight)`. LoRA on those modules is then **silently bypassed**, while the
trainable-parameter count still looks normal.

**What it does.** Before every fold, `assert_lora_active` checks two things:
- every `lora_B` receives a non-zero gradient;
- randomising `B` changes the output.

All 24 adapters per branch passed in every fold, on transformers 5.23.

### 4.4 Training (`lore/train.py`)

| Setting | Video | Audio |
|---|---|---|
| Loss | BCE-with-logits, `pos_weight = #neg/#pos` of train | same |
| Optimiser | AdamW, wd 0.01; LR 2e-4 (LoRA), 1e-3 (head) | same |
| Schedule | 1-epoch linear warm-up, then cosine to 0 | same |
| Effective batch | 16 (4 × accumulation 4) | 16 (8 × 2) |
| Epochs / early stopping | ≤ 10, patience 3 on val AUC | ≤ 15, patience 4 |
| Precision | fp16 autocast + GradScaler (T4) | same |

The epoch with the best val AUC is kept. The decision threshold is then chosen on val with
`lib.metrics.best_threshold`.

On the official protocol, the test fold is also scored after every epoch, at threshold 0.5. This gives
the **paper-style** number: the best test epoch, exactly as the paper's `train_test.py` selects it. We
report it only to compare with the paper; it is never used for any decision.

### 4.5 Late fusion (`lore/fusion.py`)

1. Each branch's logits are z-scored with the mean and std of its own val logits.
2. `s = w·z_v + (1−w)·z_a`, with `w ∈ {0, 0.05, …, 1}` chosen by val AUC. An untuned `w = 0.5` is also
   reported.
3. The threshold is chosen on val, as for a single branch.

The paper-style fused number combines each branch's predictions at its own best test epoch. It uses
the per-fold val-tuned `w` on raw logits, with threshold 0.5.

---

## 5. Experimental setup

- **Hardware.** Kaggle, 2 × Tesla T4: one branch's folds per GPU. Software: torch 2.11, transformers 5.23,
  peft 0.21.3, seed 0.
- **Cost.** Measured training time, summed over folds:

| Protocol | Video | Audio |
|---|---|---|
| subject-level, 5 folds | 98.8 min | 15.7 min |
| official, 3 folds | 44.6 min | 9.7 min |

  Peak GPU memory: 0.79 GB for video (gradient checkpointing) and 3.2 GB for audio.
- **Reproducibility.**
  - Every run writes `run.json`: configuration, library versions, GPU, and the hashes of `dataset.csv`,
    the split files and the face cache.
  - Folds resume from per-epoch checkpoints.

---

## 6. Results

### 6.1 Speaker-independent protocol (5 folds; this is the board number)

| Model | ACC | F1 | AUC | Bal. ACC | male ACC | female ACC |
|---|---|---|---|---|---|---|
| majority baseline (pooled) | 52.1 | — | — | — | — | — |
| video only | 51.6 ± 4.6 | 41.7 | 53.6 | 51.5 | 47.6 | 61.7 |
| audio only | 56.1 ± 5.5 | 56.0 | 59.5 | 56.2 | 57.8 | 51.8 |
| **fused, w tuned on val** | **56.1 ± 7.3** | 55.1 | **60.6** | 56.2 | 57.8 | 51.8 |
| fused, w = 0.5 | 55.4 ± 7.7 | 60.5 | 59.7 | 56.1 | 56.2 | 53.4 |

Per fold (holdout ACC / AUC):

| fold | video | audio | fused (tuned) | tuned w_video |
|---|---|---|---|---|
| 1 | 57.1 / 61.0 | 65.5 / 72.7 | 68.9 / 73.0 | 0.25 |
| 2 | 50.3 / 43.0 | 53.1 / 55.4 | 52.5 / 55.4 | 0.00 |
| 3 | 49.4 / 59.1 | 55.1 / 56.1 | 52.2 / 61.3 | 0.30 |
| 4 | 45.8 / 45.5 | 51.4 / 55.8 | 51.4 / 55.8 | 0.00 |
| 5 | 55.4 / 59.5 | 55.4 / 57.6 | 55.4 / 57.6 | 0.00 |

**Is it above chance?** We pooled the 886 holdout clips (135 speakers), each judged with its own
fold's threshold, and resampled whole speakers 10,000 times. On every resample the model is compared
with the majority rate *of that resample* (ACC − majority) and with 0.5 (AUC − 0.5); p is the
one-sided share of resamples in which the gain is ≤ 0. Produced by
[`scripts/cluster_test.py`](../scripts/cluster_test.py), stored in
`results/fusion-video-lora-r8+audio-lora-r8/cluster_test.json`:

| Model | pooled ACC | ACC − 52.1 % majority [95 % CI] | p | pooled AUC | AUC − 0.5 [95 % CI] | p |
|---|---|---|---|---|---|---|
| video only | 51.6 | −0.6 [−9.9, +5.3] | 0.66 | 53.5 | +3.5 [−3.2, +10.3] | 0.16 |
| audio only | 56.1 | +4.0 [−4.7, +9.1] | 0.21 | 58.7 | +8.7 [+2.8, +14.7] | 0.001 |
| fused (tuned) | 56.1 | +4.0 [−4.6, +9.1] | 0.21 | 60.4 | +10.4 [+4.1, +16.5] | 0.001 |

The accuracy gains are not significant once speakers, not clips, are the unit; the audio and fused
AUCs are. An earlier version of this table quoted a one-sided binomial p = 0.010 for both accuracy
rows: that test treats the clips of one speaker as independent and is too optimistic.

### 6.2 The paper's clip-level split (3 folds; not comparable with §6.1)

| Model | ACC | F1 | AUC | Bal. ACC |
|---|---|---|---|---|
| majority baseline (pooled) | 53.3 | — | — | — |
| video only | 65.4 ± 0.9 | 64.5 | 71.1 | 65.7 |
| audio only | 62.8 ± 0.9 | 65.1 | 67.0 | 62.6 |
| fused, w tuned on val | 66.3 ± 1.5 | 65.9 | 73.1 | 66.6 |
| fused, w = 0.5 | 67.1 ± 2.1 | 69.2 | 73.7 | 66.9 |

**Comparison with the paper (same selection rule as the paper: best test epoch, threshold 0.5):**

| Model | ours, paper-style | DOLOS paper **[paper]** |
|---|---|---|
| visual | 66.9 ± 2.0 (F1 68.6, AUC 71.1) | 61.4 |
| audio | 64.4 ± 1.9 (F1 68.5, AUC 66.7) | 59.2 |
| fused | **68.0 ± 1.8** (F1 70.3, AUC 72.9) | 64.8 (PAVF) / 66.8 (PAVF + multi-task) |

These numbers are not strictly comparable, for four reasons:
- our folds hold 1,427 of the paper's 1,646 clips;
- we train on 85 % of each training fold;
- we use early stopping, which limits the epochs the paper-style rule can choose from;
- the models differ.

The direction is clear all the same: on the paper's own protocol, these LoRA-adapted backbones are at
least as strong as PECL.

### 6.3 Same model, two protocols

| Branch | clip-level ACC / AUC | speaker-independent ACC / AUC | drop |
|---|---|---|---|
| video | 65.4 / 71.1 | 51.6 / 53.6 | **−13.8 / −17.5** |
| audio | 62.8 / 67.0 | 56.1 / 59.5 | −6.7 / −7.5 |
| fused | 66.3 / 73.1 | 56.1 / 60.6 | −10.2 / −12.5 |

---

## 7. Analysis

### 7.1 The video branch learns identity, not deception

On the speaker-independent folds, the video branch's **validation AUC never left chance** (0.50, 0.48,
0.60, 0.49, 0.40 at the selected epochs). Two holdout AUCs are below 0.5 (43.0 and 45.5). The fold-1
learning curve shows why: train loss goes from 0.62 to 0.38 while val loss rises every epoch (0.70 →
0.93). The model memorises the training people.

On the clip-level folds, the same configuration trains smoothly. Val AUC climbs to 0.65–0.69 and test
AUC to 0.69–0.74, because 95–100 % of the test clips show a face the model has already seen with its
label. Validation agrees: the tuned fusion weight gives video **w = 0 in 3 of 5** speaker-independent
folds, but 0.45–0.80 on the clip-level folds.

### 7.2 Speaker confounding inside a fold

Many speakers appear with a single label. For them, "who is speaking" and "is lying" coincide. In the
speaker-independent holdouts, 32–62 % of the clips belong to such speakers.

The table below restricts each holdout to speakers who have both labels in it (7–12 speakers and 68–121
clips per fold). On these clips identity alone cannot separate the classes:

| | subject-level folds 1–5 | mean | official folds 1–3 | mean |
|---|---|---|---|---|
| audio, within-speaker AUC | 0.71, 0.58, 0.57, 0.54, 0.52 | 0.58 | 0.62, 0.66, 0.60 | 0.63 |
| video, within-speaker AUC | 0.63, 0.63, 0.61, 0.38, 0.53 | 0.56 | 0.67, 0.69, 0.63 | 0.66 |

On unseen speakers, both branches keep a weak signal within a speaker; audio is more consistent across
folds. On the clip-level split, the within-speaker scores are higher, but these clips still share
episodes and scenes with training clips. So contextual leakage is not excluded.

### 7.3 Variance across folds

Fold 1 is an outlier: audio AUC 72.7 there, against 55–58 on folds 2–5. With about 177 holdout clips
and 27 speakers per fold, one fold is not a reliable estimate. The pilot on fold 1 alone overstated the
final result by about 13 accuracy points.

### 7.4 Thresholds and gender

Because the training folds are about 56 % deception and the holdouts about 47 %, a threshold of 0.5 is
biased. On fold 1, audio drops from 65.5 % to 58.8 % accuracy at 0.5, with recall falling to 20 %.

When val AUC is at chance, the val-tuned threshold is unstable. Video thresholds ranged from 0.16 to
0.79, and fold 2 ended with F1 = 2.2 (almost always predicting `truth`).

Male clips are the majority. On the speaker-independent folds, audio is better on male clips (57.8 vs
51.8) and video on female clips (61.7 vs 47.6). The female sets are small, so this is weak evidence.

---

## 8. Limitations

- **Smaller dataset than the paper.** 1,427 of the paper's 1,646 clips are used in the comparison (§3.1).
- **Small validation sets.** About 107 clips per subject-level fold. Early stopping, the threshold and
  the fusion weight are all chosen on it.
- **Sparse video sampling.** 16 frames over about 8 s (about 2 fps) can miss sub-second
  micro-expressions.
- **Truncated audio.** Evaluation uses the first 10 s, so the tail of 28.5 % of the clips is ignored.
- **No cross-modal interaction.** Late fusion cannot learn conditional cues; PECL's PAVF can.
- **Not a strict replication of the paper's training.** We hold out 15 % of train for validation and
  use early stopping; the paper trains for 20 epochs on 100 %.
- **The reserved test set has not been opened**, so the subject-level number is a cross-validation
  estimate, not a final test score.

## 9. Conclusion

LoRA-adapted VideoMAE and HuBERT reproduce, and on paper exceed, the published DOLOS results on the
paper's clip-level split (68.0 % paper-style fused accuracy against 64.8–66.8 % **[paper]**). Most of that
advantage disappears for unseen people. Video falls to chance, and the system settles at
**56.1 ± 7.3 %**, carried by the voice.

The 10–14 point gap between protocols is the main finding of this direction. Deception detection on
DOLOS has to be reported on speaker-independent splits, and video models need a way to stop modelling
identity before more capacity can help. Possible ways to do that:
- a speaker-adversarial loss;
- per-speaker feature normalisation;
- training on frame differences instead of appearance.

---

## Appendix A — Implementation and how to run

| File | Role |
|---|---|
| `lore/config.py` | `Paths` (overridable by `LORE_*` env vars) and `RunConfig` (all hyper-parameters, including `protocol`) |
| `lore/faces.py` | offline face cache: detection, speaker identification, smoothing, sprites |
| `lore/data.py` | `Study` (splits via the team's `lie-detector/lib`; `official` carves the val), `FaceClips`, `AudioClips`, `BucketBatchSampler` |
| `lore/models.py` | `VideoBranch`, `AudioBranch`, LoRA wrapping, `assert_lora_active` |
| `lore/train.py` | `Runner`: one fold, early stopping, threshold, resumable checkpoints, paper-style scoring |
| `lore/fusion.py` | late fusion, CV summary, pooled/per-gender report, paper-style lines, board row |
| `lore/pilot.py` | fold-1 sanity checks and report |
| `scripts/run_cv.py`, `scripts/fuse.py` | CLIs to train one branch / fuse two runs |
| `scripts/build_face_cache.py`, `scripts/pack_kaggle.py` | face cache; private Kaggle dataset zips |
| `scripts/export_results.py` | `runs/` → `results/` without the annotation columns |
| `train_lore.ipynb` (Kaggle), `train_lore_colab.ipynb` | notebooks (pilot → folds 2–5 → fusion → official split) |
| `tests/` | protocol, resume, LoRA-activity, pilot and paper-style tests |

```bash
# from lie-detector/notebooks/architectures/videomae-hubert-lora/ (data defaults to <repo>/dolos, see lore/config.py)
python scripts/build_face_cache.py                       # once
python scripts/run_cv.py --modality video                 # 5 subject-level folds
python scripts/run_cv.py --modality audio
python scripts/fuse.py --video-run video-lora-r8 --audio-run audio-lora-r8
python scripts/run_cv.py --modality video --protocol official   # paper split
python scripts/run_cv.py --modality audio --protocol official
python scripts/fuse.py --video-run video-lora-r8-official --audio-run audio-lora-r8-official
```

Robustness for free GPUs:
- `checkpoint.pt` is written atomically after every epoch, and folds with a `metrics.json` are skipped;
- `LORE_DEADLINE` stops a run cleanly before Kaggle's 12 h limit;
- resuming with a different configuration under the same run name is refused.

Optional variants (`--head-only`, `--lora-mlp-last-n`, `--dora`, `--subject-balanced`, `--eval-tta`)
exist but have not been run.

## Appendix B — Where the numbers come from

| Numbers | Source |
|---|---|
| §6.1, §6.2 tables, per-fold rows, tuned weights | `results/fusion-*/comparison.txt` and `fusion_report.json` (the official one regenerated locally with `fuse.py` after fixing the duplicate-clip merge) |
| paper-style rows | `results/*-official/fold*/metrics.json` → `paper_style`, `paper_test_preds.csv` |
| learning curves, val AUC, costs | `results/*/fold*/log.csv`, `metrics.json`, `summary.json`, `run.json` |
| §3.2 leakage table, §6.1 CIs/p-values, §7.2 within-speaker AUC | computed from `dolos/dataset.csv`, `dolos/splits/official_fold*.json` and the `holdout_preds.csv` / `cv_lore*.csv` files, joined on `clip_id` (seed 0, 2,000 resamples) |
| face-cache statistics | `dolos/cached_features/faces-yunet-t32-s224/index.csv` |
| **[paper]** numbers | DOLOS paper, as quoted in `lie-detector/dataset/README.md` §11 and `docs/evaluation.md` |
