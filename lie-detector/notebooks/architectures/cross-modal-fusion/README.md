# `tia` — frozen backbones + cross-modal fusion

Direction `tia` of the [lie-detector project](../../../README.md#directions): **Paradigm 1** in
[`docs/architectures.md`](../../../docs/architectures.md). DINOv2 and WavLM are frozen and the only
trainable part is the fusion network; the design is in
[`docs/trainable-module.md`](docs/trainable-module.md).

**Status: exploration, seed 0.** The board row wants three seeds and the configuration chosen on
`val` — see the selection section of the design note. Nothing here is on the board yet.

## Results — subject-independent 5-fold, seed 0

| Variant | Seeds | ACC | AUC | Trainable | What it is |
|---|---:|---:|---:|---:|---|
| `late` | 3 | 53.0 ± 7.7 | 55.2 | 164,737 | mean over time, concat, MLP |
| `audio` | 1 | 54.7 ± 3.9 | 58.9 | 107,009 | WavLM only |
| `video` | 1 | 52.2 ± 5.5 | 53.7 | 57,857 | DINOv2 only |
| `congruence` | 1 | 52.6 ± 2.5 | 57.3 | 147,714 | decides only on `1 − cos(video, audio)` |
| *majority class* | — | 52.0 | — | — | the floor to beat |
| *MUMIN oracle* | 1 | 51.1 | 52.5 | 41 features | hand annotations; not deployable |

The congruence row is an average over five folds one of which reached AUC 74.4: without it the
remaining four sit at 53.0, i.e. chance.

The seed spread on `late` is large: per-seed AUC is 52.5, 55.4 and 57.7, so the one-seed rows above
are single draws and are not comparable with the three-seed row. Treat the `late` row as the only
estimate, and the others as indications.

Two caveats on the ACC and F1 columns. The threshold chosen on `val` ranged from 0.00 to 0.96 across
folds and seeds — with ~100 validation clips it is a noisy argmax, so ACC and F1 move between folds
more than AUC does, and AUC is the column to compare variants on. And only the `late` row has three
seeds: one seed says nothing about seed variance, which is exactly why the other rows are marked.

## What we found

**Audio carries the signal, video does not, fusion does not beat audio.** Audio alone reaches 58.9
AUC in its single-seed run — the only variant whose interval excludes chance (52.9–64.9). The
three-seed fusion lands at 55.2, below it, but with a per-seed spread of 52.5–57.7 the two are not
separable at this sample size, and that indistinguishability is itself the finding. Video at 16
frames is at chance (53.7), and one fold sits at 42.3.

**The frozen representation is not the bottleneck.** A linear probe reads the 41 MUMIN annotations out
of the pooled embedding with a margin of +0.098 over a prevalence-matched noise band for video (+0.046
for audio): the embedding contains what the annotators saw. The cues that resist are the fast ones —
`raising` 0.549 against a band of 0.542, `nod` 0.563, `frowning` 0.567 — consistent with 16 frames
being 3 fps on the median 5-second clip.

**The annotated behaviour does not predict deception.** A classifier on the 41 annotations reaches
51.1 ACC / 52.5 AUC on our split, below the majority baseline. The paper reports 58.84 on its own
split. That gap, and the one it implies for every model here, is measured in
[`docs/protocol-effect.md`](docs/protocol-effect.md).

## Files

| File | What it is |
|---|---|
| [`extract_features.ipynb`](extract_features.ipynb) | one-off: frozen DINOv2 (or HSEmotion), WavLM and MiniLM features into the shared cache |
| [`extract_vjepa2.ipynb`](extract_vjepa2.ipynb) | optional second video cache: V-JEPA 2 ViT-L, `(32, 1024)` per clip |
| [`campaign.ipynb`](campaign.ipynb) | the batch path: every reported run, both protocols, in one kernel |
| [`training.py`](training.py) | the recipe — features, folds, the loop, the campaign driver |
| [`train.ipynb`](train.ipynb) | one configuration interactively: every variant, predictions per clip |
| [`test.ipynb`](test.ipynb) | the closed test set, scored once: `results/<variant>-test/` |
| [`model.py`](model.py) | the variants and the `VARIANTS` registry |
| [`export.py`](export.py) | writes one run to `results/`; the layout is owned by [`results/README.md`](results/README.md) |
| [`results/`](results/) | one directory per run: summary, per-fold metrics, predictions, provenance |
| [`diagnostics.ipynb`](diagnostics.ipynb) | does the embedding contain the annotated cues |
| [`reference paper/`](reference%20paper/references.md) | the literature this direction stands on, indexed with what each paper supports |
| [`docs/trainable-module.md`](docs/trainable-module.md) | design of the trainable part, parameter budget, experiment ladder |
| [`docs/protocol-effect.md`](docs/protocol-effect.md) | the same model on the subject and the paper's splits |
| [`docs/diagnostics-findings.md`](docs/diagnostics-findings.md) | what the cue probes measured |

## How to run

```
extract_features.ipynb   # once, writes cached_features/<encoder>/ into the shared folder
extract_vjepa2.ipynb     # optional: a video-native backbone, cached_features/vjepa2-vitl/
campaign.ipynb           # the six reported runs, 3 seeds each; writes results/<run>/ for all
train.ipynb              # one configuration: VARIANT, SPLIT, SEEDS at the top, last cell saves it
test.ipynb               # the closed test set: open it once, at the end, never twice
diagnostics.ipynb        # after the caches exist; writes results/diagnostics_mumin.csv
```

`test.ipynb` is the last stop, and the only notebook that reads `subject_test.json`. It retrains
the five subject folds of one configuration, scores the 27 subjects with each fitted model at that
model's own validation threshold, and writes `results/<variant>-test/`. The guard in the notebook is deliberate: the configuration is chosen
before the set is opened, and a second variant run afterwards is a draw from the same small sample,
not a second estimate.

### Other video backbones

Two alternatives to DINOv2 write the same cache contract, so switching to one is a single line in
`train.ipynb`: the notebook calls `model.configure(...)` with the widths it finds in the caches,
so the variants build the matching input layer without edits. Audio and text stay identical,
which makes each comparison one-variable — same folds, same seeds, same probe, different visual
encoder.

**V-JEPA 2 — self-supervised, video-native.** [`extract_vjepa2.ipynb`](extract_vjepa2.ipynb):
64 frames at 256 px through a ViT-L, cached as `(32, 1024)`. `VIDEO_ENCODER = "vjepa2-vitl"`.

**HSEmotion — supervised, face-specific.** One entry in `ENCODERS` of
[`extract_features.ipynb`](extract_features.ipynb): EfficientNet-B0 trained on AffectNet
emotions ([HSEmotion](https://github.com/av-savchenko/face-emotion-recognition), Apache-2.0),
cached as `(16, 1280)` (penultimate embedding) or `(16, 8)` (class posteriors).
`VIDEO_ENCODER = "hsemotion-b0-vgaf"`. It is the only encoder here trained on faces and the only
supervised one; the hypothesis it makes testable is the old Ekman one — a lie shows as an
emotion the speaker is suppressing.

Two caveats before that row is read as a like-for-like swap:

- **Not parameter-matched.** The 1280-d embedding makes the video projection 1280→128 instead of
  384→128, so `late` moves from 164,737 to 279,425 trainable parameters and `late_delta` from
  222,465 to 451,841 — past the ceiling [`docs/trainable-module.md`](docs/trainable-module.md)
  puts at 300–500k. `d=64` brings them back to 139,777 and 225,985; say which one ran.
- **The crop is DINOv2's.** HSEmotion was trained on tight AffectNet crops, ours are MTCNN boxes
  with a 0.25 margin. Keeping them identical is what makes the comparison one-variable, so any
  accuracy it costs is part of the result, not a bug in the cache.

The 8-class cache is the cheaper half of the experiment. Audio plus the average expression and
its motion (`late_delta`) is 126,209 parameters, of which the video stream is 2,304: it asks
whether the emotion output itself carries the label, with no capacity left to hide behind.
