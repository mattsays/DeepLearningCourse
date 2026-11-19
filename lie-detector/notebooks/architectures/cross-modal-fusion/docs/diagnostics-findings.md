# Diagnostics: does the embedding contain what the annotators saw?

Read this if: you are asking whether the frozen features are the bottleneck, or why the video branch
sits at chance for deception while the cues are visibly there.

The question is narrow and answerable: **can a linear probe read the 41 MUMIN annotations out of the
pooled embedding?** If yes, the information survives the crop, the 16 frames and the mean pooling, and
the problem is downstream. If no, the representation is the problem and no fusion module will fix it.

Method and code: [`diagnostics.ipynb`](../diagnostics.ipynb). Results below are from the seed-0 run on
the clips with both cached arrays.

## How the probe works

One logistic regression per annotation, from the mean-pooled embedding (384 video, 768 audio) to the
binary annotation, trained on a fold's `train` and scored on its `holdout` — the same subject-level
split as everything else. Strong regularisation (`C=0.1`), because there are ~900 clips against
384–768 dimensions.

The chance level depends on how rare the annotation is, so it is measured, not assumed: random targets
are drawn at six prevalence levels, and each feature is compared against the band interpolated at its
own prevalence (null mean + 1.645 sd, one-sided 95%).

| Prevalence | 5% | 10% | 20% | 30% | 40% | 50% |
|---|---:|---:|---:|---:|---:|---:|
| band, video | 0.609 | 0.561 | 0.557 | 0.557 | 0.548 | 0.539 |
| band, audio | 0.551 | 0.543 | 0.528 | 0.540 | 0.551 | 0.533 |

The 5% band is the widest because a handful of positives makes the estimate noisy — which is why the
sparse features have to clear a higher bar, not a lower one.

## The result

| Stream | Mean AUC over the 26 features | Mean band | Margin | Above its band |
|---|---:|---:|---:|---:|
| video | **0.665** | 0.567 | **+0.098** | 24 / 26 |
| audio | 0.585 | 0.539 | +0.046 | 18 / 26 |

**The embedding contains the annotated behaviour, and the visual stream contains it roughly twice as
clearly as the acoustic one.** The binary "above the band" count is generous — three features clear it
by less than 0.01 — so the margin is the honest number, and it is large.

## What is read, and what is not

| Cue | Prevalence | Video AUC | Band |
|---|---:|---:|---:|
| `towards_interlocutor` | 78% | 0.753 | 0.539 |
| `smile` | 13.5% | 0.736 | 0.559 |
| `open` (arms) | 17% | 0.733 | 0.558 |
| `down` (gaze) | 29% | 0.730 | 0.557 |
| `tilt` (head) | 24% | 0.698 | 0.557 |
| `frequent_blinking` | 18% | 0.681 | 0.557 |
| `frowning` | 13% | 0.567 | 0.559 |
| `nod` | 26% | 0.563 | 0.557 |
| `raising` (eyebrows) | 46.5% | 0.549 | 0.542 |
| `sideways` (gaze) | 24% | 0.539 | 0.557 |

The pattern is consistent: **coarse, static configuration is readable** (gaze, head pose, arms, mouth
corners), **fast or subtle facial action is not** — `raising` is the most frequent annotation in the
dataset and the embedding cannot read it above chance. At 16 frames over a median 5-second clip the
sampling step is ~310 ms, which is longer than the movement.

The clips are not uniform either: 90% run 3–9 seconds (median 5, max 48), so the same 16 frames mean
**5.3 fps on a short clip and 1.8 fps on a long one** — a threefold difference in temporal resolution
that a fixed frame count cannot remove. The audio has the same shape from the other side: 64 tokens
are pooled from ~250 WavLM frames at the median and ~450 at the 95th percentile, a 4× to 7×
compression. Duration itself carries no label signal (AUC 0.503 on its own), so this is variance
across clips, not a shortcut.

## The confound worth knowing

The audio probe reads **visual** cues above the band too: `corners_down` 0.666, `towards_interlocutor`
0.644, `shake` 0.633, `open` 0.611, `smile` 0.601. Audio cannot see the corners of a mouth.

The parsimonious explanation is a common factor — **how animated the clip is**. More talk means more
gesture, more gaze shifts, more laughter, and both encoders pick that up. It is the same caution the
[dataset README](../../../../dataset/README.md#L296-L304) raises about gaze towards the interlocutor
(+8.1 pp): a dataset correlation with the interaction, not a deception cue. A deception model trained
on these features can learn "animated clip" and believe it learned deception.

## What this rules out

The video branch sitting at chance for deception (53.7 AUC) is **not** because DINOv2 cannot see. The
cues are there and linearly accessible. Combined with the annotation classifier reaching only 51.1 ACC
([`protocol-effect.md`](protocol-effect.md)), the chain breaks at the second link: the behaviour is
visible, and it does not carry the label.

That is why more capacity on the same features is the wrong next move, and why the interesting
directions are the ones that change the *sampling* (more frames, delta features, attention pooling)
rather than the fusion module.

## How to reproduce

Run [`diagnostics.ipynb`](../diagnostics.ipynb) top to bottom with the caches mounted. It writes
`results/diagnostics_mumin.csv` (one row per annotation: prevalence, both AUCs, both bands) and
`results/oracle_mumin.csv` (per-clip predictions of the annotation classifier). Pointing
`VIDEO_CACHE` at a different encoder — `dinov2-small-body`, for instance — reruns the same probe on a
different crop.
