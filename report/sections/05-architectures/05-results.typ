#import "../../template.typ": *

== Results <sec:results>

@tab:main compares all models on the official 3-fold protocol, @tab:folds breaks the scores down by
fold, and @tab:unimodal isolates the contribution of each modality.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header([Model], [ACC], [F1], [AUC], [Bal. ACC], [Trainable], [Train time]),
    table.hline(stroke: 0.5pt),
    [Majority class (same clips)], [52.4–53.3], [—], [—], [50.0], [—], [—],
    [PECL @guo2023dolos (reported)], [64.75], [71.20], [62.71], [—], [5.06M], [—],
    [Theory-driven baseline], [55.1 ± 2.2], [58.0], [58.8], [55.0], [609], [≈10 s],
    table.hline(stroke: 0.3pt),
    [Temporal CNN], [56.3 ± 2.9], [56.1], [59.1], [56.4], [57,322], [≈2.6 min (CPU)],
    [*Cross-Modal Fusion*], [*72.2 ± 3.0*], [*73.1*], [*80.4*], [*72.2*], [164,737], [≈25 s],
    [VideoMAE + HuBERT, LoRA], [66.3 ± 1.5], [65.9], [73.1], [66.6], [594,434], [54 min],
    [MARLIN + ConvNeXt], [59.1 ± 3.2], [61.4], [60.3], [60.6], [97,857], [≈15 s],
    table.hline(),
  ),
  caption: [Official 3-fold protocol: mean ± standard deviation over test folds (and seeds, for
    the 3-seed models). Baseline, Temporal CNN and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT
    and MARLIN + ConvNeXt: seed 0. Each model is scored on its own subset of the official test clips
    (@tab:coverage), and the PECL row is quoted from the paper, not re-run. Training time is for one
    seed over the three folds, on a T4 GPU unless marked CPU, excluding feature extraction.],
) <tab:main>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.hline(),
    table.header([Fold], [Cross-Modal Fusion], [VideoMAE + HuBERT], [MARLIN + ConvNeXt], [Temporal CNN], [Theory-driven baseline]),
    table.hline(stroke: 0.5pt),
    [1], [69.0 / 77.1], [65.2 / 72.9], [n/a], [55.1 / 56.8], [52.3 / 55.0],
    [2], [73.4 / 81.3], [67.9 / 76.0], [n/a], [59.4 / 62.0], [56.9 / 61.3],
    [3], [74.2 / 82.9], [65.7 / 70.3], [n/a], [54.5 / 58.4], [56.0 / 60.0],
    [Mean], [72.2 / 80.4], [66.3 / 73.1], [59.1 / 60.3], [56.3 / 59.1], [55.1 / 58.8],
    table.hline(),
  ),
  caption: [ACC / AUC per official fold (mean over seeds where several were run); the last row
    repeats the three-fold means of @tab:main. n/a: the MARLIN + ConvNeXt run saved only its
    three-fold mean, not per-fold scores.],
) <tab:folds>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Architecture], [Audio only], [Video only], [Fused]),
    table.hline(stroke: 0.5pt),
    [Cross-Modal Fusion (3 seeds)], [58.8 / 62.5], [*73.2 / 80.2*], [72.2 / 80.4],
    [VideoMAE + HuBERT, LoRA], [62.8 / 67.0], [65.4 / 71.1], [66.3 / 73.1],
    table.hline(),
  ),
  caption: [Unimodal ablations, ACC / AUC on the official protocol, for the two models where both
    single-stream variants were run.],
) <tab:unimodal>

*Reading.* Every measured model beats the majority floor, and every deep model scored on the
official folds scores above the theory-driven baseline: by 17 accuracy and 22 AUC points for
Cross-Modal Fusion, by 4 and 1.5 for MARLIN + ConvNeXt, and by only 1.2 and 0.3 for the Temporal CNN,
which is within the fold-to-fold spread. The two models built on general-purpose pre-trained
visual Transformers, DINOv2 and VideoMAE, are the best, and both score above the PECL result
reported in the paper, under the related but non-identical evaluation described in @sec:data. The
frozen-DINOv2 pipeline scores higher than the LoRA-adapted VideoMAE pipeline, although VideoMAE sees
motion and is adapted to the task. The two pipelines also differ in face detector, crop, frame
count, audio window and validation rule (@tab:inputs, @tab:rules), so this is a comparison of
pipelines, not of freezing against adaptation alone.

MARLIN + ConvNeXt reaches 59.1 ± 3.2% accuracy and 60.3% AUC with 97,857 trainable parameters. A
frozen encoder pre-trained specifically on faces, followed by an explicit temporal model, therefore
does not close the gap to the general-purpose visual encoders on this protocol. No single-stream
variant of it was run, so we cannot say how its score splits between the face and the voice.

This late-fusion design, which averages each stream over time before combining them, did not
improve over the visual stream. In Cross-Modal Fusion the video branch alone matches the fused model
(73.2 vs 72.2 ACC, 80.2 vs 80.4 AUC), and the audio branch alone is 14 points lower. In
VideoMAE + HuBERT the fused model gains one accuracy point over video alone, and an untuned 50/50
fusion is slightly better still (67.1 / 73.7). Neither model lets the two streams interact over
time, so these results say nothing about richer forms of cross-modal fusion. The audio branches are
consistently weaker but not useless: WavLM and HuBERT reach 59–63% accuracy, between the
theory-driven baseline and the visual models.

The Temporal CNN, trained from scratch on 15 facial-landmark regions and 8 eGeMAPS descriptors,
reaches 56.3 ± 2.9% accuracy and 59.1% AUC. Without pre-trained representations it stays close to
the theory-driven baseline, which sees only 17 acoustic statistics, and far below the models built on
pre-trained visual encoders.

The results are stable across folds. Cross-Modal Fusion is the best model on every fold, and its
spread across folds (69.0–74.2) is comparable to its spread across seeds (70.5–73.8, @tab:seeds).
Fold 1 gives the lowest accuracy for every model in @tab:folds except the Temporal CNN, whose folds
1 and 3 are within a point of each other.
