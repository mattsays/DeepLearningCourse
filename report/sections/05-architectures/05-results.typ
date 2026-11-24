#import "../../template.typ": *

== Results <sec:results>

@tab:main compares all models on the official 3-fold protocol, @tab:folds breaks the scores down by
fold, and @tab:unimodal isolates the contribution of each modality.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.hline(),
    table.header([Model], [ACC], [F1], [AUC], [Bal. ACC], [Trainable]),
    table.hline(stroke: 0.5pt),
    [Majority class (same clips)], [52.4–53.6], [69.6], [50.0], [50.0], [0],
    [PECL @guo2023dolos (reported)], [64.75], [71.20], [62.71], [n.r.], [5.06M],
    [Theory-driven baseline], [56.5 ± 1.7], [60.2], [59.2], [56.1], [609],
    table.hline(stroke: 0.3pt),
    [Temporal CNN], [56.3 ± 2.9], [56.1], [59.1], [56.4], [57,322],
    [*Cross-Modal Fusion*], [*72.2 ± 3.0*], [*73.1*], [*80.4*], [*72.2*], [164,737],
    [VideoMAE + HuBERT, LoRA], [66.3 ± 1.5], [65.9], [73.1], [66.6], [594,434],
    [MARLIN + ConvNeXt], [58.9 ± 0.9], [59.8], [58.9], [59.0], [97,857],
    table.hline(),
  ),
  caption: [Official 3-fold protocol: mean ± standard deviation over test folds (and seeds, for
    the 3-seed models). Baseline, Temporal CNN and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT
    and MARLIN + ConvNeXt: seed 0. Each model is scored on its own subset of the official test clips
    (@tab:coverage), and the PECL row is quoted from the paper, not re-run (n.r.: not reported). The
    majority class always predicts `deception`; its F1 is computed on the 1,427 downloadable test
    clips, and a constant score has AUC 50 by definition.],
) <tab:main>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.hline(),
    table.header([Fold], [Cross-Modal Fusion], [VideoMAE + HuBERT], [MARLIN + ConvNeXt], [Temporal CNN], [Theory-driven baseline]),
    table.hline(stroke: 0.5pt),
    [1], [69.0 / 77.1], [65.2 / 72.9], [60.0 / 58.1], [55.1 / 56.8], [54.9 / 57.2],
    [2], [73.4 / 81.3], [67.9 / 76.0], [58.4 / 56.8], [59.4 / 62.0], [56.7 / 60.3],
    [3], [74.2 / 82.9], [65.7 / 70.3], [58.4 / 61.8], [54.5 / 58.4], [57.9 / 60.2],
    [Mean], [72.2 / 80.4], [66.3 / 73.1], [58.9 / 58.9], [56.3 / 59.1], [56.5 / 59.2],
    table.hline(),
  ),
  caption: [ACC / AUC per official fold (mean over seeds where several were run); the last row
    repeats the three-fold means of @tab:main. MARLIN + ConvNeXt is recomputed from its per-clip
    predictions (`fab/results/marlin-official/preds.csv`) against the `dataset.csv` labels.],
) <tab:folds>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Architecture], [Audio only], [Video only], [Fused]),
    table.hline(stroke: 0.5pt),
    [Theory-driven baseline (3 seeds)], [56.5 / 59.2], [—], [—],
    [Temporal CNN (3 seeds)], [55.9 / 59.1], [55.0 / 57.8], [56.3 / 59.1],
    [Cross-Modal Fusion (3 seeds)], [58.8 / 62.5], [*73.2 / 80.2*], [72.2 / 80.4],
    [VideoMAE + HuBERT, LoRA], [62.8 / 67.0], [65.4 / 71.1], [66.3 / 73.1],
    [MARLIN + ConvNeXt], [n/a], [n/a], [58.9 / 58.9],
    table.hline(),
  ),
  caption: [Unimodal ablations, ACC / AUC on the official protocol. —: the model has no such
    stream. n/a: single-stream variants of MARLIN + ConvNeXt were not run. The Temporal CNN's single-stream variants zero the other stream's input
    columns (`teo/features/mask_streams.py`) and keep the same training recipe.],
) <tab:unimodal>

*Reading.* Every measured model beats the majority floor, but only two clearly beat the
theory-driven baseline: Cross-Modal Fusion by 16 accuracy and 21 AUC points, and VideoMAE + HuBERT by
10 and 14. MARLIN + ConvNeXt (+2.4 ACC, −0.3 AUC) and the Temporal CNN (−0.2, −0.1) are level with
it, within the fold-to-fold spread. The two models built on general-purpose pre-trained
visual Transformers, DINOv2 and VideoMAE, are the best, and both score above the PECL result
reported in the paper, under the related but non-identical evaluation described in @sec:data. The
frozen-DINOv2 pipeline scores higher than the LoRA-adapted VideoMAE pipeline, although VideoMAE sees
motion and is adapted to the task. The two pipelines also differ in face detector, crop, frame
count, audio window and validation rule (@tab:inputs, @tab:rules), so this is a comparison of
pipelines, not of freezing against adaptation alone.

MARLIN + ConvNeXt reaches 58.9 ± 0.9% accuracy and 58.9% AUC with 97,857 trainable parameters. A
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
reaches 56.3 ± 2.9% accuracy and 59.1% AUC. Without pre-trained representations it stays level with
the theory-driven baseline, which sees only 17 acoustic statistics, and far below the models built on
pre-trained visual encoders. Its two streams are nearly interchangeable: landmarks alone (55.0 /
57.8) and eGeMAPS alone (55.9 / 59.1) each come within 1.3 points of the fused model, so neither
stream adds much to the other.

The results are stable across folds. Cross-Modal Fusion is the best model on every fold, and its
spread across folds (69.0–74.2) is comparable to its spread across seeds (70.5–73.8, @tab:seeds).
Fold 1 gives the lowest accuracy for Cross-Modal Fusion, VideoMAE + HuBERT and the baseline. For
the three weaker models the spread across folds (1.6–4.9 accuracy points) is as large as the
differences between them, so their ordering is not meaningful.
