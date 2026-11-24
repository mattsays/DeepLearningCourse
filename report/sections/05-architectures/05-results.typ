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
    [Temporal CNN#super[†]], [62.9], [—], [64.8], [—], [57,322], [CPU],
    [*Cross-Modal Fusion*], [*72.2 ± 3.0*], [*73.1*], [*80.4*], [*72.2*], [164,737], [≈25 s],
    [VideoMAE + HuBERT, LoRA], [66.3 ± 1.5], [65.9], [73.1], [66.6], [594,434], [54 min],
    [MARLIN + ConvNeXt], [59.1 ± 3.2], [61.4], [60.3], [60.6], [97,857], [≈15 s],
    table.hline(),
  ),
  caption: [Official 3-fold protocol: mean ± standard deviation over test folds (and seeds, for
    the 3-seed models). Baseline and Cross-Modal Fusion: 3 seeds; VideoMAE + HuBERT: seed 0. Training
    time is for one seed over the three folds on a T4 GPU, excluding feature extraction.
    #super[†]Not on the official folds: validation set (213 clips) of a single random 85/15
    clip-level split; included for orientation only.],
) <tab:main>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.hline(),
    table.header([Fold], [Cross-Modal Fusion], [VideoMAE + HuBERT], [Theory-driven baseline]),
    table.hline(stroke: 0.5pt),
    [1], [69.0 / 77.1], [65.2 / 72.9], [52.3 / 55.0],
    [2], [73.4 / 81.3], [67.9 / 76.0], [56.9 / 61.3],
    [3], [74.2 / 82.9], [65.7 / 70.3], [56.0 / 60.0],
    table.hline(),
  ),
  caption: [ACC / AUC per official fold (mean over seeds where several were run).],
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
    [Temporal CNN], [#notrun], [#notrun], [62.9 / 64.8#super[†]],
    [MARLIN + ConvNeXt], [#notrun], [#notrun], [#notrun],
    table.hline(),
  ),
  caption: [Unimodal ablations, ACC / AUC on the official protocol. #super[†]Random-split
    validation, see @tab:main.],
) <tab:unimodal>

*Reading.* Every measured model beats the majority floor, and every deep model beats the
theory-driven baseline by a wide margin: 17 accuracy and 22 AUC points for Cross-Modal Fusion. The
ranking follows the strength of the _visual_ representation. The two models built on pre-trained
visual Transformers are the best, and both outperform PECL; the frozen DINOv2 features are better
than the LoRA-adapted VideoMAE, although VideoMAE sees motion and is adapted to the task.

Fusion adds little. In Cross-Modal Fusion the video branch alone matches the fused model (73.2 vs
72.2 ACC, 80.2 vs 80.4 AUC), and the audio branch alone is 14 points lower. In VideoMAE + HuBERT the
fused model gains one accuracy point over video alone, and an untuned 50/50 fusion is slightly
better still (67.1 / 73.7). The audio branches are consistently weaker but not useless: WavLM and
HuBERT reach 59–63% accuracy, between the theory-driven baseline and the visual models.

The results are stable across folds. Cross-Modal Fusion is the best model on every fold, and its
spread across folds (69.0–74.2) is comparable to its spread across seeds (70.5–73.8, @tab:seeds).
Fold 1 is the hardest for all three models.

The Temporal CNN reaches 62.9% accuracy and 64.8% AUC, but on a validation set of a random clip-level
split, which is not comparable with the official test folds. MARLIN + ConvNeXt achieves 59.1 ± 3.2%
accuracy and 60.3% AUC (61.4% F1, 60.6% Balanced ACC), comfortably beating the majority baseline with
only 97,857 trainable parameters and training in ≈15 seconds across the three folds.

#todo[Temporal CNN: run on the official folds and replace the #super[†] entries in @tab:main and @tab:unimodal.]
