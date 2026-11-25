#import "../../template.typ": *

= Implementation Details <app:impl>

@tab:hparams lists the training settings of every model as found in the code. Feature extraction
and the Cross-Modal Fusion, baseline and MARLIN heads ran on Google Colab or Kaggle with an NVIDIA T4
GPU; VideoMAE + HuBERT ran on Kaggle with two T4 GPUs, one branch per GPU (torch 2.11,
transformers 5.23, peft 0.21.3); the Temporal CNN (MediaPipe 1.2.3, openSMILE 2.6.0, feature
extraction included) and the acoustic features run on CPU. All numbers in
this report come from the saved result files of each run, which record the dataset and split hashes.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    table.hline(),
    table.header([Model], [Optimiser, LR], [Weight decay], [Batch], [Epochs], [Other]),
    table.hline(stroke: 0.5pt),
    [Theory baseline], [AdamW, 1e-3], [0.01], [32], [60 (fixed) official; ≤ 60, patience 10 subject], [dropout 0.3, clip 1.0, 3 seeds],
    [Temporal CNN], [AdamW, 3e-4], [1e-4], [16], [≤ 40, early stop (val AUC), patience 15], [plateau LR ×0.5, clip 5.0, dropout 0.3, 3 seeds],
    [Cross-Modal Fusion], [AdamW, 1e-3], [0.05], [16], [30 (fixed)], [dropout 0.3, clip 1.0, 3 seeds],
    [VideoMAE + HuBERT], [AdamW, 2e-4 (LoRA), 1e-3 (head)], [0.01], [16 (accum.)], [≤ 10 video, ≤ 15 audio], [warm-up + cosine, fp16, r 8, α 16],
    [MARLIN + ConvNeXt], [AdamW, 3e-4], [0.01], [16], [≤ 35, patience 7], [cosine, clip 1.0, dropout 0.3],
    table.hline(),
  ),
  caption: [Training hyperparameters. All models use binary cross-entropy with the positive class
    weighted by the negative/positive ratio of the training data.],
) <tab:hparams>
