#import "../../template.typ": *

// ================================================================ 5
= Deep Architectures <sec:arch>

All four architectures share one template: an encoder per modality turns the clip into a sequence
or a vector, a fusion step combines the modalities, and a head outputs one deception logit. They
differ in three choices: _what is pre-trained_, _what is trained_, and _where time is modelled_
(@fig:overview).

#figure(
  {
    set align(left)
    set par(justify: false)
    grid(
      columns: (auto, 1fr),
      column-gutter: 8pt,
      row-gutter: 7pt,
      align: horizon,
      text(size: 8pt, weight: "bold")[Temporal CNN],
      [#blk[landmarks + LLDs, T×53] #arr #blk(fill: train-fill)[4× Conv1d, 64 ch] #arr #blk[masked mean] #arr #blk(fill: train-fill)[Linear → logit]],
      text(size: 8pt, weight: "bold")[Cross-Modal Fusion],
      [#blk(fill: frozen-fill)[DINOv2 ∥ WavLM] #arr #blk(fill: train-fill)[Linear → 128, ×2] #arr #blk[mean, concat] #arr #blk(fill: train-fill)[LN + MLP → logit]],
      text(size: 8pt, weight: "bold")[VideoMAE + HuBERT],
      [#blk(fill: frozen-fill)[VideoMAE ∥ HuBERT] #blk(fill: train-fill)[\+ LoRA r = 8] #arr #blk[mean] #arr #blk(fill: train-fill)[LN + Linear, ×2] #arr #blk[weighted z-sum]],
      text(size: 8pt, weight: "bold")[MARLIN + ConvNeXt],
      [#blk(fill: frozen-fill)[MARLIN ∥ ConvNeXt] #arr #blk(fill: train-fill)[LN + Linear, fuse] #arr #blk(fill: train-fill)[dilated TCN] #arr #blk[mean] #arr #blk(fill: train-fill)[MLP → logit]],
    )
  },
  caption: [The four architectures. Grey: frozen pre-trained encoder; blue: trained; white:
    parameter-free. "∥" separates the visual and the audio branch.],
) <fig:overview>
