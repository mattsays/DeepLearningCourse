# The trainable module of TIA (Paradigm 1)

Read this if: you are building `train.ipynb`, choosing a fusion strategy, or sizing
the part of the network that actually learns. It covers **only** the trainable part of
[Paradigm 1](../../../../docs/architectures.md#paradigm-1-frozen-foundation-models--cross-modal-fusion-pecl-inspired---tia);
DINOv2 and WavLM are frozen and are not discussed here. This is a design note, not a spec — what the
notebook implements is the notebook's business, and how a result is measured is
[evaluation.md](../../../../docs/evaluation.md).

## Inputs

The frozen encoders are run once by `extract_features.ipynb`, which writes one array per clip into the
shared cache:

```
<DATA>/cached_features/dinov2-small/<clip_id>.npy      float16  (16, 384)   # CLS of 16 face crops
<DATA>/cached_features/wavlm-base-plus/<clip_id>.npy   float16  (64, 768)   # WavLM, pooled to 64 tokens
```

Training reads them through `CachedFeatures` + `make_loader` ([notebooks.md](../../../../docs/notebooks.md)),
and takes the labels and the `pos_weight` from `dataset.csv` via `Clips`. The split is the frozen
protocol; [splits.md](../../../../docs/splits.md) measures what a fold hands out — about 1,160 clips to
train on, 124 for early stopping and 204 for the score.

## The budget

Per clip the trainable net receives 16 × 384 + 64 × 768 = 55,296 floats and must return one logit. Two
consequences follow, and they decide more than the architecture does.

**Capacity is bounded by n, not by VRAM.** With ~1,160 training clips per fold, the useful range is
10⁵ trainable parameters, not 10⁶. Above roughly 300–500k (≈250–400 parameters per clip) the
training loss reaches zero within a few epochs and the validation curve stops following it.

**The constraint is variance, not compute.** Cached features make an epoch cheaper than a minute, so
the expensive thing is not training a configuration — it is repeating it across 5 folds × 3 seeds. With
~204 holdout clips, ±3.5 pp is sampling noise: one fold cannot separate two configurations that differ
by 2 pp. Differences between directions are argued with McNemar on the saved predictions
(`preds/cv_<direction>.csv`), not with the gap between two means.

## The five stages

**1. Projections.** `384→d` and `768→d` cost 147,712 parameters at `d=128` and 295,424 at `d=256` — more
than most of the fusion modules in the table below. The dimension is therefore the first decision, not
the last. Options: `d=128`; a **PCA fitted on the training fold** (zero trainable parameters, and it
regularises); a `LayerNorm` after each projection. Any statistic used here — PCA, mean/std — is fitted
on the training fold only and applied unchanged to val and holdout.

**2. Token normalisation.** Features arrive as `float16` and with very different scales between the two
backbones. `LayerNorm(d)` per token is the robust default; per-dimension standardisation with
training-fold statistics is the alternative; L2-normalising each token (cosine geometry) is worth one
ablation on the DINOv2 CLS vectors.

**3. Time.** Attention is permutation-invariant, so without a temporal positional embedding the model
cannot tell the beginning of an answer from its end — which is where the signal is. A learned PE over
16 + 64 positions is close to mandatory. Then, in increasing cost: **delta features**
(`x_t − x_{t−1}`, concatenated before the projection) for micro-expressions and prosody shifts; a
dilated `Conv1d` over time; one or two self-attention layers per modality before fusion.

**4. Fusion.** The menu is the table in the next section.

**5. Pooling and head.** `mean` over time is the baseline; `mean + std` concatenated is the cheap
upgrade; learned attention pooling adds ~d² parameters and needs regularising. Head:
`LayerNorm → Dropout → Linear(d, 1)`, not a two-layer MLP at n=1,160. No BatchNorm anywhere: the batch
is small, and it interacts badly with modality dropout.

## The fusion menu

Parameter counts are **measured**, not estimated: they exclude the two input projections (every variant
pays them) and include the head, at `heads=4`. The congruence probe is the exception — the cosine needs a
shared space, so it brings its own projections. Reproduce them with the snippet at the end of this note.

| Module | Params `d=128` | Params `d=256` | What it adds |
|---|---:|---:|---|
| Late fusion: pool each modality, concat, MLP `2d→64→1` | 16,513 | 32,897 | interaction between the two mean vectors |
| Two unimodal heads + two logit weights | 260 | 516 | only the relative reliability of the channels |
| Congruence probe: project both, `1 − cos`, one weight | 147,714 | 295,426 | only disagreement matters, not content |
| GMU: `g ⊙ W_v v + (1−g) ⊙ W_a a`, per token | 65,920 | 262,912 | gated additive combination |
| Cross-attention one direction (A→V) + FFN | 131,968 | 526,080 | alignment of voice onto face |
| Co-attention symmetric (A↔V, two FFNs) | 263,936 | 1,052,160 | bidirectional incongruence |
| Joint self-attention over 80 tokens, 1 layer, + PE, + head | 143,361 | 548,865 | temporal dynamics **and** interaction in one stack |
| Perceiver: K=8 latents, 2 blocks (cross + self + FFN) | 397,441 | 1,581,313 | compression into a few latent tokens |
| PECL: correlation matrix `W ∈ R^{d×d}` + bottleneck 64 + classifier | 37,185 | 102,721 | learned cross-modal similarity |

Read it in four groups:

- **Controls, not candidates.** The two unimodal heads answer *does fusion add anything over the better
  single channel?* — if audio-only reaches the fused score, the report says so and that is a finding. The
  congruence probe answers a narrower question, *is the decision driven only by disagreement?*, and pays
  for its projections because a cosine across 384 and 768 dimensions needs a shared space.
- **The baseline.** Late fusion is what every other row has to beat. On a dataset this size it often
  wins, and a tie is a result.
- **The attention family.** Joint self-attention and cross-attention differ less than they appear to:
  the joint version is the same interaction with shared projections and one stack, which is why it is
  the recommended default. The PECL module is the cheapest attention-based option because its
  interaction is a single `d×d` matrix instead of Q/K/V projections.
- **The capacity trap.** At `d=256` nothing except late fusion fits the budget, and the Perceiver is
  already 397k at `d=128` — use `K=4` or one block if you want it.

## The alignment problem: 16 frames vs 64 audio tokens

In the paper both branches carry the same number of temporal steps (64 frames, 64 audio tokens), so the
correlation matrix is 64×64 and "the same instant" means the same index. Our cache has 16 frames and 64
audio tokens, so cross-attention has to *learn* a soft 16↔64 alignment from content alone.

The sampling is also very coarse. The median DOLOS clip is 5 s
([dataset/README.md](../../../../dataset/README.md)), so 16 frames is ~3 fps — one sample every ~310 ms,
while a micro-expression is usually placed at 40–600 ms in the literature. At the median clip, 64 frames
would be ~13 fps.

Two honest ways out, and the choice belongs in the notebook's header:

1. **Bin the audio to 16 tokens** (mean of 4 consecutive WavLM tokens). Forces a 1:1 index alignment,
   keeps the cache, throws away fine acoustic time resolution.
2. **Re-extract the video with `FRAMES=32` or `64`** (`extract_features.ipynb` records `frames` in the
   manifest, so it rebuilds cleanly). Restores the resolution the task needs, costs another extraction
   pass, and makes the comparison with the paper fair.

Whichever is chosen, a positional embedding on both sides is what lets the model exploit the alignment
it learns.

## Reference implementation

`model.py` holds what is implemented so far — `late` plus the rung-0 controls (`video`, `audio`,
`congruence`) — and the `VARIANTS` registry the notebook selects from. The class below is the next rung: joint self-attention over the 80 tokens, one
layer, `d=128` — 143,361 parameters with the embeddings and the head, before the two input projections.

```python
class Fusion(nn.Module):
    """80 tokens (16 video + 64 audio) -> one logit per clip."""

    def __init__(self, d=128, heads=4, layers=1, drop=0.3):
        super().__init__()
        self.video, self.audio = nn.Linear(384, d), nn.Linear(768, d)
        self.type_emb = nn.Parameter(torch.zeros(2, 1, d))      # video / audio
        self.time_emb = nn.Parameter(torch.zeros(1, 16 + 64, d))  # 16 + 64 positions
        block = nn.TransformerEncoderLayer(d, heads, 2 * d, drop,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, video, audio):                 # (B,16,384), (B,64,768)
        v = self.video(video) + self.type_emb[0] + self.time_emb[:, :16]
        a = self.audio(audio) + self.type_emb[1] + self.time_emb[:, 16:]
        h = self.encoder(torch.cat([v, a], dim=1))   # 80 tokens
        return self.head(h.mean(dim=1)).squeeze(-1)
```

The cross-attention variant in
[architectures.md](../../../../docs/architectures.md#paradigm-1-frozen-foundation-models--cross-modal-fusion-pecl-inspired---tia)
replaces the encoder with two `nn.MultiheadAttention(d, heads, batch_first=True)` calls — `audio` querying
`video` and the other way round — before the same pooling and head. It costs about twice as much and
states a stronger hypothesis about alignment.

The interface stays the one the template expects: `forward` returns `(batch,)`, one logit per clip, and
the loop keeps `BCEWithLogitsLoss` so that `best_threshold` and the metrics in `lib/metrics.py` apply
unchanged.

## Training recipe

These matter more than the choice of fusion module.

- **Loss.** `BCEWithLogitsLoss(pos_weight=clips.class_weights(train_ids))`. The training folds are
  ~56% deception and the holdouts ~47% ([notebooks.md](../../../../docs/notebooks.md)), so an unweighted
  loss, or a 0.5 threshold, already predicts `deception` too often.
- **Auxiliary unimodal heads** (the `multi` flag in PECL): `L = L_fused + 0.3·(L_audio + L_video)`. It
  regularises the fusion and yields the per-channel AUC for free.
- **Modality dropout**, p ≈ 0.15: zero one channel's tokens at random during training. The model cannot
  lean on audio alone, and at test time it degrades gradually instead of collapsing.
- **Optimisation.** AdamW, lr 1e-3, weight decay 5e-2, cosine schedule with 10% warmup, gradient clip
  1.0, batch 32–64, 40–60 epochs with early stopping on **validation AUC** (patience 8).
- **Checkpoint averaging.** EMA of the weights (0.999), or SWA over the last epochs. With ~124
  validation clips, selecting on a single checkpoint is noisy.
- **Threshold.** From `best_threshold` on val. AUC stays the primary metric because it is
  threshold-free.

## Experiment ladder

Run these in order; each rung is cheap and makes the next one interpretable.

| Rung | Configuration | What it answers |
|---|---|---|
| 0 | Congruence probe, video-only head, audio-only head | where the signal is, and what fusion must beat |
| 1 | Late fusion (pool + concat + MLP) | the baseline every candidate must clear |
| 2 | Joint self-attention, 80 tokens, PE, `d=128` | the recommended default |
| 3 | Cross/co-attention, and Perceiver with `K=4` | whether the paper's mechanism adds anything |
| 4 | Ablations on the winner, one at a time: PE off, delta off, aux heads off, modality dropout off, `d ∈ {64,128,256}`, 1 vs 2 layers | which part of the gain is real |

Rung 4 is where the report gets its content: an ablation that costs 1 point is worth more than a third
fusion variant.

## Selecting the one configuration that goes on the board

The board carries the configuration that represents the direction — here a single row — and that row is
chosen **before** the holdout is read. Same principle as the threshold rule in
[evaluation.md](../../../../docs/evaluation.md): the decision is taken on `val`, the holdout only
reports. Two phases:

1. **Rank on val.** Run every candidate × 5 folds × 3 seeds, early stopping on val, and write only
   validation metrics to `results/cv.csv`. The holdout is not evaluated in this phase, so the code
   cannot leak it by accident.
2. **One holdout pass.** Take the winner, evaluate it on each fold's holdout, and write the single board
   row — mean and spread over the five folds, plus the pooled number and the majority baseline on the
   same clips.

Three rules that make the choice defensible when someone asks why this configuration:

- **Rank by mean val AUC over folds and seeds**, not by one fold: a fold's val is 124 clips.
- **Require a margin.** Pooled val AUC over the ~620 val predictions of the five folds carries a
  standard error of roughly 0.02–0.03, so a gap under ~2 pp is not a win. Below it, take the cheaper
  model (fewer parameters, fewer seconds) and say so — [notebooks.md](../../../../docs/notebooks.md)
  treats cost as a result.
- **Never pool val across folds into a reported number.** Val subjects repeat across folds, so the pooled
  val score ranks candidates but estimates nothing. Only the holdout is an estimate.

The runners-up stay in the CSV, not on the board: the file shows the ranking, the margin, and that the
holdout was read once. A direction that picks its row by holdout accuracy reports a number inflated by
the selection itself, and the comparison with the other directions stops meaning anything.

## What to report

- **Per-channel AUC against the fused AUC.** This is the answer to "is the fusion earning its keep".
  A 2-point gain is a result here; so is a tie, and a tie is cheaper to report than to chase.
- **Attention maps.** Which frame and which token the model reads. One 16-frame heatmap for a false
  positive is worth more than a table.
- **Paired comparison.** McNemar on the pooled predictions saved by the template; nothing else can
  separate two configurations at this sample size.
- **Cost.** Trainable parameters, wall-clock per epoch, GPU. "Same accuracy, 5× cheaper" is a legitimate
  row.
- **Negative results.** If fusion does not beat the best unimodal channel, that goes in the report
  ([AGENTS.md](../../../../AGENTS.md), rule 5).
- **The protocol effect.** The same model scores 73.8 ACC on the paper's split and 56.5 on ours;
  [protocol-effect.md](protocol-effect.md) has the measurement, the mechanism and the wording for the
  report.

## Fidelity to PECL

The reference implementation in
[`baselines/DOLOS-PECL`](../../../../baselines/DOLOS-PECL/scripts/models/fusion_model.py) is **not**
fully frozen. [`adapter.py`](../../../../baselines/DOLOS-PECL/scripts/models/adapter.py) inserts
non-linear adapters (`768→256→768`) inside every encoder layer, and the cross-modal module runs
layer by layer on 64 frames against 64 audio tokens, with optional unimodal auxiliary classifiers.

This direction freezes both backbones completely and fuses once, on the final features. That is a
legitimate and much cheaper variant, but in the report it must be called **PECL-inspired**, and the
difference stated — otherwise the comparison with the paper is not honest.

## Decisions to record

The notebook prints these in its results cell, because the report needs them and nobody can reconstruct
them later:

| Decision | Values |
|---|---|
| Fusion module | late / joint self-attention / cross-attention / Perceiver / PECL-corr / GMU |
| `d_model`, heads, layers | 128, 4, 1 |
| Positional embedding | on / off (and modality embedding) |
| Delta features | on / off |
| Alignment | audio binned to 16 / video re-extracted at 32–64 / none |
| Modality dropout | p |
| Auxiliary heads | λ, or off |
| Pooling | mean / mean+std / attention |
| Seeds and folds | 3 × 5 |
| Threshold source | val |

## Reproducing the parameter counts

The snippet below prints the table above and is the only source for those numbers; it runs in about a
second. If a row changes, the table changes with it.

```python
import torch
import torch.nn as nn

def count(module, *extra):
    """Trainable parameters, plus any raw tensors passed separately."""
    return (sum(p.numel() for p in module.parameters() if p.requires_grad)
            + sum(t.numel() for t in extra))

def variants(d, heads=4, K=8):
    ffn = lambda: nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    mha = lambda: nn.MultiheadAttention(d, heads, batch_first=True)
    emb = lambda *shape: torch.zeros(shape)
    head = lambda: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    return {
        "late fusion":      (nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(), nn.Linear(64, 1)), []),
        "two unimodal":     (nn.Sequential(nn.Linear(d, 1), nn.Linear(d, 1)), [emb(2)]),
        "congruence probe": (nn.Linear(1, 1), []),
        "GMU":              (nn.Sequential(nn.Linear(d, d), nn.Linear(d, d), nn.Linear(2 * d, d)), []),
        "cross-attention":  (nn.Sequential(mha(), ffn()), []),
        "co-attention":     (nn.Sequential(mha(), mha(), ffn(), ffn()), []),
        "joint":            (nn.Sequential(nn.TransformerEncoderLayer(d, heads, 2 * d,
                                          batch_first=True, norm_first=True), head()),
                             [emb(2, 1, d), emb(1, 80, d)]),
        "perceiver":        (nn.Sequential(*[nn.Sequential(mha(), mha(), ffn()) for _ in range(2)],
                                           head()), [emb(1, K, d)]),
        "PECL corr":        (nn.Sequential(nn.Linear(d, d, bias=False), nn.Linear(2 * d, 64),
                                           nn.LayerNorm(64), nn.Linear(64, 64), nn.Linear(64, 1)), []),
    }

for d in (128, 256):
    print(f"d={d}  projections 384->d + 768->d: {384 * d + d + 768 * d + d:,}")
    for name, (module, extra) in variants(d).items():
        print(f"  {name:16s} {count(module, *extra):>9,d}")
```

The counts exclude the two input projections, which every variant pays, and include the head. The two
rows that are easy to get wrong: the Perceiver is the most expensive option because each block carries a
cross-attention, a self-attention and an FFN (397k at `d=128`, not the ~200k the block count suggests),
and PECL's module is the cheapest attention-based one because its interaction is a single `d×d` matrix.
