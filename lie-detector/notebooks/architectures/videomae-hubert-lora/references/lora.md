# LoRA — Low-Rank Adaptation

Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 ·
[arXiv 2106.09685](https://arxiv.org/abs/2106.09685) ·
[PDF](LoRA%20Low%20Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)

## The idea

Fine-tuning a big pretrained model normally updates **all** its weights. That is expensive:

- **Memory**: gradients and optimizer state for every parameter.
- **Storage**: a full copy of the model per task.
- **Overfitting**: easy when the task has little data.

LoRA starts from one observation: the *change* a pretrained model needs in order to adapt to a new
task is simple. It lives in a few directions, not in the whole weight space (the change has a low
"intrinsic rank"). So instead of learning a full update for a weight matrix, LoRA learns it as the
product of **two thin matrices**.

## How it works

Take any linear layer with a pretrained weight `W₀` (size d × k). LoRA leaves `W₀` frozen and adds a
trainable side path:

```
              ┌──── W₀ (frozen) ────┐
x ────────────┤                     ├──(+)──► h = W₀x + (α/r)·B·A·x
              └── A (r×k) → B (d×r) ┘
                   trainable, r ≪ d
```

- **`A` and `B`.** `A` projects the input down to a tiny dimension `r` (the *rank*), and `B` projects it
  back up. Their product `B·A` has the same shape as `W₀`, but it is described by far fewer numbers.
  With d = k = 768 and r = 8, that is 12 k numbers instead of 590 k.
- **Start at zero.** `B` is initialised **at zero**, so at the start of training the model is *exactly*
  the pretrained one. Training then only moves it as far as the task needs.
- **Scaling.** `α/r` scales the update, so changing `r` does not require re-tuning the learning rate.
- **No cost at inference.** After training, `B·A` can be added into `W₀` once. The final model has the
  same size and speed as the original. Earlier *adapter* methods (the family DOLOS's UT-Adapter belongs
  to) instead add extra layers that stay in the model and slow it down.
- **Same as fine-tuning in the limit.** If `r` grows to the full size, LoRA becomes ordinary
  fine-tuning. A small `r` is a deliberate restriction.

## Where to put it

A Transformer has four projections in each attention block — query `Wq`, key `Wk`, value `Wv` and
output `Wo` — plus an MLP. The paper studies where a fixed budget of trainable parameters helps most:

- **Q and V.** Adapting **`Wq` and `Wv`** together works best. Spreading a small rank across several
  matrices beats a large rank on one.
- **A very small rank.** The rank can be tiny: even `r = 1`–`4` is often enough, and a larger `r`
  mostly learns noise.
- **What the update does.** It does not repeat what `W₀` already does well. It **amplifies features the
  pretrained model already has but under-uses**, the ones that matter for the new task.

That last point explains why LoRA suits small datasets: it re-weights knowledge the model already has
instead of learning new knowledge from scratch.

## How it is used in `lore`

LoRA is how both `lore` branches are trained. The backbone stays frozen, and only the LoRA matrices
plus a small classification head learn.

- **`lore/models.py`:**
  - `wrap_lora` wraps an encoder with the `peft` library (`LoraConfig` + `get_peft_model`).
  - `video_targets` / `audio_targets` list the modules to adapt: the **query and value** projections
    of all 12 attention layers. In VideoMAE these are `attention.attention.{query,value}`; in HuBERT,
    `{q_proj,v_proj}`. This follows the paper's recommendation.
  - `parameter_groups` gives the LoRA weights and the head **different learning rates**.
  - `trainable_state` saves only the trainable weights, so a fold checkpoint (`trainable.pt`) is about
    1 MB instead of a full model.
- **`lore/config.py`.** The rank, `α`, dropout and learning rates are set here (`lora_r`, `lora_alpha`,
  …). It also defines ablations: `head_only` (no LoRA at all), `lora_mlp_last_n` (LoRA on the MLPs
  too) and `use_dora` (a LoRA variant).
- **`assert_lora_active`.** This is a safety check the paper does not need. In some `transformers`
  versions, VideoMAE computes Q and V in a way that **skips** the LoRA-wrapped layers, so the adapters
  silently do nothing. Before every fold, `lore` checks two things:
  - the LoRA weights receive gradients;
  - changing them changes the output.

  See [`docs/report.md`](../docs/report.md) §4.3.

The models being adapted are [VideoMAE](videomae.md) and [HuBERT](hubert.md).
