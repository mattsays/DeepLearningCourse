"""The two branches of Paradigm 2, each a frozen pretrained Transformer adapted with LoRA.

* ``VideoBranch``: VideoMAE (spatio-temporal ViT; 16 frames -> 8x14x14 = 1568 tubelet tokens)
  -> mean over tokens -> the checkpoint's ``fc_norm`` -> dropout -> one logit.
* ``AudioBranch``: HuBERT on the raw 16 kHz waveform -> mean over the real (non-padded) frames
  -> LayerNorm -> dropout -> one logit.

LoRA (``peft``) replaces a frozen ``W`` by ``W + (alpha/r) B A`` inside the attention layers; only
``A``, ``B``, the norm and the head are trained (~0.3 M parameters per branch).

**Why** :func:`assert_lora_active` **exists.** In transformers releases up to 5.5, VideoMAE's
attention computes Q and V with ``F.linear(weight=self.query.weight)`` instead of calling the
module, so a LoRA wrapper on ``query``/``value`` is silently skipped: only the head trains, while
``print_trainable_parameters()`` looks perfectly normal. Fixed in 5.6. The check fails loudly
whatever the version, the backbone or the target modules.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LoraSettings:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    targets: tuple[str, ...] = ("query", "value")
    mlp_last_n: int = 0          # also adapt the MLP of the last N blocks
    use_dora: bool = False


def _alternation(names) -> str:
    return "|".join(str(n) for n in names)


def video_targets(num_layers: int, lora: LoraSettings):
    """Module names for VideoMAE: ``encoder.layer.N.attention.attention.{query,value}`` (+ MLP)."""
    if not lora.mlp_last_n:
        return list(lora.targets)
    last = range(num_layers - lora.mlp_last_n, num_layers)
    return (rf".*encoder\.layer\.\d+\.attention\.attention\.({_alternation(lora.targets)})"
            rf"|.*encoder\.layer\.({_alternation(last)})\.(intermediate\.dense|output\.dense)")


def audio_targets(num_layers: int, lora: LoraSettings):
    """Module names for HuBERT: ``encoder.layers.N.attention.{q_proj,v_proj}`` (+ feed-forward)."""
    if not lora.mlp_last_n:
        return list(lora.targets)
    last = range(num_layers - lora.mlp_last_n, num_layers)
    return (rf".*encoder\.layers\.\d+\.attention\.({_alternation(lora.targets)})"
            rf"|.*encoder\.layers\.({_alternation(last)})\.feed_forward\.(intermediate_dense|output_dense)")


def wrap_lora(encoder: nn.Module, targets, lora: LoraSettings) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                        target_modules=targets, bias="none", use_dora=lora.use_dora)
    return get_peft_model(encoder, config)


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class _Branch(nn.Module):
    """Shared plumbing: LoRA or frozen encoder, head, train/eval behaviour, parameter groups."""

    modality = ""

    def _setup(self, encoder: nn.Module, targets, lora: LoraSettings | None, hidden: int,
               head_dropout: float, norm: nn.Module | None = None) -> None:
        self.head_only = lora is None
        if self.head_only:
            _freeze(encoder)
            self.encoder = encoder
        else:
            self.encoder = wrap_lora(encoder, targets, lora)
        self.norm = norm if norm is not None else nn.LayerNorm(hidden)
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(hidden, 1)
        for module in (self.norm, self.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.head_only:          # a frozen backbone stays deterministic (no dropout / masking)
            self.encoder.eval()
        return self

    def _encoder_context(self):
        return torch.no_grad() if self.head_only else nullcontext()

    def enable_gradient_checkpointing(self) -> None:
        if not self.head_only:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """(adapter parameters, head + norm parameters) — they get different learning rates."""
        adapters, head = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (adapters if "lora_" in name else head).append(parameter)
        return adapters, head

    def trainable_state(self) -> dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in the saved state: {unexpected[:5]}")
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        absent = trainable - set(state)
        if absent:
            raise RuntimeError(f"saved state lacks trainable parameters: {sorted(absent)[:5]}")

    def disabled_adapters(self):
        """Context in which the encoder runs without its adapters (the plain pretrained model)."""
        return nullcontext() if self.head_only else self.encoder.disable_adapter()


class VideoBranch(_Branch):
    modality = "video"

    def __init__(self, encoder: nn.Module, norm: nn.Module | None, hidden: int,
                 lora: LoraSettings | None, head_dropout: float = 0.3):
        super().__init__()
        targets = None if lora is None else video_targets(encoder.config.num_hidden_layers, lora)
        self._setup(encoder, targets, lora, hidden, head_dropout, norm)

    @classmethod
    def from_pretrained(cls, backbone: str, lora: LoraSettings | None, head_dropout: float = 0.3) -> "VideoBranch":
        from transformers import VideoMAEForVideoClassification

        base = VideoMAEForVideoClassification.from_pretrained(backbone, attn_implementation="sdpa").float()
        norm = base.fc_norm if base.fc_norm is not None else nn.LayerNorm(base.config.hidden_size)
        return cls(base.videomae, norm, base.config.hidden_size, lora, head_dropout)

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        with self._encoder_context():
            tokens = self.encoder(pixel_values=frames).last_hidden_state      # (B, 1568, H)
        return self.norm(tokens.mean(dim=1))

    def forward(self, frames: torch.Tensor):
        z = self.embed(frames)
        return self.head(self.drop(z)).squeeze(-1), z


class AudioBranch(_Branch):
    modality = "audio"

    def __init__(self, encoder: nn.Module, hidden: int, lora: LoraSettings | None, head_dropout: float = 0.3):
        super().__init__()
        self._frames_for = encoder._get_feat_extract_output_lengths     # samples -> encoder frames
        targets = None if lora is None else audio_targets(encoder.config.num_hidden_layers, lora)
        self._setup(encoder, targets, lora, hidden, head_dropout)

    @classmethod
    def from_pretrained(cls, backbone: str, lora: LoraSettings | None, head_dropout: float = 0.3) -> "AudioBranch":
        from transformers import HubertModel

        encoder = HubertModel.from_pretrained(backbone, attn_implementation="sdpa").float()
        return cls(encoder, encoder.config.hidden_size, lora, head_dropout)

    def embed(self, waves: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        with self._encoder_context():
            # group-norm HuBERT checkpoints are fed zero-padded input without an attention mask
            hidden = self.encoder(input_values=waves).last_hidden_state       # (B, F, H)
        frames = self._frames_for(lengths.to(hidden.device)).clamp(min=1, max=hidden.size(1))
        mask = (torch.arange(hidden.size(1), device=hidden.device)[None, :] < frames[:, None]).to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.norm(pooled)

    def forward(self, waves: torch.Tensor, lengths: torch.Tensor):
        z = self.embed(waves, lengths)
        return self.head(self.drop(z)).squeeze(-1), z


def build_branch(cfg) -> _Branch:
    """Branch for a :class:`lore.config.RunConfig`."""
    lora = None if cfg.head_only else LoraSettings(
        r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, targets=tuple(cfg.lora_targets),
        mlp_last_n=cfg.lora_mlp_last_n, use_dora=cfg.use_dora)
    cls = VideoBranch if cfg.modality == "video" else AudioBranch
    branch = cls.from_pretrained(cfg.backbone, lora, cfg.head_dropout)
    if cfg.grad_checkpointing:
        branch.enable_gradient_checkpointing()
    return branch


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    return {"total": total, "trainable": trainable, "adapters": adapters,
            "trainable_pct": round(100 * trainable / max(total, 1), 3)}


def assert_lora_active(branch: _Branch, *inputs: torch.Tensor) -> dict:
    """Prove that the adapters are part of the forward pass, or raise.

    1. after one backward pass every ``lora_B`` matrix has a non-zero gradient (with ``B = 0`` at
       initialisation the gradient of ``A`` is zero by construction, so only ``B`` is checked);
    2. with ``B`` set to random values, the output differs from the output of the same model with
       its adapters disabled.

    Runs in eval mode (no dropout, LayerDrop or SpecAugment masking, which could skip a layer) and
    restores the original weights. Returns a small report.
    """
    if branch.head_only:
        return {"skipped": "head-only model, no adapters"}
    lora_b = [(n, p) for n, p in branch.named_parameters() if "lora_B" in n]
    if not lora_b:
        raise RuntimeError("no LoRA matrices found: the target modules matched nothing")
    was_training = branch.training
    branch.eval()
    branch.zero_grad(set_to_none=True)

    logits, _ = branch(*inputs)
    logits.float().sum().backward()
    dead = [n for n, p in lora_b if p.grad is None or not torch.any(p.grad != 0)]
    branch.zero_grad(set_to_none=True)
    if dead:
        raise RuntimeError(
            f"{len(dead)}/{len(lora_b)} LoRA matrices receive no gradient (first: {dead[0]}). The adapters are "
            "not in the forward pass — e.g. an attention layer that calls F.linear(module.weight) directly "
            "(VideoMAE in transformers < 5.6). Pin transformers>=5.6 or change the target modules.")

    saved = {n: p.detach().clone() for n, p in lora_b}
    with torch.no_grad():
        for _, p in lora_b:
            p.normal_(0.0, 0.02)
        with_adapters, _ = branch(*inputs)
        with branch.disabled_adapters():
            without_adapters, _ = branch(*inputs)
        for n, p in lora_b:
            p.copy_(saved[n])
    change = float((with_adapters - without_adapters).abs().max())
    branch.train(was_training)
    if change == 0.0:
        raise RuntimeError("enabling the adapters does not change the output: LoRA is not in the forward pass")
    return {"lora_matrices": len(lora_b), "max_logit_change_with_random_B": round(change, 6)}
