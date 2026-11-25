"""Fusion variants for the TIA direction (Paradigm 1).

The backbones are frozen; `extract_features.ipynb` caches their output once per clip:

    video  (16, 384)   DINOv2 CLS of 16 face crops, one vector per frame
    audio  (64, 768)   WavLM last hidden state, average-pooled to 64 tokens

Every variant takes ``(video, audio)`` and returns one logit per clip, shape ``(B,)``, so
the training loop in ``train.ipynb`` is identical for all of them. ``VARIANTS`` is the
registry the notebook selects from: adding a method is one class plus one line here.

The video cache is usually DINOv2 ``(16, 384)``, but the direction also takes other backbones:
`extract_vjepa2.ipynb` writes ``cached_features/vjepa2-vitl/<clip_id>.npy`` as ``(32, 1024)``,
and `extract_features.ipynb` writes the supervised HSEmotion encoder as ``(16, 1280)`` — or
``(16, 8)`` with ``output: "emotions"``, the class posteriors. Only the input width changes, so
the classes read the widths from module constants and ``configure()`` sets them from whatever
cache the notebook loaded. The temporal axis is already pooled by each variant, so a different
token count needs no other change either.

Parameter counts, the reasoning behind each option and the experiment ladder are in
``docs/trainable-module.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

VIDEO_DIM = 384
AUDIO_DIM = 768


def configure(video_dim: int | None = None, audio_dim: int | None = None) -> None:
    """Set the input widths the variants build their first layer from.

    Call this once, after the caches are loaded and before the first model is built. Without
    it the defaults above apply, which is what DINOv2 + WavLM produce.
    """
    global VIDEO_DIM, AUDIO_DIM
    if video_dim:
        VIDEO_DIM = int(video_dim)
    if audio_dim:
        AUDIO_DIM = int(audio_dim)


class LateFusion(nn.Module):
    """Pool each modality to one vector, concatenate, classify.

    The baseline every other fusion has to beat, and the cheapest thing that can use both
    channels at once. No temporal modelling: the 16 frames and the 64 tokens are averaged
    away before the two streams meet.
    """

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.video = nn.Linear(VIDEO_DIM, d)
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Dropout(drop),
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = self.video(video).mean(dim=1)      # (B, 16, 384) -> (B, d)
        a = self.audio(audio).mean(dim=1)      # (B, 64, 768) -> (B, d)
        return self.head(torch.cat([v, a], dim=-1)).squeeze(-1)


class VideoOnly(nn.Module):
    """Late fusion with the audio stream switched off: the control for "does fusion help?"."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.video = nn.Linear(VIDEO_DIM, d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.video(video).mean(dim=1)).squeeze(-1)


class AudioOnly(nn.Module):
    """The same, with only WavLM: if this matches the fused model, fusion earns nothing."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.audio(audio).mean(dim=1)).squeeze(-1)


class Congruence(nn.Module):
    """Decide on the *disagreement* between the channels, and nothing else.

    Both streams are projected into a common d-dimensional space and collapsed to one
    scalar, `1 - cos(v, a)`, which a single weight turns into a logit. The cosine needs a
    shared space, so this costs what `LateFusion` costs — it is a different hypothesis, not
    a cheaper control: if it matches the fusion, only agreement matters, not content.
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.video = nn.Linear(VIDEO_DIM, d)
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = nn.functional.normalize(self.video(video).mean(dim=1), dim=-1)
        a = nn.functional.normalize(self.audio(audio).mean(dim=1), dim=-1)
        score = 1.0 - (v * a).sum(dim=-1)          # (B,), in [0, 2]
        return self.weight * score + self.bias


class AttentionPool(nn.Module):
    """Learned weighting over time instead of a mean.

    A small additive score per token, softmaxed over the sequence. It costs ~8k parameters
    at d=128 and keeps whatever the mean throws away: for the audio stream, the shape of the
    prosody rather than its average.
    """

    def __init__(self, dim: int, drop: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)   # (B, T)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)               # (B, D)


class VideoDelta(nn.Module):
    """Video as pose plus motion: the mean frame embedding and the mean absolute frame change.

    The mean of the raw differences `x_t - x_{t-1}` is only the net drift between the first and
    the last frame, which is another pose feature. The absolute difference per dimension measures
    how much the face actually moves, which is what the mean pooling was throwing away.
    """

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.pose = nn.Linear(VIDEO_DIM, d)
        self.motion = nn.Linear(VIDEO_DIM, d)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Dropout(drop),
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        change = (video[:, 1:] - video[:, :-1]).abs().mean(dim=1)   # (B, 384)
        pose = self.pose(video).mean(dim=1)
        motion = self.motion(change)
        return self.head(torch.cat([pose, motion], dim=-1)).squeeze(-1)


class LateDelta(nn.Module):
    """Pose + motion + audio, concatenated."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.pose = nn.Linear(VIDEO_DIM, d)
        self.motion = nn.Linear(VIDEO_DIM, d)
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Dropout(drop),
            nn.Linear(3 * d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        change = (video[:, 1:] - video[:, :-1]).abs().mean(dim=1)
        pose = self.pose(video).mean(dim=1)
        motion = self.motion(change)
        sound = self.audio(audio).mean(dim=1)
        return self.head(torch.cat([pose, motion, sound], dim=-1)).squeeze(-1)


class VideoAttention(nn.Module):
    """Video only, with attention pooling over the 16 frames."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.video = nn.Linear(VIDEO_DIM, d)
        self.pool = AttentionPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.video(video))).squeeze(-1)


class AudioAttention(nn.Module):
    """Audio only, with attention pooling over the 64 tokens."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.pool = AttentionPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.audio(audio))).squeeze(-1)


class LateAttention(nn.Module):
    """Late fusion with attention pooling on both streams."""

    def __init__(self, d: int = 128, hidden: int = 64, drop: float = 0.3):
        super().__init__()
        self.video = nn.Linear(VIDEO_DIM, d)
        self.audio = nn.Linear(AUDIO_DIM, d)
        self.pool = AttentionPool(d)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Dropout(drop),
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = self.pool(self.video(video))
        a = self.pool(self.audio(audio))
        return self.head(torch.cat([v, a], dim=-1)).squeeze(-1)


VARIANTS = {
    "late": LateFusion,
    "video": VideoOnly,
    "audio": AudioOnly,
    "congruence": Congruence,
    "video_delta": VideoDelta,
    "late_delta": LateDelta,
    "video_attention": VideoAttention,
    "audio_attention": AudioAttention,
    "late_attention": LateAttention,
}


if __name__ == "__main__":
    video = torch.randn(4, 16, VIDEO_DIM)
    audio = torch.randn(4, 64, AUDIO_DIM)
    for name, build in VARIANTS.items():
        net = build()
        logits = net(video, audio)
        assert logits.shape == (4,), (name, logits.shape)
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"{name:11s} {trainable:>8,} trainable parameters, logits {tuple(logits.shape)}")
