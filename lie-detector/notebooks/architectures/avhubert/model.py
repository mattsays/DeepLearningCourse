"""`avh` — AV-HuBERT mouth-region encoder, frozen, plus one trainable probe.

AV-HuBERT (Shi et al., 2022) is an audio-visual **speech** encoder: masked multimodal cluster
prediction over 25 fps mouth crops and the aligned audio. Here it is used video-only — the
audio stream comes from WavLM, which `tia` already cached — so its features describe *how a
mouth moves while speaking*, which is a prior no other encoder in this project has.

Cache contract (written by `extract_features.ipynb`, read here):

    <DATA>/cached_features/avhubert-video/<clip_id>.npy   float16  (64, 768)
    <DATA>/cached_features/wavlm-base-plus/<clip_id>.npy  float16  (64, 768)

Both streams are 768 wide, so the probe projects them into the same space and the
`congruence` variant compares two representations of the same speech event.

`VARIANTS` is the registry the notebook selects from: one class plus one line.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #


class _Tokens(nn.Module):
    """Project both streams into one width, concatenate, attend with a learned query.

    Mean-pooling throws away *when* something happened; a query token can learn to look at
    the frames that matter (a hesitation, a blink) or at the moments where the mouth and the
    voice disagree. The two streams are both 768 wide here, but each keeps its own projection:
    a shared one would silently be a statement that mouth features and audio features live in
    the same space, which is exactly what `congruence` exists to question.
    """

    def __init__(self, video_dim: int, audio_dim: int, d_model: int = 192,
                 num_heads: int = 8, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.video = nn.Linear(video_dim, d_model)
        self.audio = nn.Linear(audio_dim, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.ModuleList([layer for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, video=None, audio=None) -> torch.Tensor:
        streams = []
        if video is not None:
            streams.append(self.video(video))
        if audio is not None:
            streams.append(self.audio(audio))
        memory = torch.cat(streams, dim=1)
        query = self.query.expand(memory.size(0), -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        return self.head(self.norm(query)).squeeze(-1).squeeze(-1)


class MLP(nn.Module):
    """Two-layer head used by the pooling baselines."""

    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Dropout(dropout),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #


class UnifiedProbe(nn.Module):
    """One probe over audio and video tokens together — the native read-out of a joint
    encoder, and for V-JEPA 2 the variant that lets the two streams talk to each other.
    """

    def __init__(self, video_dim: int, audio_dim: int, **kwargs):
        super().__init__()
        self.probe = _Tokens(video_dim, audio_dim, **kwargs)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.probe(video=video, audio=audio)


class VideoOnly(nn.Module):
    """Visual stream alone — the floor the cross-modal pre-training has to beat."""

    def __init__(self, video_dim: int, audio_dim: int, **kwargs):
        super().__init__()
        self.probe = _Tokens(video_dim, audio_dim, **kwargs)

    def forward(self, video: torch.Tensor, audio: torch.Tensor | None = None) -> torch.Tensor:
        return self.probe(video=video)


class AudioOnly(nn.Module):
    """Acoustic stream alone. On DOLOS this is the stream that matters (see `tia`)."""

    def __init__(self, video_dim: int, audio_dim: int, **kwargs):
        super().__init__()
        self.probe = _Tokens(video_dim, audio_dim, **kwargs)

    def forward(self, video: torch.Tensor | None, audio: torch.Tensor) -> torch.Tensor:
        return self.probe(audio=audio)


class LateFusion(nn.Module):
    """Average each stream over time, concatenate, MLP.

    No temporal modelling and no cross-attention: the control that says whether the probe
    is doing anything beyond a linear read-out of the mean feature.
    """

    def __init__(self, video_dim: int, audio_dim: int, d: int = 128, **kwargs):
        super().__init__()
        self.video = nn.Linear(video_dim, d)
        self.audio = nn.Linear(audio_dim, d)
        self.head = MLP(2 * d)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([self.video(video).mean(1), self.audio(audio).mean(1)], dim=-1)
        return self.head(pooled)


class Congruence(nn.Module):
    """Decide on the audio/video agreement, not on either stream.

    Deception is often an *incongruence* — a steady voice on a tense face — so this variant
    sees only cosine similarity between the two pooled embeddings and their norms. If a
    joint pre-training helped anywhere, it should show here first.
    """

    def __init__(self, video_dim: int, audio_dim: int, **kwargs):
        super().__init__()
        self.video = nn.Linear(video_dim, 128)
        self.audio = nn.Linear(audio_dim, 128)
        self.head = MLP(3, hidden=32)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v = self.video(video).mean(1)
        a = self.audio(audio).mean(1)
        cosine = nn.functional.cosine_similarity(v, a, dim=-1, eps=1e-6)
        features = torch.stack([cosine, v.norm(dim=-1), a.norm(dim=-1)], dim=-1)
        return self.head(features)


VARIANTS: dict[str, type[nn.Module]] = {
    "unified": UnifiedProbe,
    "video": VideoOnly,
    "audio": AudioOnly,
    "late": LateFusion,
    "congruence": Congruence,
}


def build(variant: str, video_dim: int, audio_dim: int | None = None, **kwargs) -> nn.Module:
    """Instantiate a variant by name, with the widths the caches report."""
    if variant not in VARIANTS:
        raise KeyError(f"unknown variant {variant!r}; pick one of {sorted(VARIANTS)}")
    return VARIANTS[variant](video_dim, audio_dim if audio_dim is not None else video_dim,
                             **kwargs)


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# AV-HuBERT backend
# --------------------------------------------------------------------------- #

# Video-only checkpoints from the official model zoo. LRS3 + VoxCeleb2 is the one with the
# most pretraining data, which is what a dataset as small as DOLOS needs.
AVHUBERT_CHECKPOINTS = {
    "base": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt",
    "large": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt",
}

# 25 fps mouth crops, 96 x 96 grayscale, as produced by the official `align_mouth.py`.
AVHUBERT_FPS = 25
AVHUBERT_CROP = 96
# python_speech_features defaults, then four consecutive frames stacked: the 26 log-mel bins
# become a 104-dim vector at 25 fps, aligned with the video. This mirrors `hubert_dataset.py`.
FBANK_BINS, FBANK_STACK, FBANK_DIM = 26, 4, 104


def load_avhubert(checkpoint, device: str = "cpu", user_dir=None,
                  modalities: tuple[str, ...] = ("video",)):
    """Build the frozen AV-HuBERT encoder from an official checkpoint.

    The model is fairseq-based: the checkpoint carries its own task config, and `user_dir`
    must point at the `avhubert/` package inside the vendored repository (it registers the
    `av_hubert_pretraining` task and the model classes).

    `modalities` picks which branches the encoder keeps. The model was trained with modality
    dropout, so `("video",)`, `("audio",)` and `("audio", "video")` are all in-distribution —
    that is what makes the three caches comparable.

    Returns `(model, width)`.

    This needs a `fairseq` import that works on the running Python. The official code targets
    Python 3.8 and its bundled fairseq fork; on a newer Colab image the import may fail before
    any of this runs. `check_avhubert_environment()` reports that before a session is spent.
    """
    from pathlib import Path

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"no AV-HuBERT checkpoint at {path}. Download one from "
            f"{AVHUBERT_CHECKPOINTS['base']} (base, LRS3 + VoxCeleb2) and pass the path."
        )

    import fairseq

    overrides = {"data": str(path.parent)}      # fairseq insists on a task.data directory
    if user_dir is not None:
        overrides["user_dir"] = str(user_dir)
    models, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(path)], arg_overrides=overrides,
    )
    model = models[0].eval().to(device)
    model.modalities = list(modalities)
    for parameter in model.parameters():
        parameter.requires_grad = False
    width = model.encoder.embed_dim if hasattr(model, "encoder") else model.embed_dim
    return model, width


def check_avhubert_environment(user_dir=None) -> dict:
    """What the notebook prints before downloading 1.2 GB: python, fairseq, model classes."""
    import importlib
    import sys
    from pathlib import Path

    report = {"python": sys.version.split()[0], "fairseq": None, "user_dir": None}
    for name in ("fairseq", "python_speech_features", "dlib"):
        try:
            module = importlib.import_module(name)
            report[name] = getattr(module, "__version__", "ok")
        except Exception as error:                   # noqa: BLE001 - report, do not raise
            report[name] = f"missing ({type(error).__name__})"
    if user_dir is not None:
        report["user_dir"] = Path(user_dir).is_dir()
    return report


def audio_features(path, sample_rate: int = 16000) -> torch.Tensor:
    """16 kHz wav -> `(T, 104)` at 25 fps, exactly as the AV-HuBERT dataset does it.

    `logfbank` (26 bins, 10 ms hop) then four frames stacked, then a per-frame layer norm over
    the 104 bins — `hubert_dataset.py` uses `F.layer_norm(audio_feats, audio_feats.shape[1:])`,
    so this is the same normalisation, not a choice of ours.
    """
    import numpy as np
    import soundfile as sf
    from python_speech_features import logfbank

    waveform, rate = sf.read(str(path), dtype="int16")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, found {rate}")
    features = logfbank(waveform, samplerate=sample_rate).astype(np.float32)   # (T, 26)
    remainder = len(features) % FBANK_STACK
    if remainder:
        features = np.concatenate([features, np.zeros((FBANK_STACK - remainder, FBANK_BINS),
                                                      dtype=np.float32)])
    features = features.reshape(-1, FBANK_STACK * FBANK_BINS)                  # (T/4, 104)
    tensor = torch.from_numpy(features)
    return torch.nn.functional.layer_norm(tensor, (FBANK_DIM,))


@torch.no_grad()
def encode_avhubert(model, video: torch.Tensor | None = None,
                    audio: torch.Tensor | None = None):
    """One clip through the frozen encoder, in whichever modalities `model.modalities` holds.

    `video` is `(1, T, 1, 96, 96)` in `[0, 1]`, `audio` is `(1, T, 104)`; `T` must match,
    because the encoder expects one token per 25 fps frame for both. Returns `(1, T, D)`.
    """
    source = {}
    if video is not None:
        source["video"] = video
    if audio is not None:
        source["audio"] = audio
    features, _ = model.extract_features(source=source, padding_mask=None, mask=False)
    return features


def pool_avhubert(features: torch.Tensor, tokens: int = 64) -> torch.Tensor:
    """`(B, T, D)` at 25 fps -> `(B, tokens, D)` by adaptive average pooling over time.

    AV-HuBERT emits 25 vectors per second, so a 19-second clip is 475 tokens and a 2-second
    one is 50. A fixed count keeps the cache rectangular and the probe free of padding masks;
    the cost is a loss of temporal detail on the long clips.
    """
    return torch.nn.functional.adaptive_avg_pool1d(
        features.transpose(1, 2), tokens,
    ).transpose(1, 2)
