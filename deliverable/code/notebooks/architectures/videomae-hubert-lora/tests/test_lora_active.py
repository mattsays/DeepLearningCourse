"""LoRA must really be part of the forward pass — tiny random models on CPU, nothing downloaded.

    python tests/test_lora_active.py          (or: python -m pytest tests)

The negative control rebuilds the attention of transformers <= 5.5 (``F.linear`` on
``self.query.weight``): the guard must reject it, otherwise it would not protect a real run.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lore.models import AudioBranch, LoraSettings, VideoBranch, assert_lora_active, count_parameters  # noqa: E402

VIDEO_INPUT = (torch.randn(1, 4, 3, 32, 32),)                       # (B, T, C, H, W)
AUDIO_INPUT = (torch.randn(1, 4000), torch.tensor([4000]))          # (B, samples), lengths


def tiny_video(lora: LoraSettings | None = LoraSettings()) -> VideoBranch:
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    config = VideoMAEConfig(image_size=32, patch_size=16, num_frames=4, tubelet_size=2, hidden_size=32,
                            num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            use_mean_pooling=True, num_labels=2)
    base = VideoMAEForVideoClassification(config)
    return VideoBranch(base.videomae, base.fc_norm, 32, lora)


def tiny_audio(lora: LoraSettings | None = LoraSettings(targets=("q_proj", "v_proj"))) -> AudioBranch:
    from transformers import HubertConfig, HubertModel

    config = HubertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                          conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3),
                          num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2)
    return AudioBranch(HubertModel(config), 32, lora)


def lora_modules(branch) -> list[str]:
    return sorted({n.split(".lora_")[0] for n, _ in branch.named_parameters() if "lora_" in n})


def test_video_lora_active():
    torch.manual_seed(0)
    report = assert_lora_active(tiny_video(), *VIDEO_INPUT)
    assert report["lora_matrices"] == 4, report                     # 2 layers x {query, value}


def test_audio_lora_active():
    torch.manual_seed(0)
    report = assert_lora_active(tiny_audio(), *AUDIO_INPUT)
    assert report["lora_matrices"] == 4, report


def test_video_mlp_targets_skip_attention_output():
    branch = tiny_video(LoraSettings(mlp_last_n=1))
    names = lora_modules(branch)
    assert any("layer.1.intermediate.dense" in n for n in names), names
    assert any("layer.1.output.dense" in n and "attention" not in n for n in names), names
    assert not any("attention.output.dense" in n for n in names), names
    assert not any("layer.0.intermediate" in n for n in names), names
    assert_lora_active(branch, *VIDEO_INPUT)


def test_audio_mlp_targets():
    branch = tiny_audio(LoraSettings(targets=("q_proj", "v_proj"), mlp_last_n=1))
    names = lora_modules(branch)
    assert any("layers.1.feed_forward.intermediate_dense" in n for n in names), names
    assert not any("layers.0.feed_forward" in n for n in names), names
    assert_lora_active(branch, *AUDIO_INPUT)


def test_dora_active():
    assert_lora_active(tiny_video(LoraSettings(use_dora=True)), *VIDEO_INPUT)


def test_head_only_trains_only_the_head():
    branch = tiny_video(None)
    params = count_parameters(branch)
    assert params["adapters"] == 0
    assert params["trainable"] == 32 * 2 + 32 + 1, params             # LayerNorm (w, b) + Linear(32 -> 1)
    assert "skipped" in assert_lora_active(branch, *VIDEO_INPUT)


def test_trainable_state_roundtrip():
    branch = tiny_video()
    state = branch.trainable_state()
    with torch.no_grad():
        for name, parameter in branch.named_parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)
    branch.load_trainable_state(state)
    for name, parameter in branch.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.detach().cpu(), state[name]), name


def _old_videomae_attention(self, hidden_states, *args, **kwargs):
    """VideoMAE self-attention as in transformers <= 5.5: weights read directly, modules bypassed."""
    batch, tokens, _ = hidden_states.shape
    q = F.linear(hidden_states, self.query.weight, self.query.bias)
    k = F.linear(hidden_states, self.key.weight, self.key.bias)
    v = F.linear(hidden_states, self.value.weight, self.value.bias)
    shape = (batch, tokens, self.num_attention_heads, self.attention_head_size)
    q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
    context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, tokens, self.all_head_size)
    return context, None


def test_guard_rejects_old_style_attention():
    branch = tiny_video()
    for name, module in branch.named_modules():
        if type(module).__name__ == "VideoMAESelfAttention":
            module.forward = types.MethodType(_old_videomae_attention, module)
    try:
        assert_lora_active(branch, *VIDEO_INPUT)
    except RuntimeError as error:
        assert "no gradient" in str(error)
        return
    raise AssertionError("the guard accepted a model whose LoRA matrices are bypassed")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} tests passed")
