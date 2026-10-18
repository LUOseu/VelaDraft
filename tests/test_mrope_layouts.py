"""MRoPE table construction: chunked (Qwen2.5-VL) vs interleaved (Qwen3-VL)."""

from __future__ import annotations

import pytest
import torch

from vlm_dflash.dflash.draft_model import _merge_multiaxis_rotary_embeddings


class _FakeRotary(torch.nn.Module):
    """Minimal rotary module returning cos/sin tables for flattened rows."""

    def __init__(self, half_dim: int, base: float = 5000000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, 2 * half_dim, 2).float() / (2 * half_dim))))

    def forward(self, hidden_states, position_ids):
        freqs = torch.outer(position_ids.float().reshape(-1), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)


def _reference_chunked(freqs_axis: torch.Tensor, sections: tuple[int, int, int]) -> torch.Tensor:
    """HF ``apply_multimodal_rotary_pos_emb`` chunk recombination on raw angles.

    HF does ``mrope_section = mrope_section * 2`` on a *list*, so the split
    pattern is the section list repeated twice over the doubled table.
    """
    repeated = list(sections) * 2
    chunks = freqs_axis.split(repeated, dim=-1)
    return torch.cat([chunk[index % 3] for index, chunk in enumerate(chunks)], dim=-1)


def _reference_interleaved(freqs_axis: torch.Tensor, sections: tuple[int, int, int]) -> torch.Tensor:
    """HF ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`` on raw angles."""
    freqs = freqs_axis.clone()
    freqs_t = freqs[0]
    for dim, offset in ((1, 1), (2, 2)):
        length = sections[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return torch.cat((freqs_t, freqs_t), dim=-1)


@pytest.mark.parametrize(
    ("sections", "head_dim"),
    [((16, 24, 24), 128), ((24, 20, 20), 128)],
)
def test_chunked_layout_matches_reference(sections, head_dim):
    rotary = _FakeRotary(head_dim // 2)
    hidden = torch.zeros(1, 1, 4, dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
        ]
    )
    cos, sin = _merge_multiaxis_rotary_embeddings(rotary, hidden, position_ids, sections, interleaved=False)
    freqs = torch.outer(
        position_ids.reshape(3 * 2, 4).float().reshape(-1), rotary.inv_freq
    ).reshape(3, 2, 4, -1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    reference = _reference_chunked(freqs, sections)
    assert torch.allclose(cos, reference.cos(), atol=1e-6)
    assert torch.allclose(sin, reference.sin(), atol=1e-6)


@pytest.mark.parametrize(
    ("sections", "head_dim"),
    # (11, 11, 10) over 64: Qwen3.5/3.6 partial rotary (64 of 256 channels).
    [((24, 20, 20), 128), ((11, 11, 10), 64)],
)
def test_interleaved_layout_matches_reference(sections, head_dim):
    rotary = _FakeRotary(head_dim // 2)
    hidden = torch.zeros(1, 1, 4, dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 1, 0, 1], [0, 1, 0, 1]],
        ]
    )
    cos, sin = _merge_multiaxis_rotary_embeddings(rotary, hidden, position_ids, sections, interleaved=True)
    freqs = torch.outer(
        position_ids.reshape(3 * 2, 4).float().reshape(-1), rotary.inv_freq
    ).reshape(3, 2, 4, -1)
    reference = _reference_interleaved(freqs, sections)
    assert torch.allclose(cos, reference.cos(), atol=1e-6)
    assert torch.allclose(sin, reference.sin(), atol=1e-6)


def test_interleaved_rejects_oversized_sections():
    rotary = _FakeRotary(64)
    hidden = torch.zeros(1, 1, 1, dtype=torch.float32)
    position_ids = torch.zeros(3, 1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="Interleaved MRoPE"):
        _merge_multiaxis_rotary_embeddings(rotary, hidden, position_ids, (16, 30, 18), interleaved=True)


def test_interleaved_matches_hf_qwen3_vl_rotary_when_available():
    transformers = pytest.importorskip("transformers")
    rotary_class = getattr(transformers.models.qwen3_vl.modeling_qwen3_vl, "Qwen3VLTextRotaryEmbedding", None)
    if rotary_class is None:
        pytest.skip(f"transformers {transformers.__version__} has no Qwen3VLTextRotaryEmbedding")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(
        head_dim=128,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20], "mrope_interleaved": True},
    )
    hf_rotary = rotary_class(config)
    hidden = torch.zeros(1, 4, 4096, dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3]],
            [[0, 0, 1, 1]],
            [[0, 1, 0, 1]],
        ]
    )
    hf_cos, hf_sin = hf_rotary(hidden, position_ids)
    cos, sin = _merge_multiaxis_rotary_embeddings(hf_rotary, hidden, position_ids, (24, 20, 20), interleaved=True)
    assert torch.allclose(cos, hf_cos, atol=1e-6)
    assert torch.allclose(sin, hf_sin, atol=1e-6)

def test_layouts_differ_on_image_positions():
    rotary = _FakeRotary(64)
    hidden = torch.zeros(2, 4, 8, dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 1, 2], [3, 3, 2, 1]],
            [[0, 1, 3, 0], [1, 0, 1, 3]],
        ]
    )
    chunked_cos, _ = _merge_multiaxis_rotary_embeddings(rotary, hidden, position_ids, (24, 20, 20), interleaved=False)
    interleaved_cos, _ = _merge_multiaxis_rotary_embeddings(rotary, hidden, position_ids, (24, 20, 20), interleaved=True)
    assert not torch.allclose(chunked_cos, interleaved_cos, atol=1e-4)


def test_qwen25_rotated_q_matches_hf_apply_multimodal_rotary_pos_emb():
    transformers = pytest.importorskip("transformers")
    qwen25 = getattr(transformers.models, "qwen2_5_vl", None)
    if qwen25 is None:
        pytest.skip("transformers has no qwen2_5_vl")
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
    )
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    config = Qwen2_5_VLTextConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24], "rope_theta": 1000000.0},
    )
    hf_rotary = Qwen2_5_VLRotaryEmbedding(config)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 1, 2], [3, 3, 2, 1]],
            [[0, 1, 3, 0], [1, 0, 1, 3]],
        ]
    )
    hidden = torch.zeros(2, 4, 8, dtype=torch.float32)
    axis_tables = hf_rotary(hidden, position_ids)
    q = torch.randn(2, 4, 4, 128)
    k = torch.randn(2, 4, 4, 128)
    hf_q, hf_k = apply_multimodal_rotary_pos_emb(q, k, axis_tables[0], axis_tables[1], [16, 24, 24])

    class _RowRotary(torch.nn.Module):
        """Adapt HF's 3-axis rotary (requires ``[3, bs, s]``) to the draft model's flat-row interface."""

        def __init__(self, rotary):
            super().__init__()
            self.rotary = rotary

        def forward(self, hidden_states, flat_position_ids):
            batch = flat_position_ids.shape[0] // 3
            pids = flat_position_ids.reshape(3, batch, flat_position_ids.shape[-1])
            return self.rotary(hidden_states, pids)

    cos, sin = _merge_multiaxis_rotary_embeddings(
        _RowRotary(hf_rotary), hidden, position_ids, (16, 24, 24), interleaved=False
    )
    from vlm_dflash.dflash.draft_model import apply_rotary_pos_emb, rotate_half

    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    ours_q = (q * cos_u) + (rotate_half(q) * sin_u)
    ours_k = (k * cos_u) + (rotate_half(k) * sin_u)
    assert torch.allclose(ours_q, hf_q, atol=1e-5)
    assert torch.allclose(ours_k, hf_k, atol=1e-5)


def test_qwen3_rotated_q_matches_hf_rotary_and_apply():
    transformers = pytest.importorskip("transformers")
    qwen3 = getattr(transformers.models, "qwen3_vl", None)
    if qwen3 is None:
        pytest.skip("transformers has no qwen3_vl")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000.0,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    hf_rotary = Qwen3VLTextRotaryEmbedding(config)
    position_ids = torch.tensor(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 1, 2], [3, 3, 2, 1]],
            [[0, 1, 3, 0], [1, 0, 1, 3]],
        ]
    )
    hidden = torch.zeros(2, 4, 8, dtype=torch.float32)
    hf_cos, hf_sin = hf_rotary(hidden, position_ids)
    q = torch.randn(2, 4, 4, 128)
    k = torch.randn(2, 4, 4, 128)
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    cos, sin = _merge_multiaxis_rotary_embeddings(hf_rotary, hidden, position_ids, (24, 20, 20), interleaved=True)
    from vlm_dflash.dflash.draft_model import rotate_half

    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    ours_q = (q * cos_u) + (rotate_half(q) * sin_u)
    ours_k = (k * cos_u) + (rotate_half(k) * sin_u)
    assert torch.allclose(ours_q, hf_q, atol=1e-5)
    assert torch.allclose(ours_k, hf_k, atol=1e-5)
