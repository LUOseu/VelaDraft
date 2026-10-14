"""CPU tests: the draft pins the main flex-attention kernel for block masks.

Torch >= 2.10 can route the draft block attention (q_len < 128, power-of-two
GQA ratio) to the flex *decoding* kernel, whose ``BLOCK_M =
next_pow2(q_len * gqa_ratio)`` heuristic (80 * 8 -> 1024 for a ten-anchor
smoke block) exceeds the sm_86 shared-memory budget at Triton compile time
(272 KiB needed vs 99 KiB available). On torch 2.11 the same input lowers to
``NoValidChoicesError``. The draft attention therefore passes
``FORCE_USE_FLEX_ATTENTION`` so the main flex kernel is used; these tests pin
that contract without needing a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash import draft_model as draft_model_mod
from vlm_dflash.dflash.draft_model import Qwen3DFlashAttention


def _tiny_config(attn_implementation: str):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=1,
        dflash_config={
            "mask_token_id": 63,
            "target_layer_ids": [0],
            "spatial_rope_enabled": False,
        },
    )
    config._attn_implementation = attn_implementation
    return config


def _run_attention(monkeypatch, attn_implementation: str, **forward_kwargs):
    """Run one attention forward with a recording backend; return its kwargs."""
    calls: list[dict] = []

    def recorder(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs)
        batch, heads, q_len, _ = query.shape
        out = torch.zeros(batch, heads, q_len, value.shape[-1], dtype=query.dtype)
        return out, None

    monkeypatch.setattr(
        draft_model_mod,
        "ALL_ATTENTION_FUNCTIONS",
        {attn_implementation: recorder},
    )
    attn = Qwen3DFlashAttention(_tiny_config(attn_implementation), layer_idx=0)
    batch, ctx_len, q_len, head_dim = 2, 7, 5, 8
    hidden = torch.randn(batch, q_len, 32)
    target = torch.randn(batch, ctx_len, 32)
    # Baseline RoPE tables span [context | draft]; queries use the tail slice.
    cos = torch.randn(batch, ctx_len + q_len, head_dim)
    sin = torch.randn(batch, ctx_len + q_len, head_dim)
    out, weights = attn(
        hidden_states=hidden,
        target_hidden=target,
        position_embeddings=(cos, sin),
        attention_mask=None,
        **forward_kwargs,
    )
    assert weights is None
    assert out.shape == (batch, q_len, 32)
    assert len(calls) == 1
    return calls[0]


def test_flex_attention_forces_main_kernel(monkeypatch):
    kwargs = _run_attention(monkeypatch, "flex_attention")
    assert kwargs.get("kernel_options") == {"FORCE_USE_FLEX_ATTENTION": True}


def test_flex_attention_respects_caller_kernel_options(monkeypatch):
    kwargs = _run_attention(
        monkeypatch, "flex_attention", kernel_options={"BACKEND": "TRITON"}
    )
    assert kwargs.get("kernel_options") == {"BACKEND": "TRITON"}


def test_non_flex_backends_receive_no_kernel_options(monkeypatch):
    kwargs = _run_attention(monkeypatch, "sdpa")
    assert "kernel_options" not in kwargs
