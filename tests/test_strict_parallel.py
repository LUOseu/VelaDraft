"""Strict-parallel verification: near-tie guard, fallback wiring, parity.

Parallel block verification is the fast target-side path, but bf16 kernel
differences can flip near-tie argmax decisions away from ``generate()``. With
``strict_parallel_margin`` set, any round whose emitted token had a top-1/top-2
gap below the margin is re-verified with the sequential generate()-parity path.
These tests pin the guard helper and exercise the fallback (including the KV
cache rollback) on tiny CPU models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.draft_model import _min_decision_margin


def _logits_rows(*gaps: float, base: float = 10.0) -> torch.Tensor:
    return torch.tensor([[[base, base - gap, base - 5.0] for gap in gaps]], dtype=torch.float32)


def test_min_decision_margin_reads_the_worst_used_position():
    logits = _logits_rows(4.0, 0.25, 9.0)
    assert _min_decision_margin(logits, 3) == pytest.approx(0.25)
    assert _min_decision_margin(logits, 1) == pytest.approx(4.0)
    assert _min_decision_margin(logits, 0) == float("inf")


def _tiny_draft():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    torch.manual_seed(3)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    return Qwen3DFlashDraftModel(config)


def _tiny_target():
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _spec_kwargs(input_ids):
    return {
        "target": _tiny_target(),
        "input_ids": input_ids,
        "max_new_tokens": 16,
        "stop_token_ids": None,
        "temperature": 0.0,
    }


def test_strict_margin_argument_validation():
    draft = _tiny_draft()
    input_ids = torch.randint(0, 256, (1, 8))
    target = _tiny_target()
    with pytest.raises(ValueError, match="requires parallel target verification"):
        draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=4,
            stop_token_ids=None,
            temperature=0.0,
            sequential_target_verification=True,
            strict_parallel_margin=1.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=4,
            stop_token_ids=None,
            temperature=0.0,
            strict_parallel_margin=0.0,
        )
    with pytest.raises(ValueError, match="greedy"):
        draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=4,
            stop_token_ids=None,
            temperature=1.0,
            strict_parallel_margin=1.0,
        )


@torch.no_grad()
def test_strict_parallel_falls_back_to_sequential_parity():
    target = _tiny_target()
    draft = _tiny_draft()
    draft.eval()
    torch.manual_seed(5)
    input_ids = torch.randint(0, 256, (1, 12))
    kwargs = {
        "target": target,
        "input_ids": input_ids,
        "max_new_tokens": 16,
        "stop_token_ids": None,
        "temperature": 0.0,
    }

    ids_seq, stats_seq = draft.spec_generate(
        **kwargs, return_stats=True, sequential_target_verification=True
    )
    # A margin no logit gap can satisfy forces the fallback for every round.
    ids_strict, stats_strict = draft.spec_generate(
        **kwargs,
        return_stats=True,
        sequential_target_verification=False,
        strict_parallel_margin=1e9,
    )
    assert ids_strict.tolist() == ids_seq.tolist()
    assert stats_strict["strict_parallel_rounds"] > 0
    assert stats_strict["strict_fallback_rounds"] == stats_strict["strict_parallel_rounds"]
    assert stats_strict["verify_steps"] == stats_seq["verify_steps"]
    assert stats_strict["strict_min_decision_margin"] is not None

    ids_relaxed, stats_relaxed = draft.spec_generate(
        **kwargs, return_stats=True, sequential_target_verification=False
    )
    assert stats_relaxed["strict_parallel_rounds"] == 0.0
    assert stats_relaxed["strict_fallback_rounds"] == 0.0
    assert stats_relaxed["strict_min_decision_margin"] is None
