"""CPU smoke test: the ported DFlash trainer must overfit a tiny synthetic batch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule


def _tiny_module(seed: int = 0) -> DFlashTrainerModule:
    torch.manual_seed(seed)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

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
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": False,
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
    )


def test_trainer_overfits_tiny_batch():
    torch.manual_seed(0)
    batch_size, seq_len = 2, 48
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    module = _tiny_module()
    # The trainer consumes the captured target features: one hidden state per
    # routed target layer, concatenated on the last axis (see extract_context_feature).
    with torch.no_grad():
        base = module.embed_tokens(input_ids).detach()
        layer_a = base + 0.01 * torch.randn(batch_size, seq_len, 64)
        layer_b = torch.tanh(base) + 0.01 * torch.randn(batch_size, seq_len, 64)
        hidden = torch.cat([layer_a, layer_b], dim=-1)
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=2e-3)
    losses = []
    for _ in range(60):
        metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
        optimizer.zero_grad()
        metrics.loss.backward()
        optimizer.step()
        losses.append(float(metrics.loss.detach()))
        assert torch.isfinite(metrics.loss)
    assert losses[-1] < 0.75 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert metrics.pos_correct is not None and metrics.pos_count is not None
