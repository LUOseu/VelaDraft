"""Regression tests for the target output head: tied vs untied targets.

Qwen2.5-VL-3B ties its lm_head to the input embedding, Qwen3-VL-8B does not.
The draft trainer must consume the same frozen head that ``spec_generate``
samples through (``target.lm_head``); tying an untied target's head to the
embedding trains a head that inference never uses and collapses acceptance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.factory import build_trainer_module
from vlm_dflash.eval.run import check_checkpoint_head
from vlm_dflash.target.capture import save_target_head

CONTRACT = {"target": {"model_fingerprint": "deadbeef"}}


class _FakeTarget(torch.nn.Module):
    """Minimal stand-in exposing the embed/head surface the capture path uses."""

    def __init__(self, tied: bool, vocab: int = 16, hidden: int = 6) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        else:
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.randn(vocab, hidden))

    def get_input_embeddings(self):
        return self.embed


def _draft_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=16,
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=3,
        max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=1,
        dflash_config={"mask_token_id": 15, "target_layer_ids": [0], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    return config


def _build(embed_weight, lm_head_weight=None):
    drafter = {"block_size": 8, "mask_token_id": 15, "attention_backend": "sdpa"}
    return build_trainer_module(
        _draft_config(), CONTRACT, {"drafter": drafter}, embed_weight, lm_head_weight=lm_head_weight
    )


def test_tied_target_head_is_the_embedding_matrix():
    target = _FakeTarget(tied=True)
    module = _build(target.embed.weight.detach())
    assert module.lm_head.weight is module.embed_tokens.weight


def test_untied_target_head_is_its_own_frozen_copy():
    target = _FakeTarget(tied=False)
    module = _build(target.embed.weight.detach(), target.lm_head.weight.detach())
    assert module.lm_head.weight is not module.embed_tokens.weight
    assert torch.equal(module.lm_head.weight, target.lm_head.weight)
    assert not module.lm_head.weight.requires_grad
    hidden = torch.randn(2, 3, 6)
    assert torch.allclose(module.lm_head(hidden), target.lm_head(hidden))


def test_untied_head_differentiates_from_tied_head():
    """The bug's signature: tied-head logits differ from the real head's argmax path."""
    target = _FakeTarget(tied=False)
    hidden = torch.randn(4, 6)
    tied_logits = _build(target.embed.weight.detach()).lm_head(hidden)
    real_logits = _build(target.embed.weight.detach(), target.lm_head.weight.detach()).lm_head(hidden)
    assert not torch.equal(tied_logits.argmax(-1), real_logits.argmax(-1))


def test_build_trainer_module_rejects_wrong_lm_head_shape():
    target = _FakeTarget(tied=False)
    with pytest.raises(ValueError, match="lm_head_weight"):
        _build(target.embed.weight.detach(), torch.randn(3, 5))


def test_save_target_head_round_trip(tmp_path):
    tied_target = _FakeTarget(tied=True)
    assert save_target_head(tied_target, tmp_path, torch.float32, CONTRACT) is False
    tied_payload = torch.load(tmp_path / "target_head.pt", weights_only=False)
    assert tied_payload["tie_word_embeddings"] is True and "lm_head" not in tied_payload
    tied_module = _build(tied_payload["embed_tokens"], tied_payload.get("lm_head"))
    assert tied_module.lm_head.weight is tied_module.embed_tokens.weight

    untied_target = _FakeTarget(tied=False)
    assert save_target_head(untied_target, tmp_path, torch.float32, CONTRACT) is True
    payload = torch.load(tmp_path / "target_head.pt", weights_only=False)
    assert payload["tie_word_embeddings"] is False
    assert torch.equal(payload["lm_head"], untied_target.lm_head.weight)
    module = _build(payload["embed_tokens"], payload["lm_head"])
    hidden = torch.randn(2, 6)
    assert torch.allclose(module.lm_head(hidden), untied_target.lm_head(hidden))


def test_head_gate_rejects_tied_checkpoint_on_untied_target():
    with pytest.raises(RuntimeError, match="untied"):
        check_checkpoint_head("ckpt_final.pt", {"head": "tied_embedding"}, target_tied_head=False)
    # Pre-fix checkpoints carry no head tag; they are treated as embedding-tied.
    with pytest.raises(RuntimeError, match="untied"):
        check_checkpoint_head("ckpt_final.pt", {}, target_tied_head=False)


def test_head_gate_allows_matching_targets():
    check_checkpoint_head("ckpt_final.pt", {"head": "target_lm_head"}, target_tied_head=False)
    check_checkpoint_head("ckpt_final.pt", {"head": "tied_embedding"}, target_tied_head=True)
    # A tied target makes both heads numerically identical, so either tag is fine.
    check_checkpoint_head("ckpt_final.pt", {"head": "target_lm_head"}, target_tied_head=True)


def test_target_head_is_tied_detects_shared_pointer():
    from vlm_dflash.eval.run import _target_head_is_tied

    assert _target_head_is_tied(_FakeTarget(tied=True)) is True
    assert _target_head_is_tied(_FakeTarget(tied=False)) is False
