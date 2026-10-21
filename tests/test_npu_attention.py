"""Mask and chunked-attention parity tests (C03/C04).

These run on CPU with ordinary PyTorch. The fake fused-kernel case checks only
argument conversion/fallback control flow; it is not an Ascend operator test.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vlm_dflash.dflash.chunked_attention import (
    _repeat_kv,
    npu_fusion_chunked_attention,
    sdpa_chunked_attention,
)
from vlm_dflash.dflash.mask import (
    DFlashMaskSpec,
    create_dflash_sdpa_mask,
    estimate_dflash_attention_memory,
)


def _spec(*, causal: bool = False, docs: bool = True, chunk: int = 3, backend: str = "sdpa_chunked") -> DFlashMaskSpec:
    anchors = torch.tensor([[0, 5, 8], [2, 4, 10]], dtype=torch.long)
    keep = torch.tensor([[True, True, False], [True, False, True]])
    ctx_len = 11
    if docs:
        ctx_doc_id = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            ],
            dtype=torch.long,
        )
        anchor_doc_id = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    else:
        ctx_doc_id = anchor_doc_id = None
    return DFlashMaskSpec(
        anchor_positions=anchors,
        block_keep_mask=keep,
        ctx_len=ctx_len,
        block_size=4,
        causal=causal,
        ctx_doc_id=ctx_doc_id,
        anchor_doc_id=anchor_doc_id,
        query_chunk_size=chunk,
        backend=backend,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("docs", [False, True])
def test_mask_spec_matches_dense_reference_exactly(causal: bool, docs: bool):
    spec = _spec(causal=causal, docs=docs)
    dense = create_dflash_sdpa_mask(
        spec.anchor_positions,
        spec.block_keep_mask,
        spec.ctx_len,
        spec.block_size,
        torch.device("cpu"),
        torch.float32,
        causal=causal,
        ctx_doc_id=spec.ctx_doc_id,
        anchor_doc_id=spec.anchor_doc_id,
    )
    assert torch.equal(spec.full_additive(dtype=torch.float32), dense)


def test_chunked_additive_and_boolean_masks_concatenate_to_the_dense_mask():
    spec = _spec(chunk=3)
    dense = spec.full_additive()
    chunks = [spec.materialize_additive(start, min(spec.query_len, start + 3)) for start in range(0, spec.query_len, 3)]
    assert torch.equal(torch.cat(chunks, dim=2), dense)
    visible = torch.cat(
        [spec.materialize_bool(start, min(spec.query_len, start + 3)) for start in range(0, spec.query_len, 3)],
        dim=2,
    )
    inverted = torch.cat(
        [
            spec.materialize_bool(start, min(spec.query_len, start + 3), inverted=True)
            for start in range(0, spec.query_len, 3)
        ],
        dim=2,
    )
    assert torch.equal(visible, dense == 0)
    assert torch.equal(inverted, ~visible)


def test_padding_blocks_and_zero_length_context_never_produce_a_fully_masked_row():
    anchors = torch.tensor([[0, 4]], dtype=torch.long)
    keep = torch.tensor([[False, True]])
    spec = DFlashMaskSpec(anchors, keep, ctx_len=0, block_size=4, query_chunk_size=2)
    visible = spec.materialize_bool(0, spec.query_len)
    assert visible.any(dim=-1).all()


def _attention_inputs(*, heads: int = 4, kv_heads: int = 2, batch: int = 2, seed: int = 4):
    torch.manual_seed(seed)
    spec = _spec(causal=False, docs=True, chunk=5)
    q = torch.randn(batch, heads, spec.query_len, 8)
    k = torch.randn(batch, kv_heads, spec.kv_len, 8)
    v = torch.randn(batch, kv_heads, spec.kv_len, 8)
    return spec, q, k, v


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (4, 4), (4, 1)])
def test_chunked_attention_forward_matches_dense_sdpa(heads: int, kv_heads: int):
    spec, q, k, v = _attention_inputs(heads=heads, kv_heads=kv_heads)
    scale = q.shape[-1] ** -0.5
    # PyTorch's CPU GQA path in this wheelset does not broadcast a [B,1,Q,KV]
    # mask against the unexpanded KV heads, so expand the reference the same
    # way the chunked path does before comparing.
    dense_k = _repeat_kv(k, heads // kv_heads)
    dense_v = _repeat_kv(v, heads // kv_heads)
    dense = F.scaled_dot_product_attention(
        q, dense_k, dense_v, attn_mask=spec.full_additive(dtype=q.dtype), dropout_p=0.0, scale=scale
    )
    chunked, info = sdpa_chunked_attention(q, k, v, spec, dropout_p=0.0, scale=scale)
    torch.testing.assert_close(chunked, dense, rtol=1e-4, atol=1e-5)
    assert info["effective_backend"] == "sdpa_chunked"
    assert info["chunks"] == (spec.query_len + 4) // 5


def test_chunked_attention_backward_matches_dense_for_qkv_and_projection():
    spec, q0, k0, v0 = _attention_inputs(seed=9)
    scale = q0.shape[-1] ** -0.5
    upstream = torch.randn_like(q0)

    dense_inputs = [t.clone().requires_grad_(True) for t in (q0, k0, v0)]
    dense_k = _repeat_kv(dense_inputs[1], q0.shape[1] // k0.shape[1])
    dense_v = _repeat_kv(dense_inputs[2], q0.shape[1] // v0.shape[1])
    dense = F.scaled_dot_product_attention(
        dense_inputs[0], dense_k, dense_v,
        attn_mask=spec.full_additive(dtype=q0.dtype), dropout_p=0.0, scale=scale
    )
    (dense * upstream).sum().backward()

    chunk_inputs = [t.clone().requires_grad_(True) for t in (q0, k0, v0)]
    chunked, _ = sdpa_chunked_attention(*chunk_inputs, spec, dropout_p=0.0, scale=scale)
    (chunked * upstream).sum().backward()

    torch.testing.assert_close(chunked, dense, rtol=1e-4, atol=1e-5)
    for dense_tensor, chunk_tensor in zip(dense_inputs, chunk_inputs):
        torch.testing.assert_close(chunk_tensor.grad, dense_tensor.grad, rtol=2e-4, atol=2e-5)


def _tiny_module(backend: str):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

    torch.manual_seed(17)
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
        attention_backend=backend,
        num_anchors=8,
        loss_decay_gamma=7.0,
        attention_chunk_size=7,
    )


def test_whole_trainer_module_forward_backward_parity():
    dense_module = _tiny_module("sdpa")
    chunked_module = _tiny_module("sdpa_chunked")
    chunked_module.load_state_dict(dense_module.state_dict())

    torch.manual_seed(0)
    batch_size, seq_len = 2, 48
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    with torch.no_grad():
        embed = torch.nn.Embedding(256, 64)
        embed.weight.copy_(dense_module.embed_tokens.weight)
        base = embed(input_ids)
        hidden = torch.cat([base + 0.02 * torch.randn_like(base), torch.tanh(base)], dim=-1)

    torch.manual_seed(123)
    dense_metrics = dense_module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    dense_metrics.loss.backward()
    dense_loss = float(dense_metrics.loss.detach())
    dense_grads = {name: parameter.grad.detach().clone() for name, parameter in dense_module.named_parameters() if parameter.grad is not None}

    torch.manual_seed(123)
    chunked_metrics = chunked_module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    chunked_metrics.loss.backward()
    torch.testing.assert_close(chunked_metrics.loss, dense_metrics.loss, rtol=1e-4, atol=1e-5)
    for name, parameter in chunked_module.named_parameters():
        if parameter.grad is None:
            continue
        assert name in dense_grads, name
        torch.testing.assert_close(parameter.grad, dense_grads[name], rtol=2e-4, atol=2e-5, msg=name)
    assert float(chunked_metrics.loss.detach()) == pytest.approx(dense_loss, rel=1e-4, abs=1e-5)
    report = chunked_module.attention_runtime_report()
    assert report["requested_backend"] == "sdpa_chunked"
    assert report["effective_backends"] == ["sdpa_chunked"]
    assert report["chunked_layers"] > 0
    assert report["memory_plan"]["chunked_total_gb"] < report["memory_plan"]["dense_total_gb"]


def test_attention_memory_estimate_and_budget_rejection():
    plan = estimate_dflash_attention_memory(
        batch_size=2,
        num_blocks=256,
        block_size=16,
        context_len=2048,
        num_heads=28,
        head_dim=128,
        dtype=torch.bfloat16,
        query_chunk_size=128,
    )
    assert plan["dense_total_gb"] > plan["chunked_total_gb"]
    assert plan["chunks"] == 32

    module = _tiny_module("sdpa_chunked")
    module.attention_memory_budget_gb = 1e-12
    input_ids = torch.arange(48).remainder(200).add(10).unsqueeze(0).repeat(2, 1)
    loss_mask = torch.ones(2, 48)
    hidden = torch.randn(2, 48, 128)
    with pytest.raises(RuntimeError, match="configured budget"):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)


def _install_fake_fusion(monkeypatch, *, fail: bool = False):
    captured: list[dict] = []

    # TorchNPU 2.10.0's real operator schema has no return_softmax keyword.
    def fake_fusion(q, k, v, heads, layout, *, atten_mask, scale, keep_prob, sparse_mode):
        captured.append({"mask": atten_mask, "layout": layout, "sparse_mode": sparse_mode})
        if fail:
            raise RuntimeError("simulated unsupported shape")
        batch, q_len, hidden = q.shape
        kv_len = k.shape[1]
        head_dim = hidden // heads
        qh = q.view(batch, q_len, heads, head_dim).transpose(1, 2)
        kh = k.view(batch, kv_len, heads, head_dim).transpose(1, 2)
        vh = v.view(batch, kv_len, heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=~atten_mask, dropout_p=0.0, scale=scale
        )
        return (out.transpose(1, 2).reshape(batch, q_len, hidden),)

    fake_module = types.SimpleNamespace(npu_fusion_attention=fake_fusion)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_module)
    return captured


def test_fake_npu_fusion_receives_inverted_mask_and_matches_reference(monkeypatch):
    captured = _install_fake_fusion(monkeypatch)
    spec, q, k, v = _attention_inputs(seed=21)
    spec = DFlashMaskSpec(
        spec.anchor_positions,
        spec.block_keep_mask,
        spec.ctx_len,
        spec.block_size,
        ctx_doc_id=spec.ctx_doc_id,
        anchor_doc_id=spec.anchor_doc_id,
        query_chunk_size=5,
        backend="npu_fusion",
    )
    scale = q.shape[-1] ** -0.5
    fused, info = npu_fusion_chunked_attention(q, k, v, spec, dropout_p=0.0, scale=scale)
    dense = F.scaled_dot_product_attention(
        q,
        _repeat_kv(k, q.shape[1] // k.shape[1]),
        _repeat_kv(v, q.shape[1] // v.shape[1]),
        attn_mask=spec.full_additive(dtype=q.dtype),
        dropout_p=0.0,
        scale=scale,
    )
    torch.testing.assert_close(fused, dense, rtol=1e-4, atol=1e-5)
    assert info["effective_backend"] == "npu_fusion"
    assert info["fused_chunks"] == info["chunks"]
    assert info["fallback_chunks"] == 0
    assert info["mixed_backend"] is False
    assert info["fallback_mode"] == "strict"
    assert info["fallback_reason"] is None
    assert captured and all(item["layout"] == "BSH" and item["sparse_mode"] == 0 for item in captured)
    # Ascend: True = masked. The first chunk's mask is the exact inverse of SDPA visibility.
    assert torch.equal(captured[0]["mask"], spec.materialize_bool(0, 5, inverted=True))


def test_fake_npu_fusion_fallback_is_explicit_and_can_be_disabled(monkeypatch):
    _install_fake_fusion(monkeypatch, fail=True)
    spec, q, k, v = _attention_inputs(seed=22)
    scale = q.shape[-1] ** -0.5
    base = DFlashMaskSpec(
        spec.anchor_positions,
        spec.block_keep_mask,
        spec.ctx_len,
        spec.block_size,
        ctx_doc_id=spec.ctx_doc_id,
        anchor_doc_id=spec.anchor_doc_id,
        query_chunk_size=5,
        backend="npu_fusion",
        npu_fusion_fallback=True,
    )
    output, info = npu_fusion_chunked_attention(q, k, v, base, dropout_p=0.0, scale=scale)
    dense = F.scaled_dot_product_attention(
        q,
        _repeat_kv(k, q.shape[1] // k.shape[1]),
        _repeat_kv(v, q.shape[1] // v.shape[1]),
        attn_mask=base.full_additive(dtype=q.dtype),
        dropout_p=0.0,
        scale=scale,
    )
    torch.testing.assert_close(output, dense, rtol=1e-4, atol=1e-5)
    assert info["effective_backend"] == "sdpa_chunked"
    assert "simulated unsupported shape" in info["fallback_reason"]
    assert info["fused_chunks"] == 0
    assert info["fallback_chunks"] == info["chunks"]
    assert info["mixed_backend"] is False
    assert info["fallback_mode"] == "profile"

    strict = DFlashMaskSpec(
        spec.anchor_positions,
        spec.block_keep_mask,
        spec.ctx_len,
        spec.block_size,
        ctx_doc_id=spec.ctx_doc_id,
        anchor_doc_id=spec.anchor_doc_id,
        query_chunk_size=5,
        backend="npu_fusion",
        npu_fusion_fallback=False,
    )
    with pytest.raises(RuntimeError, match="simulated unsupported shape"):
        npu_fusion_chunked_attention(q, k, v, strict, dropout_p=0.0, scale=scale)

