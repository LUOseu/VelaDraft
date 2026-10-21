"""CPU telemetry checks; mocked fusion is not hardware evidence."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from tests.test_npu_attention import _attention_inputs, _tiny_module
from vlm_dflash.dflash import chunked_attention


def test_mixed_chunks_report_completed_forward_only_and_preserve_backward(monkeypatch):
    spec, q, k, v = _attention_inputs(seed=91)
    spec = replace(spec, backend="npu_fusion", npu_fusion_fallback=True)
    q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
    # Reject only the short tail; the same choice applies during recomputation.
    def fake_fusion(q, k, v, mask, *, scale, keep_prob):
        if q.shape[-2] < spec.query_chunk_size:
            raise RuntimeError("tail shape unsupported")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, scale=scale)

    monkeypatch.setattr(chunked_attention, "_npu_fusion_call", fake_fusion)
    out, info = chunked_attention.npu_fusion_chunked_attention(
        q, k, v, spec, dropout_p=0.0, scale=8 ** -0.5
    )
    assert info["fused_chunks"] == 2
    assert info["fallback_chunks"] == 1
    assert info["sdpa_chunks"] == 1
    assert info["mixed_backend"] is True
    assert info["effective_backend"] == "mixed"
    assert info["fallback_mode"] == "profile"
    before_backward = dict(info)
    out.square().sum().backward()
    assert info == before_backward
    dense_q, dense_k, dense_v = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    dense = F.scaled_dot_product_attention(
        dense_q,
        chunked_attention._repeat_kv(dense_k, 2),
        chunked_attention._repeat_kv(dense_v, 2),
        attn_mask=spec.full_additive(dtype=q.dtype),
        scale=8 ** -0.5,
    )
    dense.square().sum().backward()
    torch.testing.assert_close(out, dense, rtol=1e-4, atol=1e-5)
    for chunk_tensor, dense_tensor in zip((q, k, v), (dense_q, dense_k, dense_v)):
        torch.testing.assert_close(chunk_tensor.grad, dense_tensor.grad, rtol=2e-4, atol=2e-5)


def _batch():
    return dict(
        input_ids=torch.arange(48).remainder(200).add(10).unsqueeze(0).repeat(2, 1),
        hidden_states=torch.randn(2, 48, 128),
        loss_mask=torch.ones(2, 48),
    )


def test_measurement_window_accumulates_resets_and_returns_detached_snapshots():
    module = _tiny_module("sdpa_chunked")
    batch = _batch()
    module(**batch).loss.backward()  # Warmup must be removable.
    module.reset_attention_runtime_stats()
    assert module.attention_runtime_report()["measurement_window"]["completed_attention_calls"] == 0
    module(**batch).loss.backward()
    first = module.attention_runtime_report()
    window = first["measurement_window"]
    assert window["completed_attention_calls"] == 2  # Two layers, not two model forwards.
    assert window["chunks"] > 2
    assert window["sdpa_chunks"] == window["chunks"]
    assert window["fused_chunks"] == window["fallback_chunks"] == 0
    module(**batch).loss.backward()
    second = module.attention_runtime_report()["measurement_window"]
    assert second["completed_attention_calls"] == 4
    assert second["chunks"] == 2 * window["chunks"]
    assert first["measurement_window"]["completed_attention_calls"] == 2
    assert "recomputation" in second["count_semantics"]
    assert first["memory_plan_kind"] == "analytical_estimate_not_measured_peak_hbm"
    module.reset_attention_runtime_stats()
    assert module.attention_runtime_report()["effective_backends"] == []


def test_switch_to_dense_does_not_keep_stale_chunked_latest_report():
    module = _tiny_module("sdpa_chunked")
    batch = _batch()
    module(**batch)
    module.attention_backend = "sdpa"
    module(**batch)
    report = module.attention_runtime_report()
    assert report["effective_backends"] == ["sdpa"]
    assert report["chunked_layers"] == 0
    window = report["measurement_window"]
    assert window["effective_backends"] == ["sdpa", "sdpa_chunked"]
    assert window["mixed_backend"] is True


def test_measurement_window_retains_fallback_after_later_all_fused_call(monkeypatch):
    spec, q, k, v = _attention_inputs(seed=93)
    spec = replace(spec, backend="npu_fusion", npu_fusion_fallback=True)
    fail = True

    def fake_fusion(q, k, v, mask, *, scale, keep_prob):
        if fail:
            raise RuntimeError("unsupported first call")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, scale=scale)

    monkeypatch.setattr(chunked_attention, "_npu_fusion_call", fake_fusion)
    stats = chunked_attention.AttentionRuntimeStats()
    _, fallback = chunked_attention.npu_fusion_chunked_attention(q, k, v, spec, dropout_p=0, scale=8 ** -0.5)
    stats.record(fallback)
    fail = False
    _, fused = chunked_attention.npu_fusion_chunked_attention(q, k, v, spec, dropout_p=0, scale=8 ** -0.5)
    stats.record(fused)
    window = stats.snapshot()
    assert window["fused_chunks"] == window["fallback_chunks"] == 3
    assert window["mixed_backend"] is True
    assert window["fallback_reasons"] == ["unsupported first call"]
    window["fallback_reasons"].clear()
    assert stats.snapshot()["fallback_reasons"] == ["unsupported first call"]


def test_compile_report_distinguishes_requested_from_device_gate(monkeypatch):
    from vlm_dflash import runtime

    module = _tiny_module("sdpa_chunked")
    module.recipe_compile_tv_kl = True
    monkeypatch.setattr(runtime, "compile_supported", lambda device: False)
    gate = module.attention_runtime_report()["training_switches"]["recipe_compile_tv_kl"]
    assert gate["requested"] is True
    assert gate["effective"] is False
    assert "runtime.compile_supported" in gate["reason"]
    assert gate["scope"] == "device_gate_not_execution_evidence"
    monkeypatch.setattr(runtime, "compile_supported", lambda device: True)
    gate = module.attention_runtime_report()["training_switches"]["recipe_compile_tv_kl"]
    assert gate["effective"] is True
    module.recipe_compile_tv_kl = False
    gate = module.attention_runtime_report()["training_switches"]["recipe_compile_tv_kl"]
    assert gate["effective"] is False
    assert gate["reason"] == "not_requested"
