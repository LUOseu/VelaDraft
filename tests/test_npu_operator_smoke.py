"""Check reference-side correctness of the hardware acceptance harness."""

import importlib.util
from pathlib import Path

import torch


def _load_smoke():
    path = Path(__file__).resolve().parents[1] / "scripts/npu/operator_smoke.py"
    spec = importlib.util.spec_from_file_location("operator_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1_gqa_reference_accepts_different_query_and_kv_heads():
    module = _load_smoke()
    # Other H0 cases intentionally reject CPU. Here CPU only checks that the
    # GQA reference itself is valid; it is never evidence for NPU execution.
    checks = module._run_checks(
        torch.device("cpu"), include_fusion=False, include_h1=True, seed=2026
    )
    gqa = next(check for check in checks if check["name"] == "gqa_sdpa_forward_backward")
    assert gqa["status"] == "passed", gqa.get("error")


def test_fused_reference_sums_gradients_into_original_kv_heads(monkeypatch):
    module = _load_smoke()

    def reference_fusion(q, k, v, mask, *, dropout_p, scale):
        result = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.full_additive(dtype=q.dtype),
            dropout_p=dropout_p, scale=scale, enable_gqa=True,
        )
        return result, {"effective_backend": "npu_fusion"}

    monkeypatch.setattr(module, "npu_fusion_chunked_attention", reference_fusion)
    checks = module._run_checks(
        torch.device("cpu"), include_fusion=True, include_h1=False, seed=2026
    )
    fused = next(check for check in checks if check["name"] == "npu_fusion_forward_backward")
    assert fused["status"] == "passed", fused.get("error")
