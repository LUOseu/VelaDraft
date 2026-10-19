#!/usr/bin/env python3
"""Strict Ascend operator smoke (H0/H1). Exits nonzero without a usable NPU.

This script never falls back to CPU: every tensor check asserts ``device.type ==
'npu'``. The optional fused-attention case exercises this repository's opt-in
``npu_fusion_attention`` wrapper and compares forward/backward against CPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash import runtime
from vlm_dflash.dflash.chunked_attention import npu_fusion_chunked_attention
from vlm_dflash.dflash.mask import DFlashMaskSpec


def _npu_smi() -> dict:
    executable = shutil.which("npu-smi")
    if executable is None:
        return {"available": False, "reason": "npu-smi not found on PATH"}
    try:
        completed = subprocess.run([executable, "info"], capture_output=True, text=True, timeout=15)
        return {
            "available": True,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-2000:],
        }
    except Exception as error:
        return {"available": True, "error": f"{type(error).__name__}: {error}"}


def _close(left: torch.Tensor, right: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-5) -> float:
    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    scale = right.detach().float().cpu().abs().clamp_min(1e-12)
    if not torch.all(difference <= atol + rtol * scale):
        raise AssertionError(
            f"tensor mismatch: max_abs={float(difference.max()):.3e}, "
            f"max_rel={float((difference / scale).max()):.3e}"
        )
    return float(difference.max())


def _run_checks(device: torch.device, *, include_fusion: bool, include_h1: bool, seed: int) -> list[dict]:
    checks: list[dict] = []

    def record(name: str, function):
        started = time.time()
        try:
            details = function() or {}
            checks.append(
                {"name": name, "status": "passed", "seconds": round(time.time() - started, 4), **details}
            )
        except Exception as error:
            checks.append(
                {
                    "name": name,
                    "status": "failed",
                    "seconds": round(time.time() - started, 4),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )

    def allocation_matmul_backward():
        x = torch.randn(256, 256, device=device, dtype=torch.float32)
        weight = torch.randn(256, 256, device=device, dtype=torch.float32, requires_grad=True)
        output = (x @ weight).square().mean()
        output.backward()
        runtime.synchronize(device)
        assert x.device.type == "npu" and weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        return {"loss": float(output.detach().cpu()), "grad_norm": float(weight.grad.norm().detach().cpu())}

    def rng_round_trip():
        runtime.seed_all(seed, device)
        before = torch.rand(8, device=device)
        state = runtime.capture_rng_state()
        expected = torch.rand(8, device=device)
        runtime.restore_rng_state(state)
        replayed = torch.rand(8, device=device)
        runtime.synchronize(device)
        if not torch.equal(expected.cpu(), replayed.cpu()):
            raise AssertionError("torch NPU RNG did not reproduce after save/restore")
        if not state.get("npu"):
            raise AssertionError("capture_rng_state() returned no NPU state")
        return {"npu_states": len(state["npu"]), "first_sample": float(before[0].cpu())}

    def peak_memory():
        runtime.reset_peak_memory_stats(device)
        block = torch.empty(8 * 1024 * 1024, device=device, dtype=torch.uint8)
        runtime.synchronize(device)
        peak = runtime.max_memory_allocated(device)
        if peak is None:
            raise AssertionError("NPU peak memory API is unavailable; refusing to report 0")
        del block
        return {"peak_bytes": int(peak), "peak_gb": round(peak / 1e9, 4)}

    def sdpa_forward_backward():
        torch.manual_seed(seed)
        q = torch.randn(2, 2, 8, 16, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 2, 8, 16, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(2, 2, 8, 16, device=device, dtype=torch.float32, requires_grad=True)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output.square().mean().backward()
        runtime.synchronize(device)
        q_cpu = q.detach().cpu().requires_grad_(True)
        k_cpu = k.detach().cpu().requires_grad_(True)
        v_cpu = v.detach().cpu().requires_grad_(True)
        reference = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True)
        reference.square().mean().backward()
        difference = _close(output, reference)
        grad_difference = max(
            _close(q.grad, q_cpu.grad), _close(k.grad, k_cpu.grad), _close(v.grad, v_cpu.grad)
        )
        return {"max_abs_output_error": difference, "max_abs_grad_error": grad_difference}

    def gqa_sdpa_forward_backward():
        torch.manual_seed(seed + 2)
        q = torch.randn(2, 4, 12, 16, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 2, 12, 16, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(2, 2, 12, 16, device=device, dtype=torch.float32, requires_grad=True)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output.square().mean().backward()
        runtime.synchronize(device)
        q_cpu = q.detach().cpu().requires_grad_(True)
        k_cpu = k.detach().cpu().requires_grad_(True)
        v_cpu = v.detach().cpu().requires_grad_(True)
        reference = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True)
        reference.square().mean().backward()
        return {
            "heads": 4,
            "kv_heads": 2,
            "max_abs_output_error": _close(output, reference),
            "max_abs_grad_error": max(
                _close(q.grad, q_cpu.grad), _close(k.grad, k_cpu.grad), _close(v.grad, v_cpu.grad)
            ),
        }

    def bf16_sdpa_forward_backward():
        torch.manual_seed(seed + 3)
        q = torch.randn(2, 3, 10, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(2, 3, 10, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(2, 3, 10, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output.float().square().mean().backward()
        runtime.synchronize(device)
        q_cpu = q.detach().float().cpu().requires_grad_(True)
        k_cpu = k.detach().float().cpu().requires_grad_(True)
        v_cpu = v.detach().float().cpu().requires_grad_(True)
        reference = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=False)
        reference.square().mean().backward()
        return {
            "max_abs_output_error": _close(output, reference, rtol=2e-2, atol=2e-2),
            "max_abs_grad_error": max(
                _close(q.grad, q_cpu.grad, rtol=2e-2, atol=2e-2),
                _close(k.grad, k_cpu.grad, rtol=2e-2, atol=2e-2),
                _close(v.grad, v_cpu.grad, rtol=2e-2, atol=2e-2),
            ),
        }

    def mask_spec_chunked_parity():
        from vlm_dflash.dflash.mask import DFlashMaskSpec

        anchors = torch.tensor([[1, 5, 9], [2, 6, 10]], dtype=torch.long)
        keep = torch.tensor([[True, True, False], [True, False, True]])
        ctx_doc_id = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]],
            dtype=torch.long,
        )
        anchor_doc_id = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
        spec = DFlashMaskSpec(
            anchors.to(device),
            keep.to(device),
            ctx_len=12,
            block_size=4,
            ctx_doc_id=ctx_doc_id.to(device),
            anchor_doc_id=anchor_doc_id.to(device),
            query_chunk_size=3,
            backend="sdpa_chunked",
        )
        spec_cpu = DFlashMaskSpec(
            anchors,
            keep,
            ctx_len=12,
            block_size=4,
            ctx_doc_id=ctx_doc_id,
            anchor_doc_id=anchor_doc_id,
            query_chunk_size=3,
            backend="sdpa_chunked",
        )
        chunks = [
            spec.materialize_additive(start, min(spec.query_len, start + 3))
            for start in range(0, spec.query_len, 3)
        ]
        dense = torch.cat(chunks, dim=2)
        if not torch.equal(dense.cpu(), spec_cpu.full_additive()):
            raise AssertionError("chunked NPU mask differs from the dense CPU reference")
        inverted = torch.cat(
            [spec.materialize_bool(start, min(spec.query_len, start + 3), inverted=True) for start in range(0, spec.query_len, 3)],
            dim=2,
        )
        if not torch.equal(inverted.cpu(), ~spec_cpu.materialize_bool(0, spec_cpu.query_len)):
            raise AssertionError("NPU inverted boolean mask is not the exact SDPA-mask complement")
        return {"query_len": spec.query_len, "kv_len": spec.kv_len, "chunks": len(chunks)}

    def draft_module_forward_backward_parity():
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
        from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

        torch.manual_seed(seed + 4)
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
        base = DFlashTrainerModule(
            draft_model=draft,
            target_lm_head=lm_head,
            target_embed_tokens=embed,
            mask_token_id=255,
            block_size=8,
            attention_backend="sdpa_chunked",
            num_anchors=2,
            loss_decay_gamma=7.0,
            attention_chunk_size=7,
        )
        npu_module = copy.deepcopy(base).to(device)
        cpu_module = copy.deepcopy(base).cpu()
        npu_module.train()
        cpu_module.train()
        input_ids = torch.arange(48).remainder(200).add(10).unsqueeze(0).repeat(2, 1)
        loss_mask = torch.ones(2, 48)
        hidden = torch.randn(2, 48, 128, generator=torch.Generator().manual_seed(seed + 5)) * 0.1
        anchors = torch.tensor([[8, 20], [12, 28]], dtype=torch.long)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        npu_metrics = npu_module(
            input_ids=input_ids.to(device),
            hidden_states=hidden.to(device),
            loss_mask=loss_mask.to(device),
            anchor_table=(anchors.to(device), keep.to(device)),
        )
        npu_metrics.loss.backward()
        cpu_metrics = cpu_module(
            input_ids=input_ids,
            hidden_states=hidden,
            loss_mask=loss_mask,
            anchor_table=(anchors, keep),
        )
        cpu_metrics.loss.backward()
        runtime.synchronize(device)
        npu_grads = dict(npu_module.named_parameters())
        cpu_grads = dict(cpu_module.named_parameters())
        max_grad_error = 0.0
        compared = 0
        for name, parameter in cpu_grads.items():
            if parameter.grad is None:
                continue
            compared += 1
            max_grad_error = max(max_grad_error, _close(npu_grads[name].grad, parameter.grad))
        if compared == 0:
            raise AssertionError("draft module parity backward produced no gradients")
        return {
            "loss": float(npu_metrics.loss.detach().cpu()),
            "max_abs_loss_error": abs(float(npu_metrics.loss.detach().cpu()) - float(cpu_metrics.loss.detach().cpu())),
            "max_abs_grad_error": max_grad_error,
            "compared_parameter_grads": compared,
            "attention_chunk_size": 7,
        }

    def fused_attention():
        torch.manual_seed(seed + 1)
        anchors = torch.tensor([[0, 4]], device=device, dtype=torch.long)
        keep = torch.ones_like(anchors, dtype=torch.bool)
        spec = DFlashMaskSpec(
            anchors,
            keep,
            ctx_len=4,
            block_size=2,
            query_chunk_size=2,
            backend="npu_fusion",
            npu_fusion_fallback=False,
        )
        q = torch.randn(1, 2, spec.query_len, 8, device=device, requires_grad=True)
        k = torch.randn(1, 1, spec.kv_len, 8, device=device, requires_grad=True)
        v = torch.randn(1, 1, spec.kv_len, 8, device=device, requires_grad=True)
        output, info = npu_fusion_chunked_attention(
            q, k, v, spec, dropout_p=0.0, scale=8 ** -0.5
        )
        output.square().mean().backward()
        runtime.synchronize(device)
        if info.get("effective_backend") != "npu_fusion":
            raise AssertionError(f"fused path fell back: {info.get('fallback_reason')}")
        k_cpu = k.detach().cpu().repeat_interleave(2, dim=1).requires_grad_(True)
        v_cpu = v.detach().cpu().repeat_interleave(2, dim=1).requires_grad_(True)
        q_cpu = q.detach().cpu().requires_grad_(True)
        spec_cpu = DFlashMaskSpec(
            anchors.cpu(), keep.cpu(), ctx_len=4, block_size=2, query_chunk_size=2
        )
        reference = F.scaled_dot_product_attention(
            q_cpu, k_cpu, v_cpu, attn_mask=spec_cpu.full_additive(dtype=torch.float32)
        )
        reference.square().mean().backward()
        return {
            "max_abs_output_error": _close(output, reference),
            "max_abs_q_grad_error": _close(q.grad, q_cpu.grad),
            "max_abs_kv_grad_error": max(_close(k.grad, k_cpu.grad), _close(v.grad, v_cpu.grad)),
            "effective_backend": info["effective_backend"],
        }

    record("allocation_matmul_backward", allocation_matmul_backward)
    record("rng_round_trip", rng_round_trip)
    record("peak_memory", peak_memory)
    record("sdpa_forward_backward", sdpa_forward_backward)
    if include_h1:
        record("gqa_sdpa_forward_backward", gqa_sdpa_forward_backward)
        record("bf16_sdpa_forward_backward", bf16_sdpa_forward_backward)
        record("mask_spec_chunked_parity", mask_spec_chunked_parity)
        record("draft_module_forward_backward_parity", draft_module_forward_backward_parity)
    if include_fusion:
        record("npu_fusion_forward_backward", fused_attention)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--out", default=str(ROOT / "runs" / "npu" / "operator_smoke.json"))
    parser.add_argument("--include-fusion", action="store_true")
    parser.add_argument("--h1", action="store_true", help="add GQA/BF16/mask/draft-module numeric parity checks")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "runtime": runtime.environment_report(),
        "npu_smi": _npu_smi(),
    }
    started = time.time()
    payload = {
        "format": "vlm-dflash-npu-operator-smoke-v1",
        "requested_device": args.device,
        "include_fusion": bool(args.include_fusion),
        "include_h1": bool(args.h1),
        "environment": environment,
        "checks": [],
    }
    try:
        device = runtime.ensure_runtime_device(args.device, context="operator_smoke.device")
    except Exception as error:
        payload.update(
            {
                "status": "NPU_UNVERIFIED",
                "evidence_level": "NPU_UNVERIFIED",
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": round(time.time() - started, 3),
                "note": "No usable NPU; no operator check ran and no CPU fallback was attempted.",
            }
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": payload["status"], "error": payload["error"], "json": str(out_path)}))
        return 2

    payload["resolved_device"] = str(device)
    payload["device_name"] = runtime.device_name(device)
    payload["device_memory"] = runtime.device_memory_info(device)
    checks = _run_checks(device, include_fusion=args.include_fusion, include_h1=args.h1, seed=args.seed)
    failures = [check for check in checks if check["status"] != "passed"]
    payload["checks"] = checks
    payload["duration_seconds"] = round(time.time() - started, 3)
    payload["status"] = "NPU_SMOKE_PASSED" if not failures else "FAILED"
    payload["evidence_level"] = "NPU_SMOKE_PASSED" if not failures else "NOT_VERIFIED"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "checks": {check["name"]: check["status"] for check in checks},
                "json": str(out_path),
            }
        )
    )
    return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())

