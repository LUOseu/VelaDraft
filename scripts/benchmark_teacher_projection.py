#!/usr/bin/env python3
"""Paired synthetic-shape update probe; never a convergence or E2E benchmark.

Uses fixed anchors, the real DFlash trainer, BF16 live weights + FP32 AdamW
masters, and alternating order. lr=0 preserves an identical model for every
arm while still executing the optimizer. All raw observations are retained.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
from vlm_dflash.dflash.trainer_core import DFlashTrainerModule
from vlm_dflash.objectives import resolve_recipe


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_snapshot():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None


def require_exclusive(uuid, *, require_own=False):
    """Fail if another compute process is observed; never signal other jobs."""
    if uuid is None:
        return
    rows = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    processes = [(parts[0].strip(), int(parts[1])) for row in rows.splitlines()
                 if len(parts := row.split(",")) == 2]
    others = [pid for gpu, pid in processes if gpu == uuid and pid != os.getpid()]
    if others or (require_own and (uuid, os.getpid()) not in processes):
        raise RuntimeError(f"exclusive GPU guard failed: expected {uuid}, other PIDs={others}, own PID={os.getpid()}")


def gradient_drift(reference, candidate):
    missing = sorted(set(reference) - set(candidate))
    extra = sorted(set(candidate) - set(reference))
    common = set(reference) & set(candidate)
    bad_shapes = sorted(k for k in common if candidate[k].shape != reference[k].shape)
    finite = all(torch.isfinite(t).all().item() for t in [*reference.values(), *candidate.values()])
    valid = not missing and not extra and not bad_shapes and finite and bool(reference)
    result = {"missing": missing, "extra": extra, "bad_shapes": bad_shapes,
              "complete_and_finite": valid, "parameter_count": len(candidate),
              "relative_l2": None, "max_absolute": None}
    if valid:
        numerator = sum((candidate[k] - reference[k]).double().square().sum().item() for k in reference)
        denominator = sum(reference[k].double().square().sum().item() for k in reference)
        result.update(relative_l2=(numerator / max(denominator, 1e-30)) ** 0.5,
                      max_absolute=max((candidate[k] - reference[k]).abs().max().item() for k in reference))
    return result


def run_case(args, stride, stage):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    h, v, bs, n = args.hidden, args.vocab, args.block_size, args.anchors
    seq = max(args.sequence, 8 + (n - 1) * stride + bs)
    cfg = Qwen3Config(
        vocab_size=v, hidden_size=h, intermediate_size=2*h, num_hidden_layers=args.layers,
        num_attention_heads=h // 64, num_key_value_heads=max(1, h // 128), head_dim=64,
        max_position_embeddings=seq + n * bs,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=bs, num_target_layers=2,
        dflash_config={"mask_token_id": v-1, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    cfg._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(cfg).to(device=device, dtype=dtype)
    embed = torch.nn.Embedding(v, h, device=device, dtype=dtype).requires_grad_(False)
    head = torch.nn.Linear(h, v, bias=False, device=device, dtype=dtype).requires_grad_(False)
    head.weight = embed.weight
    # Unit-scale logits; token embeddings are deliberately separate from any
    # target checkpoint. These timings test execution, not trained quality.
    with torch.no_grad():
        embed.weight.normal_(std=h ** -0.5)
    recipe = (resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
              if args.optimization == "teacher" else None)
    module = DFlashTrainerModule(
        draft, head, embed, v-1, block_size=bs, attention_backend="sdpa",
        num_anchors=n, loss_decay_gamma=7.0, train_recipe=recipe,
        teacher_projection_dedup=False,
        recipe_compile_tv_kl=args.compile_tv_kl,
    ).to(device)
    module.set_training_progress(stage, 2)
    ids = torch.randint(0, v-1, (args.batch_size, seq), device=device)
    hidden = torch.randn(args.batch_size, seq, 2*h, device=device, dtype=dtype)
    teacher = torch.randn(args.batch_size, seq, h, device=device, dtype=dtype)
    mask = torch.ones(args.batch_size, seq, device=device)
    mask[:, :8] = 0
    anchors = (8 + torch.arange(n, device=device) * stride).view(1, n).expand(args.batch_size, -1).clone()
    keep = torch.ones_like(anchors, dtype=torch.bool)
    if args.short_anchors is not None:
        keep[1:, args.short_anchors:] = False
        anchors[~keep] = 0
        mask[1:, 8 + (args.short_anchors - 1) * stride + bs:] = 0
    params = dict(draft.named_parameters())
    masters = {key: torch.nn.Parameter(p.detach().float().clone()) for key, p in params.items()}
    optimizer = torch.optim.AdamW(list(masters.values()), lr=0.0, betas=(0.9, 0.95))
    batch = dict(input_ids=ids, hidden_states=hidden, loss_mask=mask, teacher_hidden=teacher,
                 anchor_table=(anchors, keep))

    def step(reuse, *, save_grad=False):
        module.teacher_projection_dedup = reuse and args.optimization == "teacher"
        module.ce_chunk_size = args.ce_chunk_size if reuse and args.optimization == "ce" else 0
        module.ce_backward = args.ce_backward if reuse and args.optimization == "ce" else "checkpoint"
        module.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        metrics = module(**batch)
        (metrics.loss / 3.0).backward()
        gradients = {k: p.grad.detach().float().cpu().clone() for k, p in params.items()
                     if p.grad is not None} if save_grad else None
        with torch.no_grad():
            for key, p in params.items():
                masters[key].grad = None if p.grad is None else p.grad.detach().float()
                p.grad = None
        torch.nn.utils.clip_grad_norm_(list(masters.values()), 1.0)
        optimizer.step()
        with torch.no_grad():
            for key, p in params.items():
                p.copy_(masters[key])
        return metrics, gradients

    ref, ref_grads = step(False, save_grad=True)
    repeat, repeat_grads = step(False, save_grad=True)
    out, out_grads = step(True, save_grad=True)
    parity = {
        "reference_loss": float(ref.loss.detach()), "repeat_loss": float(repeat.loss.detach()), "reuse_loss": float(out.loss.detach()),
        "repeat_gradient": gradient_drift(ref_grads, repeat_grads),
        "reuse_gradient": gradient_drift(ref_grads, out_grads),
        "metric_counts_equal": all(torch.equal(getattr(ref, k), getattr(out, k)) for k in
            ("valid_tokens", "correct_tokens", "pos_count", "pos_correct", "accept_len_sum")),
    }
    # Keep raw observations even if the numerical gate fails.
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-4
    parity["passed"] = (parity["reuse_gradient"]["complete_and_finite"]
                        and parity["repeat_gradient"]["complete_and_finite"]
                        and parity["reuse_gradient"]["relative_l2"] <= tolerance
                        and abs(parity["reference_loss"] - parity["reuse_loss"]) <= tolerance
                        and parity["metric_counts_equal"])
    del ref, repeat, out, ref_grads, repeat_grads, out_grads
    gc.collect()
    for _ in range(args.warmup):
        step(False)
        step(True)
    raw = []
    for pair in range(args.repeats):
        for reuse in ([False, True] if pair % 2 == 0 else [True, False]):
            require_exclusive(args.exclusive_gpu_uuid, require_own=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            sync(device)
            start = time.perf_counter()
            for _ in range(args.inner):
                metrics, _ = step(reuse)
            sync(device)
            elapsed = (time.perf_counter() - start) / args.inner
            require_exclusive(args.exclusive_gpu_uuid, require_own=True)
            raw.append({
                "pair": pair, "reuse": reuse, "seconds_per_update": elapsed,
                "loss": float(metrics.loss.detach()), "workload": dict(module.last_workload),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            })
    medians = {name: statistics.median(r["seconds_per_update"] for r in raw if r["reuse"] == reuse)
               for name, reuse in [("baseline", False), ("reuse", True)]}
    result = {
        "stride": stride, "stage": stage, "sequence": seq, "parity": parity, "raw": raw,
        "median_seconds": medians, "ratio_of_medians": medians["baseline"] / medians["reuse"],
    }
    del module, draft, optimizer, masters, params, metrics
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--optimization", choices=["teacher", "ce"], default="teacher")
    p.add_argument("--ce-chunk-size", type=int, default=64)
    p.add_argument("--ce-backward", choices=["checkpoint", "precompute"], default="checkpoint")
    p.add_argument("--compile-tv-kl", action="store_true")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--anchors", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--short-anchors", type=int)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--sequence", type=int, default=512)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 8])
    p.add_argument("--stages", type=int, nargs="+")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--inner", type=int, default=2)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--memory-fraction", type=float, default=0.20)
    p.add_argument("--exclusive-gpu-uuid", help="Abort if another compute process is observed on this UUID")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.stages is None:
        args.stages = [0] if args.optimization == "ce" else [0, 1]
    if args.optimization == "ce" and args.stages != [0]:
        p.error("controlled CE has no recipe stages; use --stages 0 or omit --stages")
    if args.hidden < 64 or args.hidden % 128 != 0 and args.hidden != 64:
        p.error("--hidden must be 64 or a positive multiple of 128")
    if min(args.anchors, args.layers, args.repeats, args.inner, *args.strides) < 1 or args.block_size < 2:
        p.error("sizes, repeats, strides and inner must be positive; block-size >= 2")
    if any(stage not in (0, 1) for stage in args.stages):
        p.error("--stages must contain only 0 and/or 1")
    if args.ce_chunk_size < 1:
        p.error("--ce-chunk-size must be positive")
    if args.batch_size < 1 or (args.short_anchors is not None and
        (args.batch_size < 2 or not 1 <= args.short_anchors <= args.anchors)):
        p.error("batch-size must be positive; short-anchors needs batch-size >= 2 and 1..anchors")
    device = torch.device(args.device)
    require_exclusive(args.exclusive_gpu_uuid)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "vlm_dflash/objectives.py", root / "vlm_dflash/dflash/trainer_core.py",
             root / "vlm_dflash/dflash/chunked_ce.py"]
    report = {
        "scope": "synthetic_shape_full_update_probe_not_convergence_or_pipeline_E2E",
        "measurement_context": ("shared_host; exclusive_GPU_checked_before_and_after_each_arm"
                                if args.exclusive_gpu_uuid else
                                "shared_host_and_GPU; absolute timings are provisional"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "python": platform.python_version(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "gpu_before": gpu_snapshot(), "cases": [],
    }
    def write():
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.out)
    try:
        for stride in args.strides:
            for stage in args.stages:
                case = run_case(args, stride, stage)
                report["cases"].append(case)
                write()
                print(json.dumps({"stride": stride, "stage": stage, "ratio": case["ratio_of_medians"],
                                  "parity": case["parity"]}), flush=True)
        report["gpu_after"] = gpu_snapshot()
        report["complete"] = True
        write()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        write()
        raise
    if not all(case["parity"]["passed"] for case in report["cases"]):
        raise SystemExit("numerical gate failed; raw results retained")


if __name__ == "__main__":
    main()
