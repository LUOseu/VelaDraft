#!/usr/bin/env python
"""Paired target-only vs speculative speed measurement (one GPU, exclusive).

Same prompts, same dtype/decoding/output cap for both arms; warmup runs before
the measured window and is reported separately (startup/compile overhead).
Verification is the sequential (lossless) path; output equivalence is checked
token-exact. Reports wall per sample paired ratios, speculative rounds, actual
target forward calls, fallback count, draft/verify/prefill seconds and peak
memory. The drafter is scored exactly as eval scores it (parity verify
backend), only the timing bookkeeping differs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from vlm_dflash.algorithms import get_algorithm  # noqa: E402
from vlm_dflash.config import apply_overrides, load_config, resolve_run_dir  # noqa: E402
from vlm_dflash.contract import ensure_contract  # noqa: E402
from vlm_dflash.eval.equivalence import sequences_equal  # noqa: E402
from vlm_dflash.eval.run import _stops, load_eval_rows, prepare_prompts  # noqa: E402
from vlm_dflash.target.load import load_target  # noqa: E402
from vlm_dflash.utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default="ckpt_final.pt")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--warmup-prompts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--verification", default="sequential",
                        choices=["sequential", "parallel", "strict_parallel"],
                        help="sequential is the lossless reference; parallel/strict_parallel "
                             "are measured separately and only reported as faithful if token-exact")
    parser.add_argument("--strict-margin", type=float, default=2.0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    algorithm = get_algorithm(cfg.get("algorithm"))
    rows, media_dir = load_eval_rows(None, run_dir)
    measured_rows = rows[: args.num_prompts]
    warm_rows = rows[args.num_prompts : args.num_prompts + args.warmup_prompts]

    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    stop_ids = set(_stops(model))
    block_size = int(cfg["drafter"]["block_size"])
    dtype_name = str(cfg.get("target", {}).get("dtype", "bfloat16"))
    draft_dtype = getattr(torch, dtype_name, torch.bfloat16)
    from vlm_dflash.adapters import adapter_for_model

    adapter = adapter_for_model(model)

    startup = {}
    started = time.perf_counter()
    warm_prompts = prepare_prompts(model, processor, warm_rows, media_dir, max_new_tokens=args.max_new_tokens, device=device, label="warmup")
    startup["target_only_warmup_seconds"] = time.perf_counter() - started
    draft, _ = algorithm.load_decoder(run_dir / "train" / args.ckpt)
    draft.to(device=device, dtype=draft_dtype)
    draft.eval()
    sequential_verification = args.verification == "sequential"
    strict_margin = args.strict_margin if args.verification == "strict_parallel" else None
    # Same verify-backend regime as eval: "parity" (the config default) keeps the
    # generate()-backend logits, "reference" narrows the verify SDPA backend.
    verify_backend = str(cfg.get("eval", {}).get("verify_sdpa_backend", "parity"))
    algorithm.configure_verify_backend(verify_backend == "reference")
    started = time.perf_counter()
    with torch.inference_mode():
        warm_entry = warm_prompts[0]
        draft.spec_generate(
            target=model,
            input_ids=warm_entry["inputs"]["input_ids"],
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=sorted(stop_ids),
            temperature=0.0,
            target_kwargs=adapter.forward_kwargs_from_inputs(warm_entry["inputs"]),
            return_stats=False,
            sequential_target_verification=sequential_verification,
            strict_parallel_margin=strict_margin,
        )
    startup["spec_warmup_seconds"] = time.perf_counter() - started
    torch.cuda.synchronize()

    prompts = prepare_prompts(model, processor, measured_rows, media_dir, max_new_tokens=args.max_new_tokens, device=device, label="measured")

    torch.cuda.reset_peak_memory_stats(device)
    per_sample = []
    for entry in prompts:
        inputs = entry["inputs"]
        target_kwargs = adapter.forward_kwargs_from_inputs(inputs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids, stats = draft.spec_generate(
                target=model,
                input_ids=inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=sorted(stop_ids),
                temperature=0.0,
                target_kwargs=target_kwargs,
                return_stats=True,
                sequential_target_verification=sequential_verification,
                strict_parallel_margin=strict_margin,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        prompt_len = entry["prompt_len"]
        spec_tokens = output_ids[0, prompt_len:].tolist()
        equivalent = sequences_equal(spec_tokens, entry["tokens"], stop_ids)
        per_sample.append(
            {
                "id": entry["id"],
                "target_only_wall": entry["wall_seconds"],
                "spec_wall": wall,
                "speedup": entry["wall_seconds"] / wall if wall > 0 else None,
                "new_tokens": len(spec_tokens),
                "target_new_tokens": len(entry["tokens"]),
                "equivalent": equivalent,
                "rounds": int(stats["verify_steps"]),
                "target_forward_calls": int(stats["target_forward_calls"]),
                "accepted_tokens": int(stats["accepted_tokens"]),
                "draft_tokens": int(stats["draft_tokens"]),
                "strict_parallel_rounds": float(stats.get("strict_parallel_rounds", 0.0)),
                "strict_fallback_rounds": float(stats.get("strict_fallback_rounds", 0.0)),
                "draft_seconds": float(stats.get("draft_seconds", 0.0)),
                "round_prep_seconds": float(stats.get("round_prep_seconds", 0.0)),
                "verify_prep_seconds": float(stats.get("verify_prep_seconds", 0.0)),
                "target_verify_seconds": float(stats.get("target_verify_seconds", 0.0)),
                "target_prefill_seconds": float(stats.get("target_prefill_seconds", 0.0)),
                "commit_seconds": float(stats.get("commit_seconds", 0.0)),
            }
        )
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    def mean(values: list[float]) -> float:
        return sum(values) / max(1, len(values))

    ratios = [row["speedup"] for row in per_sample if row["speedup"] is not None]
    total_target = sum(row["target_only_wall"] for row in per_sample)
    total_spec = sum(row["spec_wall"] for row in per_sample)
    report = {
        "format": "vlm-dflash-speed-paired-v1",
        "tag": args.tag,
        "checkpoint": args.ckpt,
        "num_prompts": len(per_sample),
        "max_new_tokens": args.max_new_tokens,
        "verification": args.verification,
        "gpu_name": torch.cuda.get_device_name(device),
        "startup": startup,
        "target_only_wall_per_sample_mean": mean([row["target_only_wall"] for row in per_sample]),
        "spec_wall_per_sample_mean": mean([row["spec_wall"] for row in per_sample]),
        "speedup_ratio_of_means": total_target / total_spec if total_spec else None,
        "speedup_paired_mean_of_ratios": mean(ratios),
        "speedup_paired_median": statistics.median(ratios),
        "token_exact": all(row["equivalent"] for row in per_sample),
        "token_exact_count": sum(1 for row in per_sample if row["equivalent"]),
        "new_tokens_total": sum(row["new_tokens"] for row in per_sample),
        "rounds_total": sum(row["rounds"] for row in per_sample),
        "target_forward_calls_total": sum(row["target_forward_calls"] for row in per_sample),
        "target_forwards_per_round": (
            sum(row["target_forward_calls"] for row in per_sample) / max(1, sum(row["rounds"] for row in per_sample))
        ),
        "fallback_rounds_total": sum(row["strict_fallback_rounds"] for row in per_sample),
        "draft_seconds_total": sum(row["draft_seconds"] for row in per_sample),
        "target_verify_seconds_total": sum(row["target_verify_seconds"] for row in per_sample),
        "target_prefill_seconds_total": sum(row["target_prefill_seconds"] for row in per_sample),
        "mean_al_rounds": (
            sum(row["new_tokens"] for row in per_sample) / max(1, sum(row["rounds"] for row in per_sample))
        ),
        "tokens_per_target_forward": (
            sum(row["new_tokens"] for row in per_sample)
            / max(1, sum(row["target_forward_calls"] for row in per_sample))
        ),
        # Dual acceptance metrics (see vlm_dflash/eval/accept.py): draft-only vs
        # progress-inclusive. Published tau values are usually the draft one.
        "al_progress_tokens_per_round": (
            sum(row["new_tokens"] for row in per_sample) / max(1, sum(row["rounds"] for row in per_sample))
        ),
        "al_draft_tokens_per_round": (
            sum(row["accepted_tokens"] for row in per_sample) / max(1, sum(row["rounds"] for row in per_sample))
        ),
        # Wall-time decomposition inside the speculative generate loop. ``misc``
        # is the residual (python glue, sampling, stop-token scans, anything not
        # bracketed by a cuda event pair).
        "phase_seconds_total": {
            phase: sum(row[f"{phase}_seconds"] for row in per_sample)
            for phase in ("round_prep", "draft", "verify_prep", "target_verify", "target_prefill", "commit")
        },
        "peak_gpu_mem_gb": peak_mem,
        "per_sample": per_sample,
    }
    phases = report["phase_seconds_total"]
    report["phase_fraction_of_spec_wall"] = {
        phase: (seconds / total_spec if total_spec else None) for phase, seconds in phases.items()
    }
    report["phase_seconds_total"]["misc"] = total_spec - sum(phases.values())
    report["phase_fraction_of_spec_wall"]["misc"] = (
        (total_spec - sum(phases.values())) / total_spec if total_spec else None
    )
    out = run_dir / "eval" / f"speed_paired_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_sample"}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
