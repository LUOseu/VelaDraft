#!/usr/bin/env python
"""Target/server-side profile of the speculative path (E2 in the ViSpec task doc).

Measures everything the speedup model needs on one exclusive GPU, with the same
prompts / dtype / decoding / output cap as the official harness:

    S_model ~= AL_progress / (r + d)        S_wall = T_target_only / T_spec

with

    r = mean target block-verify time / mean single-token target decode time
    d = mean draft-round time          / mean single-token target decode time

The single-token target decode time is measured on the *reference* path (HF
``generate``) with a two-point probe: ``generate(max_new_tokens=1)`` is
``prefill + 1 decode step`` and ``generate(max_new_tokens=K)`` is
``prefill + K decode steps``, so the delta divided by ``K - 1`` cancels the
prefill and the per-call python overhead. Phase timings on the speculative side
come from the CUDA-event brackets inside ``spec_generate`` (round_prep, draft,
verify_prep, target_verify, target_prefill, commit) plus a `misc` residual.
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

PHASES = ("round_prep", "draft", "verify_prep", "target_verify", "target_prefill", "commit")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default="ckpt_final.pt")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--decode-probe-prompts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--verification", default="parallel", choices=["sequential", "parallel", "strict_parallel"])
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
    warm_prompts = prepare_prompts(
        model, processor, warm_rows, media_dir, max_new_tokens=args.max_new_tokens, device=device, label="warmup"
    )
    startup["target_only_warmup_seconds"] = time.perf_counter() - started
    draft, _ = algorithm.load_decoder(run_dir / "train" / args.ckpt)
    draft.to(device=device, dtype=draft_dtype)
    draft.eval()
    sequential = args.verification == "sequential"
    strict_margin = args.strict_margin if args.verification == "strict_parallel" else None
    verify_backend = str(cfg.get("eval", {}).get("verify_sdpa_backend", "parity"))
    algorithm.configure_verify_backend(verify_backend == "reference")
    started = time.perf_counter()
    with torch.inference_mode():
        entry = warm_prompts[0]
        draft.spec_generate(
            target=model,
            input_ids=entry["inputs"]["input_ids"],
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=sorted(stop_ids),
            temperature=0.0,
            target_kwargs=adapter.forward_kwargs_from_inputs(entry["inputs"]),
            return_stats=False,
            sequential_target_verification=sequential,
            strict_parallel_margin=strict_margin,
        )
    _sync()
    startup["spec_warmup_seconds"] = time.perf_counter() - started

    # ---- reference path: target-only greedy (identical to the eval harness) ----
    prompts = prepare_prompts(
        model, processor, measured_rows, media_dir, max_new_tokens=args.max_new_tokens, device=device, label="measured"
    )

    # ---- decode-step probe: prefill vs per-token decode on the reference path ----
    probe = []
    for entry in prompts[: args.decode_probe_prompts]:
        inputs = entry["inputs"]
        torch.cuda.reset_peak_memory_stats(device)
        one = dict(inputs)
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            model.generate(**one, max_new_tokens=1, do_sample=False, repetition_penalty=1.0)
        _sync()
        t_one = time.perf_counter() - started
        full = dict(inputs)
        _sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated_full = model.generate(
                **full, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.0
            )
        _sync()
        t_full = time.perf_counter() - started
        # ``generate`` may stop on EOS before the cap, so the number of decode
        # steps actually executed is the generated length, not the cap. Dividing
        # by the cap silently inflates the single-token decode time whenever
        # prompts finish early.
        generated_tokens_full = int(generated_full.shape[1] - inputs["input_ids"].shape[1])
        steps = max(1, generated_tokens_full - 1)
        decode_s = (t_full - t_one) / steps
        probe.append(
            {
                "id": entry["id"],
                "wall_1_token": t_one,
                "wall_k_tokens": t_full,
                "generated_tokens_full": generated_tokens_full,
                "prefill_plus_one_s": t_one,
                "single_token_decode_s": decode_s,
                "prefill_s": t_one - decode_s,
                "target_only_harness_wall_s": entry["wall_seconds"],
                "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            }
        )
    single_token_decode_s = statistics.median([row["single_token_decode_s"] for row in probe]) if probe else None

    # ---- speculative run with phase timings ----
    per_sample = []
    for entry in prompts:
        inputs = entry["inputs"]
        target_kwargs = adapter.forward_kwargs_from_inputs(inputs)
        _sync()
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
                sequential_target_verification=sequential,
                strict_parallel_margin=strict_margin,
            )
        _sync()
        wall = time.perf_counter() - started
        prompt_len = entry["prompt_len"]
        spec_tokens = output_ids[0, prompt_len:].tolist()
        phases = {phase: float(stats.get(f"{phase}_seconds", 0.0)) for phase in PHASES}
        per_sample.append(
            {
                "id": entry["id"],
                "target_only_wall": entry["wall_seconds"],
                "spec_wall": wall,
                "new_tokens": len(spec_tokens),
                "equivalent": sequences_equal(spec_tokens, entry["tokens"], stop_ids),
                "rounds": int(stats["verify_steps"]),
                "accepted_tokens": int(stats["accepted_tokens"]),
                "draft_tokens": int(stats["draft_tokens"]),
                "target_forward_calls": int(stats.get("target_forward_calls", 0)),
                "strict_fallback_rounds": float(stats.get("strict_fallback_rounds", 0.0)),
                **phases,
            }
        )

    rounds = sum(row["rounds"] for row in per_sample)
    new_tokens = sum(row["new_tokens"] for row in per_sample)
    accepted = sum(row["accepted_tokens"] for row in per_sample)
    total_spec = sum(row["spec_wall"] for row in per_sample)
    total_target_only = sum(row["target_only_wall"] for row in per_sample)
    phases_total = {phase: sum(row[phase] for row in per_sample) for phase in PHASES}
    phases_total["misc"] = total_spec - sum(phases_total.values())
    al_progress = new_tokens / rounds if rounds else None
    al_draft = accepted / rounds if rounds else None
    r = d = s_model = None
    if single_token_decode_s:
        r = (phases_total["target_verify"] / rounds) / single_token_decode_s if rounds else None
        d = (phases_total["draft"] / rounds) / single_token_decode_s if rounds else None
        if r is not None and d is not None:
            s_model = al_progress / (r + d) if (r + d) else None
    report = {
        "format": "vlm-dflash-parallel-profile-v1",
        "tag": args.tag,
        "checkpoint": args.ckpt,
        "verification": args.verification,
        "num_prompts": len(per_sample),
        "max_new_tokens": args.max_new_tokens,
        "gpu_name": torch.cuda.get_device_name(device),
        "startup": startup,
        "startup_note": "excluded from every measured window (warmup prefills + one warmup spec generate)",
        "target_only_total_s": total_target_only,
        "spec_total_s": total_spec,
        "s_wall_ratio_of_means": (total_target_only / total_spec) if total_spec else None,
        "al_progress_tokens_per_round": al_progress,
        "al_draft_tokens_per_round": al_draft,
        "al_progress_minus_draft": (al_progress - al_draft) if (al_progress is not None and al_draft is not None) else None,
        "rounds_total": rounds,
        "target_forward_calls_total": sum(row["target_forward_calls"] for row in per_sample),
        "tokens_per_target_forward": (new_tokens / max(1, sum(row["target_forward_calls"] for row in per_sample))),
        "single_token_target_decode_ms": (single_token_decode_s * 1e3) if single_token_decode_s else None,
        "single_token_target_decode_source": "two-point HF generate probe (median over probes)",
        "r_block_verify_over_single_decode": r,
        "d_draft_over_single_decode": d,
        "s_model_al_over_r_plus_d": s_model,
        "phase_seconds_total": phases_total,
        "phase_fraction_of_spec_wall": {k: (v / total_spec if total_spec else None) for k, v in phases_total.items()},
        "phase_ms_per_round": {k: (v / rounds * 1e3 if rounds else None) for k, v in phases_total.items()},
        "decode_probe": probe,
        "token_exact": all(row["equivalent"] for row in per_sample),
        "token_exact_count": sum(1 for row in per_sample if row["equivalent"]),
        "non_equivalent_ids": [row["id"] for row in per_sample if not row["equivalent"]],
        "per_sample": per_sample,
    }
    out = run_dir / "eval" / f"profile_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_sample", "decode_probe"}}, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
