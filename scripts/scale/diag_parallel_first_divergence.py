#!/usr/bin/env python
"""First-divergence diagnosis for the parallel verification path.

Runs the same prompt through several paths and localises the *first* position
where they disagree, then reports the per-forward decision evidence on both
sides of the disagreement:

  target_only        reference greedy tokens (HF generate)
  sequential         lossless reference speculative path (one target forward per token)
  parallel           current fast path (one target forward per block)
  parallel_parity    same as parallel but with the narrowed/parity SDPA backend
  strict_parallel    parallel with the near-tie margin guard (falls back to parity
                     verification round by round)

For the first divergence position the decision trace (captured inside
``spec_generate`` with ``capture_decision_trace=True``) gives, for each path:
absolute position, the shape of the forward that produced it (sequential vs
block), the KV length at that forward, argmax, top-1/top-2 logits and their gap.
That is what separates "logical/offset bug" (Case A) from "bf16/kernel
numerical-path divergence" (Case B) from "the baseline itself is not bit-stable"
(Case C; measured with ``--target-only-repeats``).
"""
from __future__ import annotations

import argparse
import json
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


def first_divergence(a: list[int], b: list[int], stops: set[int]) -> int | None:
    if sequences_equal(a, b, stops):
        return None
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def trace_at(stats: dict, position: int) -> list[dict]:
    return [row for row in stats.get("decision_trace", []) if row["position"] == position]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default="ckpt_final.pt")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target-only-repeats", type=int, default=2)
    parser.add_argument("--ids", default=None, help="comma-separated row ids to restrict to")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    algorithm = get_algorithm(cfg.get("algorithm"))
    rows, media_dir = load_eval_rows(None, run_dir)
    if args.ids:
        wanted = {value.strip() for value in args.ids.split(",") if value.strip()}
        rows = [row for row in rows if str(row.get("id")) in wanted]
    measured_rows = rows[: args.num_prompts]

    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    stop_ids = set(_stops(model))
    dtype_name = str(cfg.get("target", {}).get("dtype", "bfloat16"))
    draft_dtype = getattr(torch, dtype_name, torch.bfloat16)
    from vlm_dflash.adapters import adapter_for_model

    adapter = adapter_for_model(model)
    draft, _ = algorithm.load_decoder(run_dir / "train" / args.ckpt)
    draft.to(device=device, dtype=draft_dtype)
    draft.eval()

    prompts = prepare_prompts(
        model, processor, measured_rows, media_dir, max_new_tokens=args.max_new_tokens, device=device, label="diag"
    )

    # Case C probe: is the reference path itself bit-stable across reruns?
    stability = []
    for entry in prompts:
        signals = []
        for _ in range(max(1, args.target_only_repeats)):
            with torch.inference_mode():
                generated = model.generate(
                    **dict(entry["inputs"]),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                )
            signals.append(generated[0, entry["prompt_len"] :].tolist())
        stable = all(sequences_equal(signals[0], other, stop_ids) for other in signals[1:])
        stability.append({"id": entry["id"], "target_only_bit_stable": stable, "repeats": len(signals)})

    def run_spec(entry, *, sequential: bool, strict_margin: float | None, parity_backend: bool) -> dict:
        algorithm.configure_verify_backend(parity_backend)
        target_kwargs = adapter.forward_kwargs_from_inputs(entry["inputs"])
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids, stats = draft.spec_generate(
                target=model,
                input_ids=entry["inputs"]["input_ids"],
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=sorted(stop_ids),
                temperature=0.0,
                target_kwargs=target_kwargs,
                return_stats=True,
                sequential_target_verification=sequential,
                strict_parallel_margin=strict_margin,
                capture_decision_trace=True,
            )
        wall = time.perf_counter() - started
        tokens = output_ids[0, entry["prompt_len"] :].tolist()
        return {"tokens": tokens, "stats": stats, "wall_seconds": wall}

    algorithm.configure_verify_backend(False)
    results = []
    for entry in prompts:
        reference = entry["tokens"]
        paths = {
            "sequential": run_spec(entry, sequential=True, strict_margin=None, parity_backend=False),
            "parallel": run_spec(entry, sequential=False, strict_margin=None, parity_backend=False),
            "parallel_parity": run_spec(entry, sequential=False, strict_margin=None, parity_backend=True),
            "strict_parallel": run_spec(entry, sequential=False, strict_margin=2.0, parity_backend=False),
        }
        row = {"id": entry["id"], "prompt_len": entry["prompt_len"], "reference_tokens": reference, "paths": {}}
        for name, result in paths.items():
            divergence = first_divergence(result["tokens"], reference, stop_ids)
            stats = result["stats"]
            record = {
                "equivalent": divergence is None,
                "decision_trace": stats.get("decision_trace", []),
                "first_divergence": divergence,
                "new_tokens": len(result["tokens"]),
                "rounds": int(stats["verify_steps"]),
                "accepted_tokens": int(stats["accepted_tokens"]),
                "target_forward_calls": int(stats.get("target_forward_calls", 0)),
                "strict_fallback_rounds": float(stats.get("strict_fallback_rounds", 0.0)),
                "wall_seconds": result["wall_seconds"],
                "token_at_divergence": (result["tokens"][divergence] if divergence is not None else None),
                "reference_token_at_divergence": (reference[divergence] if divergence is not None else None),
            }
            if divergence is not None:
                # The token at output index ``divergence`` sits at absolute
                # position prompt_len + divergence and is decided by the verify
                # forward row whose trace ``position`` equals that absolute
                # position (the row predicts the token *after* its own position).
                position = entry["prompt_len"] + divergence
                record["decision_at_divergence"] = trace_at(stats, position)
                record["decision_next"] = trace_at(stats, position + 1)
            row["paths"][name] = record
        # Paired view of the decision evidence at the first parallel divergence.
        reference_divergence = row["paths"]["parallel"]["first_divergence"]
        if reference_divergence is not None:
            position = entry["prompt_len"] + reference_divergence
            row["divergence_evidence"] = {
                "position": position,
                "sequential": trace_at(paths["sequential"]["stats"], position),
                "parallel": trace_at(paths["parallel"]["stats"], position),
                "parallel_parity": trace_at(paths["parallel_parity"]["stats"], position),
                "strict_parallel": trace_at(paths["strict_parallel"]["stats"], position),
            }
        results.append(row)

    mismatches = [row for row in results if not row["paths"]["parallel"]["equivalent"]]
    case_counts = {"A_parallel_and_parity_agree_wrong": 0, "B_parity_ok_perf_flips": 0, "C_reference_unstable": 0, "other": 0}
    for row in mismatches:
        paths = row["paths"]
        if row.get("divergence_evidence"):
            seq = row["divergence_evidence"]["sequential"]
            par = row["divergence_evidence"]["parallel"]
            margin = min([entry["margin"] for entry in par], default=None)
            if seq and par:
                if any(a["argmax"] != b["argmax"] for a in seq for b in par):
                    case_counts["B_parity_ok_perf_flips"] += 1
                else:
                    case_counts["A_parallel_and_parity_agree_wrong"] += 1
            else:
                case_counts["other"] += 1
        else:
            case_counts["other"] += 1
    if any(not row["target_only_bit_stable"] for row in stability):
        case_counts["C_reference_unstable"] = sum(1 for row in stability if not row["target_only_bit_stable"])

    report = {
        "format": "vlm-dflash-first-divergence-v1",
        "tag": args.tag,
        "checkpoint": args.ckpt,
        "num_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "gpu_name": torch.cuda.get_device_name(device),
        "target_dtype": dtype_name,
        "verify_sdpa_backend_cfg": str(cfg.get("eval", {}).get("verify_sdpa_backend", "parity")),
        "strict_parallel_margin": 2.0,
        "mismatch_counts": {
            "sequential": sum(1 for row in results if not row["paths"]["sequential"]["equivalent"]),
            "parallel": sum(1 for row in results if not row["paths"]["parallel"]["equivalent"]),
            "parallel_parity": sum(1 for row in results if not row["paths"]["parallel_parity"]["equivalent"]),
            "strict_parallel": sum(1 for row in results if not row["paths"]["strict_parallel"]["equivalent"]),
        },
        "case_counts": case_counts,
        "target_only_stability": stability,
        "rows": results,
    }
    out = run_dir / "eval" / f"diag_first_divergence_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=str))
    for row in results:
        print(
            row["id"],
            {name: (path["first_divergence"], path["new_tokens"]) for name, path in row["paths"].items()},
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
