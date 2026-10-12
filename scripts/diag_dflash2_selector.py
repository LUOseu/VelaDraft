#!/usr/bin/env python3
"""Dev diagnostic: how much does the DFlash2 candidate selector change serving?

Runs one trained checkpoint over the same prompts through the same
sequential-verification path twice:

* ``selector``: the production walk - unary top-K candidates reranked by the
  transition scores, predecessor-chained left to right;
* ``unary``:    the same candidate set ranked by unary logits only (the reranker
  contribution removed without touching the architecture).

Reports AL, per-slot acceptance and the fraction of proposal positions where the
selector's pick differs from the unary argmax. Not part of the pipeline; used to
interpret a DFlash2 eval result (``docs/algorithms/dflash2.md``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.config import apply_overrides, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.eval.accept import summarize_samples
from vlm_dflash.eval.equivalence import sequences_equal
from vlm_dflash.eval.run import _stops, load_eval_rows, prepare_prompts
from vlm_dflash.target.load import load_target
from vlm_dflash.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default="ckpt_final.pt")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", default=None, help="JSON output path (default: runs/<run>/eval/selector_ablation.json)")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    algorithm = get_algorithm(cfg.get("algorithm"))
    if algorithm.name != "dflash2":
        raise SystemExit(f"This diagnostic is DFlash2-specific; config resolves algorithm={algorithm.name!r}.")
    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    draft_dtype = getattr(torch, str(cfg["target"].get("dtype", "bfloat16")), torch.bfloat16)
    block_size = int(cfg["drafter"]["block_size"])
    stop_ids = set(_stops(model))
    rows, media_dir = load_eval_rows(None, run_dir)
    rows = rows[: int(args.num_prompts)]
    prompts = prepare_prompts(
        model, processor, rows, media_dir, max_new_tokens=int(args.max_new_tokens), device=device, label="selector-ablation"
    )

    draft, _ = algorithm.load_decoder(run_dir / "train" / args.checkpoint)
    draft.to(device=device, dtype=draft_dtype)
    draft.eval()
    selector = draft.candidate_selector
    original_select = selector.select

    counters = {"positions": 0, "overrides": 0}
    mode = {"name": "selector"}

    def instrumented_select(unary_logits, hidden_states, predecessor_ids, top_k=None):
        candidate_ids, scores = original_select(unary_logits, hidden_states, predecessor_ids, top_k)
        unary_scores = unary_logits.gather(-1, candidate_ids)
        counters["positions"] += int(candidate_ids.numel() // candidate_ids.shape[-1])
        counters["overrides"] += int(
            (scores.argmax(dim=-1) != unary_scores.argmax(dim=-1)).sum()
        )
        return candidate_ids, (unary_scores if mode["name"] == "unary" else scores)

    selector.select = instrumented_select  # instance attribute; unbound wrapper
    results = {}
    for name in ("selector", "unary"):
        mode["name"] = name
        counters["positions"] = 0
        counters["overrides"] = 0
        samples = []
        for entry in prompts:
            inputs = entry["inputs"]
            target_kwargs = {
                "pixel_values": inputs.get("pixel_values"),
                "image_grid_thw": inputs.get("image_grid_thw"),
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            }
            target_kwargs = {key: value for key, value in target_kwargs.items() if value is not None}
            started = time.perf_counter()
            with torch.inference_mode():
                output_ids, stats = draft.spec_generate(
                    target=model,
                    input_ids=inputs["input_ids"],
                    max_new_tokens=int(args.max_new_tokens),
                    stop_token_ids=sorted(stop_ids),
                    temperature=0.0,
                    target_kwargs=target_kwargs,
                    return_stats=True,
                    sequential_target_verification=True,
                )
            wall = time.perf_counter() - started
            prompt_len = entry["prompt_len"]
            spec_tokens = output_ids[0, prompt_len:].tolist()
            samples.append(
                {
                    "id": entry["id"],
                    "new_tokens": len(spec_tokens),
                    "verify_steps": int(stats["verify_steps"]),
                    "accepted_tokens": int(stats["accepted_tokens"]),
                    "draft_tokens": int(stats["draft_tokens"]),
                    "acceptance_events": stats["acceptance_events"],
                    "wall_seconds": wall,
                    # Same gate as the canonical eval (stop-token aware), so this
                    # diagnostic cannot disagree with eval_result.json by accident.
                    "matches_target": sequences_equal(spec_tokens, entry["tokens"], stop_ids),
                }
            )
        summary = summarize_samples(samples, block_size)
        summary["selector_overrides"] = counters["overrides"]
        summary["selector_positions"] = counters["positions"]
        summary["selector_override_rate"] = (
            counters["overrides"] / counters["positions"] if counters["positions"] else None
        )
        summary["token_exact"] = all(sample["matches_target"] for sample in samples)
        summary.pop("acceptance_events", None)
        results[name] = summary
        print(
            f"{name}: AL={summary['mean_al']:.4f} per_slot={['%.4f' % value for value in summary['per_slot_acceptance']]} "
            f"override_rate={summary['selector_override_rate']} token_exact={summary['token_exact']}"
        )

    out_path = Path(args.out) if args.out else run_dir / "eval" / "selector_ablation.json"
    write_json(
        out_path,
        {
            "format": "vlm-dflash-selector-ablation-v1",
            "checkpoint": args.checkpoint,
            "num_prompts": len(prompts),
            "max_new_tokens": int(args.max_new_tokens),
            "verification": "sequential",
            "runs": results,
            "note": (
                "unary = same unary top-K candidate set ranked by unary logits (reranker removed); "
                "selector_override_rate counts proposal positions where the transition scores change "
                "the argmax inside the candidate set."
            ),
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
