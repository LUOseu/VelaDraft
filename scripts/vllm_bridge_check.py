#!/usr/bin/env python
"""vLLM bridge check: greedy token-exactness + acceptance for a DFlash draft.

Phase 1 runs the target model alone (greedy); phase 2 re-runs the same prompts
with a converted DFlash draft attached through vLLM's native ``dflash``
speculative-decoding method. The script asserts the two greedy transcripts
match token-for-token and reports vLLM's speculative metrics (acceptance
length and per-position acceptance) next to each phase's wall time.

Requires a vLLM with the native DFlash method (pinned: 0.22.1). Run on a GPU
host, e.g.:

    python scripts/vllm_bridge_check.py --target <Qwen2.5-VL-3B> \\
        --draft /tmp/wp5/vllm_draft --source runs/.../data/val.jsonl \\
        --images-dir runs/.../data/images --num-prompts 32 \\
        --max-new-tokens 128 --min-pixels 200704 --max-pixels 1003520 \\
        --out /tmp/wp5/bridge_check.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_conversations(rows: list[dict], images_dir: Path) -> list[list[dict]]:
    from PIL import Image

    conversations = []
    for row in rows:
        question = str(row["conversations"][0]["value"]).replace("<image>", "").strip()
        image_path = images_dir / row["images"][0]
        image = Image.open(image_path).convert("RGB")
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        )
    return conversations


def _spec_metrics(all_metrics) -> dict:
    """Aggregate the spec-decode counters vLLM exposes for the acceptance math."""

    totals: dict[str, float] = {}
    per_pos: dict[str, float] = {}
    for metric in all_metrics or []:
        name = getattr(metric, "name", "")
        if "spec_decode" not in name:
            continue
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            # vLLM exposes this one as a Vector (one counter per token position).
            values = getattr(metric, "values", None)
            if values is None:
                continue
            for position, value in enumerate(values):
                per_pos[str(position)] = per_pos.get(str(position), 0.0) + float(value)
        else:
            totals[name] = totals.get(name, 0.0) + float(getattr(metric, "value", 0.0) or 0.0)
    summary: dict[str, float | list] = {"raw": totals}
    drafts = totals.get("vllm:spec_decode_num_drafts", 0.0)
    accepted = totals.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    draft_tokens = totals.get("vllm:spec_decode_num_draft_tokens", 0.0)
    if drafts > 0:
        summary["acceptance_length"] = 1.0 + accepted / drafts
        summary["acceptance_rate"] = accepted / draft_tokens if draft_tokens > 0 else None
        positions = sorted(per_pos, key=lambda key: int(key))
        summary["per_position_acceptance"] = [per_pos[key] / drafts for key in positions]
    return summary


def run_phase(
    *,
    model_path: str,
    draft_dir: str | None,
    speculative_tokens: int,
    conversations: list,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    min_pixels: int | None,
    max_pixels: int | None,
    label: str,
) -> dict:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=False,
        seed=0,
        limit_mm_per_prompt={"image": 1},
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if min_pixels is not None or max_pixels is not None:
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        kwargs["mm_processor_kwargs"] = processor_kwargs
    if draft_dir is not None:
        kwargs.update(spec_method="dflash", spec_model=str(draft_dir), spec_tokens=int(speculative_tokens))

    print(f"[{label}] loading vLLM ({'dflash' if draft_dir else 'target-only'})", flush=True)
    llm = LLM(**kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    print(f"[{label}] generating {len(conversations)} prompts", flush=True)
    started = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
    wall = time.perf_counter() - started
    records = []
    for output in outputs:
        completed = output.outputs[0]
        records.append(
            {
                "request_id": output.request_id,
                "prompt_len": len(output.prompt_token_ids),
                "token_ids": list(completed.token_ids),
                "finish_reason": completed.finish_reason,
            }
        )
    spec = _spec_metrics(llm.get_metrics())
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(2)
    return {
        "label": label,
        "wall_seconds": wall,
        "num_prompts": len(records),
        "total_new_tokens": sum(len(r["token_ids"]) for r in records),
        "records": records,
        "spec_decode": spec if draft_dir is not None else None,
    }


def compare(target_phase: dict, draft_phase: dict) -> dict:
    exact = 0
    divergences = []
    for base, spec in zip(target_phase["records"], draft_phase["records"]):
        if base["token_ids"] == spec["token_ids"]:
            exact += 1
            continue
        first = next(
            (index for index, (a, b) in enumerate(zip(base["token_ids"], spec["token_ids"])) if a != b),
            min(len(base["token_ids"]), len(spec["token_ids"])),
        )
        divergences.append(
            {
                "request_id": base["request_id"],
                "first_divergence": first,
                "target_tokens": len(base["token_ids"]),
                "draft_tokens": len(spec["token_ids"]),
            }
        )
    return {
        "num_prompts": len(target_phase["records"]),
        "token_exact": exact,
        "all_exact": exact == len(target_phase["records"]),
        "divergences": divergences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", default=None, help="converted speculators draft dir (enables phase 2)")
    parser.add_argument("--speculative-tokens", type=int, default=7)
    parser.add_argument("--source", required=True, help="sharegpt-style rows (repo val split)")
    parser.add_argument("--images-dir", default=None, help="images dir (default: alongside --source)")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    images_dir = Path(args.images_dir) if args.images_dir else source.parent / "images"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.num_prompts]
    conversations = build_conversations(rows, images_dir)

    import vllm

    phase_target = run_phase(
        model_path=args.target,
        draft_dir=None,
        speculative_tokens=args.speculative_tokens,
        conversations=conversations,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        label="target-only",
    )
    result = {
        "format": "vlm-dflash-vllm-bridge-check-v1",
        "vllm_version": vllm.__version__,
        "target": args.target,
        "draft": args.draft,
        "source": str(source),
        "num_prompts": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "target_only": phase_target,
    }
    if args.draft:
        phase_draft = run_phase(
            model_path=args.target,
            draft_dir=args.draft,
            speculative_tokens=args.speculative_tokens,
            conversations=conversations,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            label="dflash",
        )
        result["dflash"] = phase_draft
        result["equivalence"] = compare(phase_target, phase_draft)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("target_only", "dflash")}, indent=2))
    print(
        f"target-only: {phase_target['num_prompts']} prompts, "
        f"{phase_target['total_new_tokens']} tokens, {phase_target['wall_seconds']:.1f}s"
    )
    if args.draft:
        print(
            f"dflash: {result['dflash']['total_new_tokens']} tokens, "
            f"{result['dflash']['wall_seconds']:.1f}s, spec={result['dflash']['spec_decode']}"
        )
        return 0 if result["equivalence"]["all_exact"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
