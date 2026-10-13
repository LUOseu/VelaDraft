#!/usr/bin/env python
"""Measure target logit margins at vLLM DFlash spec-decode divergences.

Reads a ``vllm_bridge_check.py`` result (target-only + dflash greedy
transcripts), picks the prompt positions where the two transcripts first
differ, teaches-force the target-only prefix through the repo's own target
implementation, and reports, at each divergence step: the top-1/top-2 logit
margin, the rank of the token vLLM-dflash emitted, and the logit gap between
the two. Tiny margins are the signature of the documented bf16 near-tie flips
under block-parallel verification; large margins would instead point at an
integration bug.

    python scripts/probe_divergence_margin.py --run-dir /tmp/wp5/run_1d \\
        --bridge-json /tmp/wp5/bridge_check.json --max-cases 4 \\
        --out /tmp/wp5/divergence_margins.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.contract import ensure_contract  # noqa: E402
from vlm_dflash.eval.run import load_eval_rows, load_target  # noqa: E402
from vlm_dflash.target.prompts import encode_prompt  # noqa: E402
from vlm_dflash.utils import read_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run dir with config.json/contract.json")
    parser.add_argument("--bridge-json", required=True, help="vllm_bridge_check.py result")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bridge = read_json(Path(args.bridge_json))
    cfg = read_json(run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    rows, media_dir = load_eval_rows(None, run_dir)
    rows = rows[: args.num_prompts]

    target_records = bridge["target_only"]["records"]
    draft_records = bridge["dflash"]["records"]
    cases = []
    for index, (base, spec) in enumerate(zip(target_records, draft_records)):
        if base["token_ids"] == spec["token_ids"]:
            continue
        divergence = next(
            (
                position
                for position, (left, right) in enumerate(zip(base["token_ids"], spec["token_ids"]))
                if left != right
            ),
            min(len(base["token_ids"]), len(spec["token_ids"])),
        )
        cases.append((index, divergence))
    cases = cases[: args.max_cases]
    print(f"probing {len(cases)} divergences: {cases}")

    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    results = []
    for index, divergence in cases:
        row = rows[index]
        image = Image.open(media_dir / row["images"][0]).convert("RGB")
        inputs = encode_prompt(processor, row["conversations"][0]["value"], image)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=divergence + 1,
                do_sample=False,
                repetition_penalty=1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )
        repo_tokens = outputs.sequences[0, prompt_len:].tolist()
        vllm_target = target_records[index]["token_ids"]
        prefix_match = repo_tokens[:divergence] == vllm_target[:divergence]
        logits = outputs.scores[divergence][0].float()
        top2 = torch.topk(logits, 2)
        vllm_token = draft_records[index]["token_ids"][divergence]
        vllm_logit = float(logits[vllm_token])
        results.append(
            {
                "row_id": row.get("id"),
                "index": index,
                "divergence": divergence,
                "prompt_len": prompt_len,
                "vllm_prompt_len": target_records[index]["prompt_len"],
                "prefix_reproduced": prefix_match,
                "target_top1": int(top2.indices[0]),
                "target_top1_logit": float(top2.values[0]),
                "margin_top1_top2": float(top2.values[0] - top2.values[1]),
                "vllm_dflash_token": int(vllm_token),
                "vllm_dflash_logit": vllm_logit,
                "logit_gap_top1_minus_vllm": float(top2.values[0] - vllm_logit),
                "vllm_token_rank": int((logits > vllm_logit).sum().item()) + 1,
            }
        )
        print(json.dumps(results[-1], indent=2))
    write_json(Path(args.out), {"format": "vlm-dflash-divergence-margin-v1", "cases": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
