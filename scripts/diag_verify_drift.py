#!/usr/bin/env python3
"""Dev diagnostic: how far does parallel block verification drift from step mode?

Runs one prompt through ``spec_generate`` twice (sequential and parallel target
verification) with a forward hook on the target, then compares the prefill
logits and the first verify-step logits computed at the *same position with the
same inputs*. The delta is the bf16 block-vs-step kernel drift that makes an
unguarded parallel verify flip near-tie argmax decisions; it is the number
behind the numbers quoted in ``docs/REFERENCE.md``.

    CUDA_VISIBLE_DEVICES=0 python scripts/diag_verify_drift.py --run-dir runs/qwen3vl8b_dflash

Not part of the pipeline; kept because it documents the parallel-verification
numerics claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from vlm_dflash.config import apply_overrides, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.data.sharegpt import load_sharegpt, row_image_path, row_question
from vlm_dflash.dflash.factory import load_checkpoint
from vlm_dflash.eval.run import encode_prompt
from vlm_dflash.target.load import load_target


def capture_verify_logits(model, draft, inputs, sequential: bool, max_new_tokens: int):
    records: list[torch.Tensor] = []

    def hook(module, args, kwargs, output):
        logits = getattr(output, "logits", None)
        if logits is not None:
            records.append(logits.detach().float().cpu())

    handle = model.register_forward_hook(hook, with_kwargs=True)
    target_kwargs = {
        key: inputs[key]
        for key in ("pixel_values", "image_grid_thw", "attention_mask", "mm_token_type_ids")
        if key in inputs
    }
    with torch.inference_mode():
        output_ids, stats = draft.spec_generate(
            target=model,
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            stop_token_ids=None,
            temperature=0.0,
            target_kwargs=target_kwargs,
            return_stats=True,
            sequential_target_verification=sequential,
        )
    handle.remove()
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_vl_8b.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--row", type=int, default=0, help="which val prompt to probe")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    rows, media_dir = load_sharegpt(run_dir / "data" / "val.jsonl")
    row = rows[args.row]
    image = Image.open(row_image_path(row, media_dir)).convert("RGB")
    inputs = encode_prompt(processor, row_question(row), image)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    draft, _ = load_checkpoint(run_dir / "train" / "ckpt_final.pt")
    draft.to(device=device, dtype=torch.bfloat16).eval()

    seq_records, seq_stats = capture_verify_logits(model, draft, inputs, True, args.max_new_tokens)
    par_records, par_stats = capture_verify_logits(model, draft, inputs, False, args.max_new_tokens)

    prefill_delta = (seq_records[0] - par_records[0]).abs().max().item()
    print(f"prompt id           : {row.get('id')}")
    print(f"prefill max|delta|  : {prefill_delta:.4f}  (must be 0)")
    positions = min(par_records[1].shape[1], len(seq_records) - 1)
    for position in range(positions):
        step = par_records[1][0, position]
        if position + 1 >= len(seq_records):
            break
        step_reference = seq_records[position + 1][0, 0]
        delta = (step - step_reference).abs()
        margin_seq = torch.topk(step_reference, 2).values
        margin_par = torch.topk(step, 2).values
        print(
            f"position {position}: max|delta|={delta.max():.4f} mean={delta.mean():.5f} "
            f"margin_seq={(margin_seq[0] - margin_seq[1]).item():.4f} "
            f"margin_par={(margin_par[0] - margin_par[1]).item():.4f} "
            f"argmax_equal={int(step.argmax()) == int(step_reference.argmax())}"
        )
    print(f"verify_steps seq/par: {seq_stats['verify_steps']:.0f}/{par_stats['verify_steps']:.0f}")


if __name__ == "__main__":
    main()
