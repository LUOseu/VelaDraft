#!/usr/bin/env python3
"""Score a released drafter export on the tiny synthetic sample (library API).

Same code path as ``vlm-dflash demo --drafter <export>``, without the CLI:

    python examples/score_pretrained.py \
        --target /path/to/Qwen2.5-VL-3B-Instruct \
        --drafter qwen25vl-3b-dflash        # extracted zoo export directory

Prints acceptance length, per-slot acceptance and the token-exactness check
(the speculative output must equal the target's greedy output token-by-token).
"""

from __future__ import annotations

import argparse

from vlm_dflash.config import apply_overrides, dump_effective_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.demo import (
    attach_drafter,
    build_demo_config,
    detect_drafter_algorithm,
    drafter_identity,
    write_demo_source,
)
from vlm_dflash.eval.run import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="local Qwen2.5-VL-3B model directory")
    parser.add_argument("--drafter", required=True, help="export dir, run dir or .pt checkpoint")
    parser.add_argument("--run-dir", default="runs/example_pretrained")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    run_dir = resolve_run_dir({"run": {"dir": args.run_dir}}, None)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool_path, images_dir = write_demo_source(run_dir, 8)

    cfg = build_demo_config(
        target=args.target,
        algorithm=detect_drafter_algorithm(args.drafter) or "dflash",
        run_dir=run_dir,
        pool_path=pool_path,
        images_dir=images_dir,
        num_train=4,
        num_val=4,
        num_prompts=4,
        max_new_tokens=args.max_new_tokens,
        steps=0,
        source_desc="synthetic sample (examples/score_pretrained.py)",
    )
    identity = drafter_identity(args.drafter)
    for key in ("image_min_pixels", "image_max_pixels"):
        if identity.get(key) is not None:
            cfg["target"][key] = int(identity[key])
    cfg = apply_overrides(cfg, None)
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    build_dataset(cfg, run_dir / "data", limit=4)

    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    print(attach_drafter(args.drafter, train_dir))
    cfg["eval"]["checkpoints"] = ["ckpt_final.pt"]

    result = evaluate(cfg, run_dir, contract, num_prompts=4, max_new_tokens=args.max_new_tokens)
    summary = result["runs"]["ckpt_final.pt"]
    exact = result["equivalence_check"]["runs"]["ckpt_final.pt"]
    passed = exact["num_prompts"] - len(exact["failed_ids"])
    slots = ", ".join(f"{value:.2f}" for value in summary["per_slot_acceptance"])
    print(f"AL {summary['mean_al']:.3f} | token-exact {passed}/{exact['num_prompts']} | per-slot [{slots}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
