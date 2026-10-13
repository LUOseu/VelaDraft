#!/usr/bin/env python3
"""One capture, two algorithms: compare drafters trained from the same features.

    python examples/reuse_features_compare.py \
        --target /path/to/Qwen2.5-VL-3B-Instruct \
        --jsonl data/rows.jsonl --images-dir data/images --steps 20

Step 1 runs the expensive part once (target-native regen + hidden-state
capture). Step 2 trains DFlash and DSpark from that captured store via
``train_dflash(..., features_dir=...)`` -- the target is never loaded again.
The trainer re-verifies the store's target identity and saved head before
training, and records the reuse (directory, sample count, seq length, dtype,
manifest digest) in each run's ``provenance.json``.
"""

from __future__ import annotations

import argparse

from vlm_dflash.config import apply_overrides, dump_effective_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.dflash.train import train_dflash
from vlm_dflash.eval.run import evaluate
from vlm_dflash.target.capture import run_capture
from vlm_dflash.target.regen import run_regen

ALGORITHMS = ("dflash", "dspark")


def build_config(args, algorithm: str, run_dir) -> dict:
    return {
        "algorithm": algorithm,
        "run": {"dir": str(run_dir)},
        "target": {
            "model": args.target,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "device": "cuda:0",
            "local_files_only": True,
        },
        "dataset": {
            "source": "jsonl",
            "path": args.jsonl,
            "images_dir": args.images_dir,
            "slice": args.limit,
            "val_size": 4,
            "seed": 42,
            "min_answer_chars": 0,
        },
        "regen": {"max_new_tokens": 96, "min_response_tokens": 8, "batch_size": 8, "temperature": 0.0},
        "capture": {"seq_length": 1024, "shard_size": 8, "dtype": "bfloat16", "visual_path_check": True},
        "drafter": {
            "num_hidden_layers": 3,
            "block_size": 8,
            "num_anchors": 32,
            "loss_decay_gamma": 7.0,
            "mask_token_id": 151669,  # Qwen2.5-VL default; re-validated at runtime
            "attention_backend": "flex_attention",
        },
        "train": {
            "micro_batch_size": 4,
            "num_epochs": 1,
            "lr": 6.0e-4,
            "warmup_ratio": 0.0,
            "max_grad_norm": 1.0,
            "log_every_steps": 1,
            "seed": 42,
            "device": "cuda:0",
            "overfit_sanity": {"enabled": False},
        },
        "eval": {"num_prompts": 4, "max_new_tokens": 32, "checkpoints": ["ckpt_final.pt"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--run-dir", default="runs/example_reuse_compare")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    parent = resolve_run_dir({"run": {"dir": args.run_dir}}, None)
    parent.mkdir(parents=True, exist_ok=True)

    # Step 1: one capture shared by every algorithm below.
    cfg = apply_overrides(build_config(args, "dflash", parent), None)
    dump_effective_config(cfg, parent / "config.json")
    contract = ensure_contract(cfg, parent)
    build_dataset(cfg, parent / "data", limit=args.limit)
    run_regen(cfg, parent, contract, limit=args.limit)
    run_capture(cfg, parent, contract, limit=args.limit)
    features_dir = parent / "features"
    print(f"shared capture: {features_dir}")

    # Step 2: train each algorithm from the same store (no target in the loop).
    scores = {}
    for algorithm in ALGORITHMS:
        run_dir = parent / algorithm
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = apply_overrides(build_config(args, algorithm, run_dir), None)
        dump_effective_config(cfg, run_dir / "config.json")
        build_dataset(cfg, run_dir / "data", limit=args.limit)
        summary = train_dflash(cfg, run_dir, contract, steps=args.steps, features_dir=features_dir)
        result = evaluate(cfg, run_dir, contract, num_prompts=4, max_new_tokens=32)
        scores[algorithm] = result["runs"]["ckpt_final.pt"]["mean_al"]
        print(f"{algorithm}: {summary['steps']} steps, final loss {summary['final_loss']:.3f}, AL {scores[algorithm]:.3f}")

    print("acceptance length: " + " | ".join(f"{name} {al:.3f}" for name, al in scores.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
