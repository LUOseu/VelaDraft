#!/usr/bin/env python3
"""Train a DFlash drafter end to end from a small image-text JSONL pool.

    python examples/train_from_jsonl.py \
        --target /path/to/Qwen2.5-VL-3B-Instruct \
        --jsonl data/rows.jsonl --images-dir data/images --steps 60

This is the same staged pipeline as ``vlm-dflash train``
(prepare -> target-native regen -> capture -> train -> eval), written out as a
script you can read top to bottom. The target is loaded for regen/capture
only; training reads the captured feature store.

JSONL rows: one image + one question/answer pair, any of the shapes accepted
by ``vlm_dflash.data.schema`` (e.g. ``{"question", "answer", "image"}``).
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


def build_config(args) -> dict:
    return {
        "algorithm": "dflash",
        "run": {"dir": args.run_dir},
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
        "regen": {"max_new_tokens": 160, "min_response_tokens": 16, "batch_size": 8, "temperature": 0.0},
        "capture": {"seq_length": 1024, "shard_size": 16, "dtype": "bfloat16", "visual_path_check": True},
        "drafter": {
            "num_hidden_layers": 3,
            "block_size": 8,
            "num_anchors": 64,
            "loss_decay_gamma": 7.0,
            # Qwen2.5-VL default; the contract re-validates it against the
            # tokenizer at runtime (no silent fallback).
            "mask_token_id": 151669,
            "attention_backend": "flex_attention",
        },
        "train": {
            "micro_batch_size": 4,
            "num_epochs": 1,
            "lr": 6.0e-4,
            "betas": [0.9, 0.95],
            "warmup_ratio": 0.04,
            "max_grad_norm": 1.0,
            "log_every_steps": 1,
            "seed": 42,
            "device": "cuda:0",
            "overfit_sanity": {"enabled": False},
        },
        "eval": {
            "num_prompts": 4,
            "max_new_tokens": 32,
            "checkpoints": ["ckpt_final.pt", "ckpt_init.pt"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--jsonl", required=True, help="image-text pool (JSONL)")
    parser.add_argument("--images-dir", required=True, help="image root for the JSONL")
    parser.add_argument("--run-dir", default="runs/example_from_jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="number of training rows to use; the validation split (val_size=4 here) is drawn on top, so the pool needs at least limit + 4 valid rows",
    )
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    run_dir = resolve_run_dir({"run": {"dir": args.run_dir}}, None)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = apply_overrides(build_config(args), None)
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)

    build_dataset(cfg, run_dir / "data", limit=args.limit)
    run_regen(cfg, run_dir, contract, limit=args.limit)
    run_capture(cfg, run_dir, contract, limit=args.limit)
    summary = train_dflash(cfg, run_dir, contract, steps=args.steps)
    print(f"training: {summary['steps']} steps, final loss {summary['final_loss']:.3f}")

    result = evaluate(cfg, run_dir, contract, num_prompts=4, max_new_tokens=32)
    trained = result["runs"]["ckpt_final.pt"]["mean_al"]
    baseline = result["runs"]["ckpt_init.pt"]["mean_al"]
    exact = result["equivalence_check"]["runs"]["ckpt_final.pt"]
    passed = exact["num_prompts"] - len(exact["failed_ids"])
    print(f"AL {trained:.3f} (random-init baseline {baseline:.3f}) | token-exact {passed}/{exact['num_prompts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
