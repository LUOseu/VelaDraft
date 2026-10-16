#!/usr/bin/env python
"""Emit the R3/R5 run configs for a frozen corpus size N.

Both configs differ only in ``drafter.num_hidden_layers`` (3 vs 5): same data
(hardlinked shared corpus), same seed, same recipe, same optimizer/schedule and
eval protocol. Target feature layers stay [1,17,33] for both (R5 adds drafter
depth only).

The ``dataset`` and ``regen`` blocks are copied verbatim from the data config
that produced the shared corpus: the trainer's stage-digest gate
(``vlm_dflash/cli.py``) hashes those blocks, so a run config with a different
``dataset``/``regen`` section is refused as "produced from different inputs".
``dataset.slice`` therefore keeps describing the *input* slice the shared corpus
was built from (24000 rows); the frozen training member list is what
``shared/regen/target_data.jsonl`` + ``freeze_record.json`` contain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HEADER = """\
# R3/R5 next-round run config (2026-10-15). Frozen corpus N={n} rows
# (regen-kept, image-grouped split seed 42; see shared/regen/freeze_record.json).
# R3 = 3 drafter layers, R5 = 5 -- identical in every other knob, including the
# target feature layers [1,17,33] and the dataset/regen blocks below, which are
# copied verbatim from the data config so the stage-digest gate accepts the
# hardlinked shared corpus. This is the round's candidate recipe, not a claim
# about an official optimum.
"""


def run_block(layers: int) -> dict:
    return {
        "algorithm": {
            "name": "dfly",
            "train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.25},
            "dfly": {
                "context_layer_ids": [1, 17, 33],
                "fusion_trainable": True,
                "fusion_init": "distance",
                "enable_hidden_correction": True,
                "hidden_correction_intermediate_size": None,
            },
        },
        "target": {
            "model": "/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct",
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "device": "cuda:0",
            "local_files_only": True,
            "image_min_pixels": 200704,
            "image_max_pixels": 1003520,
        },
        "capture": {
            "seq_length": 2048,
            "shard_size": 32,
            "dtype": "bfloat16",
            "store_final_hidden": True,
            "visual_path_check": True,
            "visual_check_samples": 2,
        },
        "drafter": {
            "num_hidden_layers": layers,
            "block_size": 8,
            "num_anchors": 256,
            "loss_decay_gamma": 7.0,
            "capture_layer_ids": [1, 17, 33],
            "mask_token_id": 151669,
            "attention_backend": "flex_attention",
            "spatial_rope_enabled": True,
            "spatial_rope_mode": "replace",
            "spatial_rope_sections": [16, 24, 24],
        },
        "train": {
            "feature_mode": "online",
            "online_batch_size": 8,
            "online_feature_cache_gb": 32,
            "micro_batch_size": 8,
            "grad_accumulation_steps": 1,
            "num_epochs": 3,
            "save_every_fraction_of_epoch": 0.5,
            "milestone_snapshots": True,
            "resume_state_dir": "/dev/shm/r3r5_{tag}/train_state",
            "lr": 3.0e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_ratio": 0.04,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
            "log_every_steps": 1,
            "seed": 42,
            "device": "cuda:0",
            "overfit_sanity": {
                "enabled": True,
                "num_samples": 4,
                "steps": 200,
                "ce_drop": 0.6,
                "slot0_acc": 0.9,
            },
        },
        "eval": {
            "verification": "sequential",
            "strict_parallel_margin": 2.0,
            "num_prompts": 64,
            "max_new_tokens": 128,
            "sequential_target_verification": True,
            "verify_sdpa_backend": "parity",
            "parallel_diagnostic_prompts": 0,
            "checkpoints": ["ckpt_final.pt"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="frozen row count")
    parser.add_argument("--data-config", default="scripts/scale/r3r5_data_config.yaml")
    parser.add_argument("--outdir", default="configs/scale")
    args = parser.parse_args()

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    shared_blocks = {"dataset": data_cfg["dataset"], "regen": data_cfg["regen"]}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, layers in (("r3", 3), ("r5", 5)):
        cfg = dict(run_block(layers))
        cfg["run"] = {"name": f"qwen25vl3b_dfly_{tag}_n{args.n}", "dir": f"runs/r3r5_n{args.n}/{tag}"}
        # key order: run, dataset, regen, target, capture, drafter, train, eval
        ordered = {
            "algorithm": cfg["algorithm"],
            "run": cfg["run"],
            "dataset": shared_blocks["dataset"],
            "regen": shared_blocks["regen"],
            "target": cfg["target"],
            "capture": cfg["capture"],
            "drafter": cfg["drafter"],
            "train": {k: (v.format(tag=tag) if isinstance(v, str) else v) for k, v in cfg["train"].items()},
            "eval": cfg["eval"],
        }
        body = yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False)
        path = outdir / f"{tag}_dfly_n{args.n}.yaml"
        path.write_text(HEADER.format(n=args.n) + body)
        written.append(str(path))
    print("wrote " + ", ".join(written))


if __name__ == "__main__":
    main()
