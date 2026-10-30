"""Build the DFlash2 draft config/model/trainer from the target contract.

The draft backbone is the shared DFlash config (Qwen3-style stack, three layers,
MRoPE converted to the draft's 1-D rope); DFlash2 adds the conv/selector knobs
under ``dflash_config.dflash2`` so the serialized draft config alone rebuilds the
right architecture (checkpoint, eval, export, visual check).
"""

from __future__ import annotations

import torch

DEFAULTS: dict = {
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "selector_rank": 256,
    "selector_top_k": 16,
    "selector_loss_alpha": 1.0,
    "kd_chunk_size": 1024,
}


def resolve_dflash2_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.dflash2`` (defaults fill gaps)."""
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("dflash2") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.dflash2 must be a mapping of DFlash2 knobs.")
        resolved.update(block)
    return resolved


def build_draft_config(contract: dict, cfg: dict):
    """DFlash draft config + the DFlash2 architecture fields."""
    from ..dflash.factory import build_draft_config as build_dflash_draft_config

    dflash2 = resolve_dflash2_config(cfg)
    config = build_dflash_draft_config(contract, cfg)
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    dflash_config["dflash2"] = {
        "conv_kernel_size": int(dflash2["conv_kernel_size"]),
        "conv_group_size": int(dflash2["conv_group_size"]),
        "selector_rank": int(dflash2["selector_rank"]),
        "selector_top_k": int(dflash2["selector_top_k"]),
    }
    config.dflash_config = dflash_config

    block_size = int(config.block_size)
    if int(dflash2["conv_kernel_size"]) > block_size:
        raise ValueError(
            f"algorithm.dflash2.conv_kernel_size ({dflash2['conv_kernel_size']}) cannot exceed "
            f"drafter.block_size ({block_size})."
        )
    hidden_size = int(config.hidden_size)
    group_size = int(dflash2["conv_group_size"])
    if hidden_size % group_size:
        raise ValueError(
            f"algorithm.dflash2.conv_group_size ({group_size}) must divide the draft hidden size "
            f"({hidden_size})."
        )
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module
    from .model import Qwen3DFlash2DraftModel
    from .trainer import DFlash2TrainerModule

    dflash2 = resolve_dflash2_config(cfg)
    # Execution-only switches live under ``drafter`` (like recipe_row_compaction),
    # outside the training-plan digest: Reference and Optimized arms of one
    # plan resume from the same state.
    drafter = cfg.get("drafter") or {}
    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        model_class=Qwen3DFlash2DraftModel,
        module_class=DFlash2TrainerModule,
        module_kwargs={
            # DFlash slot convention (slots 1..block_size-1); the anchor is the
            # verifier's bonus token, exactly like upstream DFlash2.
            "sample_from_anchor": False,
            "selector_loss_alpha": float(dflash2["selector_loss_alpha"]),
            "kd_chunk_size": int(dflash2["kd_chunk_size"]),
            "teacher_projection_reuse": bool(drafter.get("dflash2_teacher_projection_reuse", False)),
            "scoring_row_compaction": bool(drafter.get("dflash2_scoring_row_compaction", False)),
            "fused_teacher_kl": bool(drafter.get("dflash2_fused_teacher_kl", False)),
        },
    )


def load_checkpoint(path, map_location="cpu"):
    """DFlash2 checkpoints load through the shared DFlash container."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint
    from .model import Qwen3DFlash2DraftModel

    return load_dflash_checkpoint(path, map_location=map_location, model_class=Qwen3DFlash2DraftModel)
