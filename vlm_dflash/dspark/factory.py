"""Build DSpark draft configs / trainer modules / decoders from the run config."""

from __future__ import annotations

from ..dflash.factory import build_draft_config as build_dflash_draft_config

__all__ = [
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dspark_config",
]

DEFAULTS: dict = {
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "confidence_head_alpha": 1.0,
    # Upstream DSpark default. True drafts the anchor slot too (block_size
    # proposals per round); False keeps the DFlash slot convention
    # (block_size - 1 proposals) and is the DFlash-compatible ablation.
    "sample_from_anchor": True,
}


def resolve_dspark_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.dspark`` (defaults fill gaps)."""
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("dspark") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.dspark must be a mapping of DSpark knobs.")
        resolved.update(block)
    return resolved


def build_draft_config(contract: dict, cfg: dict):
    """DFlash draft config plus the DSpark head fields in ``dflash_config``."""
    config = build_dflash_draft_config(contract, cfg)
    dspark = resolve_dspark_config(cfg)
    config.architectures = ["Qwen3DSparkDraftModel"]
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    dflash_config.update(
        {
            "markov_rank": int(dspark["markov_rank"]),
            "markov_head_type": str(dspark["markov_head_type"]),
            "enable_confidence_head": bool(dspark["enable_confidence_head"]),
            "confidence_head_with_markov": bool(dspark["confidence_head_with_markov"]),
            "sample_from_anchor": bool(dspark["sample_from_anchor"]),
        }
    )
    config.dflash_config = dflash_config
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module
    from ..objectives import resolve_recipe
    from .model import Qwen3DSparkDraftModel
    from .trainer import DSparkTrainerModule

    dspark = resolve_dspark_config(cfg)
    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        model_class=Qwen3DSparkDraftModel,
        module_class=DSparkTrainerModule,
        module_kwargs={
            "markov_rank": int(dspark["markov_rank"]),
            "markov_head_type": str(dspark["markov_head_type"]),
            "enable_confidence_head": bool(dspark["enable_confidence_head"]),
            "confidence_head_with_markov": bool(dspark["confidence_head_with_markov"]),
            "confidence_head_alpha": float(dspark["confidence_head_alpha"]),
            "sample_from_anchor": bool(dspark["sample_from_anchor"]),
            "train_recipe": resolve_recipe(cfg, "dspark"),
        },
    )


def load_checkpoint(path, map_location="cpu"):
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint
    from .model import Qwen3DSparkDraftModel

    return load_dflash_checkpoint(path, map_location=map_location, model_class=Qwen3DSparkDraftModel)
