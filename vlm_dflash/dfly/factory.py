"""Build DFly draft configs / trainer modules / decoders from the run config."""

from __future__ import annotations

from ..contract import contract_capture_layer_ids
from ..dflash.factory import build_draft_config as build_dflash_draft_config
from ..dflare.factory import resolve_context_layer_ids

__all__ = [
    "DEFAULTS",
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dfly_config",
]

DEFAULTS: dict = {
    # Which captured target layers the base FC + fusion residual consume
    # (absolute layer indices, ordered); ``None`` -> every captured layer.
    "context_layer_ids": None,
    "fusion_trainable": True,
    "fusion_init": "distance",
    # Upstream AngelSpec default config ships enable_hidden_correction=true.
    "enable_hidden_correction": True,
    # None -> SwiGLU bottleneck width = hidden_size (upstream default).
    "hidden_correction_intermediate_size": None,
}


def resolve_dfly_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.dfly`` (defaults fill gaps)."""
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("dfly") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.dfly must be a mapping of DFly knobs.")
        resolved.update(block)
    return resolved


def build_draft_config(contract: dict, cfg: dict):
    """DFlash draft config with DFly's fusion + correction fields."""
    config = build_dflash_draft_config(contract, cfg)
    dfly = resolve_dfly_config(cfg)
    capture = contract_capture_layer_ids(contract)
    context = resolve_context_layer_ids(dfly, capture)
    fusion_init = str(dfly["fusion_init"])
    if fusion_init not in {"distance", "uniform"}:
        raise ValueError(f"algorithm.dfly.fusion_init must be 'distance' or 'uniform', got {fusion_init!r}.")
    intermediate = dfly["hidden_correction_intermediate_size"]
    if intermediate is not None and int(intermediate) < 1:
        raise ValueError(
            f"algorithm.dfly.hidden_correction_intermediate_size must be >= 1, got {intermediate!r}."
        )
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    dflash_config.update(
        {
            "target_layer_ids": context,
            "context_layer_ids": context,
            # DFly keeps DFlash's shared context/draft K/V projections.
            "separate_context_kv": False,
            "sample_from_anchor": False,
            "fusion_trainable": bool(dfly["fusion_trainable"]),
            "fusion_init": fusion_init,
            "enable_hidden_correction": bool(dfly["enable_hidden_correction"]),
            "hidden_correction_intermediate_size": (
                None if intermediate is None else int(intermediate)
            ),
        }
    )
    config.dflash_config = dflash_config
    config.architectures = ["Qwen3DFlyDraftModel"]
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module

    from .model import Qwen3DFlyDraftModel
    from .trainer import DFlyTrainerModule

    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        model_class=Qwen3DFlyDraftModel,
        module_class=DFlyTrainerModule,
    )


def load_checkpoint(path, map_location="cpu"):
    """Load a DFly draft checkpoint (own format; trusted local file)."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint

    from .model import Qwen3DFlyDraftModel

    return load_dflash_checkpoint(path, map_location=map_location, model_class=Qwen3DFlyDraftModel)

