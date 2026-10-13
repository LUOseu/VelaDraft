"""Build DFlare draft configs / trainer modules / decoders from the run config."""

from __future__ import annotations

import torch

from ..contract import contract_capture_layer_ids
from ..dflash.factory import build_draft_config as build_dflash_draft_config

__all__ = [
    "DEFAULTS",
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dflare_config",
    "resolve_context_layer_ids",
]

DEFAULTS: dict = {
    # Which captured target layers the fusion consumes (absolute layer indices,
    # ordered). ``None`` -> every captured layer, in capture order.
    "context_layer_ids": None,
    # Upstream DFlare: context K/V use dedicated k_proj_target / v_proj_target.
    # False is the shared-KV control (DFlash projection layout + fusion only).
    "separate_context_kv": True,
    # Upstream trains the fusion logits; False freezes the distance prior
    # (explicit ablation, reported as such).
    "fusion_trainable": True,
    # "distance" = upstream prior; "uniform" = zero-logit ablation only.
    "fusion_init": "distance",
}


def resolve_dflare_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.dflare`` (defaults fill gaps).

    Bounded schema: unknown keys fail loud instead of being silently ignored
    (a typo would otherwise train a different configuration than the one the
    config claims to describe).
    """
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("dflare") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.dflare must be a mapping of DFlare knobs.")
        unknown = sorted(set(block) - set(DEFAULTS))
        if unknown:
            raise ValueError(
                f"algorithm.dflare has unsupported keys {unknown}. Supported keys: "
                f"{sorted(DEFAULTS)}."
            )
        resolved.update(block)
    return resolved


def resolve_context_layer_ids(dflare: dict, capture_layer_ids: list[int]) -> list[int]:
    """Validate the consumed layer subset against the captured layer list."""
    raw = dflare.get("context_layer_ids")
    capture = [int(layer) for layer in capture_layer_ids]
    context = [int(layer) for layer in raw] if raw is not None else list(capture)
    if not context:
        raise ValueError("algorithm.dflare.context_layer_ids must not be empty.")
    if len(set(context)) != len(context):
        raise ValueError(f"algorithm.dflare.context_layer_ids must be unique, got {context}.")
    missing = [layer for layer in context if layer not in capture]
    if missing:
        raise ValueError(
            f"algorithm.dflare.context_layer_ids {context} include layers {missing} that the feature "
            f"store does not capture ({capture}); set drafter.capture_layer_ids to a superset."
        )
    return context


def build_draft_config(contract: dict, cfg: dict):
    """DFlash draft config with the DFlare fusion fields in ``dflash_config``."""
    config = build_dflash_draft_config(contract, cfg)
    dflare = resolve_dflare_config(cfg)
    capture = contract_capture_layer_ids(contract)
    context = resolve_context_layer_ids(dflare, capture)
    fusion_init = str(dflare["fusion_init"])
    if fusion_init not in {"distance", "uniform"}:
        raise ValueError(f"algorithm.dflare.fusion_init must be 'distance' or 'uniform', got {fusion_init!r}.")
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    dflash_config.update(
        {
            # The fusion axis; also what inference extracts from the target.
            "target_layer_ids": context,
            "context_layer_ids": context,
            "separate_context_kv": bool(dflare["separate_context_kv"]),
            "fusion_trainable": bool(dflare["fusion_trainable"]),
            "fusion_init": fusion_init,
        }
    )
    config.dflash_config = dflash_config
    config.architectures = ["Qwen3DFlareDraftModel"]
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module

    from .model import Qwen3DFlareDraftModel
    from .trainer import DFlareTrainerModule

    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        model_class=Qwen3DFlareDraftModel,
        module_class=DFlareTrainerModule,
    )


def load_checkpoint(path, map_location="cpu"):
    """Load a DFlare draft checkpoint (own format; trusted local file)."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint

    from .model import Qwen3DFlareDraftModel

    return load_dflash_checkpoint(path, map_location=map_location, model_class=Qwen3DFlareDraftModel)

