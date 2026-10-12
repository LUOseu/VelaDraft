"""Build JetSpec draft configs / trainer modules from the run config."""

from __future__ import annotations

__all__ = [
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_jetspec_config",
]

# Defaults follow the maintainer JetSpec recipe (``qwen3_jetspec.yaml``, see
# docs/upstream/jetspec.md): temperature 1.0 (no rescaling) and no position
# chunking upstream; the staged port chunks the teacher projection by default so
# peak memory stays bounded.
DEFAULTS: dict = {
    "kd_temperature": 1.0,
    "kd_chunk_size": 2048,
}


def resolve_jetspec_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.jetspec`` (defaults fill gaps)."""
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("jetspec") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.jetspec must be a mapping of JetSpec knobs.")
        resolved.update(block)
    return resolved


def build_draft_config(contract: dict, cfg: dict):
    """JetSpec uses the unmodified DFlash draft backbone (no new parameters)."""
    from ..dflash.factory import build_draft_config as build_dflash_draft_config

    return build_dflash_draft_config(contract, cfg)


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module
    from .trainer import JetSpecTrainerModule

    jetspec = resolve_jetspec_config(cfg)
    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        module_class=JetSpecTrainerModule,
        module_kwargs={
            # DFlash slot convention (slots 1..block_size-1); sampled anchor.
            "sample_from_anchor": False,
            "kd_temperature": float(jetspec["kd_temperature"]),
            "kd_chunk_size": int(jetspec["kd_chunk_size"]),
        },
    )


def load_checkpoint(path, map_location="cpu"):
    """JetSpec checkpoints load through the shared DFlash container."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint

    return load_dflash_checkpoint(path, map_location=map_location)
