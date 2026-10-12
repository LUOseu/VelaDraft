"""Build Domino draft configs / trainer modules / decoders from the run config."""

from __future__ import annotations

from ..dflash.factory import build_draft_config as build_dflash_draft_config

__all__ = [
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_domino_config",
]

# Defaults follow the maintainer's multimodal Domino recipe
# (``qwen2_5_vl_domino.yaml``, see docs/upstream/domino.md): a 256-dim low-rank
# bottleneck, a 1024-dim GRU, one backbone-only leading position, shifted
# labels, and a full-training linear decay of the base-anchor weight from 1 to 0.
DEFAULTS: dict = {
    "emb_dim": 256,
    "gru_hidden_dim": 1024,
    "pure_draft_prefix_len": 1,
    "shift_label": True,
    "lambda_base_start": 1.0,
    "lambda_base_decay_ratio": 1.0,
    # Ablation-only switch (not an upstream knob): False trains the exact DFlash
    # objective on the same architecture to isolate the correction head.
    "enable_correction_head": True,
}


def resolve_domino_config(cfg: dict) -> dict:
    """Algorithm-namespaced knobs: ``algorithm.domino`` (defaults fill gaps)."""
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("domino") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.domino must be a mapping of Domino knobs.")
        resolved.update(block)
    return resolved


def build_draft_config(contract: dict, cfg: dict):
    """DFlash draft config plus the Domino correction-head fields."""
    config = build_dflash_draft_config(contract, cfg)
    domino = resolve_domino_config(cfg)
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    dflash_config.update(
        {
            "projector_type": "domino",
            "emb_dim": int(domino["emb_dim"]),
            "gru_hidden_dim": int(domino["gru_hidden_dim"]),
            "pure_draft_prefix_len": int(domino["pure_draft_prefix_len"]),
            "shift_label": bool(domino["shift_label"]),
        }
    )
    config.dflash_config = dflash_config
    return config


def build_trainer_module(draft_config, contract: dict, cfg: dict, embed_weight, lm_head_weight=None):
    from ..dflash.factory import build_trainer_module as build_dflash_trainer_module
    from .trainer import DominoTrainerModule

    domino = resolve_domino_config(cfg)
    return build_dflash_trainer_module(
        draft_config,
        contract,
        cfg,
        embed_weight,
        lm_head_weight=lm_head_weight,
        module_class=DominoTrainerModule,
        module_kwargs={
            # The shared slot machinery (``sample_from_anchor``) *is* Domino's
            # shifted-label convention; keep it in sync with the draft model's
            # ``shift_label`` (the trainer module validates the pair).
            "sample_from_anchor": bool(domino["shift_label"]),
            "lambda_base_start": float(domino["lambda_base_start"]),
            "lambda_base_decay_ratio": float(domino["lambda_base_decay_ratio"]),
            "enable_correction_head": bool(domino["enable_correction_head"]),
        },
    )


def load_checkpoint(path, map_location="cpu"):
    """Domino checkpoints load through the shared DFlash container."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint

    return load_dflash_checkpoint(path, map_location=map_location)
