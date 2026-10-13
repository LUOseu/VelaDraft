"""Build DFly draft configs / trainer modules / decoders from the run config."""

from __future__ import annotations

from ..contract import contract_capture_layer_ids
from ..dflash.factory import build_draft_config as build_dflash_draft_config
from ..dflare.factory import resolve_context_layer_ids

__all__ = [
    "DEFAULTS",
    "OBJECTIVE_PROFILE",
    "UNSUPPORTED_RECIPE_KEYS",
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dfly_config",
]

#: Default objective profile of DFly runs in this repo: the shared DFlash
#: block-CE training profile (anchors + block mask + decayed CE through the
#: frozen LM head), recorded in the draft config, the checkpoint and the export.
#: Selecting ``algorithm.train_recipe`` replaces it with the recipe name
#: (``upstream_two_stage`` = upstream's LK / D-PACE cold start -> end-to-end TV).
OBJECTIVE_PROFILE = "controlled_ce"

#: Upstream's raw recipe knob names. They stay rejected: the recipe is selected
#: through the first-class ``algorithm.train_recipe`` interface instead
#: (``upstream_two_stage``), and accepting both spellings could describe one run
#: two conflicting ways.
UNSUPPORTED_RECIPE_KEYS = frozenset(
    {
        "loss_objective",
        "lk_loss_weight",
        "tv_loss",
        "tv_loss_weight",
        "two_stage",
        "d_pace",
        "d_pace_weight",
    }
)

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
    """Algorithm-namespaced knobs: ``algorithm.dfly`` (defaults fill gaps).

    Bounded schema: only the supported keys are accepted. Upstream's raw recipe
    knobs (LK / D-PACE / e2e TV) are rejected explicitly -- select the ported
    recipe through ``algorithm.train_recipe: upstream_two_stage`` instead; a
    config must never be able to ask for an objective by a spelling the run
    would silently ignore.
    """
    resolved = dict(DEFAULTS)
    algorithm_cfg = cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        block = algorithm_cfg.get("dfly") or {}
        if not isinstance(block, dict):
            raise ValueError("algorithm.dfly must be a mapping of DFly knobs.")
        unknown = sorted(set(block) - set(DEFAULTS))
        if unknown:
            hinted = sorted(set(unknown) & UNSUPPORTED_RECIPE_KEYS)
            detail = (
                f" (unsupported upstream-recipe knobs: {', '.join(hinted)}; select the ported recipe "
                "via algorithm.train_recipe=upstream_two_stage instead)"
                if hinted
                else ""
            )
            raise ValueError(
                f"algorithm.dfly has unsupported keys {unknown}{detail}. Supported keys: "
                f"{sorted(DEFAULTS)}."
            )
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
    from ..objectives import resolve_recipe

    recipe = resolve_recipe(cfg, "dfly")
    dflash_config.update(
        {
            "target_layer_ids": context,
            "context_layer_ids": context,
            # Machine-readable objective profile: the default shared block-CE
            # profile, or the selected ``algorithm.train_recipe`` name
            # (e.g. upstream_two_stage); re-read by checkpoints and exports.
            "objective_profile": OBJECTIVE_PROFILE if recipe is None else recipe.name,
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


def build_trainer_module(
    draft_config,
    contract: dict,
    cfg: dict,
    embed_weight,
    lm_head_weight=None,
    module_kwargs: dict | None = None,
):
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
        module_kwargs=module_kwargs,
    )


def load_checkpoint(path, map_location="cpu"):
    """Load a DFly draft checkpoint (own format; trusted local file)."""
    from ..dflash.factory import load_checkpoint as load_dflash_checkpoint

    from .model import Qwen3DFlyDraftModel

    return load_dflash_checkpoint(path, map_location=map_location, model_class=Qwen3DFlyDraftModel)

