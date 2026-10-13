"""DFly (DFlareV2): shared FC context + per-layer fusion residual + optional
TreeFlash hidden-states correction (AngelSpec port).

See ``docs/upstream/dfly.md`` for the audited upstream semantics (including the
two-stage D-PACE + LK -> e2e-TV training recipe, ported as
``algorithm.train_recipe: upstream_two_stage`` in ``vlm_dflash/objectives.py``)
and this repo's adaptation.
"""

from .factory import build_draft_config, build_trainer_module, load_checkpoint, resolve_dfly_config

__all__ = [
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dfly_config",
]
