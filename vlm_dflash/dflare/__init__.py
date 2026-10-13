"""DFlare: DFlash with learnable per-draft-layer target fusion (AngelSpec port).

See ``docs/upstream/dflare.md`` for the audited upstream semantics and the
explicit list of VLM/local adaptations.
"""

from .factory import build_draft_config, build_trainer_module, load_checkpoint, resolve_dflare_config

__all__ = [
    "build_draft_config",
    "build_trainer_module",
    "load_checkpoint",
    "resolve_dflare_config",
]
