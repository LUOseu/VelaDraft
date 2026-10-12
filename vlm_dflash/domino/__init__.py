"""Domino: DFlash backbone + a causal GRU logit-correction head.

Ported from the maintainer implementation (SpecForge#571 semantics, see
``docs/upstream/domino.md``). The correction head itself lives on the draft
model (``projector_type="domino"``, ported with the frozen DFlash backbone);
this package adds the trainer-side dual-logit loss, the base-anchor
curriculum, and the build/load factories.
"""

from .trainer import DominoTrainerModule, get_lambda_base

__all__ = ["DominoTrainerModule", "get_lambda_base"]
