"""JetSpec: causal block-parallel drafting trained by forward-KL distillation.

Reference implementation (see ``docs/upstream/jetspec.md``): NVIDIA-NeMo's
``nemo_automodel/components/speculative/dflash/jetspec_core.py`` (Apache-2.0).
JetSpec reuses the unmodified DFlash draft backbone and changes only the
in-block attention mask (causal) and the training objective (temperature-scaled
forward KL against the target's per-position distribution instead of hard-label
CE). The production parallel-tree runtime is NOT ported (see the audit doc).
"""

from .factory import (
    build_draft_config,
    build_trainer_module,
    load_checkpoint,
    resolve_jetspec_config,
)
from .kd_loss import ForwardKDLoss, forward_kl_per_token
from .trainer import JetSpecTrainerModule

__all__ = [
    "ForwardKDLoss",
    "JetSpecTrainerModule",
    "build_draft_config",
    "build_trainer_module",
    "forward_kl_per_token",
    "load_checkpoint",
    "resolve_jetspec_config",
]
