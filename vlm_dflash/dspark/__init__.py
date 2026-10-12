"""DSpark: DFlash backbone + Markov logit-bias head + acceptance confidence head."""

from .model import Qwen3DSparkDraftModel
from .trainer import DSparkTrainerModule

__all__ = ["DSparkTrainerModule", "Qwen3DSparkDraftModel"]
