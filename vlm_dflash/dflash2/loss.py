"""DFlash2 selector candidates and per-position objectives.

``selector_training_candidates`` is ported from the Z-Lab DFlash2 architecture as
vendored by vllm-project/speculators @ 55ebbd5b
(`src/speculators/models/dflash2/metrics.py`; MIT License, Copyright (c) 2026
Z Lab — see ``conv.py`` for the full notice): unary top-k is the serving
candidate set, and training replaces its weakest candidate only when the hard
target is absent, so every position has a well-defined K-way cross-entropy label
without expanding the selector to the full vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "candidate_cross_entropy",
    "decay_weights",
    "selector_training_candidates",
]


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build top-k selector candidates, injecting a missing target at rank K.

    Returns ``(training_candidate_ids, target_positions, contains_target)`` where
    ``contains_target`` marks positions whose target was already in the unary
    top-k (the only positions that are servable without the injection).
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(contains_target, target_positions, top_k - 1)

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target,
        training_candidate_ids[..., -1],
        target_ids,
    )
    return training_candidate_ids, target_positions, contains_target


def candidate_cross_entropy(logits: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
    """Per-position K-way cross-entropy over the candidate set."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_positions.reshape(-1),
        reduction="none",
    ).view_as(target_positions)


def decay_weights(mask: torch.Tensor, slots: int, gamma: float | None) -> torch.Tensor:
    """DFlash fixed-exponential position weights for a supervised-slot mask.

    ``mask`` is ``[..., slots]`` over the *supervised* block offsets (slot index
    ``i`` is block offset ``k = i + 1`` for the DFlash slot convention), so the
    weights are ``exp(-i / gamma)`` — upstream ``dflash_loss_decay`` with the
    anchor slot already sliced off. ``gamma=None`` disables decay (uniform).
    """
    if gamma is None:
        return mask.float()
    depth = torch.arange(slots, device=mask.device, dtype=torch.float32)
    return mask.float() * torch.exp(-depth / float(gamma))
