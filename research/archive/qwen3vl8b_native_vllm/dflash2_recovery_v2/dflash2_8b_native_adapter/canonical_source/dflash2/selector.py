"""DFlash2 candidate selector.

Ported from the Z-Lab DFlash2 architecture as vendored by
vllm-project/speculators @ 55ebbd5b
(`src/speculators/models/dflash2/model_definitions.py::CandidateSelector`;
MIT License, Copyright (c) 2026 Z Lab — see ``conv.py`` for the full notice).

Serving semantics implemented here follow the speculators user guide: the unary
logits propose ``top_k`` candidates per position, and a low-rank bilinear edge
score reranks them while a sequential walk conditions each position on the token
selected at the preceding position. Parameter names match the canonical
checkpoint keys (``predecessor_codebook`` / ``successor_codebook`` /
``hidden_projection.weight``).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["CandidateSelector"]


class CandidateSelector(nn.Module):
    """Bilinear predecessor/hidden/candidate scorer used after unary top-k."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        top_k: int,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if top_k > vocab_size:
            raise ValueError(f"top_k ({top_k}) cannot exceed vocab_size ({vocab_size}).")
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)
        nn.init.normal_(self.predecessor_codebook, std=initializer_range)
        nn.init.normal_(self.successor_codebook, std=initializer_range)
        nn.init.normal_(self.hidden_projection.weight, std=initializer_range)

    def context(self, hidden_states: torch.Tensor, predecessor_ids: torch.Tensor) -> torch.Tensor:
        """Elementwise predecessor and hidden-state interaction."""
        predecessor = self.predecessor_codebook[predecessor_ids.long()]
        projected_hidden = self.hidden_projection(hidden_states)
        return predecessor * projected_hidden.to(predecessor.dtype)

    def transition_scores(
        self,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score selected candidate IDs (the bilinear edge term only)."""
        context = self.context(hidden_states, predecessor_ids)
        successors = self.successor_codebook[candidate_ids.long()]
        return (context.unsqueeze(-2) * successors).sum(dim=-1)

    def score_candidates(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add transition scores to a selected subset of unary logits."""
        unary_scores = unary_logits.gather(-1, candidate_ids)
        transition_scores = self.transition_scores(hidden_states, predecessor_ids, candidate_ids)
        return unary_scores + transition_scores.to(unary_scores.dtype)

    def select(
        self,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unary top-k IDs and their selector-corrected scores."""
        candidate_ids = unary_logits.topk(top_k or self.top_k, dim=-1).indices
        scores = self.score_candidates(unary_logits, hidden_states, predecessor_ids, candidate_ids)
        return candidate_ids, scores
