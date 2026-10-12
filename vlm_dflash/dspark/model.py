"""DSpark draft model: the DFlash draft backbone plus the DSpark heads.

At inference the DSpark draft is semi-autoregressive inside a block: the
block-parallel backbone produces all slot logits in one forward pass, and the
Markov head then biases slot k with the token sampled (or, during training,
the ground-truth token) at the previous position — restoring the in-block
conditional dependency that pure block-parallel drafting lacks.
"""

from __future__ import annotations

import torch
from torch import nn

from ..dflash.draft_model import Qwen3DFlashDraftModel, sample
from .heads import ConfidenceHead, build_markov_head

__all__ = ["Qwen3DSparkDraftModel"]


class Qwen3DSparkDraftModel(Qwen3DFlashDraftModel):
    """DFlash draft with the DSpark Markov logit-bias and confidence heads."""

    def __init__(self, config) -> None:
        super().__init__(config)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        hidden_size = int(config.hidden_size)
        self.markov_rank = int(dflash_config.get("markov_rank", 256))
        self.markov_head_type = str(dflash_config.get("markov_head_type", "vanilla"))
        self.markov_head = build_markov_head(
            vocab_size=int(config.vocab_size),
            markov_rank=self.markov_rank,
            markov_head_type=self.markov_head_type,
            hidden_size=hidden_size,
        )
        self.enable_confidence_head = bool(dflash_config.get("enable_confidence_head", True))
        self.confidence_head_with_markov = bool(dflash_config.get("confidence_head_with_markov", True))
        if self.enable_confidence_head and self.confidence_head_with_markov and self.markov_head is None:
            raise ValueError("confidence_head_with_markov=True requires markov_rank > 0.")
        input_dim = hidden_size + (self.markov_rank if self.confidence_head_with_markov else 0)
        self.confidence_head = ConfidenceHead(input_dim) if self.enable_confidence_head else None
        if self.projector_type is not None:
            raise ValueError(
                "DSpark does not combine with a block-correction projector "
                f"(got projector_type={self.projector_type!r})."
            )

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample this round's proposals with the Markov head chained in-block.

        Returns ``block_size - 1`` proposals for the DFlash slot convention
        (``sample_from_anchor=False``) or ``block_size`` when the anchor slot is
        drafted too (DSpark's ``sample_from_anchor=True``).
        """
        block_size = block_output_ids.shape[1]
        if block_size != self.block_size:
            raise ValueError(f"Expected a draft block of length {self.block_size}, got {block_size}.")
        if draft_hidden.ndim != 3 or draft_hidden.shape[0] != block_output_ids.shape[0]:
            raise ValueError(
                "draft_hidden must have shape [batch, sequence, hidden] with the same batch as "
                f"block_output_ids, got {tuple(draft_hidden.shape)} and {tuple(block_output_ids.shape)}."
            )
        if draft_hidden.shape[1] < block_size:
            raise ValueError(
                f"draft_hidden must contain at least {block_size} current-block positions, "
                f"got {draft_hidden.shape[1]}."
            )
        current_hidden = draft_hidden[:, -block_size:, :]
        base_logits = target.lm_head(current_hidden)
        if self.markov_head is None:
            if self.sample_from_anchor:
                return sample(base_logits)
            return sample(base_logits[:, 1:, :])
        anchor_token_ids = block_output_ids[:, 0]
        if self.sample_from_anchor:
            # Slot k drafts ``anchor + 1 + k``; slot 0 conditions on the anchor.
            tokens, _ = self.markov_head.sample_block_tokens(
                base_logits,
                first_prev_token_ids=anchor_token_ids,
                hidden_states=current_hidden,
            )
        else:
            # Slot k drafts ``anchor + k``; the first proposal sits at slot 1
            # and conditions on the anchor token.
            tokens, _ = self.markov_head.sample_block_tokens(
                base_logits[:, 1:, :],
                first_prev_token_ids=anchor_token_ids,
                hidden_states=current_hidden[:, 1:, :],
            )
        return tokens
