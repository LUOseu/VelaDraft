"""DFlash2 draft model: DFlash backbone + local dynamic convolutions + selector.

The draft backbone is the repo's DFlash stack; DFlash2 changes two things (see
``docs/upstream/dflash2.md``):

* every decoder layer wraps its attention and MLP sublayers in a
  :class:`~.conv.GroupedDynamicCausalConv` (causal within the anchor block);
* a :class:`~.selector.CandidateSelector` reranks the unary top-K candidates at
  serving time, walking the candidate path one position at a time.

Both additions are modality-agnostic (they act on the draft block), so the
multimodal plumbing — MRoPE, image spans, captured target features — is exactly
the DFlash one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dflash.draft_model import Qwen3DFlashDecoderLayer, Qwen3DFlashDraftModel
from .conv import GroupedDynamicCausalConv
from .selector import CandidateSelector

__all__ = ["Qwen3DFlash2DecoderLayer", "Qwen3DFlash2DraftModel"]


class Qwen3DFlash2DecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer with local convolution around attention and MLP."""

    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
    ) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        conv_kwargs = {
            "block_size": block_size,
            "kernel_size": conv_kernel_size,
            "group_size": conv_group_size,
        }
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size, **conv_kwargs)
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, **conv_kwargs)

    def reset_convolutions(self) -> None:
        """Restore the DFlash-equivalent identity initialization."""
        self.attention_conv.reset_parameters()
        self.mlp_conv.reset_parameters()

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        multimodal_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        spatial_rope_gate: torch.Tensor | None = None,
        visual_context: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, output_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            multimodal_position_embeddings=multimodal_position_embeddings,
            spatial_rope_gate=spatial_rope_gate,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(hidden_states, output_kernel)
        hidden_states = residual + hidden_states
        if self.visual_fusion is not None:
            if visual_context is None:
                raise ValueError("visual_context is required when DFlash visual conditioning is enabled.")
            hidden_states = self.visual_fusion(hidden_states, visual_context)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, output_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, output_kernel)
        return residual + hidden_states


class Qwen3DFlash2DraftModel(Qwen3DFlashDraftModel):
    """DFlash draft model with local convolutions and a candidate selector."""

    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        dflash2 = dflash_config.get("dflash2") or {}
        if not isinstance(dflash2, dict):
            raise ValueError("dflash_config.dflash2 must be a mapping of DFlash2 knobs.")

        vocab_size = int(config.vocab_size)
        self.conv_kernel_size = int(dflash2.get("conv_kernel_size", 2))
        self.conv_group_size = int(dflash2.get("conv_group_size", 16))
        self.selector_rank = int(dflash2.get("selector_rank", 256))
        self.selector_top_k = int(dflash2.get("selector_top_k", 16))
        if self.selector_top_k > vocab_size:
            raise ValueError(
                f"selector_top_k ({self.selector_top_k}) cannot exceed verifier_vocab_size ({vocab_size})."
            )
        if self.sample_from_anchor:
            raise ValueError(
                "DFlash2 follows the DFlash slot convention (the anchor is the bonus token); "
                "sample_from_anchor must be false."
            )
        if self.projector_type is not None:
            raise ValueError("DFlash2 does not combine with the Domino correction head (projector_type).")

        # Built after ``post_init`` so the selector keeps its own initialization
        # (the conv kernels were re-initialized there and are reset below).
        initializer_range = float(getattr(config, "initializer_range", 0.02))
        self.candidate_selector = CandidateSelector(
            vocab_size=vocab_size,
            hidden_size=int(config.hidden_size),
            rank=self.selector_rank,
            top_k=self.selector_top_k,
            initializer_range=initializer_range,
        )
        for layer in self.layers:
            reset = getattr(layer, "reset_convolutions", None)
            if not callable(reset):
                raise TypeError(
                    f"{type(layer).__name__} has no reset_convolutions(); a DFlash2 decoder layer must "
                    "wrap its sublayers in resettable convolutions."
                )
            reset()

    def _make_decoder_layer(self, config, layer_idx: int) -> Qwen3DFlash2DecoderLayer:
        """Build a conv-wrapped decoder layer (the DFlash2 architecture hook)."""
        dflash_config = getattr(config, "dflash_config", {}) or {}
        dflash2 = dflash_config.get("dflash2") or {}
        return Qwen3DFlash2DecoderLayer(
            config,
            layer_idx,
            block_size=int(getattr(config, "block_size", 8)),
            conv_kernel_size=int(dflash2.get("conv_kernel_size", 2)),
            conv_group_size=int(dflash2.get("conv_group_size", 16)),
        )

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Greedy selector walk over the unary top-K of one parallel draft forward.

        Upstream inference keeps the single parallel draft forward and lets the
        selector walk the candidate path: the walk starts at the anchor token and
        conditions each position on the token selected at the preceding position.
        The predecessor ids therefore come from the block's own (already
        selected) tokens, matching training's shifted-block predecessor
        convention (slot ``k`` predicts the token after block token ``k - 1``).
        """
        block_size = block_output_ids.shape[1]
        if block_size != self.block_size:
            raise ValueError(f"Expected a draft block of length {self.block_size}, got {block_size}.")
        if draft_hidden.ndim != 3 or draft_hidden.shape[0] != block_output_ids.shape[0]:
            raise ValueError(
                "draft_hidden must have shape [batch, sequence, hidden] with the same batch as "
                f"block_output_ids, got {tuple(draft_hidden.shape)} and {tuple(block_output_ids.shape)}."
            )
        current_hidden = draft_hidden[:, -block_size:, :]
        # Slot k (k = 1..block_size-1) predicts the token at ``anchor + k`` from
        # the hidden state at block position k.
        unary_logits = target.lm_head(current_hidden[:, 1:, :])
        completed_ids = block_output_ids[:, :block_size].clone()
        for offset in range(1, block_size):
            logits_at = unary_logits[:, offset - 1 : offset, :]
            hidden_at = current_hidden[:, offset : offset + 1, :]
            predecessor_ids = completed_ids[:, offset - 1 : offset]
            candidate_ids, scores = self.candidate_selector.select(
                logits_at, hidden_at, predecessor_ids
            )
            picked = scores.argmax(dim=-1, keepdim=True)
            completed_ids[:, offset] = candidate_ids.gather(-1, picked).squeeze(-1)
        return completed_ids[:, 1:]
