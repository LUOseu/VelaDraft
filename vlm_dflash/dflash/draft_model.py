# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DFlash draft model (Qwen3-style).

Ported from SpecForge's ``specforge/modeling/draft/dflash.py``. DFlash drafts a
whole block of ``block_size`` tokens in parallel: the block's first position
holds the real anchor token and the rest are ``MASK`` tokens, and the draft
predicts the whole block in a single non-causal forward conditioned on the
target model's context hidden states.

The draft attention is therefore **not causal** -- a draft block's queries
attend to (a) the projected target-hidden context strictly before its anchor and
(b) bidirectionally to the other (noise) tokens of the same block. The attention
mask that enforces this is built by the trainer wrapper in
``nemo_automodel.components.speculative.dflash.core``.
"""

from __future__ import annotations

import inspect
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Callable, Optional, Tuple, TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


# --- vlm-dflash local modification -------------------------------------
# Verification-time SDPA-backend switch (reference behavior). The reference
# narrows verify forwards to EFFICIENT_ATTENTION+MATH to avoid cuDNN plan
# construction on short dynamic shapes. That switch changes the numerics of
# near-tie logits relative to ``generate()``; ``set_verify_sdpa_backend_switch``
# lets the parity evaluation mode disable it (documented in docs/REFERENCE.md).
_VERIFY_SDPA_BACKEND_SWITCH = True


def set_verify_sdpa_backend_switch(enabled: bool) -> None:
    """Enable/disable the verify-time SDPA backend narrowing (parity mode)."""
    global _VERIFY_SDPA_BACKEND_SWITCH
    _VERIFY_SDPA_BACKEND_SWITCH = bool(enabled)


def verify_sdpa_backend_switch_enabled() -> bool:
    return _VERIFY_SDPA_BACKEND_SWITCH


@contextmanager
def verify_sdpa_backend_parity():
    """Force generate()-parity verify numerics for the duration of the block.

    Used by the strict-parallel fallback: when a near-tie decision is detected,
    the round is re-verified with sequential target steps *and* the parity
    backend, which is the configuration the token-exact gate validates.
    """
    global _VERIFY_SDPA_BACKEND_SWITCH
    saved = _VERIFY_SDPA_BACKEND_SWITCH
    _VERIFY_SDPA_BACKEND_SWITCH = False
    try:
        yield
    finally:
        _VERIFY_SDPA_BACKEND_SWITCH = saved


class _DFlashAcceptanceEvent(TypedDict):
    """Acceptance counters for one verification round."""

    generated_position: int
    accepted_tokens: int
    draft_tokens: int


class _DFlashGenerationStats(TypedDict):
    """Counters and timings returned by speculative generation."""

    draft_tokens: float
    accepted_tokens: float
    verify_steps: float
    acceptance_events: list[_DFlashAcceptanceEvent]
    target_prefill_seconds: float
    draft_seconds: float
    target_verify_seconds: float
    strict_parallel_rounds: float
    strict_fallback_rounds: float
    strict_min_decision_margin: Optional[float]


def _min_decision_margin(logits: torch.Tensor, num_positions: int) -> float:
    """Smallest top-1/top-2 logit gap over the first ``num_positions`` positions.

    The strict-parallel guard uses the raw difference between the two highest
    logits as a cheap near-tie detector: bf16 verification rounds that move a
    decision by a fraction of a logit unit are exactly the ones where parallel
    and sequential kernels can disagree, so they are routed to the parity path.
    """
    if num_positions <= 0:
        return float("inf")
    top2 = torch.topk(logits[0, :num_positions].float(), k=2, dim=-1).values
    return float((top2[:, 0] - top2[:, 1]).min())


def _bounded_acceptance_counts(
    *,
    start: int,
    max_length: int,
    block_size: int,
    acceptance_length: int,
) -> tuple[int, int]:
    """Clip proposal and acceptance counts to requested output positions.

    Args:
        start: Absolute sequence position of the verification-round anchor.
        max_length: Exclusive maximum sequence length requested by the caller.
        block_size: Number of tokens in the draft block, including its anchor.
        acceptance_length: Number of draft tokens accepted by the target.

    Returns:
        Number of in-range draft opportunities and accepted draft tokens.
    """
    remaining_tokens = max(0, max_length - start)
    draft_tokens = min(block_size - 1, max(0, remaining_tokens - 1))
    return draft_tokens, min(acceptance_length, draft_tokens)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Greedy (temperature ~ 0) or temperature sampling over the last dim."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _ablate_image_token_hidden_states(
    target_hidden: torch.Tensor,
    image_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Transform image-token rows in target context for inference diagnostics.

    Args:
        target_hidden: Target features of shape [batch, sequence, hidden].
        image_mask: Boolean image-token mask of shape [batch, sequence].
        mode: ``keep`` preserves features, ``zero`` removes them, and
            ``shuffle`` reverses their order independently in each sample.

    Returns:
        Target features with the requested image-token transformation.
    """
    if mode not in {"keep", "zero", "shuffle"}:
        raise ValueError(f"image-token context mode must be keep, zero, or shuffle, got {mode!r}.")
    if target_hidden.ndim != 3 or image_mask.shape != target_hidden.shape[:2]:
        raise ValueError(
            "image_mask must match the batch and sequence axes of target_hidden, "
            f"got {tuple(image_mask.shape)} and {tuple(target_hidden.shape)}."
        )
    if mode == "keep":
        return target_hidden

    image_mask = image_mask.to(device=target_hidden.device, dtype=torch.bool)
    transformed = target_hidden.clone()
    if mode == "zero":
        return transformed.masked_fill(image_mask.unsqueeze(-1), 0)
    for batch_index in range(target_hidden.shape[0]):
        image_indices = image_mask[batch_index].nonzero(as_tuple=True)[0]
        transformed[batch_index, image_indices] = target_hidden[batch_index, image_indices.flip(0)]
    return transformed


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE where queries (draft block) are a suffix of the key positions.

    The keys span ``[context | noise-block]`` while the queries are only the
    noise block, so ``q`` is rotated with the trailing ``q_len`` slice of the
    rotary tables and ``k`` with the full table.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _merge_multiaxis_rotary_embeddings(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    sections: tuple[int, int, int],
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one MRoPE table from temporal, height, and width positions.

    Two layouts are supported, matching the two Qwen-VL generations:

    * chunked (Qwen2.5-VL): the half-head channels are split into three
      contiguous runs ``[t, h, w]`` of ``sections`` each;
    * interleaved (Qwen3-VL, ``mrope_interleaved=true``): the first
      ``3*max(sections[1:])`` half-head channels cycle ``t, h, w`` and any
      remaining channels stay temporal. Because cosine/sine are elementwise in
      the frequencies, the interleave is applied directly on the per-axis
      cosine/sine tables (each half-head channel ``j`` above ``head_dim/2``
      duplicates channel ``j``), which reproduces
      ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`` exactly without
      reconstructing angles.

    Args:
        rotary_emb: Rotary module producing tables for flattened position rows.
        hidden_states: Tensor of shape [batch, tokens, hidden], used for dtype and device.
        position_ids: Long tensor of shape [3, batch, sequence], with axes ordered as
            temporal, height, and width.
        sections: Number of half-head channels assigned to the three position axes.
        interleaved: Select the Qwen3-VL interleaved layout instead of the
            Qwen2.5-VL chunked layout.

    Returns:
        Cosine and sine tensors, each of shape [batch, sequence, head_dim].
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            f"multimodal_position_ids must have shape [3, batch, sequence], got {tuple(position_ids.shape)}."
        )
    axes, batch_size, sequence_length = position_ids.shape
    flat_position_ids = position_ids.reshape(axes * batch_size, sequence_length)
    axis_cos, axis_sin = rotary_emb(hidden_states, flat_position_ids)
    axis_cos = axis_cos.reshape(axes, batch_size, sequence_length, -1)
    axis_sin = axis_sin.reshape(axes, batch_size, sequence_length, -1)
    head_dim = axis_cos.shape[-1]
    half = head_dim // 2
    if interleaved:
        if 2 * sum(sections) != head_dim:
            raise ValueError(
                "Twice the sum of spatial_rope_sections must equal the attention head dimension, "
                f"got sections={sections} and head_dim={head_dim}."
            )
        for axis in (1, 2):
            if 3 * sections[axis] > half:
                raise ValueError(
                    f"Interleaved MRoPE needs 3*sections[axis] <= head_dim/2; axis {axis} has "
                    f"3*{sections[axis]} > {half}."
                )
        cos = axis_cos[0].clone()
        sin = axis_sin[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            indices = torch.arange(offset, 3 * sections[axis], 3, device=cos.device)
            duplicated = indices + half
            cos[..., indices] = axis_cos[axis][..., indices]
            cos[..., duplicated] = axis_cos[axis][..., duplicated]
            sin[..., indices] = axis_sin[axis][..., indices]
            sin[..., duplicated] = axis_sin[axis][..., duplicated]
        return cos, sin
    split_sizes = sections * 2
    if sum(split_sizes) != head_dim:
        raise ValueError(
            "Twice the sum of spatial_rope_sections must equal the attention head dimension, "
            f"got sections={sections} and head_dim={head_dim}."
        )
    cos = torch.cat([chunk[index % 3] for index, chunk in enumerate(axis_cos.split(split_sizes, dim=-1))], dim=-1)
    sin = torch.cat([chunk[index % 3] for index, chunk in enumerate(axis_sin.split(split_sizes, dim=-1))], dim=-1)
    return cos, sin


class Qwen3DFlashAttention(nn.Module):
    """Non-causal attention whose keys/values are ``[context | noise-block]``.

    Queries come from the draft (noise) tokens only; keys and values are the
    concatenation of the projected target-hidden context and the noise tokens.
    The bidirectional/block structure is supplied entirely by ``attention_mask``.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        multimodal_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        spatial_rope_gate: torch.Tensor | None = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Attend from draft tokens to target context and the draft block.

        Args:
            hidden_states: Tensor of shape [batch, draft_tokens, hidden].
            target_hidden: Tensor of shape [batch, context_tokens, hidden].
            position_embeddings: Tuple of cosine and sine tensors, each of shape
                [batch, context_tokens + draft_tokens, head_dim], for baseline 1D RoPE.
            multimodal_position_embeddings: Optional tuple of cosine and sine tensors,
                each of shape [batch, context_tokens + draft_tokens, head_dim], for MRoPE.
            spatial_rope_gate: Optional scalar tensor interpolating between baseline and
                multimodal rotary outputs. When omitted with multimodal embeddings, MRoPE
                replaces baseline RoPE directly.
            attention_mask: Optional attention mask broadcastable to shape
                [batch, heads, draft_tokens, context_tokens + draft_tokens].
            past_key_values: Optional per-layer key/value cache.
            cache_position: Optional long tensor containing cache write positions.
            **kwargs: Attention backend arguments.

        Returns:
            Tuple containing the attention output tensor of shape [batch, draft_tokens,
            hidden] and optional attention weights.
        """
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k, v = self._context_noise_kv(hidden_states, target_hidden)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        unrotated_q, unrotated_k = q, k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if multimodal_position_embeddings is not None:
            spatial_cos, spatial_sin = multimodal_position_embeddings
            spatial_q, spatial_k = apply_rotary_pos_emb(
                unrotated_q,
                unrotated_k,
                spatial_cos,
                spatial_sin,
            )
            if spatial_rope_gate is None:
                q, k = spatial_q, spatial_k
                cos, sin = spatial_cos, spatial_sin
            else:
                strength = torch.tanh(spatial_rope_gate).to(dtype=q.dtype)
                q = q + strength * (spatial_q - q)
                k = k + strength * (spatial_k - k)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        backend_context = nullcontext()
        if not self.training and q.device.type == "cuda" and self.config._attn_implementation == "sdpa":
            # DFlash changes the cached context length after nearly every block.
            # cuDNN SDPA repeatedly plans those dynamic shapes, while efficient
            # attention supports the same additive non-causal block mask without
            # that host overhead. Keep math enabled for older CUDA architectures.
            backend_context = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        if (
            os.environ.get("VLM_DFLASH_FLEX_FORCE_MAIN") == "1"
            and self.config._attn_implementation == "flex_attention"
            and kwargs.get("kernel_options") is None
        ):
            # torch 2.11 can route GQA block-sparse attention (q_len < 128) to
            # the flex-decoding Triton template, where the candidate filter
            # (SPARSE_Q_BLOCK_SIZE % BLOCK_M) can reject every config and
            # Inductor raises NoValidChoicesError. Opt-in: force the main
            # flex-attention kernel instead.
            kwargs["kernel_options"] = {"FORCE_USE_FLEX_ATTENTION": True}
        with backend_context:
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights

    def _context_noise_kv(
        self, hidden_states: torch.Tensor, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the two K/V sources and concatenate them pre-norm/pre-RoPE.

        DFlash shares one ``k_proj`` / ``v_proj`` for the target context and the
        draft (noise) tokens; DFlare overrides this seam with dedicated
        ``k_proj_target`` / ``v_proj_target`` context projections. Layout (and
        the concat -> k_norm -> RoPE order) is identical in both cases:
        ``[batch, context_tokens + draft_tokens, kv_heads * head_dim]``.
        """
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        return k, v


class DFlashVisualLayer(nn.Module):
    """Compress one target layer's image tokens and fuse them into one draft layer.

    Each draft layer owns its two-query resampler because it is paired with a
    different captured target layer. The compressed context can be built once
    during prefill and reused for every draft round.

    Args:
        config: Qwen draft configuration supplying the hidden size and RMSNorm epsilon.
        num_query_tokens: Number of learned visual queries.
        adapter_dim: Hidden size of the lightweight visual attention branch.
        num_attention_heads: Number of visual attention heads.
        gate_init: Initial scalar residual-gate value.
    """

    def __init__(
        self,
        config: Qwen3Config,
        num_query_tokens: int,
        adapter_dim: int,
        num_attention_heads: int,
        gate_init: float,
    ) -> None:
        super().__init__()
        if num_query_tokens < 1:
            raise ValueError(f"visual_num_query_tokens must be >= 1, got {num_query_tokens}.")
        if num_attention_heads <= 0 or adapter_dim <= 0 or adapter_dim % num_attention_heads != 0:
            raise ValueError(
                "visual_adapter_dim must be positive and divisible by visual_num_attention_heads, "
                f"got {adapter_dim} and {num_attention_heads}."
            )
        self.hidden_size = config.hidden_size
        self.adapter_dim = adapter_dim
        self.num_heads = num_attention_heads
        self.head_dim = adapter_dim // num_attention_heads
        self.num_query_tokens = num_query_tokens
        self.query = nn.Parameter(torch.empty(num_query_tokens, self.num_heads, self.head_dim))
        self.target_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_proj = nn.Linear(config.hidden_size, adapter_dim, bias=False)
        self.resampler_k_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.resampler_v_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.resampler_o_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.draft_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_q_proj = nn.Linear(config.hidden_size, adapter_dim, bias=False)
        self.cross_k_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.cross_v_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.cross_o_proj = nn.Linear(adapter_dim, config.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.normal_(self.query, mean=0.0, std=self.head_dim**-0.5)

    def build_context(self, target_hidden: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """Compress image-token features with learned queries.

        Args:
            target_hidden: Tensor of shape [batch, sequence, hidden] from the
                target layer paired with this draft layer.
            image_mask: Bool tensor of shape [batch, sequence], with True at
                image-token positions.

        Returns:
            Tensor of shape [batch, visual_queries, visual_dim]. Samples without an
            image receive an all-zero context.
        """
        if target_hidden.ndim != 3:
            raise ValueError(f"target_hidden must have shape [batch, sequence, hidden], got {target_hidden.shape}.")
        if image_mask.shape != target_hidden.shape[:2]:
            raise ValueError(
                "image_mask must have shape [batch, sequence] matching target_hidden, "
                f"got {tuple(image_mask.shape)} and {tuple(target_hidden.shape)}."
            )
        batch_size, sequence_length, _ = target_hidden.shape
        if sequence_length == 0:
            raise ValueError("target_hidden must contain at least one sequence position.")
        image_mask = image_mask.to(device=target_hidden.device, dtype=torch.bool)
        has_image = image_mask.any(dim=-1)
        # SDPA must not receive a fully-masked row. For text-only samples expose
        # one harmless token, then zero the compressed result below.
        safe_mask = image_mask.clone()
        safe_mask[:, 0] |= ~has_image
        features = self.target_proj(self.target_norm(target_hidden))
        query = self.query.to(dtype=features.dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
        query = query.transpose(1, 2)
        key = self.resampler_k_proj(features).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.resampler_v_proj(features).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        compressed = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=safe_mask[:, None, None, :],
            is_causal=False,
        )
        compressed = compressed.transpose(1, 2).reshape(batch_size, self.num_query_tokens, self.adapter_dim)
        compressed = self.resampler_o_proj(compressed)
        return compressed * has_image[:, None, None].to(dtype=compressed.dtype)

    def forward(self, hidden_states: torch.Tensor, visual_context: torch.Tensor) -> torch.Tensor:
        """Apply gated cross-attention from draft tokens to cached visual queries.

        Args:
            hidden_states: Tensor of shape [batch, draft_tokens, hidden].
            visual_context: Tensor of shape [batch, visual_queries, visual_dim].

        Returns:
            Tensor of shape [batch, draft_tokens, hidden]. The input is unchanged
            when ``gate`` is exactly zero.
        """
        if hidden_states.ndim != 3 or visual_context.ndim != 3:
            raise ValueError(
                "hidden_states and visual_context must have shapes [batch, tokens, hidden], "
                f"got {tuple(hidden_states.shape)} and {tuple(visual_context.shape)}."
            )
        if hidden_states.shape[0] != visual_context.shape[0] or hidden_states.shape[2] != self.hidden_size:
            raise ValueError(
                "visual_context must share hidden_states' batch and configured hidden size, "
                f"got {tuple(hidden_states.shape)} and {tuple(visual_context.shape)}."
            )
        if visual_context.shape[2] != self.adapter_dim:
            raise ValueError(
                f"visual_context hidden dimension must be {self.adapter_dim}, got {visual_context.shape[2]}."
            )
        batch_size, draft_tokens, _ = hidden_states.shape
        visual_tokens = visual_context.shape[1]
        query = self.cross_q_proj(self.draft_norm(hidden_states)).view(
            batch_size, draft_tokens, self.num_heads, self.head_dim
        )
        key = self.cross_k_proj(visual_context).view(batch_size, visual_tokens, self.num_heads, self.head_dim)
        value = self.cross_v_proj(visual_context).view(batch_size, visual_tokens, self.num_heads, self.head_dim)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, draft_tokens, self.adapter_dim)
        return hidden_states + torch.tanh(self.gate) * self.cross_o_proj(attended)


class DFlashPooledMLPVisualLayer(nn.Module):
    """Pool one target layer's image tokens and fuse them with a gated MLP.

    The masked mean and target-side MLP run once during prefill. Each subsequent
    draft round reuses the single cached visual vector and applies only a
    bottleneck MLP to the draft tokens. The residual gate starts at zero so a
    warm-started DFlash checkpoint initially preserves its original behavior.

    Args:
        config: Qwen draft configuration supplying the hidden size and RMSNorm epsilon.
        adapter_dim: Hidden size of the target and draft bottleneck MLPs.
    """

    def __init__(self, config: Qwen3Config, adapter_dim: int) -> None:
        super().__init__()
        if adapter_dim <= 0:
            raise ValueError(f"visual_adapter_dim must be positive, got {adapter_dim}.")
        self.hidden_size = config.hidden_size
        self.adapter_dim = adapter_dim
        self.target_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.target_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, adapter_dim, bias=False),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim, bias=False),
        )
        self.draft_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.draft_proj = nn.Linear(config.hidden_size, adapter_dim, bias=False)
        self.output_proj = nn.Linear(adapter_dim, config.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def build_context(self, target_hidden: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        """Build one cached visual vector with masked mean pooling and an MLP.

        Args:
            target_hidden: Tensor of shape [batch, sequence, hidden] from the
                target layer paired with this draft layer.
            image_mask: Bool tensor of shape [batch, sequence], with True at
                image-token positions.

        Returns:
            Tensor of shape [batch, 1, visual_dim]. Samples without an image
            receive an all-zero context.
        """
        if target_hidden.ndim != 3:
            raise ValueError(f"target_hidden must have shape [batch, sequence, hidden], got {target_hidden.shape}.")
        if image_mask.shape != target_hidden.shape[:2]:
            raise ValueError(
                "image_mask must have shape [batch, sequence] matching target_hidden, "
                f"got {tuple(image_mask.shape)} and {tuple(target_hidden.shape)}."
            )
        if target_hidden.shape[1] == 0:
            raise ValueError("target_hidden must contain at least one sequence position.")
        image_mask = image_mask.to(device=target_hidden.device, dtype=torch.bool)
        mask = image_mask.unsqueeze(-1).to(dtype=target_hidden.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        pooled = (self.target_norm(target_hidden) * mask).sum(dim=1) / count
        context = self.target_mlp(pooled).unsqueeze(1)
        return context * image_mask.any(dim=-1)[:, None, None].to(dtype=context.dtype)

    def forward(self, hidden_states: torch.Tensor, visual_context: torch.Tensor) -> torch.Tensor:
        """Apply cached visual conditioning through a zero-gated bottleneck MLP.

        Args:
            hidden_states: Tensor of shape [batch, draft_tokens, hidden].
            visual_context: Tensor of shape [batch, 1, visual_dim].

        Returns:
            Tensor of shape [batch, draft_tokens, hidden]. The input is unchanged
            when ``gate`` is exactly zero.
        """
        if hidden_states.ndim != 3 or visual_context.ndim != 3:
            raise ValueError(
                "hidden_states and visual_context must have shapes [batch, tokens, hidden], "
                f"got {tuple(hidden_states.shape)} and {tuple(visual_context.shape)}."
            )
        if hidden_states.shape[0] != visual_context.shape[0] or hidden_states.shape[2] != self.hidden_size:
            raise ValueError(
                "visual_context must share hidden_states' batch and configured hidden size, "
                f"got {tuple(hidden_states.shape)} and {tuple(visual_context.shape)}."
            )
        if visual_context.shape[1:] != (1, self.adapter_dim):
            raise ValueError(
                f"visual_context must have shape [batch, 1, {self.adapter_dim}], got {tuple(visual_context.shape)}."
            )
        fused = F.silu(self.draft_proj(self.draft_norm(hidden_states)) + visual_context)
        return hidden_states + torch.tanh(self.gate) * self.output_proj(fused)


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    """A DFlash decoder block: non-causal attention over ``[context | noise]`` + MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        dflash_config = getattr(config, "dflash_config", {}) or {}
        visual_num_query_tokens = int(dflash_config.get("visual_num_query_tokens", 0) or 0)
        visual_adapter_type = str(dflash_config.get("visual_adapter_type", "cross_attention"))
        if visual_num_query_tokens <= 0:
            self.visual_fusion = None
        elif visual_adapter_type == "cross_attention":
            self.visual_fusion = DFlashVisualLayer(
                config,
                num_query_tokens=visual_num_query_tokens,
                adapter_dim=int(dflash_config.get("visual_adapter_dim", 256)),
                num_attention_heads=int(dflash_config.get("visual_num_attention_heads", 4)),
                gate_init=float(dflash_config.get("visual_gate_init", 1.0e-3)),
            )
        elif visual_adapter_type == "pooled_mlp":
            self.visual_fusion = DFlashPooledMLPVisualLayer(
                config,
                adapter_dim=int(dflash_config.get("visual_adapter_dim", 128)),
            )
        else:
            raise ValueError(
                f"visual_adapter_type must be 'cross_attention' or 'pooled_mlp', got {visual_adapter_type!r}."
            )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        multimodal_position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        spatial_rope_gate: torch.Tensor | None = None,
        visual_context: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run one DFlash layer with optional cached visual conditioning.

        Args:
            target_hidden: Tensor of shape [batch, context, hidden].
            hidden_states: Tensor of shape [batch, draft_tokens, hidden].
            attention_mask: Optional tensor or block mask describing attention
                from draft tokens to [context, draft_tokens].
            position_ids: Optional long tensor of shape [batch, context + draft_tokens].
            past_key_value: Optional per-layer key/value cache.
            use_cache: Whether to update ``past_key_value``.
            cache_position: Optional long tensor containing cache write positions.
            position_embeddings: Optional cosine and sine tensors with shape
                [batch, context + draft_tokens, head_dim].
            multimodal_position_embeddings: Optional MRoPE cosine and sine tensors
                with shape [batch, context + draft_tokens, head_dim].
            spatial_rope_gate: Optional scalar tensor interpolating between 1D RoPE
                and MRoPE inside this layer's attention.
            visual_context: Optional tensor of shape [batch, visual_queries, visual_dim].
            **kwargs: HuggingFace attention backend arguments.

        Returns:
            Tensor of shape [batch, draft_tokens, hidden].
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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
        hidden_states = residual + hidden_states
        if self.visual_fusion is not None:
            if visual_context is None:
                raise ValueError("visual_context is required when DFlash visual conditioning is enabled.")
            hidden_states = self.visual_fusion(hidden_states, visual_context)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Pick ``num_draft_layers`` target layers spread across the target's depth."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def extract_context_feature(hidden_states: list[torch.Tensor], layer_ids: list[int]) -> torch.Tensor:
    """Concatenate the selected target layers' hidden states along the feature dim.

    ``hidden_states`` follows HF's ``output_hidden_states`` convention where
    index 0 is the embedding output, so layer ``i``'s output is at index
    ``i + 1``.
    """
    offset = 1
    return torch.cat([hidden_states[layer_id + offset] for layer_id in layer_ids], dim=-1)


def _prepend_text_position_ids(
    multimodal_position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Add the text-position axis Qwen VL uses to construct its causal mask.

    Args:
        multimodal_position_ids: Tensor of shape [3, batch, sequence] containing
            temporal, height, and width MRoPE positions.
        attention_mask: Tensor of shape [batch, sequence], where nonzero entries
            identify active tokens.

    Returns:
        Tensor of shape [4, batch, sequence]. Axis 0 contains monotonically
        increasing text positions; axes 1-3 contain the input MRoPE positions.
    """
    if multimodal_position_ids.ndim != 3 or multimodal_position_ids.shape[0] != 3:
        raise ValueError(
            "Expected multimodal_position_ids with shape [3, batch, sequence], "
            f"got {tuple(multimodal_position_ids.shape)}."
        )
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(multimodal_position_ids.shape[1:]):
        raise ValueError(
            "attention_mask must have shape [batch, sequence] matching multimodal_position_ids, "
            f"got {tuple(attention_mask.shape)} and {tuple(multimodal_position_ids.shape)}."
        )
    text_position_ids = attention_mask.long().cumsum(dim=-1) - 1
    text_position_ids.masked_fill_(attention_mask == 0, 0)
    return torch.cat((text_position_ids.unsqueeze(0), multimodal_position_ids), dim=0)


class Qwen3DFlashDraftModel(Qwen3PreTrainedModel):
    """DFlash draft model: a small non-causal Qwen3 stack over ``[context | noise]``."""

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [self._make_decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.spatial_rope_enabled = bool(dflash_config.get("spatial_rope_enabled", False))
        self.spatial_rope_mode = str(dflash_config.get("spatial_rope_mode", "gated"))
        self.spatial_rope_interleaved = bool(dflash_config.get("spatial_rope_interleaved", False))
        # Inference-side in-block attention. JetSpec trains with a block-causal
        # in-block mask; serving such a checkpoint under the default
        # (non-causal) block forward measurably changes its proposals. When
        # this flag is set, spec_generate passes the causal in-block mask so
        # the served factorization matches training. Default off: every other
        # algorithm's runtime is byte-identical to before.
        self.inference_in_block_causal = bool(dflash_config.get("inference_in_block_causal", False))
        spatial_rope_sections = dflash_config.get("spatial_rope_sections", None)
        self.spatial_rope_sections = (
            tuple(int(section) for section in spatial_rope_sections) if spatial_rope_sections is not None else ()
        )
        if self.spatial_rope_enabled:
            if self.spatial_rope_mode not in {"gated", "replace"}:
                raise ValueError(f"spatial_rope_mode must be 'gated' or 'replace', got {self.spatial_rope_mode!r}.")
            head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
            if len(self.spatial_rope_sections) != 3 or 2 * sum(self.spatial_rope_sections) != head_dim:
                raise ValueError(
                    "Spatial-rope DFlash requires three spatial_rope_sections whose doubled sum equals head_dim, "
                    f"got sections={self.spatial_rope_sections} and head_dim={head_dim}."
                )
            if self.spatial_rope_interleaved:
                for axis in (1, 2):
                    if 3 * self.spatial_rope_sections[axis] > head_dim // 2:
                        raise ValueError(
                            "Interleaved spatial rope needs 3*sections[axis] <= head_dim/2; "
                            f"axis {axis} has 3*{self.spatial_rope_sections[axis]} > {head_dim // 2}."
                        )
            if self.spatial_rope_mode == "gated":
                self.spatial_rope_gates = nn.Parameter(torch.zeros(len(self.layers)))
            else:
                self.register_parameter("spatial_rope_gates", None)
        else:
            self.register_parameter("spatial_rope_gates", None)
        self.layer_routing_enabled = bool(dflash_config.get("layer_routing_enabled", False))
        if self.layer_routing_enabled:
            if len(self.target_layer_ids) != len(self.layers):
                raise ValueError(
                    "Layer-routed DFlash requires one captured target layer per draft layer, "
                    f"got {len(self.target_layer_ids)} target layers and {len(self.layers)} draft layers."
                )
            self.layer_route_gates = nn.Parameter(torch.zeros(len(self.layers)))
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.visual_num_query_tokens = int(dflash_config.get("visual_num_query_tokens", 0) or 0)
        self.visual_image_token_id = dflash_config.get("visual_image_token_id", None)
        # Optional Domino correction head (ported from SpecForge#571). DFlash drafts
        # a block in parallel and is non-causal; the Domino head adds a *causal*
        # low-rank logit correction conditioned on a GRU state built from the
        # block's previous tokens. ``projector_type=None`` leaves DFlash untouched.
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        # Block-slot convention at inference (upstream DFlash family switch).
        # False (DFlash): the anchor is the verifier's bonus token and the draft
        # proposes ``block_size - 1`` suffix tokens. True (DSpark default): the
        # anchor slot itself is drafted, so a round proposes ``block_size``
        # tokens and its verification window is one position wider.
        self.sample_from_anchor = bool(dflash_config.get("sample_from_anchor", False))
        if self.projector_type == "domino":
            self.emb_dim = dflash_config["emb_dim"]
            self.gru_hidden_dim = dflash_config["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
        elif self.projector_type is not None:
            raise ValueError(f"Unknown draft projector_type: {self.projector_type}")
        self.post_init()

    def _make_decoder_layer(self, config, layer_idx: int):
        """Build one draft decoder layer (DFlash2 overrides this to add its local convs)."""
        return Qwen3DFlashDecoderLayer(config, layer_idx)

    @property
    def visual_conditioning_enabled(self) -> bool:
        """Whether this checkpoint contains the cached visual-conditioning branch."""
        return self.visual_num_query_tokens > 0

    def set_training_stage(self, stage: str) -> None:
        """Apply the supported two-stage visual-training freeze policy.

        Args:
            stage: ``"visual_adaptor"`` trains only the new visual modules;
                ``"joint"`` trains the visual modules and existing DFlash backbone.
        """
        if stage not in {"visual_adaptor", "joint"}:
            raise ValueError(f"training_stage must be 'visual_adaptor' or 'joint', got {stage!r}.")
        if not self.visual_conditioning_enabled:
            raise ValueError("A visual training stage requires visual_num_query_tokens > 0.")
        self.requires_grad_(stage == "joint")
        for layer in self.layers:
            if layer.visual_fusion is not None:
                layer.visual_fusion.requires_grad_(True)

    def visual_parameters(self) -> list[nn.Parameter]:
        """Return all parameters owned by the visual adaptor and fusion layers."""
        parameters: list[nn.Parameter] = []
        for layer in self.layers:
            if layer.visual_fusion is not None:
                parameters.extend(layer.visual_fusion.parameters())
        return parameters

    def set_domino_training_stage(self, stage: str) -> None:
        """Freeze the DFlash backbone for head-only Domino warm-start training.

        Args:
            stage: ``"domino_head"`` trains only ``prefix_gru`` and
                ``embed_proj``; ``"joint"`` trains the full draft.
        """
        if self.projector_type != "domino":
            raise ValueError("A Domino training stage requires projector_type='domino'.")
        if stage not in {"domino_head", "joint"}:
            raise ValueError(f"Domino training_stage must be 'domino_head' or 'joint', got {stage!r}.")
        self.requires_grad_(stage == "joint")
        for parameter in self.domino_parameters():
            parameter.requires_grad_(True)

    def domino_parameters(self) -> list[nn.Parameter]:
        """Return parameters owned by the Domino causal correction head."""
        if self.projector_type != "domino":
            return []
        return [*self.prefix_gru.parameters(), *self.embed_proj.parameters()]

    def set_layer_routing_training_stage(self, stage: str) -> None:
        """Freeze the draft except for the layer-routing gates.

        Args:
            stage: ``"layer_routing"`` trains only the routing gates;
                ``"joint"`` trains the full draft.
        """
        if not self.layer_routing_enabled:
            raise ValueError("Layer-routing training requires layer_routing_enabled=true.")
        if stage not in {"layer_routing", "joint"}:
            raise ValueError(f"Layer-routing stage must be 'layer_routing' or 'joint', got {stage!r}.")
        self.requires_grad_(stage == "joint")
        self.layer_route_gates.requires_grad_(True)

    def set_spatial_rope_training_stage(self, stage: str) -> None:
        """Apply the configured MRoPE training policy.

        Args:
            stage: ``"spatial_rope"`` trains only the MRoPE gates; ``"joint"``
                trains the full draft.
        """
        if not self.spatial_rope_enabled:
            raise ValueError("Spatial-rope training requires spatial_rope_enabled=true.")
        if stage not in {"spatial_rope", "joint"}:
            raise ValueError(f"Spatial-rope stage must be 'spatial_rope' or 'joint', got {stage!r}.")
        if self.spatial_rope_mode == "replace" and stage != "joint":
            raise ValueError("spatial_rope_mode='replace' requires training_stage='joint'.")
        self.requires_grad_(stage == "joint")
        if self.spatial_rope_gates is not None:
            self.spatial_rope_gates.requires_grad_(True)

    def layer_routing_parameters(self) -> list[nn.Parameter]:
        """Return the parameters controlling target-layer routing."""
        if not self.layer_routing_enabled:
            return []
        return [self.layer_route_gates]

    def _prepare_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Hook: raw concatenated target features -> backbone context input.

        DFlash feeds the concatenation straight into its FC projection; fusion
        models (DFlare / DFly) override this to stack the captured layers into
        ``[batch, sequence, num_context_layers, hidden]`` so per-draft-layer
        fusion can weight them along the layer axis. The reshape is only ever
        done with the model's own known layer list and width.
        """
        return target_hidden

    def _build_layer_contexts(self, target_hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build one target context per draft layer without changing sequence length.

        Args:
            target_hidden: Tensor of shape [batch, context, target_layers * hidden],
                with captured target layers concatenated on the final axis.

        Returns:
            Tuple containing one tensor of shape [batch, context, hidden] per
            draft layer. At zero routing gates every entry exactly follows the
            original shared-context path.
        """
        shared_context = self.fc(target_hidden)
        if not self.layer_routing_enabled:
            normalized = self.hidden_norm(shared_context)
            return (normalized,) * len(self.layers)

        expected_hidden = len(self.target_layer_ids) * self.config.hidden_size
        if target_hidden.shape[-1] != expected_hidden:
            raise ValueError(
                "Layer-routed target_hidden has the wrong final dimension: "
                f"expected {expected_hidden}, got {target_hidden.shape[-1]}."
            )
        target_layers = target_hidden.split(self.config.hidden_size, dim=-1)
        projection_slices = self.fc.weight.split(self.config.hidden_size, dim=1)
        contexts = []
        for layer_hidden, projection, route_gate in zip(
            target_layers,
            projection_slices,
            self.layer_route_gates,
        ):
            routed_context = F.linear(layer_hidden, projection)
            route_strength = torch.tanh(route_gate).to(dtype=shared_context.dtype)
            mixed_context = shared_context + route_strength * (routed_context - shared_context)
            contexts.append(self.hidden_norm(mixed_context))
        return tuple(contexts)

    def _domino_gru_step(
        self,
        token_embedding: torch.Tensor,
        hidden_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the bias-free Domino GRU for one token without cuDNN repacking."""
        token_embedding = token_embedding.squeeze(1)
        if hidden_state is None:
            hidden_state = token_embedding.new_zeros((token_embedding.shape[0], self.gru_hidden_dim))
        input_reset, input_update, input_new = F.linear(token_embedding, self.prefix_gru.weight_ih_l0).chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = F.linear(hidden_state, self.prefix_gru.weight_hh_l0).chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)
        hidden_state = new_gate + update_gate * (hidden_state - new_gate)
        return hidden_state.unsqueeze(1), hidden_state

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Return existing DFlash parameters, excluding the visual branch."""
        visual_ids = {id(parameter) for parameter in self.visual_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in visual_ids]

    def build_visual_context(
        self,
        target_hidden: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Build one cached visual context per draft layer.

        Args:
            target_hidden: Tensor of shape [batch, sequence, target_layers * hidden],
                concatenated in the same order as ``target_layer_ids``.
            image_mask: Bool tensor of shape [batch, sequence], with True at
                target image-token positions.

        Returns:
            Tuple with one tensor of shape [batch, visual_tokens, visual_dim] per
            draft layer. ``visual_tokens`` is the learned-query count for the
            cross-attention adaptor and one for the pooled-MLP adaptor. Entry ``i``
            is computed only from target layer ``i``.
        """
        if not self.visual_conditioning_enabled:
            return ()
        expected_hidden = len(self.layers) * self.config.hidden_size
        if target_hidden.shape[-1] != expected_hidden:
            raise ValueError(
                "Visual DFlash expects one target feature per draft layer: "
                f"last dimension must be {expected_hidden}, got {target_hidden.shape[-1]}."
            )
        target_layers = target_hidden.split(self.config.hidden_size, dim=-1)
        contexts = []
        for layer, layer_target_hidden in zip(self.layers, target_layers):
            if layer.visual_fusion is None:
                raise RuntimeError("Visual DFlash layer is missing its visual fusion module.")
            contexts.append(layer.visual_fusion.build_context(layer_target_hidden, image_mask))
        return tuple(contexts)

    def _apply(self, fn, recurse=True):
        """Keep the RoPE ``inv_freq`` buffer in fp32 across dtype casts.

        ``Qwen3RotaryEmbedding`` computes the rotary angles in fp32 but reads the
        frequencies from a stored ``inv_freq`` buffer. ``model.to(bfloat16)`` -- the
        training build path -- rounds that buffer to bf16, whereas the serving
        runtime (SGLang keeps an fp32 RoPE cache) and HF's ``from_pretrained`` reload
        keep it in fp32. The resulting train/inference RoPE mismatch grows with
        absolute position (the bf16 frequencies dephase) and erodes draft
        acceptance, so ``inv_freq`` must stay fp32 on both the training and reload
        paths. A bf16 round-trip cannot be undone by upcasting, so when a cast
        rounds the buffer we recompute fresh fp32 frequencies from the rotary
        config (the same values HF derives on the fp32 paths) instead of upcasting
        the corrupted ones.
        """
        module = super()._apply(fn, recurse=recurse)
        rotary_emb = getattr(self, "rotary_emb", None)
        inv_freq = getattr(rotary_emb, "inv_freq", None) if rotary_emb is not None else None
        if (
            inv_freq is not None
            and inv_freq.is_floating_point()
            and not inv_freq.is_meta
            and inv_freq.dtype != torch.float32
        ):
            fresh = type(rotary_emb)(rotary_emb.config).inv_freq.to(device=inv_freq.device)
            rotary_emb.inv_freq = fresh
            if hasattr(rotary_emb, "original_inv_freq"):
                rotary_emb.original_inv_freq = fresh.clone()
        return module

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        image_mask: torch.Tensor | None = None,
        multimodal_position_ids: torch.Tensor | None = None,
        visual_context: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the DFlash draft over target context and parallel noise blocks.

        Args:
            position_ids: Long tensor of shape [batch, context + draft_tokens].
            attention_mask: Optional tensor or block mask describing attention
                from draft tokens to [context, draft_tokens].
            noise_embedding: Tensor of shape [batch, draft_tokens, hidden].
            target_hidden: Tensor of shape [batch, context, target_layers * hidden].
            past_key_values: Optional per-layer key/value cache.
            use_cache: Whether to update ``past_key_values``.
            image_mask: Optional bool tensor of shape [batch, context], with True
                at image-token positions.
            multimodal_position_ids: Optional long tensor of shape [3, batch,
                context + draft_tokens], with temporal, height, and width positions.
            visual_context: Optional tuple containing one tensor of shape [batch,
                visual_queries, visual_dim] per draft layer.
            **kwargs: HuggingFace attention backend arguments.

        Returns:
            Tensor of shape [batch, draft_tokens, hidden].
        """
        hidden_states = noise_embedding
        if self.visual_conditioning_enabled and visual_context is None:
            if image_mask is None:
                image_mask = torch.zeros(target_hidden.shape[:2], dtype=torch.bool, device=target_hidden.device)
            visual_context = self.build_visual_context(target_hidden, image_mask)
        layer_contexts = self._build_layer_contexts(self._prepare_target_hidden(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        multimodal_position_embeddings = None
        if self.spatial_rope_enabled:
            if multimodal_position_ids is None:
                raise ValueError("multimodal_position_ids is required when spatial_rope_enabled=true.")
            if tuple(multimodal_position_ids.shape[1:]) != tuple(position_ids.shape):
                raise ValueError(
                    "multimodal_position_ids must have shape [3, batch, sequence] matching position_ids, "
                    f"got {tuple(multimodal_position_ids.shape)} and {tuple(position_ids.shape)}."
                )
            multimodal_position_embeddings = _merge_multiaxis_rotary_embeddings(
                self.rotary_emb,
                hidden_states,
                multimodal_position_ids,
                self.spatial_rope_sections,
                interleaved=self.spatial_rope_interleaved,
            )
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=layer_contexts[layer_idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                multimodal_position_embeddings=multimodal_position_embeddings,
                spatial_rope_gate=(self.spatial_rope_gates[layer_idx] if self.spatial_rope_gates is not None else None),
                visual_context=visual_context[layer_idx] if visual_context else None,
                **kwargs,
            )
        return self.norm(hidden_states)

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative suffix, applying the Domino head when configured.

        Args:
            target: Target model that owns the token embedding and LM head used by
                the draft checkpoint.
            draft_hidden: Tensor of shape [batch, sequence, hidden] whose final
                ``block`` positions contain the current DFlash hidden states.
            block_output_ids: Tensor of shape [batch, block] whose first token is
                the target-produced anchor and whose remaining positions are masks.
                A cloned tensor is updated causally; the input is not mutated.

        Returns:
            Tensor of shape [batch, block - 1] containing the sampled draft suffix.
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
                f"draft_hidden must contain at least {block_size} current-block positions, got {draft_hidden.shape[1]}."
            )

        current_hidden = draft_hidden[:, -block_size:, :]
        if self.projector_type is None:
            if self.sample_from_anchor:
                return sample(target.lm_head(current_hidden))
            draft_logits = target.lm_head(current_hidden[:, 1:, :])
            return sample(draft_logits)

        completed_ids = block_output_ids.clone()
        base_logits = target.lm_head(current_hidden)
        suffix_start = self.pure_draft_prefix_len if self.shift_label else 1 + self.pure_draft_prefix_len
        target_embeddings = target.get_input_embeddings()
        gru_state = None

        for token_position in range(1, block_size):
            previous_token_ids = completed_ids[:, token_position - 1 : token_position]
            prefix_state, gru_state = self._domino_gru_step(target_embeddings(previous_token_ids), gru_state)
            head_position = token_position - 1 if self.shift_label else token_position
            next_token_logits = base_logits[:, head_position : head_position + 1, :]
            if head_position >= suffix_start:
                correction_features = torch.cat(
                    (current_hidden[:, head_position : head_position + 1, :], prefix_state),
                    dim=-1,
                )
                next_token_logits = next_token_logits + self.embed_proj(correction_features)
            completed_ids[:, token_position] = sample(next_token_logits).squeeze(1)

        return completed_ids[:, 1:]

    @torch.inference_mode()
    def _inference_block_mask(
        self,
        past_key_values_draft,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        block_output_ids: torch.Tensor,
    ):
        """Causal in-block additive mask for serving a JetSpec checkpoint.

        Training restricts in-block attention to offsets ``j <= i``; the default
        runtime forward passes no mask (non-causal in-block), which changes
        ~7.6% of this zoo's JetSpec checkpoint's greedy proposals. The mask
        built here forbids exactly the in-call future noise columns: the KV
        layout inside every attention layer is
        ``[cached past | this call's context | this call's noise block]``, so
        only the last ``block_size`` columns are restricted. Everything else
        (the whole cached prefix and the current context) stays visible, as in
        training. Returns ``None`` when the checkpoint did not train causally.
        """
        if not getattr(self, "inference_in_block_causal", False):
            return None
        cache_len = int(past_key_values_draft.get_seq_length())
        q_len = int(noise_embedding.shape[1])
        kv_len = cache_len + int(target_hidden.shape[1]) + q_len
        neg = torch.finfo(noise_embedding.dtype).min
        mask = torch.zeros(1, 1, q_len, kv_len, dtype=noise_embedding.dtype, device=noise_embedding.device)
        future = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=mask.device), diagonal=1)
        base = cache_len + int(target_hidden.shape[1])
        mask[0, 0, :, base : base + q_len] = torch.where(future, neg, torch.zeros((), dtype=mask.dtype, device=mask.device))
        return mask

    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]],
        temperature: float,
        target_kwargs: Optional[dict[str, torch.Tensor]] = None,
        return_stats: bool = False,
        sequential_target_verification: bool = False,
        strict_parallel_margin: float | None = None,
        draft_image_context_mode: str = "keep",
        draft_image_token_id: int | None = None,
    ) -> torch.LongTensor | tuple[torch.LongTensor, _DFlashGenerationStats]:
        """Run block-parallel speculative decoding against a Transformers target.

        Args:
            target: Frozen Transformers target model. It must expose input
                embeddings, an LM head, and a cache-aware forward method.
            input_ids: Tensor of shape [1, prompt_tokens].
            max_new_tokens: Maximum number of output tokens after the prompt.
            stop_token_ids: Token ids that terminate generation, or ``None``.
            temperature: Sampling temperature; values below ``1e-5`` use greedy
                argmax decoding.
            target_kwargs: Prompt tensors produced by the target processor. Text
                masks have shape [1, prompt_tokens]; vision tensors retain the
                target model's processor-defined flattened patch layout.
            return_stats: Whether to return draft acceptance counters.
            sequential_target_verification: Whether the target verifies draft
                candidates one token at a time. This preserves parity with
                ``generate()`` at the cost of target-side acceleration.
            strict_parallel_margin: Greedy-only guard for parallel block
                verification. When set, every verification round that would
                emit a token whose top-1/top-2 logit gap is below this margin
                is re-verified with the sequential generate()-parity path, so
                bf16 near-ties cannot flip the output away from target-only
                greedy decoding. Rounds handled by the fallback are counted in
                the returned stats.
            draft_image_context_mode: Diagnostic transformation applied only
                to image-token rows entering the draft prompt cache.
            draft_image_token_id: Image token id required when the diagnostic
                mode is ``zero`` or ``shuffle``.

        Returns:
            Tensor of shape [1, prompt_tokens + generated_tokens]. When
            ``return_stats`` is true, returns that tensor and a dictionary of
            draft, accepted, verification-step, per-round position, and timing
            statistics.
        """
        self.eval()
        if strict_parallel_margin is not None:
            if sequential_target_verification:
                raise ValueError(
                    "strict_parallel_margin requires parallel target verification "
                    "(sequential_target_verification=False)."
                )
            if float(strict_parallel_margin) <= 0.0:
                raise ValueError(f"strict_parallel_margin must be positive, got {strict_parallel_margin!r}.")
            if float(temperature) > 1e-5:
                raise ValueError(
                    "strict_parallel_margin guards argmax decisions and is only defined for greedy "
                    f"decoding (temperature={temperature!r})."
                )
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        # Verification-round width: one anchor position plus this round's
        # proposals (block_size - 1 for DFlash, block_size for DSpark's
        # sample_from_anchor convention).
        round_width = block_size + 1 if self.sample_from_anchor else block_size

        output_ids = torch.full(
            (1, max_length + round_width), self.mask_token_id, dtype=torch.long, device=target.device
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
        # Qwen2.5-VL derives 3-D MRoPE positions (and rope deltas) inside its
        # model forward. Passing the text-only 2-D arange used by Qwen3 would
        # silently disable multimodal RoPE, so let the VLM compute positions.
        target_is_vlm = callable(getattr(getattr(target, "model", None), "compute_3d_position_ids", None))
        if target_is_vlm:
            from ..adapters import adapter_for_model

            adapter = adapter_for_model(target)
        else:
            adapter = None
        target_forward_params = inspect.signature(target.forward).parameters
        target_embeddings = target.get_input_embeddings()
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()
        target_kwargs = dict(target_kwargs or {})
        draft_tokens = 0
        accepted_tokens = 0
        verify_steps = 0
        strict_parallel_rounds = 0
        strict_fallback_rounds = 0
        strict_min_decision_margin = float("inf")
        acceptance_events: list[_DFlashAcceptanceEvent] = []
        timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "target_prefill": [],
            "draft": [],
            "target_verify": [],
        }

        def start_timing() -> torch.cuda.Event | None:
            if not return_stats or not torch.cuda.is_available():
                return None
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event

        def end_timing(phase: str, start_event: torch.cuda.Event | None) -> None:
            if start_event is None:
                return
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            timing_events[phase].append((start_event, end_event))

        def target_verify_backend() -> AbstractContextManager[None]:
            if not _VERIFY_SDPA_BACKEND_SWITCH:
                return nullcontext()
            target_config = getattr(target, "config", None)
            if (
                target.device.type == "cuda"
                and target_config is not None
                and getattr(target_config, "_attn_implementation", None) == "sdpa"
            ):
                # Verification repeatedly changes the cached sequence length.
                # Avoid cuDNN SDPA plan construction for those short dynamic
                # blocks while leaving the long target prefill unchanged.
                return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
            return nullcontext()

        # Prefill the target on the prompt.
        if target_is_vlm:
            prefill_kwargs = target.prepare_inputs_for_generation(
                input_ids,
                next_sequence_length=num_input_tokens,
                past_key_values=past_key_values_target,
                is_first_iteration=True,
                use_cache=True,
                **target_kwargs,
            )
            prefill_kwargs["logits_to_keep"] = 1
            prefill_kwargs["output_hidden_states"] = True
            prefill_input_ids = prefill_kwargs.pop("input_ids")
        else:
            prefill_kwargs = {
                "past_key_values": past_key_values_target,
                "use_cache": True,
                "logits_to_keep": 1,
                "output_hidden_states": True,
                **target_kwargs,
            }
            prefill_kwargs["position_ids"] = position_ids[:, :num_input_tokens]
            prefill_input_ids = input_ids
        phase_start = start_timing()
        output = target(prefill_input_ids, **prefill_kwargs)
        end_timing("target_prefill", phase_start)
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)
        draft_multimodal_position_ids = None
        if self.spatial_rope_enabled:
            if not target_is_vlm:
                raise ValueError("Spatial-rope DFlash requires a target that exposes 3D position construction.")
            prompt_multimodal_position_ids = adapter.call_position_builder(
                target,
                input_ids=input_ids,
                image_grid_thw=target_kwargs.get("image_grid_thw"),
                video_grid_thw=target_kwargs.get("video_grid_thw"),
                inputs_embeds=None,
                attention_mask=target_kwargs.get("attention_mask"),
                past_key_values=None,
                second_per_grid_ts=target_kwargs.get("second_per_grid_ts"),
                mm_token_type_ids=target_kwargs.get("mm_token_type_ids"),
            )
            if prompt_multimodal_position_ids is None:
                raise ValueError(
                    "The target could not construct multimodal positions; image/video grid and token-type inputs "
                    "must be present."
                )
            future_length = output_ids.shape[1] - num_input_tokens
            future_offsets = torch.arange(
                1,
                future_length + 1,
                device=prompt_multimodal_position_ids.device,
                dtype=prompt_multimodal_position_ids.dtype,
            ).view(1, 1, -1)
            future_positions = prompt_multimodal_position_ids[:, :, -1:] + future_offsets
            draft_multimodal_position_ids = torch.cat(
                (prompt_multimodal_position_ids, future_positions),
                dim=-1,
            )
        if draft_image_context_mode != "keep":
            if draft_image_token_id is None:
                raise ValueError("draft_image_token_id is required for image-token context ablation.")
            draft_image_mask = input_ids == int(draft_image_token_id)
            if not draft_image_mask.any():
                raise ValueError("Image-token context ablation requires at least one image token in the prompt.")
            target_hidden = _ablate_image_token_hidden_states(
                target_hidden,
                draft_image_mask,
                draft_image_context_mode,
            )
        visual_context = None
        if self.visual_conditioning_enabled:
            if self.visual_image_token_id is None:
                raise ValueError("visual_image_token_id is required for visual DFlash inference.")
            image_mask = input_ids == int(self.visual_image_token_id)
            visual_context = self.build_visual_context(target_hidden, image_mask)

        start = num_input_tokens

        def _sequential_round(
            round_start: int,
            round_block_ids: torch.LongTensor,
        ) -> tuple[torch.Tensor, int, torch.Tensor]:
            """Verify the drafted block one target step at a time (generate()-parity path)."""
            posterior_tokens: list[torch.Tensor] = []
            verified_hidden_states: list[torch.Tensor] = []
            acceptance_length = 0
            for block_index in range(round_block_ids.shape[1]):
                token_ids = round_block_ids[:, block_index : block_index + 1]
                if target_is_vlm:
                    prompt_attention_mask = target_kwargs.get("attention_mask")
                    if prompt_attention_mask is None:
                        prompt_attention_mask = torch.ones_like(input_ids)
                    active_length = round_start + block_index + 1
                    full_attention_mask = torch.cat(
                        (
                            prompt_attention_mask,
                            torch.ones(
                                (1, active_length - num_input_tokens),
                                dtype=prompt_attention_mask.dtype,
                                device=target.device,
                            ),
                        ),
                        dim=1,
                    )
                    full_input_ids = torch.cat(
                        (output_ids[:, : round_start + block_index], token_ids),
                        dim=1,
                    )
                    generation_kwargs = {
                        key: value
                        for key, value in target_kwargs.items()
                        if key not in {"attention_mask", "mm_token_type_ids", "pixel_values", "pixel_values_videos"}
                    }
                    verify_kwargs = target.prepare_inputs_for_generation(
                        full_input_ids,
                        next_sequence_length=1,
                        past_key_values=past_key_values_target,
                        attention_mask=full_attention_mask,
                        is_first_iteration=False,
                        use_cache=True,
                        **generation_kwargs,
                    )
                    verify_input_ids = verify_kwargs.pop("input_ids")
                    verify_kwargs["attention_mask"] = full_attention_mask
                    multimodal_position_ids = adapter.call_position_builder(
                        target,
                        input_ids=token_ids,
                        image_grid_thw=None,
                        video_grid_thw=None,
                        inputs_embeds=target_embeddings(token_ids),
                        attention_mask=full_attention_mask,
                        past_key_values=past_key_values_target,
                        mm_token_type_ids=None,
                    )
                    if multimodal_position_ids is not None:
                        verify_kwargs["position_ids"] = _prepend_text_position_ids(
                            multimodal_position_ids,
                            full_attention_mask,
                        )[:, :, -1:]
                    else:
                        verify_kwargs.pop("position_ids", None)
                    if "cache_position" in target_forward_params:
                        verify_kwargs["cache_position"] = torch.tensor(
                            [round_start + block_index],
                            dtype=torch.long,
                            device=target.device,
                        )
                else:
                    verify_input_ids = token_ids
                    verify_kwargs = {
                        "position_ids": position_ids[:, round_start + block_index : round_start + block_index + 1],
                        "past_key_values": past_key_values_target,
                        "use_cache": True,
                    }
                verify_kwargs["output_hidden_states"] = True
                phase_start = start_timing()
                with target_verify_backend():
                    step_output = target(verify_input_ids, **verify_kwargs)
                end_timing("target_verify", phase_start)
                step_posterior = sample(step_output.logits[:, -1:], temperature)
                posterior_tokens.append(step_posterior)
                verified_hidden_states.append(
                    extract_context_feature(step_output.hidden_states, self.target_layer_ids)
                )
                if block_index == round_block_ids.shape[1] - 1:
                    break
                if round_block_ids[0, block_index + 1] != step_posterior[0, 0]:
                    break
                acceptance_length += 1
            posterior = torch.cat(posterior_tokens, dim=1)
            verified_target_hidden = torch.cat(verified_hidden_states, dim=1)
            return posterior, acceptance_length, verified_target_hidden

        def _parallel_round(
            round_start: int,
            round_block_ids: torch.LongTensor,
        ) -> tuple[torch.Tensor, int, torch.Tensor, float]:
            """Verify the whole drafted block in one target forward (fast path)."""
            if target_is_vlm:
                prompt_attention_mask = target_kwargs.get("attention_mask")
                if prompt_attention_mask is None:
                    prompt_attention_mask = torch.ones_like(input_ids)
                full_attention_mask = torch.cat(
                    (
                        prompt_attention_mask,
                        torch.ones(
                            (1, round_start + round_block_ids.shape[1] - num_input_tokens),
                            dtype=prompt_attention_mask.dtype,
                            device=target.device,
                        ),
                    ),
                    dim=1,
                )
                full_input_ids = torch.cat((output_ids[:, :round_start], round_block_ids), dim=1)
                generation_kwargs = {
                    key: value
                    for key, value in target_kwargs.items()
                    if key not in {"attention_mask", "mm_token_type_ids", "pixel_values", "pixel_values_videos"}
                }
                verify_kwargs = target.prepare_inputs_for_generation(
                    full_input_ids,
                    next_sequence_length=round_block_ids.shape[1],
                    past_key_values=past_key_values_target,
                    attention_mask=full_attention_mask,
                    is_first_iteration=False,
                    use_cache=True,
                    **generation_kwargs,
                )
                verify_input_ids = verify_kwargs.pop("input_ids")
                verify_kwargs["output_hidden_states"] = True
                # Qwen2.5-VL uses the full mask when deriving incremental
                # MRoPE positions. Omitting it makes the cache path fall back
                # to plain arange positions and diverge from generate().
                verify_kwargs["attention_mask"] = full_attention_mask
                # ``compute_3d_position_ids`` returns the three MRoPE axes for
                # the full mask. The fourth text-position axis is required by
                # the current Transformers causal-mask path.
                multimodal_position_ids = adapter.call_position_builder(
                    target,
                    input_ids=round_block_ids,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    inputs_embeds=target_embeddings(round_block_ids),
                    attention_mask=full_attention_mask,
                    past_key_values=past_key_values_target,
                    mm_token_type_ids=None,
                )
                if multimodal_position_ids is not None:
                    verify_kwargs["position_ids"] = _prepend_text_position_ids(
                        multimodal_position_ids,
                        full_attention_mask,
                    )[:, :, -round_block_ids.shape[1] :]
                else:
                    verify_kwargs.pop("position_ids", None)
                if "cache_position" in target_forward_params:
                    verify_kwargs["cache_position"] = torch.arange(
                        round_start,
                        round_start + round_block_ids.shape[1],
                        dtype=torch.long,
                        device=target.device,
                    )
            else:
                verify_input_ids = round_block_ids
                verify_kwargs = {
                    "past_key_values": past_key_values_target,
                    "use_cache": True,
                    "output_hidden_states": True,
                }
                verify_kwargs["position_ids"] = position_ids[:, round_start : round_start + round_block_ids.shape[1]]
            phase_start = start_timing()
            with target_verify_backend():
                output = target(verify_input_ids, **verify_kwargs)
            end_timing("target_verify", phase_start)
            posterior = sample(output.logits, temperature)
            acceptance_length = (round_block_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            verified_target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

            min_decision_margin = float("inf")
            if strict_parallel_margin is not None:
                used = min(int(acceptance_length) + 1, posterior.shape[1])
                min_decision_margin = _min_decision_margin(output.logits, used)
            return posterior, acceptance_length, verified_target_hidden, min_decision_margin

        while start < max_length:
            block_output_ids = output_ids[:, start : start + round_width].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            # The draft always consumes its own block window (anchor + block_size-1
            # mask slots); sample_from_anchor rounds only widen the verification
            # window, not the draft's input block.
            noise_embedding = target_embeddings(block_output_ids[:, :block_size])
            phase_start = start_timing()
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                multimodal_position_ids=(
                    draft_multimodal_position_ids[:, :, past_key_values_draft.get_seq_length() : start + block_size]
                    if draft_multimodal_position_ids is not None
                    else None
                ),
                past_key_values=past_key_values_draft,
                use_cache=True,
                visual_context=visual_context,
                attention_mask=self._inference_block_mask(
                    past_key_values_draft, target_hidden, noise_embedding, block_output_ids
                ),
            )
            block_output_ids[:, 1:] = self._sample_draft_tokens(
                target,
                draft_hidden,
                # Contract: the sampler receives the draft's own block window
                # (anchor + block_size - 1 mask slots). In sample_from_anchor
                # mode the round window is one position wider; the extra slot is
                # the verifier's bonus token, so the wide block must not be
                # passed through (it would also trip the block_size check).
                block_output_ids[:, :block_size],
            )
            end_timing("draft", phase_start)
            past_key_values_draft.crop(start)

            if sequential_target_verification:
                posterior, acceptance_length, verified_target_hidden = _sequential_round(start, block_output_ids)
            else:
                posterior, acceptance_length, verified_target_hidden, min_decision_margin = _parallel_round(
                    start, block_output_ids
                )
                if strict_parallel_margin is not None:
                    strict_parallel_rounds += 1
                    strict_min_decision_margin = min(strict_min_decision_margin, min_decision_margin)
                    if min_decision_margin < float(strict_parallel_margin):
                        strict_fallback_rounds += 1
                        # The parallel attempt already extended the target KV
                        # cache with the unverified block; drop it and verify
                        # the same draft again with the parity path before any
                        # token is emitted.
                        past_key_values_target.crop(start)
                        with verify_sdpa_backend_parity():
                            posterior, acceptance_length, verified_target_hidden = _sequential_round(
                                start, block_output_ids
                            )

            round_draft_tokens, round_accepted_tokens = _bounded_acceptance_counts(
                start=start,
                max_length=max_length,
                block_size=block_output_ids.shape[1],
                acceptance_length=int(acceptance_length),
            )
            draft_tokens += round_draft_tokens
            accepted_tokens += round_accepted_tokens
            verify_steps += 1
            acceptance_events.append(
                {
                    "generated_position": start - num_input_tokens + 1,
                    "accepted_tokens": round_accepted_tokens,
                    "draft_tokens": round_draft_tokens,
                }
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = verified_target_hidden[:, : acceptance_length + 1, :]
            if stop_token_ids is not None and any(
                stop_id in output_ids[:, num_input_tokens:] for stop_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_ids).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
        if not return_stats:
            return output_ids
        timing_seconds = {f"{phase}_seconds": 0.0 for phase in timing_events}
        if any(timing_events.values()):
            torch.cuda.synchronize()
            timing_seconds = {
                f"{phase}_seconds": sum(start.elapsed_time(end) for start, end in events) / 1000.0
                for phase, events in timing_events.items()
            }
        return output_ids, {
            "draft_tokens": float(draft_tokens),
            "accepted_tokens": float(accepted_tokens),
            "verify_steps": float(verify_steps),
            "acceptance_events": acceptance_events,
            "strict_parallel_rounds": float(strict_parallel_rounds),
            "strict_fallback_rounds": float(strict_fallback_rounds),
            "strict_min_decision_margin": (
                None if strict_parallel_rounds == 0 or strict_min_decision_margin == float("inf")
                else float(strict_min_decision_margin)
            ),
            **timing_seconds,
        }
