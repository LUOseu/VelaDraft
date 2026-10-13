"""DFlare draft model: per-draft-layer softmax fusion over captured target layers.

Ported from AngelSpec ``angelspec/models/draft/dflare.py`` (commit
``d3412bed231025ea85d35dbf5e0af44f1ac62a5b``, MIT) onto this repo's Qwen3-VL
draft stack; see ``docs/upstream/dflare.md`` for the file-level audit and the
VLM adaptations (MRoPE, image-token handling, block mask).

Two structural changes over :class:`Qwen3DFlashDraftModel`:

1. ``Qwen3DFlareAttention`` gives the target-context K/V dedicated
   ``k_proj_target`` / ``v_proj_target`` projections (the draft tokens keep the
   shared ``k_proj`` / ``v_proj``). Upstream DFlare shares one projection for
   both sources; that ablation is available as ``separate_context_kv=false``.
2. ``layer_fusion_weights`` of shape ``[num_draft_layers, num_context_layers]``
   replaces DFlash's single FC context: each draft layer attends to its own
   softmax-weighted mixture of the captured target layers (the fusion is
   normalized by ``hidden_norm`` before entering attention, playing the role of
   DFlare's ``context_norm``).

The training stack (anchor sampling, block-causal mask, decay CE loss) is
reused unchanged, matching upstream.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from ..dflash.draft_model import (
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    Qwen3DFlashDraftModel,
)

__all__ = [
    "Qwen3DFlareAttention",
    "Qwen3DFlareDecoderLayer",
    "Qwen3DFlareDraftModel",
    "init_fusion_weights_",
]


def init_fusion_weights_(weights: torch.Tensor, mode: str = "distance", alpha: float = 2.0) -> None:
    """In-place fusion-logit initialization (upstream semantics).

    ``distance`` (upstream default): draft layer ``d`` of ``D`` prefers the
    target layer at the same relative depth, ``t* = round(d/(D-1) * (T-1))``,
    with logits decaying linearly in ``|t - t*|`` (sharpness ``alpha``); the
    softmax over the layer axis therefore starts as a soft shallow->shallow /
    deep->deep prior instead of uniform pooling.

    ``uniform`` (zero logits) is kept only as an explicit ablation; it is *not*
    the upstream default and must be reported as such.
    """
    with torch.no_grad():
        num_draft, num_target = weights.shape
        if mode == "distance":
            distance = torch.arange(num_target, dtype=torch.float32)
            for draft_layer in range(num_draft):
                preferred = round(draft_layer / max(num_draft - 1, 1) * (num_target - 1))
                weights[draft_layer].copy_(-alpha * (distance - preferred).abs())
        elif mode == "uniform":
            weights.zero_()
        else:
            raise ValueError(f"fusion_init must be 'distance' or 'uniform', got {mode!r}.")


class Qwen3DFlareAttention(Qwen3DFlashAttention):
    """Dual-source KV attention with dedicated target-context projections.

    Same layout as DFlash (``[context | noise]`` concat -> K-norm -> RoPE); the
    only difference is where the context K/V come from when
    ``separate_context_kv`` is enabled.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        dflash_config = dict(getattr(config, "dflash_config", {}) or {})
        self.separate_context_kv = bool(dflash_config.get("separate_context_kv", True))
        if self.separate_context_kv:
            self.k_proj_target = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj_target = nn.Linear(
                config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )

    def _context_noise_kv(
        self, hidden_states: torch.Tensor, target_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.separate_context_kv:
            return super()._context_noise_kv(hidden_states, target_hidden)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        k_ctx = self.k_proj_target(target_hidden)
        v_ctx = self.v_proj_target(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        return k, v


class Qwen3DFlareDecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer with the DFlare attention variant."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3DFlareAttention(config=config, layer_idx=layer_idx)


class Qwen3DFlareDraftModel(Qwen3DFlashDraftModel):
    """DFlash draft model with learnable per-draft-layer target fusion."""

    def __init__(self, config) -> None:
        super().__init__(config)
        dflash_config = dict(getattr(config, "dflash_config", {}) or {})
        context_layer_ids = dflash_config.get("context_layer_ids") or list(self.target_layer_ids)
        self.context_layer_ids = [int(layer) for layer in context_layer_ids]
        if not self.context_layer_ids:
            raise ValueError("DFlare requires at least one context layer.")
        if len(set(self.context_layer_ids)) != len(self.context_layer_ids):
            raise ValueError(f"DFlare context_layer_ids must be unique, got {self.context_layer_ids}.")
        # The consumed layers are the fusion axis; keep ``target_layer_ids``
        # aligned so every shared code path (inference extraction, visual
        # context checks) uses the same list as the fusion weights.
        self.target_layer_ids = list(self.context_layer_ids)
        self.num_context_layers = len(self.context_layer_ids)

        target_hidden_size = int(getattr(config, "target_hidden_size", config.hidden_size))
        if target_hidden_size != int(config.hidden_size):
            raise ValueError(
                "DFlare per-layer fusion requires target_hidden_size == hidden_size, got "
                f"target_hidden_size={target_hidden_size} and hidden_size={config.hidden_size}."
            )
        self.target_hidden_size = target_hidden_size
        if bool(dflash_config.get("layer_routing_enabled", False)):
            raise ValueError("DFlare does not combine with layer_routing_enabled=true (one fusion mechanism only).")
        if int(getattr(self, "visual_num_query_tokens", 0) or 0) > 0:
            raise ValueError("DFlare does not support the visual-conditioning branch; keep visual_num_query_tokens=0.")

        # Replace DFlash's single shared FC context with per-layer fusion logits
        # (upstream deletes context_proj as well; dead parameters would still be
        # optimizer-sharded and checkpointed otherwise).
        del self.fc
        self.fusion_trainable = bool(dflash_config.get("fusion_trainable", True))
        fusion_init = str(dflash_config.get("fusion_init", "distance"))
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(len(self.layers), self.num_context_layers, dtype=torch.float32)
        )
        init_fusion_weights_(self.layer_fusion_weights, mode=fusion_init)
        self.layer_fusion_weights.requires_grad_(self.fusion_trainable)
        self.fusion_init = fusion_init
        # Non-persistent snapshot of the prior logits: diagnostics report the
        # drift of the trained fusion distributions from this initialisation.
        self.register_buffer(
            "fusion_init_logits",
            self.layer_fusion_weights.detach().clone(),
            persistent=False,
        )
        # Optimizer fp32 master reference, attached by the training loop; the
        # live parameter is bf16, where a single AdamW step is below the
        # quantisation grid and logged fusion metrics would look frozen.
        self._master_fusion_weights: torch.Tensor | None = None

    def set_master_parameters(self, master_parameters: dict) -> None:
        """Attach the optimizer's fp32 master copies (training diagnostics only)."""
        self._master_fusion_weights = master_parameters.get("layer_fusion_weights")

    def _make_decoder_layer(self, config, layer_idx):
        return Qwen3DFlareDecoderLayer(config, layer_idx)

    def _prepare_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Concat ``[B, S, T*H]`` -> stacked ``[B, S, T, H]`` (fusion axis)."""
        hidden_size = int(self.target_hidden_size)
        num_layers = self.num_context_layers
        expected = num_layers * hidden_size
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                f"DFlare target_hidden last dimension must be {expected} "
                f"({num_layers} layers x {hidden_size}), got {target_hidden.shape[-1]}. "
                "The feature layout and the model's context_layer_ids disagree."
            )
        return target_hidden.reshape(*target_hidden.shape[:-1], num_layers, hidden_size)

    def fusion_probabilities(self) -> torch.Tensor:
        """Softmax fusion distributions, shape ``[num_draft_layers, T]``."""
        return F.softmax(self.layer_fusion_weights, dim=-1)

    def _build_layer_contexts(self, target_hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One fused, normalized context per draft layer (upstream ``forward`` loop)."""
        probabilities = self.fusion_probabilities().to(target_hidden.dtype)
        contexts = []
        for draft_layer in range(len(self.layers)):
            fused = torch.einsum("t,bstd->bsd", probabilities[draft_layer], target_hidden)
            contexts.append(self.hidden_norm(fused))
        return tuple(contexts)


