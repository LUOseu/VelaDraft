"""DFly draft model: DFlash shared-FC context + DFlare fusion residual, plus an
optional hidden-states correction applied before the frozen LM head.

Ported from AngelSpec ``angelspec/models/draft/dfly.py`` and
``angelspec/models/dfly.py`` (commit
``d3412bed231025ea85d35dbf5e0af44f1ac62a5b``, Apache-2.0); see
``docs/upstream/dfly.md``.

Per draft layer ``i`` (upstream ``DFlyDraftModel._build_layer_context``)::

    layer_context_i = RMSNorm( FC(concat(target_hidden)) + fusion_i(target_hidden) )

with ``fusion_i`` the DFlare softmax mixture of the captured layers. Layers are
DFlash's shared-KV decoder layers (context and draft K/V through the same
``k_proj`` / ``v_proj``), not DFlare's separate context projections.

Hidden-states correction (TreeFlash formula (1), residual + zero-init)::

    h' = h + SwiGLU( norm(h) :: norm(embed(prev_token)) )

``down_proj`` is zero-initialized, so the correction starts as an exact identity
for the *DFly* backbone (it does **not** degenerate to plain DFlash: the fusion
residual is still present).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ..dflare.model import Qwen3DFlareDraftModel
from ..dflash.draft_model import Qwen3DFlashDecoderLayer, sample

__all__ = [
    "HiddenStatesCorrection",
    "Qwen3DFlyDraftModel",
    "build_hidden_correction",
]


class HiddenStatesCorrection(nn.Module):
    """TreeFlash hidden-states correction (upstream formula (1)).

    Args:
        hidden_size: Draft hidden width (also the previous-token embedding width,
            because the drafter reuses the target token embedding).
        intermediate_size: SwiGLU bottleneck width (upstream default: hidden).
        rms_norm_eps: Norm epsilon; both streams are normalized independently so
            the differently-scaled hidden state and embedding contribute
            comparably.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_norm = Qwen3RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.embed_norm = Qwen3RMSNorm(self.hidden_size, eps=rms_norm_eps)
        in_features = self.hidden_size + self.hidden_size
        self.gate_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Zero-init the residual projection so the correction starts at exactly
        # zero (upstream). The gate/up first-step gradient is therefore zero;
        # later steps update the module (checked by the gradient-coverage audit
        # and the correction-delta diagnostic, not by assuming step-1 grads).
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, hidden_states: torch.Tensor, prev_token_embeds: torch.Tensor) -> torch.Tensor:
        """Apply the residual correction; same shape/dtype as ``hidden_states``."""
        hidden_norm = self.hidden_norm(hidden_states)
        embed_norm = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        gate_input = torch.cat([hidden_norm, embed_norm], dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(gate_input)) * self.up_proj(gate_input))
        return hidden_states + delta.to(hidden_states.dtype)


def build_hidden_correction(config) -> Optional[HiddenStatesCorrection]:
    """Build the correction module from the draft config (None when disabled)."""
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    if not bool(dflash_config.get("enable_hidden_correction", False)):
        return None
    hidden_size = int(config.hidden_size)
    intermediate = dflash_config.get("hidden_correction_intermediate_size", None)
    intermediate = int(intermediate) if intermediate else hidden_size
    return HiddenStatesCorrection(
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
    )


class Qwen3DFlyDraftModel(Qwen3DFlareDraftModel):
    """DFlash shared-KV layers + FC context + fusion residual + correction."""

    def __init__(self, config) -> None:
        super().__init__(config)
        dflash_config = dict(getattr(config, "dflash_config", {}) or {})
        # Restore DFlash's shared-KV decoder layers (DFly keeps the FC context
        # efficiency), while the DFlare fusion weights / context norm are kept.
        if bool(dflash_config.get("separate_context_kv", False)):
            raise ValueError("DFly uses DFlash's shared context/draft K/V; separate_context_kv must be false.")
        if bool(dflash_config.get("sample_from_anchor", False)):
            raise ValueError(
                "DFly follows the DFlash slot convention (the anchor is the verifier's bonus token); "
                "sample_from_anchor must be false."
            )

        target_hidden_size = int(getattr(config, "target_hidden_size", config.hidden_size))
        if target_hidden_size != int(config.hidden_size):
            raise ValueError(
                "DFly residual fusion requires target_hidden_size == hidden_size, got "
                f"target_hidden_size={target_hidden_size} and hidden_size={config.hidden_size}."
            )
        # DFlare's __init__ deleted the FC projection; DFly re-adds it as the
        # base-context path (Linear over the flattened layer axis).
        self.fc = nn.Linear(self.num_context_layers * target_hidden_size, config.hidden_size, bias=False)
        # Match the base model's fc initialization (post_init ran before this
        # module existed, so the normal_ init would otherwise be skipped).
        nn.init.normal_(self.fc.weight, mean=0.0, std=float(getattr(config, "initializer_range", 0.02)))
        self.hidden_correction = build_hidden_correction(config)
        self.enable_hidden_correction = self.hidden_correction is not None
        # Cheap last-forward diagnostics (consumed by the trainer metrics).
        self._last_base_abs_mean: float | None = None
        self._last_residual_abs_mean: float | None = None

    def _make_decoder_layer(self, config, layer_idx):
        return Qwen3DFlashDecoderLayer(config, layer_idx)

    def _project_base_context(self, target_stacked: torch.Tensor) -> torch.Tensor:
        """DFlash FC over the stacked ``[B, S, T, H]`` target features."""
        flat = target_stacked.flatten(start_dim=2)
        return self.fc(flat.to(self.fc.weight.dtype))

    def _build_layer_contexts(self, target_stacked: torch.Tensor) -> tuple[torch.Tensor, ...]:
        base_context = self._project_base_context(target_stacked)
        probabilities = self.fusion_probabilities().to(base_context.dtype)
        contexts = []
        residual_scale = 0.0
        for draft_layer in range(len(self.layers)):
            residual = torch.einsum(
                "t,bstd->bsd",
                probabilities[draft_layer],
                target_stacked.to(base_context.dtype),
            )
            contexts.append(self.hidden_norm(base_context + residual))
            residual_scale += float(residual.detach().float().abs().mean()) / len(self.layers)
        with torch.no_grad():
            self._last_base_abs_mean = float(base_context.detach().float().abs().mean())
            self._last_residual_abs_mean = residual_scale
        return tuple(contexts)

    def context_diagnostics(self) -> dict:
        """Scalars captured on the last forward (fusion residual vs base scale)."""
        diagnostics: dict[str, float] = {}
        if self._last_base_abs_mean is not None:
            diagnostics["context_base_abs_mean"] = float(self._last_base_abs_mean)
        if self._last_residual_abs_mean is not None:
            diagnostics["context_fusion_residual_abs_mean"] = float(self._last_residual_abs_mean)
        return diagnostics

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample the draft suffix with the correction applied to real prefixes.

        Training conditions slot ``k``'s correction on the teacher-forced token at
        ``anchor + k - 1``. At inference the only correct counterpart is the
        token that slot ``k-1`` actually produced in this round (the anchor token
        for the first proposal) -- never a ground-truth token. The correction is
        therefore applied in a small sequential walk over the block: per step we
        correct one position's hidden state, project it through the frozen LM
        head, and sample. The parallel backbone forward is unchanged; only the
        ``block_size - 1`` head projections are serial (the same total FLOPs as
        one full block projection, split into per-slot GEMMs).
        """
        if self.hidden_correction is None:
            return super()._sample_draft_tokens(target, draft_hidden, block_output_ids)
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
        completed_ids = block_output_ids.clone()
        target_embeddings = target.get_input_embeddings()
        for token_position in range(1, block_size):
            previous_ids = completed_ids[:, token_position - 1 : token_position]
            previous_embeds = target_embeddings(previous_ids).to(current_hidden.dtype)
            hidden = current_hidden[:, token_position : token_position + 1, :]
            corrected = self.hidden_correction(hidden, previous_embeds)
            next_logits = target.lm_head(corrected)
            completed_ids[:, token_position] = sample(next_logits).squeeze(1)
        return completed_ids[:, 1:]

