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

"""DFlash online training wrapper.

Ported from SpecForge's ``specforge/core/dflash.py``. ``DFlashTrainerModule``
samples a set of anchor positions per sequence, builds one parallel draft block
per anchor (the block's first token is the real anchor token, the rest are
``MASK``), runs the draft model under a bespoke block attention mask, and
computes a block-wise cross-entropy loss against the ground-truth continuation
of each anchor.

Two training objectives are supported via ``loss_type``:

* ``"dflash"`` (default): the DFlash paper's fixed-anchor objective. Only block
  position 0 is a real token; positions ``1..block_size-1`` are supervised with
  the decay-weighted CE of Eq. 4 (``w_k = exp(-(k-1)/gamma)``).
* ``"variable_prefix"``: the D2SD VP-Drafter objective (arXiv:2606.04446). Each
  block draws a visible-prefix length ``l`` from a truncated geometric prior
  (``Pr(l) ~ prefix_weight_base ** l``), positions ``< l`` are filled with the
  real tokens, and only the masked positions ``>= l`` are supervised, with the
  decay re-anchored at the prefix boundary (``w_k = exp(-(k-l)/gamma)``). This
  trains the draft to re-draft block suffixes behind a variable accepted
  prefix, the regime a variable-prefix drafter sees at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sum_nonzero_weights(weights: torch.Tensor) -> torch.Tensor:
    """Reduce the same ordered effective terms regardless of zero padding.

    Recipe compaction changes the zero-padded layout. The legacy decayed-CE
    weight is reporting-only for those recipes, but shape-dependent FP32
    reductions can otherwise report different totals for identical terms.
    """
    flat = weights.reshape(-1)
    return flat[flat != 0].sum()

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

from .mask import (
    DFlashMaskSpec,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
    estimate_dflash_attention_memory,
)
from .chunked_attention import AttentionRuntimeStats
from .decay_loss import DFlashDecayLoss
from .draft_model import Qwen3DFlashDraftModel
from ..objectives import CONFIDENCE_LOGITS_AUX_KEY, TeacherLogitsStream, recipe_loss
from .. import runtime


def _context_doc_ids(seq_lens: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Per-context-token document id ``[B, S]`` from packed ``seq_lens`` ``[B, max_docs]``.

    Mirrors the ``doc_id`` construction in ``build_block_causal_additive_mask``: a
    token's id is the number of document boundaries at or before its position, so
    0-length padding entries never split a real document.
    """
    boundaries = seq_lens.to(device).cumsum(dim=1)  # [B, max_docs]
    positions = torch.arange(seq_len, device=device)
    return (boundaries.unsqueeze(1) <= positions.view(1, -1, 1)).sum(dim=2)


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialise a (possibly tensor-parallel) tensor as a plain local tensor.

    Under tensor parallelism the target's column-parallel ``lm_head`` and
    vocab-parallel ``embed_tokens`` return ``DTensor`` outputs. The draft and the
    block-wise loss consume plain tensors, so gather the full tensor. A no-op for
    an already-plain (unsharded / replicated) tensor.
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


class NoValidAnchorsError(ValueError):
    """Raised when a batch has no sample long enough to form a DFlash block.

    A DFlash anchor needs at least ``block_size + 1`` supervised tokens (the
    anchor plus its block). Datasets always contain some short conversations;
    the training loop catches this and skips the offending micro-batch rather
    than aborting the run.
    """


@dataclass
class DFlashStepMetrics:
    """Per-step training outputs for the DFlash draft.

    Attributes:
        loss: Scalar tensor containing the differentiable training loss.
        loss_weight: Scalar tensor containing the effective loss denominator.
        accuracy: Scalar tensor containing greedy token accuracy.
        valid_tokens: Scalar tensor containing the supervised-token count.
        correct_tokens: Scalar tensor containing the greedy-correct token count.
        accept_len: Scalar tensor containing mean bonus-token-inclusive acceptance length.
        accept_len_sum: Scalar tensor containing the additive acceptance-length sum.
        valid_blocks: Scalar tensor containing the number of evaluated draft blocks.

    The count and sum fields retain the additive statistics needed for
    token-weighted, distributed validation. Averaging ``accuracy`` or
    ``accept_len`` per micro-batch would bias batches with fewer valid tokens or blocks.
    """

    loss: torch.Tensor
    loss_weight: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    correct_tokens: torch.Tensor
    accept_len: torch.Tensor
    accept_len_sum: torch.Tensor
    valid_blocks: torch.Tensor
    pos_correct: Optional[torch.Tensor] = None
    pos_count: Optional[torch.Tensor] = None
    extra_metrics: Optional[dict] = None


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Count consecutive accepted predictions for every draft block.

    Args:
        pred_ids_4d: Long tensor of shape ``[batch, blocks, depth]``.
        target_ids_4d: Long tensor of shape ``[batch, blocks, depth]``.
        valid_mask_4d: Bool tensor of shape ``[batch, blocks, depth]``.

    Returns:
        Float tensor of shape ``[batch, blocks]`` containing the accepted draft
        tokens before the first mismatch. Invalid positions do not truncate the
        prefix and do not contribute to its length.
    """
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def compute_acceptance_stats(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, sum, and count for bonus-token-inclusive acceptance length.

    Args:
        pred_ids_4d: Long tensor of shape ``[batch, blocks, depth]``.
        target_ids_4d: Long tensor of shape ``[batch, blocks, depth]``.
        valid_mask_4d: Bool tensor of shape ``[batch, blocks, depth]``.

    Returns:
        Three scalar tensors: mean acceptance length, its additive sum, and the
        number of blocks containing at least one valid drafted token.
    """
    block_accept = compute_accept_len(pred_ids_4d, target_ids_4d, valid_mask_4d)
    valid_block_mask = valid_mask_4d.any(dim=2)
    valid_blocks = valid_block_mask.sum()
    accept_len_sum = ((block_accept + 1.0) * valid_block_mask).sum()
    accept_len = accept_len_sum / valid_blocks.clamp_min(1).float()
    return accept_len, accept_len_sum, valid_blocks


_DFLASH_LOSS_TYPES = ("dflash", "variable_prefix")
SUPPORTED_ATTENTION_BACKENDS = ("flex_attention", "sdpa", "sdpa_chunked", "npu_fusion")


class DFlashTrainerModule(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    # Which batch-level errors the training loop may skip (algorithms can
    # override this; the loop falls back to NoValidAnchorsError when unset).
    skippable_errors = (NoValidAnchorsError,)

    # In-block attention mask. DFlash (and Domino/DSpark, which inherit the
    # bidirectional objective) train with the bidirectional block mask; JetSpec
    # trains causally so the draft factorization mirrors the target's
    # autoregressive order (its trainer subclass sets this to True).
    causal_in_block: bool = False

    # Padding-block row compaction (``recipe_row_compaction``) is exact only for
    # algorithms whose scoring tail is row-local. DFlare/DFly set this True;
    # algorithms that score logits with block structure (DSpark's Markov bias,
    # Domino's second scoring pass) must leave it False.
    supports_recipe_row_compaction: bool = False
    supports_chunked_ce: bool = True

    def __init__(
        self,
        draft_model: Qwen3DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        prefix_weight_base: float = 0.9,
        sample_from_anchor: bool = False,
        train_recipe=None,
        recipe_row_compaction: Optional[bool] = None,
        recipe_compile_tv_kl: bool = False,
        teacher_projection_dedup: bool = False,
        ce_chunk_size: int = 0,
        ce_backward: str = "checkpoint",
        attention_chunk_size: int = 128,
        attention_memory_budget_gb: float | None = None,
        npu_fusion_fallback: bool = False,
    ):
        super().__init__()
        if loss_type not in _DFLASH_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {_DFLASH_LOSS_TYPES}, got {loss_type!r}")
        if prefix_weight_base <= 0:
            raise ValueError(f"prefix_weight_base must be > 0, got {prefix_weight_base}")
        self.draft_model = draft_model
        # Keep the frozen target lm_head / embed_tokens as NON-registered
        # references. Under tensor parallelism their weights are DTensors; a
        # registered (DDP-wrapped) sharded param would break DDP's parameter
        # broadcast/bucketing. Non-registering keeps the DDP-wrapped trainer to
        # the plain draft params only, while ``self.lm_head`` / ``self.embed_tokens``
        # attribute access still works. (Mirrors EAGLE core_v12's _target_lm_head.)
        object.__setattr__(self, "lm_head", target_lm_head)
        object.__setattr__(self, "embed_tokens", target_embed_tokens)
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = str(attention_backend)
        if self.attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {SUPPORTED_ATTENTION_BACKENDS}, "
                f"got {attention_backend!r}."
            )
        self.attention_chunk_size = int(attention_chunk_size)
        if self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be a positive integer.")
        self.attention_memory_budget_gb = (
            None if attention_memory_budget_gb in (None, "", 0, 0.0) else float(attention_memory_budget_gb)
        )
        if self.attention_memory_budget_gb is not None and self.attention_memory_budget_gb <= 0:
            raise ValueError("attention_memory_budget_gb must be positive when set.")
        self.npu_fusion_fallback = bool(npu_fusion_fallback)
        self.last_attention_memory_plan: dict | None = None
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.prefix_weight_base = float(prefix_weight_base)
        # Block-slot convention. DFlash (sample_from_anchor=False) treats the
        # anchor as the clean bonus token: slot k predicts ``anchor + k`` and
        # only slots 1..block_size-1 are supervised. DSpark with
        # sample_from_anchor=True drafts the anchor position too: slot k
        # predicts ``anchor + 1 + k`` and every slot is supervised.
        self.sample_from_anchor = bool(sample_from_anchor)
        self.input_shift = 1 if self.sample_from_anchor else 0
        self.first_supervised_slot = 0 if self.sample_from_anchor else 1
        # Smallest visible-prefix length variable-prefix training samples (and the
        # slice point of its loss); single source of truth for both methods.
        self._min_prefix = min(2, block_size - 1)
        # Optional training recipe (``vlm_dflash.objectives.ResolvedRecipe``).
        # ``None`` keeps the historical decayed block-CE path byte-identical;
        # a recipe swaps the objective (and its position weighting) per stage.
        self.train_recipe = train_recipe
        self.ce_chunk_size = int(ce_chunk_size)
        self.ce_backward = str(ce_backward)
        if self.ce_backward not in {"checkpoint", "precompute"}:
            raise ValueError("ce_backward must be 'checkpoint' or 'precompute'.")
        if self.ce_backward == "precompute" and not self.ce_chunk_size:
            raise ValueError("ce_backward='precompute' requires ce_chunk_size > 0.")
        if self.ce_chunk_size < 0:
            raise ValueError("ce_chunk_size must be nonnegative (0 keeps dense scoring).")
        if self.ce_chunk_size and (
            not self.supports_chunked_ce or self._recipe_active() or loss_type != "dflash"
        ):
            raise ValueError("ce_chunk_size supports only the shared fixed-anchor controlled-CE objective.")
        self.teacher_projection_dedup = bool(teacher_projection_dedup)
        if self.teacher_projection_dedup and not self.recipe_needs_teacher:
            raise ValueError(
                "teacher_projection_dedup requires a shared training recipe that consumes teacher logits. "
                "Use upstream_two_stage (DFlash/DFlare/DFly) or upstream_dspark (DSpark)."
            )
        self._recipe_stage_index = 0 if train_recipe is not None else None
        # Recipe runs may drop zero-masked padding blocks from the LM head and
        # the scoring step: kept rows are bitwise equivalent and padding rows
        # cannot contribute loss or gradient (their weight is exactly 0).
        # ``None`` enables it wherever the algorithm declares a row-local
        # scoring tail; an explicit True on an unsupported algorithm is
        # rejected instead of silently ignored.
        if recipe_row_compaction is None:
            self.recipe_row_compaction = bool(self.supports_recipe_row_compaction)
        else:
            self.recipe_row_compaction = bool(recipe_row_compaction)
            if self.recipe_row_compaction and not self.supports_recipe_row_compaction:
                raise ValueError(
                    "recipe_row_compaction=True is not supported by "
                    f"{type(self).__name__}: its scoring tail is not row-local."
                )
        # Execution-optimization knob (default off): compile the fp32 TV/KL
        # chunk arithmetic of recipe stages whose objective routes through it
        # (``lk_coef`` > 0). Stages that do not call ``_tv_kl`` (e.g.
        # ``e2e_tv``) are exempt by construction, so the flag is a no-op there.
        self.recipe_compile_tv_kl = bool(recipe_compile_tv_kl)
        # Feature-store layout (configured by the training script once the
        # store/source manifest is known; see ``configure_feature_layout``).
        # ``None`` keeps the legacy behavior: the batch's ``hidden`` already is
        # the concatenation of the layers this module consumes.
        self.capture_layer_ids: list[int] | None = None
        self.capture_hidden_size: int | None = None
        # Per-forward workload counters (rows allocated / kept / scored, head
        # projection calls and rows). Read by the training loop after the
        # backward, so projections recomputed under activation checkpointing
        # are counted too.
        self.last_workload: dict = {}

        # Block-wise decay-weighted CE for the fixed-anchor objective.
        # ``normalize="mean"`` gives a local per-micro-batch decay-weighted mean;
        # ``loss_decay_gamma=None`` disables decay (uniform weights). The
        # variable-prefix objective needs per-block data-dependent weights and
        # computes its loss inline instead (see _variable_prefix_loss).
        self.loss_fn = (
            DFlashDecayLoss(
                loss_gamma=loss_decay_gamma,
                normalize="mean",
                first_slot=self.first_supervised_slot,
            )
            if loss_type == "dflash"
            else None
        )

        # Per-block offset constant (block_size,) for label gathering / position ids.
        self.register_buffer("_block_offsets", torch.arange(block_size).view(1, 1, -1), persistent=False)

    def _count(self, **counts) -> None:
        """Add to this forward's workload counters."""
        for key, value in counts.items():
            self.last_workload[key] = self.last_workload.get(key, 0) + int(value)

    def _defer_student_head(self) -> bool:
        """True when the algorithm projects its own (compacted) student rows."""
        return False

    def execution_switches(self) -> dict:
        """Execution-only switches in effect (they never change the objective)."""
        return {
            "recipe_row_compaction": bool(self.recipe_row_compaction),
            "recipe_compile_tv_kl": bool(self.recipe_compile_tv_kl),
            "teacher_projection_dedup": bool(self.teacher_projection_dedup),
            "ce_chunk_size": self.ce_chunk_size,
            "ce_backward": self.ce_backward,
        }

    # ------------------------------------------------------------------ #
    # feature layout
    # ------------------------------------------------------------------ #

    def consumed_layer_ids(self) -> list[int]:
        """Ordered target layers this module feeds to its draft model.

        Base DFlash consumes the draft model's routed layers; fusion methods
        (DFlare/DFly) consume their explicit context list. Both are absolute
        target layer indices, so the feature store can be sliced by identity.
        """
        model = self.draft_model
        context_ids = getattr(model, "context_layer_ids", None)
        if context_ids is not None:
            return [int(layer) for layer in context_ids]
        return [int(layer) for layer in getattr(model, "target_layer_ids")]

    def configure_feature_layout(self, capture_layer_ids, hidden_size: int | None = None) -> None:
        """Record the feature store's layer layout (ordered ids + per-layer width).

        Called by the training script with the manifest/source layout. The
        module then slices every batch's concatenated ``hidden`` to the layers
        it consumes -- by explicit layer id, never by guessing the layout from
        the tensor shape. Refuses stores that lack any consumed layer instead
        of silently training on the wrong context.
        """
        capture = [int(layer) for layer in capture_layer_ids]
        if len(set(capture)) != len(capture):
            raise ValueError(f"capture_layer_ids {capture} contain duplicates; cannot slice a store safely.")
        needed = self.consumed_layer_ids()
        missing = [layer for layer in needed if layer not in capture]
        if missing:
            raise ValueError(
                f"algorithm {type(self).__name__} needs target layers {needed}, but the feature source "
                f"holds {capture} (missing {missing}). Recapture with drafter.capture_layer_ids set to a "
                "superset of the method's layers."
            )
        expected_hidden = int(getattr(self.draft_model.config, "target_hidden_size",
                                      self.draft_model.config.hidden_size))
        if hidden_size is not None and int(hidden_size) != expected_hidden:
            raise ValueError(
                f"feature store hidden size {int(hidden_size)} disagrees with the draft's expected "
                f"target hidden size {expected_hidden}."
            )
        self.capture_layer_ids = capture
        self.capture_hidden_size = int(hidden_size) if hidden_size is not None else expected_hidden

    def _select_context_layers(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, S, len(capture)*H] -> [B, S, len(consumed)*H] in consumed order."""
        if self.capture_layer_ids is None:
            return hidden
        from .features import select_capture_layers

        expected_width = len(self.capture_layer_ids) * int(self.capture_hidden_size)
        if hidden.shape[-1] != expected_width:
            raise ValueError(
                f"batch hidden width {hidden.shape[-1]} does not match the feature layout "
                f"len(capture_layer_ids)={len(self.capture_layer_ids)} * hidden_size="
                f"{int(self.capture_hidden_size)} = {expected_width}; the store and its manifest disagree "
                "(recapture the features)."
            )
        return select_capture_layers(hidden, self.capture_layer_ids, self.consumed_layer_ids())

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        doc_remaining: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns ``(anchors, keep_mask)``.

        ``doc_remaining`` ``[B, S]`` (sequence packing) restricts anchors so the
        whole block stays inside one document (``doc_remaining >= block_size - 1``),
        the per-document analogue of the ``anchor <= seq_len - block_size`` bound.
        This is required for correctness -- ``_build_block_targets`` gathers labels
        by absolute offset and does not encode document boundaries, so a block that
        crossed one would be supervised on the next document's tokens. A side effect
        is that a packed document shorter than ``block_size`` yields no anchors (the
        unpacked path still supervises such a short sequence's partial block); pack
        with documents at least ``block_size`` long to avoid dropping their signal.
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if doc_remaining is not None:
            # Keep the block within the anchor's document: its last predicted
            # token (anchor + block_size - 1) must still be a real token of that
            # document, i.e. at least block_size - 1 real tokens follow the anchor.
            valid = valid & (doc_remaining[:, : max_anchor + 1] >= bs - 1)
        valid_counts = valid.sum(dim=1)
        # ``valid`` already restricts positions to ``[0, seq_len - block_size]``, so
        # every valid position has room for a full block and is a legitimate anchor.
        # Draw up to the richest sample's valid count (per-sample padding is handled
        # by ``keep_mask`` below); no -1, which would spuriously raise when the
        # richest sample has exactly one valid anchor and always drop one otherwise.
        max_n = min(self.num_anchors, int(valid_counts.max().item()))
        if max_n <= 0:
            raise NoValidAnchorsError(
                "No valid anchor positions in this batch; every sample has fewer than "
                f"block_size+1 ({bs + 1}) supervised tokens."
            )

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device))
        return anchors, keep_mask

    def _sample_prefix_lengths(self, bsz: int, n_blocks: int, device: torch.device) -> torch.Tensor:
        """Sample a visible-prefix length per block for variable-prefix training.

        A prefix length ``l`` means block positions ``[0, l)`` are visible real
        tokens and positions ``[l, block_size)`` are masked prediction targets.
        Lengths follow D2SD's truncated geometric prior ``Pr(l) ~ base ** l`` over
        ``[min(2, block_size - 1), block_size - 1]``; a base below 1 biases toward
        short prefixes. The lower bound skips the degenerate fixed-anchor DFlash
        case, and the upper bound keeps at least one masked target per block.

        Returns:
            prefix_lengths: Long tensor of shape ``[batch, blocks]``.
        """
        min_prefix = self._min_prefix
        max_prefix = self.block_size - 1
        if max_prefix <= min_prefix:
            return torch.full((bsz, n_blocks), min_prefix, dtype=torch.long, device=device)
        prefix_ids = torch.arange(min_prefix, max_prefix + 1, device=device, dtype=torch.float32)
        weights = self.prefix_weight_base**prefix_ids
        samples = torch.multinomial(weights, num_samples=bsz * n_blocks, replacement=True)
        return samples.view(bsz, n_blocks) + min_prefix

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, context_position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Position ids for the parallel draft blocks (anchor position + offset).

        Without packing the anchor's position equals its row index, so the block
        positions are ``anchor + offset``. Under packing ``context_position_ids``
        ``[B, S]`` holds per-document reset positions, so the block's base position
        is gathered from it at the anchor (``context_position_ids[anchor] + offset``)
        to keep the draft's RoPE phase document-local.
        """
        bsz = anchor_positions.shape[0]
        offsets = torch.arange(self.block_size, device=anchor_positions.device).view(1, 1, -1)
        if context_position_ids is None:
            base = anchor_positions.unsqueeze(-1)
        else:
            base = torch.gather(context_position_ids, 1, anchor_positions.clamp(min=0)).unsqueeze(-1)
        pos_ids = base + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        """Embed each block as ``[anchor_token, MASK, MASK, ...]`` (invalid blocks all MASK)."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full((bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device)
        block_starts = (torch.arange(n, device=device) * bs).unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        # A tensor-parallel target's embed_tokens is vocab-parallel and returns a
        # DTensor; gather it so the (plain) draft can consume the noise embedding.
        return _to_full_tensor(self.embed_tokens(noise_ids))

    def _create_vp_noise_embed(self, input_ids, anchor_positions, block_keep_mask, prefix_lengths):
        """Embed the draft blocks with a visible prefix of real tokens, then ``MASK``.

        The variable-prefix analogue of :meth:`_create_noise_embed`: block
        positions ``< prefix_lengths`` hold the real sequence tokens, the rest
        (and every position of an invalid block) hold ``MASK``.

        Args:
            input_ids: Long tensor of shape ``[batch, sequence]``.
            anchor_positions: Long tensor of shape ``[batch, blocks]``; each block's
                start position in the sequence.
            block_keep_mask: Bool tensor of shape ``[batch, blocks]``; invalid
                (padding) blocks are embedded as all ``MASK``.
            prefix_lengths: Long tensor of shape ``[batch, blocks]``; this block's
                visible-prefix length.

        Returns:
            noise_embedding: Tensor of shape ``[batch, blocks * block_size, hidden]``.
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]

        token_positions = anchor_positions.unsqueeze(-1) + self._block_offsets  # [B, N, bs]
        safe_positions = token_positions.clamp(0, seq_len - 1)
        real_tokens = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_positions)
        visible_prefix = self._block_offsets < prefix_lengths.unsqueeze(-1)
        fill_mask = visible_prefix & block_keep_mask.unsqueeze(-1) & (token_positions < seq_len)
        mask_tokens = torch.full_like(real_tokens, self.mask_token_id)
        noise_ids = torch.where(fill_mask, real_tokens, mask_tokens).view(bsz, n * self.block_size)
        # A tensor-parallel target's embed_tokens is vocab-parallel and returns a
        # DTensor; gather it so the (plain) draft can consume the noise embedding.
        return _to_full_tensor(self.embed_tokens(noise_ids))

    def _build_block_targets(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
        label_start: int = 0,
        doc_remaining: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-block ground-truth tokens and the supervised-position mask.

        Returns ``(label_indices, target_ids, block_mask)`` each of shape
        ``[B, N, block_size]``. ``label_indices[..., k]`` is the sequence position
        block position ``k`` predicts (``anchor + label_start + k``); ``block_mask``
        is the product of block validity, in-bounds, and the gathered loss mask.
        Shared by the block-wise trainers (DFlash, JetSpec, and Domino, which passes
        ``label_start=1`` for ``shift_label``) so the label/mask gathering lives in
        one place. Under packing, ``doc_remaining`` ``[B, S]`` truncates labels at
        the anchor's document boundary: anchor sampling only keeps offsets up to
        ``block_size - 1`` inside the anchor's document, and a shifted label window
        (``label_start > 0``) reaches one past that guarantee.
        """
        n = anchor_positions.size(1)
        label_offsets = self._block_offsets + label_start  # [1, 1, bs]
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, N, bs]
        valid_label_mask = label_indices < seq_len
        if doc_remaining is not None:
            doc_rem_at_anchor = torch.gather(doc_remaining, 1, anchor_positions.clamp(min=0)).unsqueeze(-1)
            valid_label_mask = valid_label_mask & (label_offsets <= doc_rem_at_anchor)
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)
        gathered_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)
        block_mask = block_keep_mask.unsqueeze(-1).float() * valid_label_mask.float() * gathered_loss_mask
        return label_indices, target_ids, block_mask

    def _prepare_block_inputs(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
        causal: bool = False,
        anchor_table: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | BlockMask", Optional[torch.Tensor]
    ]:
        """Shared block-drafting prologue: anchors, noise embedding, positions, mask.

        Centralises the sequence-packing handling for every block-wise trainer
        (DFlash and its Domino / JetSpec subclasses): anchors are sampled so each
        block stays inside one document, the block's context prefix is restricted
        to its anchor's document, and the draft RoPE uses per-document positions.
        ``anchor_table`` replays a frozen ``(anchors, keep)`` pair instead of
        sampling (``train.regroup``; the caller pads each group to
        ``max(anchor) + block_size`` so the replay stays in bounds).

        Under ``loss_type="variable_prefix"`` the block embedding shows a sampled
        real-token prefix (``_create_vp_noise_embed``) instead of the single anchor
        token, and the sampled ``prefix_lengths`` are returned for the loss; every
        other ``loss_type`` returns ``prefix_lengths=None``.

        Args:
            input_ids:     ``[B, S]`` context token ids (long).
            loss_mask:     ``[B, S]`` supervised-token mask.
            position_ids:  ``[B, S]`` per-document reset positions (packing), or ``None``.
            seq_lens:      ``[B, max_docs]`` packed document lengths, or ``None`` (unpacked).
            doc_remaining: ``[B, S]`` remaining real tokens of each position's document.
            causal:        When True, build the in-block-causal (JetSpec) mask instead
                of the bidirectional (DFlash / Domino) one.

        Returns:
            ``(anchor_positions [B, N], block_keep_mask [B, N], noise_embedding
            [B, N*block_size, H], full_position_ids [B, S + N*block_size],
            attention mask, prefix_lengths [B, N] or None)``; the mask is a flex
            ``BlockMask`` or a dense additive ``[B, 1, N*block_size, S + N*block_size]``
            tensor, per backend.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        packed = seq_lens is not None
        if packed and (position_ids is None or doc_remaining is None):
            # A partial set would silently drop the in-document anchor constraint
            # (cross-document supervision) or the per-document RoPE positions.
            raise ValueError(
                "Sequence packing requires position_ids, seq_lens, and doc_remaining together; "
                "got seq_lens without the other packing metadata."
            )

        if anchor_table is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device, doc_remaining=doc_remaining if packed else None
            )
        else:
            anchor_positions, block_keep_mask = anchor_table
            anchor_positions = anchor_positions.to(device=device, dtype=torch.long)
            block_keep_mask = block_keep_mask.to(device=device, dtype=torch.bool)
            if (
                anchor_positions.dim() != 2
                or anchor_positions.shape[0] != bsz
                or block_keep_mask.shape != anchor_positions.shape
            ):
                raise ValueError(
                    "anchor_table must be (anchor_positions [B, N], keep [B, N]) matching the "
                    f"batch size; got {tuple(anchor_positions.shape)} / {tuple(block_keep_mask.shape)} "
                    f"for B={bsz}."
                )
            kept = anchor_positions[block_keep_mask]
            if kept.numel() and (
                bool((kept < 0).any()) or bool((kept + self.block_size > seq_len).any())
            ):
                raise ValueError(
                    "anchor_table contains kept anchors outside [0, seq_len - block_size]; "
                    "pad the group to max(anchor) + block_size before replaying."
                )
        if self.loss_type == "variable_prefix":
            prefix_lengths = self._sample_prefix_lengths(bsz, anchor_positions.shape[1], device)
            noise_embedding = self._create_vp_noise_embed(input_ids, anchor_positions, block_keep_mask, prefix_lengths)
        else:
            prefix_lengths = None
            noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        if packed:
            context_position_ids = position_ids
            ctx_doc_id = _context_doc_ids(seq_lens, seq_len, device)
            anchor_doc_id = torch.gather(ctx_doc_id, 1, anchor_positions.clamp(min=0))
        else:
            context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            ctx_doc_id = None
            anchor_doc_id = None
        draft_position_ids = self._create_position_ids(anchor_positions, context_position_ids if packed else None)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        config = getattr(self.draft_model, "config", None)
        heads = int(getattr(config, "num_attention_heads", 1))
        head_dim = int(
            getattr(config, "head_dim", None)
            or getattr(config, "hidden_size", heads * 1) // max(heads, 1)
        )
        plan = estimate_dflash_attention_memory(
            batch_size=bsz,
            num_blocks=anchor_positions.shape[1],
            block_size=self.block_size,
            context_len=seq_len,
            num_heads=heads,
            head_dim=head_dim,
            dtype=noise_embedding.dtype,
            query_chunk_size=(
                self.attention_chunk_size
                if self.attention_backend in ("sdpa_chunked", "npu_fusion")
                else None
            ),
        )
        plan["requested_backend"] = self.attention_backend
        budget = self.attention_memory_budget_gb
        if budget is not None:
            # Compare raw bytes: the JSON-facing GB fields are rounded to four
            # decimals, which would make a sub-0.0001 GB budget look satisfied.
            relevant_bytes = (
                plan["chunked"]["total_bytes"] if plan["chunked"] is not None else plan["dense"]["total_bytes"]
            )
            relevant = relevant_bytes / 1e9
            plan["budget_gb"] = budget
            plan["fits_budget"] = bool(relevant <= budget)
            if not plan["fits_budget"]:
                hint = (
                    "lower drafter.attention_chunk_size"
                    if plan["chunked"] is not None
                    else "select attention_backend=sdpa_chunked or raise drafter.attention_memory_budget_gb"
                )
                raise RuntimeError(
                    f"Attention memory plan needs {relevant:.3f} GB for "
                    f"B={bsz}, anchors={anchor_positions.shape[1]}, context={seq_len}, "
                    f"block={self.block_size}, heads={heads} but the configured budget is {budget:.3f} GB; {hint}."
                )
        self.last_attention_memory_plan = plan

        if self.attention_backend == "flex_attention":
            if device.type == "npu":
                raise RuntimeError(
                    "attention_backend=flex_attention is not enabled on Ascend NPU; "
                    "use sdpa_chunked or opt into npu_fusion explicitly."
                )
            attn_mask = create_dflash_block_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                causal=causal,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        elif self.attention_backend == "sdpa":
            attn_mask = create_dflash_sdpa_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                dtype=noise_embedding.dtype,
                causal=causal,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        else:
            if self.attention_backend == "npu_fusion" and device.type != "npu":
                raise RuntimeError(
                    "attention_backend=npu_fusion requires an Ascend NPU training device; "
                    f"the batch is on {device}. Use sdpa_chunked for CPU/CUDA execution."
                )
            attn_mask = DFlashMaskSpec(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                ctx_len=seq_len,
                block_size=self.block_size,
                causal=causal,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
                query_chunk_size=self.attention_chunk_size,
                backend=self.attention_backend,
                npu_fusion_fallback=self.npu_fusion_fallback,
            )
        return anchor_positions, block_keep_mask, noise_embedding, full_position_ids, attn_mask, prefix_lengths

    def reset_attention_runtime_stats(self) -> None:
        """Start a local measurement window after warmup; never changes model state."""
        for layer in getattr(self.draft_model, "layers", ()):
            attention = getattr(layer, "self_attn", None)
            stats = getattr(attention, "attention_runtime_stats", None)
            if stats is not None:
                stats.reset()
                attention.last_attention_backend_info = None

    def attention_runtime_report(self) -> dict:
        """Snapshot latest layer paths plus cumulative local measurement-window counts.

        See ``measurement_window.count_semantics`` before interpreting counts.
        This does not measure hardware memory, backward kernels, or update time.
        """
        layers = getattr(self.draft_model, "layers", ())
        infos = [
            info
            for info in (
                getattr(getattr(layer, "self_attn", None), "last_attention_backend_info", None)
                for layer in layers
            )
            if info
        ]
        window = AttentionRuntimeStats()
        for layer in layers:
            stats = getattr(getattr(layer, "self_attn", None), "attention_runtime_stats", None)
            if stats is not None:
                window.merge(stats.snapshot())
        effective = sorted({str(info.get("effective_backend")) for info in infos})
        fallback_reasons = sorted(
            {str(info["fallback_reason"]) for info in infos if info.get("fallback_reason")}
        )
        device = next(self.draft_model.parameters()).device
        compile_requested = bool(self.recipe_compile_tv_kl)
        compile_allowed = runtime.compile_supported(device)
        compile_gate = {
            "requested": compile_requested,
            "effective": compile_requested and compile_allowed,
            "reason": (
                "not_requested" if not compile_requested else
                f"runtime.compile_supported disabled for {device.type}" if not compile_allowed else
                "device_gate_allows_compile; actual use depends on active objective"
            ),
            "scope": "device_gate_not_execution_evidence",
        }
        return {
            "requested_backend": self.attention_backend,
            "effective_backends": effective,
            "training_switches": {"recipe_compile_tv_kl": compile_gate},
            "fallback_reasons": fallback_reasons,
            "chunked_layers": sum(1 for info in infos if info.get("activation_checkpointing")),
            "memory_plan": self.last_attention_memory_plan,
            "memory_plan_kind": "analytical_estimate_not_measured_peak_hbm",
            "measurement_window": window.snapshot(),
            "mixed_backend": any(info.get("mixed_backend", False) for info in infos) or len(effective) > 1,
            "fused_chunks": sum(info.get("fused_chunks", 0) for info in infos),
            "fallback_chunks": sum(info.get("fallback_chunks", 0) for info in infos),
        }

    def _apply_algorithm_terms(
        self,
        *,
        logits: torch.Tensor,
        output_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        label_indices: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        bsz: int,
        n: int,
        bs: int,
        seq_len: int,
        teacher_hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """Algorithm seam between the draft backbone and the block CE loss.

        Receives the raw block logits ``[bsz, n * bs, vocab]`` (and the draft
        hidden states ``[bsz, n * bs, hidden]`` that produced them) and returns
        ``(corrected_logits, auxiliary_loss, extra_metrics)``. The base DFlash
        implementation is the identity; subclasses add their own terms (DSpark
        adds the Markov logit bias here and its confidence-head BCE as the
        auxiliary loss).
        """
        return logits, None, None

    def _teacher_forced_prev_tokens(
        self, input_ids: torch.Tensor, label_indices: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Ground-truth token immediately before each supervised position.

        Args:
            input_ids: ``[B, S]`` context token ids.
            label_indices: ``[B, N, block_size]`` positions the block shadows.
            n: Number of anchor slots (``label_indices.size(1)``).

        Returns:
            Long tensor ``[B, N, block_size]``; ``input_ids[label - 1]`` with the
            first context position clamped (its slot is never supervised).
        """
        prev_indices = (label_indices - 1).clamp(min=0)
        return torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, prev_indices)

    def _correct_draft_hidden(
        self,
        output_hidden: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        label_indices: torch.Tensor,
        bsz: int,
        n: int,
        bs: int,
    ) -> torch.Tensor:
        """Pre-head hidden-state hook.

        Base DFlash (and the logit-space subclasses DSpark/Domino) return the
        tensor unchanged. DFly overrides this to add its TreeFlash correction
        conditioned on the teacher-forced previous token, so the correction
        runs before the frozen LM head exactly like at inference.
        """
        return output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        multimodal_position_ids: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
        anchor_table: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        sample_weights: torch.Tensor | None = None,
    ) -> DFlashStepMetrics:
        """Parallel block-wise training forward pass.

        Sequence packing (``position_ids`` ``[B, S]`` per-document reset positions,
        ``seq_lens`` ``[B, max_docs]`` document lengths, ``doc_remaining`` ``[B, S]``)
        keeps every block inside one document: anchors are constrained so the block
        does not cross a boundary, the block's context prefix attends only within the
        anchor's document, and the draft's RoPE uses the per-document positions.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            hidden_states: Tensor of shape [batch, sequence, target_layers * hidden].
            loss_mask: Tensor of shape [batch, sequence].
            position_ids: Optional long tensor of shape [batch, sequence].
            seq_lens: Optional long tensor of shape [batch, documents].
            doc_remaining: Optional long tensor of shape [batch, sequence].
            image_mask: Optional bool tensor of shape [batch, sequence], with True
                at image-token positions.
            multimodal_position_ids: Optional long tensor of shape [3, batch,
                sequence], with temporal, height, and width positions.
            teacher_hidden: Optional target final hidden states ``[batch, sequence,
                hidden]`` from a capture-v2 feature store. Distillation
                objectives consume it: JetSpec's forward-KL term, and for the
                DFlash-family trainers any active ``train_recipe`` stage whose
                LK / end-to-end TV terms need the teacher distribution. Other
                algorithms ignore it.
            anchor_table: Optional frozen ``(anchor_positions [B, N], keep [B, N])``
                table replayed instead of sampling anchors; used by ``train.regroup``.
                Requires the fixed-anchor dflash objective and non-packed inputs;
                kept anchors must satisfy ``anchor + block_size <= seq_len``.
            sample_weights: Optional per-sample weights ``[B]`` multiplied into the
                block-mask rows before the loss (preserved weights for regrouped
                physical batches); None (default) keeps the shipped path unchanged.

        Returns:
            Scalar loss and additive DFlash training metrics.
        """
        bsz, seq_len = input_ids.shape

        if anchor_table is not None:
            if self.loss_type == "variable_prefix":
                raise ValueError("anchor_table replay is not supported for loss_type='variable_prefix'.")
            if seq_lens is not None or doc_remaining is not None:
                raise ValueError("anchor_table replay does not support packed inputs (seq_lens/doc_remaining).")
        if sample_weights is not None:
            if sample_weights.dim() != 1 or sample_weights.shape[0] != bsz:
                raise ValueError(
                    f"sample_weights must have shape [batch]=({bsz},), got {tuple(sample_weights.shape)}."
                )
            if not torch.isfinite(sample_weights).all():
                raise ValueError("sample_weights contains non-finite entries.")

        if self.draft_model.visual_conditioning_enabled and image_mask is not None:
            if image_mask.shape != loss_mask.shape:
                raise ValueError(
                    "image_mask must have the same [batch, sequence] shape as loss_mask, "
                    f"got {tuple(image_mask.shape)} and {tuple(loss_mask.shape)}."
                )
            # The two-query context pools every image token in the sample. An
            # image after an earlier supervised token would therefore reveal a
            # future turn while training that token. Reject such conversations
            # instead of silently optimizing a non-causal objective.
            supervised_prefix = loss_mask.to(dtype=torch.bool).cumsum(dim=1) > 0
            if torch.any(image_mask.to(dtype=torch.bool) & supervised_prefix):
                raise ValueError(
                    "Visual DFlash requires every image token to appear before the first supervised token "
                    "in its sample; later images would leak future visual information into the cached context."
                )

        anchor_positions, block_keep_mask, noise_embedding, full_position_ids, dflash_attn_mask, prefix_lengths = (
            self._prepare_block_inputs(
                input_ids,
                loss_mask,
                position_ids=position_ids,
                seq_lens=seq_lens,
                doc_remaining=doc_remaining,
                causal=self.causal_in_block,
                anchor_table=anchor_table,
            )
        )

        full_multimodal_position_ids = None
        if self.draft_model.spatial_rope_enabled:
            if multimodal_position_ids is None:
                raise ValueError("multimodal_position_ids is required when spatial_rope_enabled=true.")
            if tuple(multimodal_position_ids.shape) != (3, bsz, seq_len):
                raise ValueError(
                    "multimodal_position_ids must have shape [3, batch, sequence], "
                    f"got {tuple(multimodal_position_ids.shape)} for batch={bsz}, sequence={seq_len}."
                )
            gather_indices = anchor_positions.clamp(min=0).unsqueeze(0).expand(3, -1, -1)
            anchor_multimodal_positions = torch.gather(multimodal_position_ids, 2, gather_indices)
            block_offsets = torch.arange(
                self.block_size,
                device=multimodal_position_ids.device,
                dtype=multimodal_position_ids.dtype,
            )
            draft_multimodal_position_ids = (
                anchor_multimodal_positions.unsqueeze(-1) + block_offsets.view(1, 1, 1, -1)
            ).flatten(2)
            full_multimodal_position_ids = torch.cat(
                (multimodal_position_ids, draft_multimodal_position_ids),
                dim=-1,
            )

        n = anchor_positions.size(1)
        bs = self.block_size

        # Block position k predicts the token at anchor + input_shift + k.
        label_indices, target_ids, block_mask = self._build_block_targets(
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
            seq_len,
            label_start=self.input_shift,
        )
        if sample_weights is not None:
            block_mask = block_mask * sample_weights.to(
                device=block_mask.device, dtype=block_mask.dtype
            ).view(-1, 1, 1)
        blocks_per_sample = block_keep_mask.sum(dim=1)
        self.last_workload = {
            "microbatches": 1,
            "samples": bsz,
            "allocated_blocks": bsz * n,
            "kept_blocks": int(blocks_per_sample.sum()),
            "max_sample_blocks": int(blocks_per_sample.max()),
            "allocated_rows": bsz * n * bs,
            "kept_rows": int(blocks_per_sample.sum()) * bs,
            "effective_positions": int((block_mask[:, :, self.first_supervised_slot :] > 0).sum()),
        }

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            multimodal_position_ids=full_multimodal_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=self._select_context_layers(hidden_states),
            attention_mask=dflash_attn_mask,
            image_mask=image_mask,
        )
        # Pre-head hidden-state hook (identity for DFlash/DSpark/Domino; DFly
        # applies its optional TreeFlash correction here, before the LM head).
        # The previous token of each slot is teacher-forced from the context:
        # slot k's target sits at ``label_indices[..., k]``, so the token right
        # before it is ``input_ids[label_indices - 1]``. Slot 0 (unsupervised in
        # the DFlash convention) clamps to position 0 like the label gather.
        prev_token_ids = self._teacher_forced_prev_tokens(input_ids, label_indices, n)
        output_hidden = self._correct_draft_hidden(
            output_hidden,
            prev_token_ids=prev_token_ids,
            label_indices=label_indices,
            bsz=bsz,
            n=n,
            bs=bs,
        )
        if self.ce_chunk_size:
            _, aux_loss, aux_metrics = self._apply_algorithm_terms(
                logits=None, output_hidden=output_hidden, input_ids=input_ids,
                label_indices=label_indices, target_ids=target_ids, block_mask=block_mask,
                bsz=bsz, n=n, bs=bs, seq_len=seq_len, teacher_hidden=teacher_hidden,
            )
            return self._chunked_ce_step(output_hidden, target_ids, block_mask, aux_loss, aux_metrics)
        # Recipe runs may compact zero-masked padding blocks out of the
        # scoring tail: the frozen LM head and every loss chunk then see kept
        # rows only. Blocks are independent units (each attends only its own
        # context window) and padding rows carry weight exactly 0, so dropping
        # them changes neither the loss nor the gradients; it only removes rows
        # that cannot contribute. ``compact_rows`` holds the (batch, block)
        # index of every kept block for the target/mask gathers below.
        compact_rows = None
        if self.recipe_row_compaction and self._recipe_active():
            if self.loss_type == "variable_prefix":
                raise ValueError(
                    "recipe_row_compaction is not supported for loss_type='variable_prefix'."
                )
            compact_rows = block_keep_mask.nonzero(as_tuple=False)  # [U, 2] (batch, block)
            kept_hidden = output_hidden.view(bsz * n, bs, -1)[
                compact_rows[:, 0] * n + compact_rows[:, 1]
            ].reshape(-1, output_hidden.size(-1))
            # A tensor-parallel target's lm_head is column-parallel and returns
            # vocab-sharded (DTensor) logits; gather to a full tensor for the loss.
            logits = _to_full_tensor(self.lm_head(kept_hidden.reshape(-1, kept_hidden.size(-1))))
            self._count(student_head_calls=1, student_head_rows=int(kept_hidden.shape[0]))
        elif self._defer_student_head():
            # The algorithm seam projects its own (compacted) rows.
            logits = None
        else:
            # A tensor-parallel target's lm_head is column-parallel and returns
            # vocab-sharded (DTensor) logits; gather to a full tensor for the loss.
            logits = _to_full_tensor(self.lm_head(output_hidden))
            self._count(student_head_calls=1, student_head_rows=bsz * n * bs)

        # Algorithm seam: logit correction (e.g. DSpark's Markov bias) plus any
        # auxiliary objective (e.g. the DSpark confidence BCE). The base DFlash
        # implementation is the identity; see ``algorithms/dspark.py``.
        logits, aux_loss, aux_metrics = self._apply_algorithm_terms(
            logits=logits,
            output_hidden=output_hidden,
            input_ids=input_ids,
            label_indices=label_indices,
            target_ids=target_ids,
            block_mask=block_mask,
            bsz=bsz,
            n=n,
            bs=bs,
            seq_len=seq_len,
            teacher_hidden=teacher_hidden,
        )

        if self.loss_type == "variable_prefix":
            return self._variable_prefix_loss(logits.view(bsz, n, bs, -1), target_ids, block_mask, prefix_lengths)

        block_kwargs = dict(
            logits=logits,
            target_ids=target_ids,
            block_mask=block_mask,
            aux_loss=aux_loss,
            aux_metrics=aux_metrics,
            bsz=bsz,
            n=n,
            bs=bs,
        )
        # Only a recipe run consumes the teacher signal in the scoring step, and
        # recipe algorithms all use this base implementation (the Domino /
        # DFlash2 / JetSpec overrides carry their own teacher flow through
        # ``aux_metrics`` and do not accept these keywords).
        if self.train_recipe is not None:
            block_kwargs["teacher_hidden"] = teacher_hidden
            block_kwargs["label_indices"] = label_indices
        if compact_rows is not None:
            rows_b, rows_n = compact_rows[:, 0], compact_rows[:, 1]
            block_kwargs["target_ids"] = target_ids[rows_b, rows_n].unsqueeze(1)  # [U, 1, bs]
            block_kwargs["block_mask"] = block_mask[rows_b, rows_n].unsqueeze(1)
            block_kwargs["label_indices"] = label_indices[rows_b, rows_n].unsqueeze(1)
            # The teacher stream maps its flat rows back to (sample, position)
            # through this index; the recipe's slot-major row order is unchanged.
            block_kwargs["teacher_sample_index"] = rows_b
            block_kwargs["bsz"] = int(compact_rows.shape[0])
            block_kwargs["n"] = 1
        return self._block_loss_step(**block_kwargs)

    def _recipe_active(self) -> bool:
        """True when a train recipe replaces ``loss_fn``'s decay-CE total."""
        recipe = self.train_recipe
        return recipe is not None and recipe.name != "controlled_ce"

    def _chunked_ce_step(self, hidden, targets, mask, aux_loss, aux_metrics) -> DFlashStepMetrics:
        from .chunked_ce import chunked_block_ce, precomputed_block_ce

        bsz, n, bs = targets.shape
        first = self.first_supervised_slot
        slots = bs - first
        target = targets[:, :, first:].reshape(bsz, -1)
        pred_mask = mask[:, :, first:].reshape(bsz, -1)
        predicted_hidden = hidden.view(bsz, n, bs, -1)[:, :, first:].reshape(bsz, n * slots, -1)
        score = precomputed_block_ce if self.ce_backward == "precompute" else chunked_block_ce
        out, pred_ids = score(
            predicted_hidden, self.lm_head, target, pred_mask, loss_fn=self.loss_fn,
            block_size=bs, chunk_size=self.ce_chunk_size, counter=self._count,
        )
        count, correct = out.draft_count_per_pos, out.draft_correct_per_pos
        valid, right = count.sum(), correct.sum()
        accept, accept_sum, blocks = compute_acceptance_stats(
            pred_ids.view(bsz, n, slots), target.view(bsz, n, slots), pred_mask.view(bsz, n, slots).bool(),
        )
        weights = pred_mask.view(bsz, n, slots)
        if self.loss_decay_gamma is not None:
            weights = weights * torch.exp(-torch.arange(slots, device=weights.device, dtype=weights.dtype)
                                          / self.loss_decay_gamma)
        return DFlashStepMetrics(
            loss=out.total_loss if aux_loss is None else out.total_loss + aux_loss,
            loss_weight=weights.sum().detach(), accuracy=(right / valid.clamp_min(1)).detach(),
            valid_tokens=valid.detach(), correct_tokens=right.detach(), accept_len=accept.detach(),
            accept_len_sum=accept_sum.detach(), valid_blocks=blocks.detach(),
            pos_correct=correct, pos_count=count, extra_metrics=self._public_extra_metrics(aux_metrics),
        )

    @staticmethod
    def _public_extra_metrics(aux_metrics: dict | None) -> dict | None:
        """Drop algorithm-internal (underscore-prefixed) entries from a metrics dict."""
        if not aux_metrics:
            return None
        public = {key: value for key, value in aux_metrics.items() if not str(key).startswith("_")}
        return public or None

    def _block_loss_step(
        self,
        *,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        aux_loss: torch.Tensor | None,
        aux_metrics: dict | None,
        bsz: int,
        n: int,
        bs: int,
        teacher_hidden: torch.Tensor | None = None,
        label_indices: torch.Tensor | None = None,
        teacher_sample_index: torch.Tensor | None = None,
    ) -> DFlashStepMetrics:
        """Decay-weighted block CE and standard metrics on (possibly corrected) logits.

        The scoring half of the training forward, factored out so that
        causal-correction algorithms (Domino) can score *two* logit tensors with
        the same mask and decay conventions and mix them (see
        ``domino/trainer.py``). ``aux_metrics`` entries whose key starts with
        ``_`` are algorithm-internal state (often non-scalar tensors passed from
        ``_apply_algorithm_terms``) and are never logged.
        """
        # For DFlash, drop block position 0 (the clean anchor token, never a
        # target); DSpark with sample_from_anchor=True supervises every slot.
        first = self.first_supervised_slot
        slots = bs - first
        pred_logits = logits.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1)
        pred_targets = target_ids[:, :, first:].reshape(bsz, n * slots)
        pred_mask = block_mask[:, :, first:].reshape(bsz, n * slots)
        self._count(scored_rows=bsz * n * slots)

        loss_fn = self.loss_fn
        assert loss_fn is not None, "loss_fn is always constructed for loss_type='dflash'"
        # A recipe run replaces ``loss_fn``'s decay-CE total with the recipe's
        # own objective and consumes only the per-slot counts; the full-vocab
        # cross-entropy behind that scalar would be dead work, so the counts are
        # computed directly (same argmax, same reductions, same dtype for the
        # final loss).
        if self._recipe_active():
            correct_per_pos, count_per_pos = loss_fn.accuracy_counts(
                pred_logits, pred_targets, pred_mask, block_size=bs
            )
            total_loss_dtype = loss_fn.total_loss_dtype(pred_logits)
            base_total_loss = None
        else:
            loss_out = loss_fn(pred_logits, pred_targets, pred_mask, num_tokens=None, block_size=bs)
            count_per_pos = loss_out.draft_count_per_pos
            correct_per_pos = loss_out.draft_correct_per_pos
            total_loss_dtype = loss_out.total_loss.dtype
            base_total_loss = loss_out.total_loss

        loss_weights = pred_mask.view(bsz, n, slots)
        if self.loss_decay_gamma is not None:
            depth_weights = torch.exp(
                -torch.arange(slots, device=pred_mask.device, dtype=pred_mask.dtype) / self.loss_decay_gamma
            )
            loss_weights = loss_weights * depth_weights
        loss_weight = (
            _sum_nonzero_weights(loss_weights)
            if self._recipe_active() and self.supports_recipe_row_compaction
            else loss_weights.sum()
        )

        valid_tokens = count_per_pos.sum()
        correct_tokens = correct_per_pos.sum()
        accuracy = correct_tokens / valid_tokens.clamp_min(1)
        accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(
            pred_logits.argmax(dim=-1).view(bsz, n, slots),
            pred_targets.view(bsz, n, slots),
            pred_mask.view(bsz, n, slots).bool(),
        )

        total_loss = base_total_loss
        recipe_metrics: dict = {}
        recipe = self.train_recipe
        if recipe is not None and recipe.name != "controlled_ce":
            stage = recipe.stages[self._recipe_stage_index or 0]
            teacher_chunk_fn = None
            if stage.lk_coef > 0.0 or stage.e2e_tv_coef > 0.0 or stage.confidence_coef > 0.0:
                if teacher_hidden is None or label_indices is None:
                    raise ValueError(
                        f"recipe stage {stage.name!r} needs the target's final hidden states (the "
                        "teacher signal), but the batch carries none. Train from a capture with "
                        "store_final_hidden=true / a feature store with stores_final_hidden=true."
                    )
                # Stream the teacher projection in bounded chunks: materializing
                # the full [B, n, S, V] tensor costs ~8.7 GB at the canonical
                # 16-sample sanity batch and pushes a 48 GB card into OOM.
                teacher_chunk_fn = TeacherLogitsStream(
                    self.lm_head,
                    teacher_hidden,
                    label_indices,
                    first_slot=first,
                    sample_index=teacher_sample_index,
                    counter=self._count,
                    projection_dedup=self.teacher_projection_dedup,
                ).logits
            # A confidence-head stage receives its per-slot logits from the
            # algorithm seam (DSpark's ``_apply_algorithm_terms``); the analytic
            # target is then computed against the same teacher chunks here, so
            # the teacher projection runs once per step.
            confidence_logits = None
            if stage.confidence_coef > 0.0 and isinstance(aux_metrics, dict):
                confidence_logits = aux_metrics.get(CONFIDENCE_LOGITS_AUX_KEY)
            recipe_total, recipe_metrics = recipe_loss(
                stage,
                student_logits=pred_logits.view(bsz, n, slots, -1),
                target_ids=pred_targets.view(bsz, n, slots),
                block_mask=pred_mask.view(bsz, n, slots),
                teacher_logits=None,
                teacher_chunk_fn=teacher_chunk_fn,
                loss_decay_gamma=self.loss_decay_gamma,
                confidence_logits=confidence_logits,
                compile_tv_kl=self.recipe_compile_tv_kl,
            )
            recipe_metrics["recipe_stage"] = float(self._recipe_stage_index or 0)
            total_loss = recipe_total.to(total_loss_dtype)
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
        merged_extra = {**(self._public_extra_metrics(aux_metrics) or {}), **recipe_metrics}
        return DFlashStepMetrics(
            loss=total_loss,
            loss_weight=loss_weight.detach(),
            accuracy=accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            pos_correct=correct_per_pos,
            pos_count=count_per_pos,
            extra_metrics=merged_extra or None,
        )

    def _variable_prefix_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> DFlashStepMetrics:
        """Decay-weighted CE over each block's masked suffix (D2SD Eq. for L_VP).

        Only positions at or past the sampled visible prefix are supervised, and
        the exponential decay restarts at the prefix boundary:
        ``w_k = exp(-(k - l) / gamma)`` for block position ``k >= l``
        (``loss_decay_gamma=None`` disables decay). The loss is the weighted mean
        ``sum(nll * w) / sum(w)``, mirroring the fixed-anchor path's
        ``normalize="mean"``. Assumes every ``prefix_lengths`` entry is at least
        ``min(2, block_size - 1)`` (what :meth:`_sample_prefix_lengths` produces),
        so the leading always-visible positions can be sliced off before the CE.

        Args:
            logits: Tensor of shape ``[batch, blocks, block_size, vocab]``.
            target_ids: Long tensor of shape ``[batch, blocks, block_size]``; the
                ground-truth token at ``anchor + k`` for block position ``k``.
            block_mask: Tensor of shape ``[batch, blocks, block_size]``; 0/1
                product of block validity, in-bounds, and the loss mask.
            prefix_lengths: Long tensor of shape ``[batch, blocks]``; this block's
                visible-prefix length.

        Returns:
            DFlashStepMetrics with the weighted-mean loss, the argmax accuracy
            over supervised positions, and the supervised-position count.
        """
        # Positions below the minimum prefix are visible in every block; drop them
        # before the CE like the fixed-anchor path drops position 0.
        min_prefix = self._min_prefix
        offsets = self._block_offsets[..., min_prefix:]
        logits = logits[:, :, min_prefix:, :]
        target_ids = target_ids[:, :, min_prefix:]
        block_mask = block_mask[:, :, min_prefix:]

        supervised = block_mask * (offsets >= prefix_lengths.unsqueeze(-1)).float()
        weights = supervised
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            effective_pos = (offsets - prefix_lengths.unsqueeze(-1)).clamp(min=0).float()
            weights = supervised * torch.exp(-effective_pos / self.loss_decay_gamma)

        vocab = logits.shape[-1]
        token_nll = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none").view_as(
            supervised
        )
        loss = (token_nll * weights).sum() / (weights.sum() + 1e-6)

        valid_tokens = supervised.sum()
        pred_ids = logits.argmax(dim=-1)
        correct = ((pred_ids == target_ids).float() * supervised).sum()
        accuracy = correct / valid_tokens.clamp_min(1)
        accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(pred_ids, target_ids, supervised.bool())
        return DFlashStepMetrics(
            loss=loss,
            loss_weight=weights.sum().detach(),
            accuracy=accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
        )

    @property
    def recipe_needs_teacher(self) -> bool:
        """Whether the active recipe consumes the target's final hidden states."""
        return bool(self.train_recipe is not None and self.train_recipe.needs_teacher)

    @property
    def recipe_stage_index(self) -> int | None:
        """Current recipe stage (``None`` when no recipe is active)."""
        return self._recipe_stage_index

    def current_recipe_stage(self):
        """The recipe stage the upcoming step trains (``None`` without a recipe)."""
        if self.train_recipe is None:
            return None
        stages = self.train_recipe.stages
        return stages[min(int(self._recipe_stage_index or 0), len(stages) - 1)]

    def set_training_progress(self, step: int, total_steps: int) -> None:
        """Training-loop progress hook (also used by Domino's curriculum).

        For a recipe run this selects the stage for the upcoming step: stage
        boundaries are a deterministic function of the plan (fraction × the
        run's total steps), so a resumed run lands on the same stage.
        """
        if self.train_recipe is not None:
            self._recipe_stage_index = self.train_recipe.stage_index(int(step), int(total_steps))

