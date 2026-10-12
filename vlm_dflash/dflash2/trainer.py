"""DFlash2 trainer module: local convolutions + candidate-selector objective.

The draft architecture is the DFlash block stack with the DFlash2 decoder layers
(``vlm_dflash.dflash2.model``); this trainer implements the training objective of
``docs/upstream/dflash2.md``:

* **unary term** - the configured DFlash-family loss on the unary logits. This
  port follows the upstream default (forward KL to the verifier distribution)
  and computes the teacher distribution exactly, by projecting the target's
  captured final hidden state through the frozen target ``lm_head`` chunk-wise
  under ``no_grad`` (the same staged teacher as JetSpec; max absolute deviation
  from the target's own logits measured at 0.0).
* **selector term** - a hard K-way cross-entropy over the unary top-K candidates
  with the previous sequence token teacher-forced. When the target id is absent
  from the top-K, the weakest candidate (rank K-1) is replaced by the target so
  the CE stays defined; metrics that describe *serving* behavior (`accuracy`,
  accepted length) are gated on the un-injected candidate set, matching upstream.

Both terms share the configured fixed-exponential position decay and divide by
the masked position count (not by the decay-weight sum), mirroring upstream
``loss_function``. ``selector_loss_alpha`` weights the selector term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..dflash.trainer_core import (
    DFlashStepMetrics,
    DFlashTrainerModule,
    NoValidAnchorsError,
    _to_full_tensor,
    compute_acceptance_stats,
)
from .loss import candidate_cross_entropy, decay_weights, selector_training_candidates

__all__ = ["DFlash2TrainerModule"]


class DFlash2TrainerModule(DFlashTrainerModule):
    """DFlash backbone trained with the experimental DFlash2 split objective."""

    causal_in_block = False

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        *,
        selector_loss_alpha: float = 1.0,
        kd_chunk_size: int = 1024,
        **kwargs,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            **kwargs,
        )
        if self.sample_from_anchor:
            raise ValueError(
                "DFlash2 follows the DFlash slot convention (the anchor is the verifier's bonus token); "
                "sample_from_anchor=True would shift the labels and break the selector alignment."
            )
        if self.loss_type != "dflash":
            raise ValueError(f"DFlash2 trains the fixed-anchor objective, got loss_type={self.loss_type!r}.")
        selector = getattr(draft_model, "candidate_selector", None)
        if selector is None:
            raise TypeError(
                "DFlash2 training requires a draft model with a candidate_selector; build the draft with "
                "Qwen3DFlash2DraftModel (vlm_dflash.dflash2.factory.build_draft_config)."
            )
        if float(selector_loss_alpha) < 0:
            raise ValueError(f"selector_loss_alpha must be >= 0, got {selector_loss_alpha}.")
        if int(kd_chunk_size) < 0:
            raise ValueError(f"kd_chunk_size must be >= 0, got {kd_chunk_size}.")
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.kd_chunk_size = int(kd_chunk_size)

    # ------------------------------------------------------------------ #
    # teacher / predecessor gathering
    # ------------------------------------------------------------------ #

    def _teacher_index(self, label_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Sequence positions that carry the teacher signal for the drafted slots.

        Block offset ``k`` (``k = 1..block_size-1``) predicts the token at
        ``anchor + k``; the target's autoregressive distribution for that token is
        its output at ``anchor + k - 1``, and the selector's predecessor token is
        the same position's token id. Returns ``[batch, blocks * (block_size - 1)]``.
        """
        bsz = label_indices.shape[0]
        return label_indices[:, :, :-1].clamp(min=0, max=seq_len - 1).reshape(bsz, -1)

    def _gather_rows(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(tensor, 1, indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1]))

    def _teacher_argmax(self, teacher_hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Target argmax ids for the valid positions (chunked full-vocab projection)."""
        flat_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        flat_valid = valid.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]
        ids = torch.zeros(flat_hidden.shape[0], dtype=torch.long, device=flat_hidden.device)
        num_valid = int(valid_indices.numel())
        if num_valid == 0:
            raise NoValidAnchorsError("No supervised draft positions in this batch.")
        chunk = self.kd_chunk_size if self.kd_chunk_size > 0 else num_valid
        for start in range(0, num_valid, chunk):
            indices = valid_indices[start : start + chunk]
            with torch.no_grad():
                logits = _to_full_tensor(self.lm_head(flat_hidden[indices]))
                ids[indices] = logits.float().argmax(dim=-1)
        return ids.view_as(valid)

    # ------------------------------------------------------------------ #
    # algorithm seam
    # ------------------------------------------------------------------ #

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
    ):
        if teacher_hidden is None:
            raise ValueError(
                "DFlash2 training requires capture-v2 features that store the target's final hidden states "
                "(capture.store_final_hidden=true): the unary objective distills the target distribution and "
                "the selector's hard label is the target argmax. Re-run capture_features for this run directory."
            )
        first = self.first_supervised_slot
        slots = bs - first
        teacher_index = self._teacher_index(label_indices, seq_len)
        # Internal state consumed by ``_block_loss_step``; private keys are never logged.
        payload = {
            "unary": logits.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1),
            "hidden": output_hidden.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1),
            "pred_mask": block_mask[:, :, first:].reshape(bsz, n * slots),
            "teacher_hidden": self._gather_rows(teacher_hidden, teacher_index),
            "predecessor_ids": torch.gather(input_ids, 1, teacher_index),
        }
        return logits, None, {"_dflash2": payload}

    # ------------------------------------------------------------------ #
    # objective
    # ------------------------------------------------------------------ #

    def _unary_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_hidden: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Decay-weighted forward KL ``KL(target || draft)`` over valid positions.

        Upstream ``kl_div_loss`` sums over the vocabulary per position and
        ``loss_function`` divides the decay-weighted sum by the masked count.
        """
        vocab = student_logits.shape[-1]
        flat_student = student_logits.reshape(-1, vocab)
        flat_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        flat_valid = valid.reshape(-1)
        flat_weights = weights.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]
        num_valid = int(valid_indices.numel())
        if num_valid == 0:
            raise NoValidAnchorsError("No supervised draft positions in this batch.")
        chunk = self.kd_chunk_size if self.kd_chunk_size > 0 else num_valid
        total = flat_student.new_zeros(())
        for start in range(0, num_valid, chunk):
            indices = valid_indices[start : start + chunk]
            with torch.no_grad():
                teacher_logits = _to_full_tensor(self.lm_head(flat_hidden[indices])).float()
                teacher_prob = F.softmax(teacher_logits, dim=-1)
            student = flat_student[indices].float()
            per_position = F.kl_div(F.log_softmax(student, dim=-1), teacher_prob, reduction="none").sum(dim=-1)
            total = total + (per_position * flat_weights[indices]).sum()
        return total / num_valid

    def _selector_loss(
        self,
        candidate_logits: torch.Tensor,
        target_positions: torch.Tensor,
        valid: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Decay-weighted K-way CE over the (possibly injected) candidate set."""
        per_position = candidate_cross_entropy(candidate_logits, target_positions)
        num_valid = valid.sum().clamp_min(1)
        return (per_position * weights).sum() / num_valid

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
    ) -> DFlashStepMetrics:
        payload = (aux_metrics or {}).get("_dflash2")
        if payload is None:
            raise RuntimeError("DFlash2TrainerModule requires the _dflash2 payload from _apply_algorithm_terms.")

        unary = payload["unary"]
        hidden = payload["hidden"]
        pred_mask = payload["pred_mask"]
        valid = pred_mask > 0.5
        slots = bs - self.first_supervised_slot

        # Upstream fixed-exp decay on both terms; the denominator is the masked
        # position count, so ``decay_weights`` keeps the anchor slot at zero.
        # ``decay_weights`` expects the [..., slots] view over supervised offsets
        # (slot index i is block offset k = i + 1), which is exactly this slice.
        weights = decay_weights(pred_mask.view(bsz, n, slots), slots, self.loss_decay_gamma).reshape(
            bsz, n * slots
        )

        teacher_ids = self._teacher_argmax(payload["teacher_hidden"], valid)
        selector = self.draft_model.candidate_selector
        candidate_ids = unary.topk(selector.top_k, dim=-1).indices
        training_candidate_ids, target_positions, contains_target = selector_training_candidates(
            candidate_ids, teacher_ids
        )
        candidate_logits = selector.score_candidates(
            unary,
            hidden,
            payload["predecessor_ids"],
            training_candidate_ids,
        )

        unary_loss = self._unary_kl_loss(unary, payload["teacher_hidden"], valid, weights)
        selector_loss = self._selector_loss(candidate_logits, target_positions, valid, weights)
        loss = unary_loss + self.selector_loss_alpha * selector_loss
        if aux_loss is not None:
            loss = loss + aux_loss

        # Serving-side path: the candidate the selector would emit, over the
        # un-injected candidates. Positions whose target was absent from the unary
        # top-k cannot be served (the injection is training-only), so they are
        # excluded from the servable acceptance metrics like upstream.
        picked = candidate_logits.detach().argmax(dim=-1)
        draft_ids = training_candidate_ids.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
        servable = (valid & contains_target).view(bsz, n, slots)
        matches = (draft_ids == teacher_ids) & servable.reshape(bsz, n * slots)
        valid_tokens = servable.sum()
        correct_tokens = matches.sum()
        accuracy = correct_tokens.float() / valid_tokens.clamp_min(1).float()
        # Same definition as upstream ``eal``: the leading run of correct draft
        # slots per block plus the verifier's always-emitted bonus token, over the
        # blocks that hold servable positions.
        accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(
            draft_ids.view(bsz, n, slots),
            teacher_ids.view(bsz, n, slots),
            servable,
        )

        with torch.no_grad():
            unary_oracle = unary.detach().argmax(dim=-1)
            oracle_matches = (candidate_ids.eq(teacher_ids.unsqueeze(-1)).any(dim=-1)) & valid
            oracle_al, _, _ = compute_acceptance_stats(
                unary_oracle.view(bsz, n, slots),
                teacher_ids.view(bsz, n, slots),
                (oracle_matches & valid).view(bsz, n, slots),
            )
            recall = (contains_target & valid).sum().float() / valid.sum().clamp_min(1)

        pos_correct = matches.view(bsz, n, slots).sum(dim=(0, 1))
        pos_count = servable.sum(dim=(0, 1))
        return DFlashStepMetrics(
            loss=loss,
            loss_weight=valid_tokens.float().detach(),
            accuracy=accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            pos_correct=pos_correct,
            pos_count=pos_count,
            extra_metrics=self._public_extra_metrics(
                {
                    "unary_kl": float(unary_loss.detach()),
                    "selector_ce": float(selector_loss.detach()),
                    "selector_loss_alpha": self.selector_loss_alpha,
                    "candidate_recall": float(recall),
                    "unary_oracle_al": float(oracle_al.detach()),
                }
            ),
        )
