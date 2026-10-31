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

Execution switches (default off; the objective is unchanged by every one):

* ``teacher_projection_reuse`` (D1) - the reference path projects the same
  valid teacher hidden states through the frozen head twice, once for the
  selector's argmax label and once for the KL distribution. With reuse on, the
  KL chunk loop also emits the argmax ids, so each teacher row is projected once.
* ``scoring_row_compaction`` (D2) - the student head, top-K and selector run on
  the supervised valid rows only (slot 0, padding blocks and masked positions
  are dropped before the head). Every scoring step is row-local, so dropped
  rows contributed exactly zero loss and gradient.
* ``fused_teacher_kl`` (D2, needs reuse) - the KL chunk loop forms the student
  gradient in the forward pass and keeps it in the logits dtype, instead of
  autograd retaining each chunk's fp32 log-softmax and teacher probabilities
  (8 bytes per vocabulary entry) until the backward. No recompute.
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


class _FusedTeacherKL(torch.autograd.Function):
    """Chunked ``sum_i w_i * KL(teacher_i || student_i) / num_valid`` with a forward-built gradient.

    Same math as the reference chunk loop (``F.kl_div`` on the fp32 student
    log-softmax against the fp32 teacher softmax). The gradient with respect to
    the student logits, ``w_i / num_valid * (q_i * sum(p_i) - p_i)``, is exactly
    what autograd derives through ``log_softmax`` and ``kl_div``; it is formed
    per chunk and stored in the logits dtype (autograd casts it to that dtype at
    the ``.float()`` boundary anyway). The second output is the teacher argmax
    of every row, from the same projection (D1).
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_hidden, weights, head, chunk, num_valid):
        rows = student_logits.shape[0]
        want_grad = bool(ctx.needs_input_grad[0])
        grad = torch.empty_like(student_logits) if want_grad else None
        ids = torch.empty(rows, dtype=torch.long, device=student_logits.device)
        total = student_logits.new_zeros((), dtype=torch.float32)
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            teacher_logits = _to_full_tensor(head(teacher_hidden[start:stop])).float()
            ids[start:stop] = teacher_logits.argmax(dim=-1)
            teacher_prob = F.softmax(teacher_logits, dim=-1)
            del teacher_logits
            log_student = F.log_softmax(student_logits[start:stop].float(), dim=-1)
            per_position = F.kl_div(log_student, teacher_prob, reduction="none").sum(dim=-1)
            weight = weights[start:stop]
            total = total + (per_position * weight).sum()
            if want_grad:
                scale = (weight / num_valid).unsqueeze(-1)
                mass = teacher_prob.sum(dim=-1, keepdim=True)
                grad[start:stop] = ((log_student.exp_() * mass - teacher_prob) * scale).to(grad.dtype)
        if want_grad:
            ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(ids)
        return total / num_valid, ids

    @staticmethod
    def backward(ctx, grad_loss, _grad_ids):
        (grad,) = ctx.saved_tensors
        # The upstream factor is 1 / grad_accumulation_steps, a power of two for
        # every supported geometry, so this scaling adds no rounding.
        return grad.mul_(grad_loss), None, None, None, None, None


class DFlash2TrainerModule(DFlashTrainerModule):
    """DFlash backbone trained with the experimental DFlash2 split objective."""

    causal_in_block = False
    supports_chunked_ce = False

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        *,
        selector_loss_alpha: float = 1.0,
        kd_chunk_size: int = 1024,
        teacher_projection_reuse: bool = False,
        scoring_row_compaction: bool = False,
        fused_teacher_kl: bool = False,
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
        if fused_teacher_kl and not teacher_projection_reuse:
            raise ValueError(
                "fused_teacher_kl emits the selector labels from the KL projection; it requires "
                "teacher_projection_reuse=True."
            )
        self.selector_loss_alpha = float(selector_loss_alpha)
        self.kd_chunk_size = int(kd_chunk_size)
        self.teacher_projection_reuse = bool(teacher_projection_reuse)
        self.scoring_row_compaction = bool(scoring_row_compaction)
        self.fused_teacher_kl = bool(fused_teacher_kl)

    def _defer_student_head(self) -> bool:
        return self.scoring_row_compaction

    def execution_switches(self) -> dict:
        return {
            "dflash2_teacher_projection_reuse": self.teacher_projection_reuse,
            "dflash2_scoring_row_compaction": self.scoring_row_compaction,
            "dflash2_fused_teacher_kl": self.fused_teacher_kl,
            "kd_chunk_size": self.kd_chunk_size,
        }

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
            self._count(teacher_head_calls=1, teacher_head_rows=int(indices.numel()))
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
        pred_mask = block_mask[:, :, first:].reshape(bsz, n * slots)
        hidden = output_hidden.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1)
        valid_flat = (pred_mask.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
        sample_of_row = valid_flat // (n * slots)
        teacher_pos = teacher_index.reshape(-1)[valid_flat]
        self._count(
            unique_teacher_positions=int(torch.unique(sample_of_row * seq_len + teacher_pos).numel()),
            supervised_rows=int(valid_flat.numel()),
        )
        if logits is None:
            # Row compaction: only supervised valid rows reach the student head.
            if valid_flat.numel() == 0:
                raise NoValidAnchorsError("No supervised draft positions in this batch.")
            hidden_rows = hidden.reshape(-1, hidden.shape[-1])[valid_flat]
            self._count(student_head_calls=1, student_head_rows=int(hidden_rows.shape[0]))
            payload = {
                "compact": True,
                "valid_flat": valid_flat,
                "unary": _to_full_tensor(self.lm_head(hidden_rows)),
                "hidden": hidden_rows,
                "pred_mask": pred_mask,
                "teacher_hidden": teacher_hidden[sample_of_row, teacher_pos],
                "predecessor_ids": input_ids[sample_of_row, teacher_pos],
            }
            return logits, None, {"_dflash2": payload}
        # Internal state consumed by ``_block_loss_step``; private keys are never logged.
        payload = {
            "unary": logits.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1),
            "hidden": hidden,
            "pred_mask": pred_mask,
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
        teacher_ids_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decay-weighted forward KL ``KL(target || draft)`` over valid positions.

        Upstream ``kl_div_loss`` sums over the vocabulary per position and
        ``loss_function`` divides the decay-weighted sum by the masked count.
        With ``teacher_ids_out`` (flat, one slot per position) the teacher argmax
        of every valid position is written there from the same projection (D1).
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
            self._count(teacher_head_calls=1, teacher_head_rows=int(indices.numel()))
            with torch.no_grad():
                teacher_logits = _to_full_tensor(self.lm_head(flat_hidden[indices])).float()
                if teacher_ids_out is not None:
                    teacher_ids_out[indices] = teacher_logits.argmax(dim=-1)
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

        if payload.get("compact"):
            return self._compact_block_loss_step(payload, aux_loss=aux_loss, bsz=bsz, n=n, bs=bs)

        unary = payload["unary"]
        hidden = payload["hidden"]
        pred_mask = payload["pred_mask"]
        valid = pred_mask > 0.5
        slots = bs - self.first_supervised_slot
        self._count(scored_rows=int(valid.numel()))

        # Upstream fixed-exp decay on both terms; the denominator is the masked
        # position count, so ``decay_weights`` keeps the anchor slot at zero.
        # ``decay_weights`` expects the [..., slots] view over supervised offsets
        # (slot index i is block offset k = i + 1), which is exactly this slice.
        weights = decay_weights(pred_mask.view(bsz, n, slots), slots, self.loss_decay_gamma).reshape(
            bsz, n * slots
        )

        if self.teacher_projection_reuse:
            # D1: one projection yields both the KL distribution and the argmax label.
            flat_ids = torch.zeros(valid.numel(), dtype=torch.long, device=unary.device)
            unary_loss = self._unary_kl_loss(
                unary, payload["teacher_hidden"], valid, weights, teacher_ids_out=flat_ids
            )
            teacher_ids = flat_ids.view_as(valid)
        else:
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

        if not self.teacher_projection_reuse:
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
        with torch.no_grad():
            unary_oracle = unary.detach().argmax(dim=-1)
            oracle_hit = candidate_ids.eq(teacher_ids.unsqueeze(-1)).any(dim=-1)
        return self._dflash2_metrics(
            loss=loss,
            unary_loss=unary_loss,
            selector_loss=selector_loss,
            draft_ids=draft_ids,
            teacher_ids=teacher_ids,
            valid=valid,
            contains_target=contains_target,
            unary_oracle=unary_oracle,
            oracle_hit=oracle_hit,
            bsz=bsz,
            n=n,
            slots=slots,
        )

    def _compact_block_loss_step(self, payload: dict, *, aux_loss, bsz: int, n: int, bs: int) -> DFlashStepMetrics:
        """D2 scoring on the supervised valid rows only (``scoring_row_compaction``).

        Rows are the valid positions in flat ``[bsz, n * slots]`` order; the
        metrics scatter them back to the dense grid, where the dropped positions
        are exactly the ones the dense path masks out.
        """
        unary = payload["unary"]  # [U, V]
        hidden = payload["hidden"]
        valid_flat = payload["valid_flat"]
        teacher_hidden = payload["teacher_hidden"]
        slots = bs - self.first_supervised_slot
        rows = int(valid_flat.numel())
        self._count(scored_rows=rows)
        weights = decay_weights(payload["pred_mask"].view(bsz, n, slots), slots, self.loss_decay_gamma).reshape(-1)[
            valid_flat
        ]

        chunk = self.kd_chunk_size if self.kd_chunk_size > 0 else rows
        if self.fused_teacher_kl:
            self._count(teacher_head_calls=-(-rows // chunk), teacher_head_rows=rows)
            unary_loss, teacher_ids = _FusedTeacherKL.apply(unary, teacher_hidden, weights, self.lm_head, chunk, rows)
        else:
            all_valid = torch.ones(rows, dtype=torch.bool, device=unary.device)
            if self.teacher_projection_reuse:
                teacher_ids = torch.zeros(rows, dtype=torch.long, device=unary.device)
                unary_loss = self._unary_kl_loss(unary, teacher_hidden, all_valid, weights, teacher_ids_out=teacher_ids)
            else:
                teacher_ids = self._teacher_argmax(teacher_hidden, all_valid)
                unary_loss = self._unary_kl_loss(unary, teacher_hidden, all_valid, weights)

        selector = self.draft_model.candidate_selector
        candidate_ids = unary.topk(selector.top_k, dim=-1).indices
        training_candidate_ids, target_positions, contains_target = selector_training_candidates(
            candidate_ids, teacher_ids
        )
        candidate_logits = selector.score_candidates(
            unary, hidden, payload["predecessor_ids"], training_candidate_ids
        )
        selector_loss = (candidate_cross_entropy(candidate_logits, target_positions) * weights).sum() / rows
        loss = unary_loss + self.selector_loss_alpha * selector_loss
        if aux_loss is not None:
            loss = loss + aux_loss

        picked = candidate_logits.detach().argmax(dim=-1)
        draft_ids = training_candidate_ids.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
        dense = bsz * n * slots

        def scatter(values: torch.Tensor) -> torch.Tensor:
            out = values.new_zeros(dense)
            out[valid_flat] = values
            return out.view(bsz, n * slots)

        with torch.no_grad():
            valid = scatter(torch.ones(rows, dtype=torch.bool, device=unary.device))
            return self._dflash2_metrics(
                loss=loss,
                unary_loss=unary_loss,
                selector_loss=selector_loss,
                draft_ids=scatter(draft_ids),
                teacher_ids=scatter(teacher_ids),
                valid=valid,
                contains_target=scatter(contains_target),
                unary_oracle=scatter(unary.detach().argmax(dim=-1)),
                oracle_hit=scatter(candidate_ids.eq(teacher_ids.unsqueeze(-1)).any(dim=-1)),
                bsz=bsz,
                n=n,
                slots=slots,
            )

    def _dflash2_metrics(
        self,
        *,
        loss,
        unary_loss,
        selector_loss,
        draft_ids,
        teacher_ids,
        valid,
        contains_target,
        unary_oracle,
        oracle_hit,
        bsz: int,
        n: int,
        slots: int,
    ) -> DFlashStepMetrics:
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
            oracle_matches = oracle_hit & valid
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
