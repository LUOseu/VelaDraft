"""JetSpec trainer module: causal in-block drafting + forward-KL distillation.

Wraps :class:`~vlm_dflash.dflash.trainer_core.DFlashTrainerModule` and changes
exactly the two things the maintainer implementation changes (see
``docs/upstream/jetspec.md``):

- ``causal_in_block = True``: the shared mask builder emits the block-causal
  in-block mask (a query at within-block offset ``i`` attends to offsets
  ``j <= i``) instead of DFlash's bidirectional one, so the draft factorization
  mirrors the target's autoregressive order;
- the loss: the decay-weighted hard-label CE is replaced by the
  temperature-scaled forward KL of :mod:`.kd_loss` against the target's
  per-position soft distribution. The draft architecture and the anchor /
  noise-block machinery are unmodified.

Teacher signal. The upstream trainer consumes full-vocab target logits captured
online (``[B, S, V]``, ~hundreds of MB per sample). This staged port instead
reads the target's final hidden state from a capture-v2 feature store and
projects it through the frozen target ``lm_head`` at training time: for block
offset ``k`` (predicting the token at ``anchor + k``) the teacher distribution
is the target's logits at ``anchor + k - 1``, i.e. ``lm_head`` applied to the
gathered final hidden state at that position - an exact computation, not a
top-K or hidden-state-space approximation. The projection runs chunk-wise under
``no_grad``, so peak memory stays bounded by ``kd_chunk_size`` positions.

Metrics follow the upstream trainer: ``accuracy`` is draft-vs-teacher greedy
agreement (the target's argmax, not the ground-truth token) and ``accept_len``
is the expected accepted-prefix length per block under that agreement - the
metric that matters for speculative decoding.
"""

from __future__ import annotations

import torch

from ..dflash.trainer_core import (
    DFlashStepMetrics,
    DFlashTrainerModule,
    NoValidAnchorsError,
    _to_full_tensor,
    compute_acceptance_stats,
)
from .kd_loss import IGNORE_INDEX, ForwardKDLoss

__all__ = ["JetSpecTrainerModule"]


class JetSpecTrainerModule(DFlashTrainerModule):
    """Causal block-parallel draft trained with forward-KL distillation."""

    causal_in_block = True

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        *,
        kd_temperature: float = 1.0,
        kd_chunk_size: int = 2048,
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
                "JetSpec supervises block offsets 1..block_size-1 (the DFlash slot convention); "
                "sample_from_anchor=True would shift the labels and break the teacher alignment."
            )
        if self.loss_decay_gamma is not None:
            raise ValueError(
                "JetSpec's objective is a forward KL with uniform position weighting (no depth decay); "
                "set drafter.loss_decay_gamma to null in the config."
            )
        if kd_temperature <= 0:
            raise ValueError(f"algorithm.jetspec.kd_temperature must be > 0, got {kd_temperature}.")
        if kd_chunk_size < 0:
            raise ValueError(f"algorithm.jetspec.kd_chunk_size must be >= 0, got {kd_chunk_size}.")
        self.kd_temperature = float(kd_temperature)
        self.kd_chunk_size = int(kd_chunk_size)
        # Reference logits-in KD (used by unit tests against the trainer's
        # chunked hidden-state path).
        self.kd_loss_fn = ForwardKDLoss(
            ignore_index=IGNORE_INDEX, temperature=self.kd_temperature, chunk_size=self.kd_chunk_size
        )

    # ------------------------------------------------------------------ #
    # teacher gathering / projection
    # ------------------------------------------------------------------ #

    def _gather_teacher_hidden(
        self,
        teacher_hidden: torch.Tensor,
        label_indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Target hidden states for every predicted draft position.

        Block offset ``k`` (``k = 1..block_size-1``) predicts the token at
        sequence position ``anchor + k``; the target's autoregressive
        distribution for that token (on the ground-truth prefix) is its output at
        position ``anchor + k - 1``, i.e. ``label_indices[..., :-1]``. Returns
        ``[batch, blocks * (block_size - 1), hidden]``.
        """
        bsz = label_indices.shape[0]
        teacher_idx = label_indices[:, :, :-1].clamp(min=0, max=seq_len - 1).reshape(bsz, -1)
        return torch.gather(
            teacher_hidden,
            1,
            teacher_idx.unsqueeze(-1).expand(-1, -1, teacher_hidden.shape[-1]),
        )

    def _forward_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Chunked forward KL and teacher-agreement statistics.

        Args:
            student_logits: ``[batch, blocks * slots, vocab]`` draft logits.
            teacher_hidden: ``[batch, blocks * slots, hidden]`` gathered target
                final hidden states, aligned with ``student_logits``.
            valid_mask: ``[batch, blocks * slots]`` float/bool supervision mask.

        Returns:
            ``(loss, stats)`` where ``stats`` carries per-position draft and
            teacher argmax ids plus the valid mask, for the acceptance metrics.
        """
        vocab = student_logits.shape[-1]
        flat_student = student_logits.reshape(-1, vocab)
        flat_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        flat_valid = valid_mask.reshape(-1).to(dtype=torch.bool)
        valid_indices = flat_valid.nonzero(as_tuple=True)[0]
        num_valid = int(valid_indices.numel())
        if num_valid == 0:
            # Anchors exist but none of their continuations is supervised;
            # nothing to distill (skipped by the shared training loop).
            raise NoValidAnchorsError("No supervised draft positions in this batch.")

        chunk = self.kd_chunk_size if self.kd_chunk_size > 0 else num_valid
        kl_sum = flat_student.new_zeros(())
        # Ids are scattered back into per-position tensors (zeros at invalid
        # positions); only the valid entries are consumed by the metrics.
        draft_ids = torch.zeros(flat_student.shape[0], dtype=torch.long, device=flat_student.device)
        teacher_ids = torch.zeros_like(draft_ids)
        for start in range(0, num_valid, chunk):
            end = min(start + chunk, num_valid)
            indices = valid_indices[start:end]
            student_chunk = flat_student[indices]
            # Teacher logits are a frozen-target quantity: project under no_grad
            # (gradients must only flow into the draft).
            with torch.no_grad():
                teacher_chunk = _to_full_tensor(self.lm_head(flat_hidden[indices]))
                if self.kd_temperature != 1.0:
                    teacher_chunk = teacher_chunk.mul(1.0 / self.kd_temperature)
                teacher_logprob = torch.log_softmax(teacher_chunk.float(), dim=-1)
                teacher_ids[indices] = teacher_logprob.argmax(dim=-1)
                draft_ids[indices] = student_chunk.detach().argmax(dim=-1)
            student_scaled = student_chunk.float()
            if self.kd_temperature != 1.0:
                student_scaled = student_scaled.mul(1.0 / self.kd_temperature)
            student_logprob = torch.log_softmax(student_scaled, dim=-1)
            # KL(P||Q) per token, masking underflowed teacher probabilities.
            teacher_prob = teacher_logprob.exp()
            log_ratio = teacher_logprob - student_logprob
            log_ratio = torch.where(teacher_prob > 0, log_ratio, torch.zeros_like(log_ratio))
            kl_sum = kl_sum + (teacher_prob * log_ratio).sum()
        loss = kl_sum / num_valid
        if self.kd_temperature != 1.0:
            loss = loss * (self.kd_temperature**2)
        stats = {"draft_ids": draft_ids, "teacher_ids": teacher_ids, "valid": flat_valid}
        return loss, stats

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
                "JetSpec training requires capture-v2 features that store the target's final hidden "
                "states (capture.store_final_hidden=true in the config). The loaded feature store does "
                "not carry them - re-run capture_features for this run directory."
            )
        gathered = self._gather_teacher_hidden(teacher_hidden, label_indices, seq_len)
        # ``_teacher_hidden`` is internal state consumed by ``_block_loss_step``;
        # private keys are never logged.
        return logits, None, {"_teacher_hidden": gathered}

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
        """Forward-KL objective and draft-vs-teacher acceptance metrics."""
        assert aux_metrics is not None and "_teacher_hidden" in aux_metrics
        teacher_hidden = aux_metrics.pop("_teacher_hidden")

        # DFlash slot convention: block position 0 is the clean anchor token and
        # positions 1..bs-1 are drafted; the teacher gather above is aligned with
        # exactly this slice (label offsets 0..bs-2).
        first = self.first_supervised_slot
        slots = bs - first
        student = logits.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1)
        pred_mask = block_mask[:, :, first:].reshape(bsz, n * slots)
        valid = pred_mask > 0.5

        loss, stats = self._forward_kl_loss(student, teacher_hidden, valid)
        draft_ids = stats["draft_ids"].view(bsz, n, slots)
        teacher_ids = stats["teacher_ids"].view(bsz, n, slots)
        valid4 = valid.view(bsz, n, slots).bool()

        matches = (draft_ids == teacher_ids) & valid4
        valid_tokens = valid4.sum()
        correct_tokens = matches.sum()
        accuracy = correct_tokens.float() / valid_tokens.clamp_min(1).float()
        accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(draft_ids, teacher_ids, valid4)
        pos_correct = matches.sum(dim=(0, 1))
        pos_count = valid4.sum(dim=(0, 1))

        if aux_loss is not None:
            loss = loss + aux_loss
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
                {"kd_loss": float(loss.detach()), "kd_temperature": self.kd_temperature}
            ),
        )
