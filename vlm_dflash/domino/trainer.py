"""Domino trainer module: DFlash backbone + causal GRU correction + curriculum.

Wraps :class:`~vlm_dflash.dflash.trainer_core.DFlashTrainerModule` and uses its
algorithm seam (``_apply_algorithm_terms``) plus its scoring seam
(``_block_loss_step``) to add the two Domino terms:

- the causal correction: the draft model's ``prefix_gru`` encodes a GRU state
  from each block's *previous* tokens and ``embed_proj`` maps
  ``[backbone hidden | GRU state]`` to a low-rank logit delta that is added to
  the block-parallel base logits on the corrected suffix, and
- the base-anchor curriculum: ``loss = (1 - lambda_base) * final_loss +
  lambda_base * base_loss`` with ``lambda_base`` decaying linearly from
  ``lambda_base_start`` to 0 over ``lambda_base_decay_ratio`` of training.

The GRU consumes ground-truth previous tokens during training (teacher
forcing) and the module keeps both heads' metrics (final/base loss, accuracy,
acceptance length) in the training log.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..dflash.trainer_core import DFlashStepMetrics, DFlashTrainerModule, compute_acceptance_stats

__all__ = ["DominoTrainerModule", "get_lambda_base"]


def get_lambda_base(
    global_step: int,
    total_steps: int,
    lambda_start: float = 1.0,
    decay_ratio: float = 0.5,
) -> float:
    """Base-anchor curriculum weight at ``global_step`` (upstream semantics).

    ``lambda_base`` starts at ``lambda_start`` and decays linearly to 0 over the
    first ``decay_ratio`` fraction of ``total_steps``, then stays at 0. The
    result is clamped to ``[0, 1]``.
    """
    decay_steps = max(1, int(total_steps * decay_ratio))
    progress = min(global_step / decay_steps, 1.0)
    lambda_base = lambda_start * (1.0 - progress)
    return max(0.0, min(1.0, lambda_base))


class DominoTrainerModule(DFlashTrainerModule):
    """DFlash block-CE training plus the Domino causal correction and curriculum.

    The correction head is owned by the draft model (``prefix_gru`` /
    ``embed_proj``, constructed when ``dflash_config.projector_type ==
    "domino"``); this module only orchestrates it. ``enable_correction_head`` is
    an ablation-only switch (not an upstream knob): with it disabled the module
    trains the exact DFlash objective on the same architecture, which isolates
    the correction head's contribution from training randomness.
    """

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        *,
        lambda_base_start: float = 1.0,
        lambda_base_decay_ratio: float = 1.0,
        enable_correction_head: bool = True,
        **kwargs,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            **kwargs,
        )
        if getattr(draft_model, "projector_type", None) != "domino":
            raise ValueError(
                "DominoTrainerModule requires a draft model built with "
                "dflash_config.projector_type='domino' (so prefix_gru / embed_proj exist)."
            )
        self.shift_label = bool(getattr(draft_model, "shift_label", False))
        # The shared slot machinery (``sample_from_anchor``) encodes the same
        # switch: shifted labels supervise every slot and start at the anchor+1
        # target, which is exactly ``sample_from_anchor=True``'s convention.
        if bool(self.sample_from_anchor) != self.shift_label:
            raise ValueError(
                "Domino requires sample_from_anchor == draft_model.shift_label "
                f"(got {bool(self.sample_from_anchor)} vs {self.shift_label}); build the module "
                "through vlm_dflash.domino.factory so the slot conventions stay in sync."
            )
        self.pure_draft_prefix_len = int(getattr(draft_model, "pure_draft_prefix_len", 0))
        # First block position that receives the correction (upstream
        # ``DominoTrainerModule._suffix_start``).
        self.correction_suffix_start = (
            self.pure_draft_prefix_len if self.shift_label else 1 + self.pure_draft_prefix_len
        )
        if not 0 <= self.correction_suffix_start < self.block_size:
            raise ValueError(
                "pure_draft_prefix_len must leave at least one corrected block position: got "
                f"suffix_start={self.correction_suffix_start} for block_size={self.block_size}."
            )
        self.lambda_base_start = float(lambda_base_start)
        if not 0.0 <= self.lambda_base_start <= 1.0:
            raise ValueError(f"lambda_base_start must be in [0, 1], got {self.lambda_base_start}.")
        self.lambda_base_decay_ratio = float(lambda_base_decay_ratio)
        if not 0.0 < self.lambda_base_decay_ratio <= 1.0:
            raise ValueError(
                f"lambda_base_decay_ratio must be in (0, 1], got {self.lambda_base_decay_ratio}."
            )
        self.enable_correction_head = bool(enable_correction_head)
        # Until the training loop reports progress (or for one-off forwards),
        # start at the curriculum's initial weight.
        self.current_lambda_base = self.lambda_base_start
        # λ=1 at step 0 means the correction head intentionally receives *no*
        # gradient on the first optimizer step (the loss is pure base CE), so
        # the gradient-coverage audit is only meaningful after the curriculum
        # has started; the training loop audits at this fraction of the run.
        self.coverage_audit_fraction = 0.1

    def set_training_progress(self, step: int, total_steps: int) -> None:
        """Curriculum hook: the training loop calls this before each forward."""
        self.current_lambda_base = get_lambda_base(
            step, total_steps, self.lambda_base_start, self.lambda_base_decay_ratio
        )

    def _domino_correction(
        self,
        *,
        block_logits: torch.Tensor,
        hidden4d: torch.Tensor,
        input_ids: torch.Tensor,
        label_indices: torch.Tensor,
        target_ids: torch.Tensor,
        bsz: int,
        n: int,
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the GRU-conditioned low-rank delta to the corrected suffix logits.

        Returns ``(corrected_block_logits, delta)``. Kept structurally identical
        to the upstream maintainer implementation (``domino_core.py``) so the
        training math is a faithful port; see ``docs/upstream/domino.md``.
        """
        if self.shift_label:
            # Slot k supervises ``anchor + 1 + k`` and its GRU state is built
            # from the ground-truth token at ``anchor + k`` (the token right
            # before the supervised one); ``label_indices`` are those targets.
            prev_indices = (label_indices - 1).clamp(min=0)
            prev_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, prev_indices)
            block_emb = self.embed_tokens(prev_ids)
            gru_out, _ = self.draft_model.prefix_gru(block_emb.reshape(bsz * n, bs, -1))
            gru_out = gru_out.reshape(bsz, n, bs, -1)
            prefix_states = gru_out[:, :, self.correction_suffix_start :, :]
        else:
            # Unshifted labels: slot k supervises ``anchor + k`` and the GRU
            # consumes the supervised tokens themselves; the first proposal
            # (slot 1) conditions on the block's first token.
            block_emb = self.embed_tokens(target_ids)
            gru_inputs = block_emb[:, :, : bs - 1, :].reshape(bsz * n, bs - 1, -1)
            gru_out, _ = self.draft_model.prefix_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n, bs - 1, -1)
            prefix_states = gru_out[:, :, self.correction_suffix_start - 1 :, :]
        backbone_states = hidden4d[:, :, self.correction_suffix_start :, :]
        delta = self.draft_model.embed_proj(torch.cat([backbone_states, prefix_states], dim=-1))
        corrected = torch.cat(
            [
                block_logits[:, :, : self.correction_suffix_start, :],
                block_logits[:, :, self.correction_suffix_start :, :] + delta,
            ],
            dim=2,
        )
        return corrected, delta

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
    ):
        block_logits = logits.view(bsz, n, bs, -1)
        metrics: dict = {
            "lambda_base": float(self.current_lambda_base),
            "correction_enabled": float(self.enable_correction_head),
        }
        # ``_base_logits`` is internal state consumed by ``_block_loss_step``
        # (the base-anchor curriculum needs the uncorrected logits); private
        # keys are never logged.
        metrics["_base_logits"] = block_logits
        if not self.enable_correction_head:
            return logits, None, metrics
        hidden4d = output_hidden.view(bsz, n, bs, -1)
        corrected, delta = self._domino_correction(
            block_logits=block_logits,
            hidden4d=hidden4d,
            input_ids=input_ids,
            label_indices=label_indices,
            target_ids=target_ids,
            bsz=bsz,
            n=n,
            bs=bs,
        )
        with torch.no_grad():
            metrics["correction_abs_mean"] = float(delta.detach().abs().mean())
        return corrected.reshape(bsz, n * bs, -1), None, metrics

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
        """Dual-logit objective: ``(1-λ)·CE(final) + λ·CE(base)`` + both metrics."""
        assert aux_metrics is not None and "_base_logits" in aux_metrics
        base_logits = aux_metrics["_base_logits"].view(bsz, n, bs, -1)
        lambda_base = float(aux_metrics.get("lambda_base", 0.0))

        first = self.first_supervised_slot
        slots = bs - first
        final_slots = logits.view(bsz, n, bs, -1)[:, :, first:, :].reshape(bsz, n * slots, -1)
        base_slots = base_logits[:, :, first:, :].reshape(bsz, n * slots, -1)
        pred_targets = target_ids[:, :, first:].reshape(bsz, n * slots)
        pred_mask = block_mask[:, :, first:].reshape(bsz, n * slots)

        loss_fn = self.loss_fn
        assert loss_fn is not None, "loss_fn is always constructed for loss_type='dflash'"
        final_out = loss_fn(final_slots, pred_targets, pred_mask, num_tokens=None, block_size=bs)
        base_out = loss_fn(base_slots, pred_targets, pred_mask, num_tokens=None, block_size=bs)
        loss = (1.0 - lambda_base) * final_out.total_loss + lambda_base * base_out.total_loss

        loss_weights = pred_mask.view(bsz, n, slots)
        if self.loss_decay_gamma is not None:
            depth_weights = torch.exp(
                -torch.arange(slots, device=pred_mask.device, dtype=pred_mask.dtype) / self.loss_decay_gamma
            )
            loss_weights = loss_weights * depth_weights
        loss_weight = loss_weights.sum()

        count_per_pos = final_out.draft_count_per_pos
        valid_tokens = count_per_pos.sum()
        correct_tokens = final_out.draft_correct_per_pos.sum()
        accuracy = correct_tokens / valid_tokens.clamp_min(1)
        accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(
            final_slots.argmax(dim=-1).view(bsz, n, slots),
            pred_targets.view(bsz, n, slots),
            pred_mask.view(bsz, n, slots).bool(),
        )
        base_valid_tokens = base_out.draft_count_per_pos.sum()
        base_accuracy = base_out.draft_correct_per_pos.sum() / base_valid_tokens.clamp_min(1)
        base_accept_len, base_accept_len_sum, _ = compute_acceptance_stats(
            base_slots.argmax(dim=-1).view(bsz, n, slots),
            pred_targets.view(bsz, n, slots),
            pred_mask.view(bsz, n, slots).bool(),
        )

        aux_metrics = dict(aux_metrics)
        aux_metrics.pop("_base_logits", None)
        aux_metrics["final_loss"] = float(final_out.total_loss.detach())
        aux_metrics["base_loss"] = float(base_out.total_loss.detach())
        aux_metrics["base_accuracy"] = float(base_accuracy.detach())
        aux_metrics["base_accept_len"] = float(base_accept_len.detach())
        return DFlashStepMetrics(
            loss=loss,
            loss_weight=loss_weight.detach(),
            accuracy=accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            pos_correct=final_out.draft_correct_per_pos,
            pos_count=final_out.draft_count_per_pos,
            extra_metrics=self._public_extra_metrics(aux_metrics),
        )
