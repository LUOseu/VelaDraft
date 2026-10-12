"""DSpark trainer module: DFlash backbone + Markov logit bias + confidence head.

Wraps :class:`~vlm_dflash.dflash.trainer_core.DFlashTrainerModule` and uses its
algorithm seam (``_apply_algorithm_terms``) to add the two DSpark terms:

- the low-rank Markov logit bias, teacher-forced with the ground-truth previous
  token of each supervised slot (slot k predicts ``anchor + input_shift + k``,
  and its bias conditions on the token immediately before that position), and
- the confidence head BCE, whose target is the per-slot greedy acceptance
  indicator (draft argmax == target token); see ``heads.py`` for why the staged
  pipeline uses this adaptation of the upstream ``1 - TV`` target.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..dflash.trainer_core import DFlashTrainerModule

__all__ = ["DSparkTrainerModule"]


class DSparkTrainerModule(DFlashTrainerModule):
    """DFlash block-CE training plus the DSpark Markov bias and confidence BCE.

    The two DSpark heads are constructed by ``Qwen3DSparkDraftModel`` (from the
    ``algorithm.dspark`` knobs baked into its ``dflash_config``); this module
    reuses those instances, and the optional ``markov_rank`` /
    ``markov_head_type`` / ``enable_confidence_head`` /
    ``confidence_head_with_markov`` arguments are validation-only overrides of
    what the model already holds.
    """

    # Optional sink for the post-training calibration report (``None`` during
    # training: zero overhead). Each entry is a detached CPU triple
    # ``(prob [B, S], accepted [B, S], mask [B, S])`` over supervised slots.
    calibration_sink: Optional[list] = None

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        *,
        markov_rank: Optional[int] = None,
        markov_head_type: Optional[str] = None,
        enable_confidence_head: Optional[bool] = None,
        confidence_head_with_markov: Optional[bool] = None,
        confidence_head_alpha: float = 1.0,
        sample_from_anchor: bool = True,
        **kwargs,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            sample_from_anchor=sample_from_anchor,
            **kwargs,
        )
        # The heads are owned by the draft model (``Qwen3DSparkDraftModel``), not
        # re-created here. One shared instance keeps three things consistent by
        # construction:
        #   * ``module.draft_model.to(dtype)`` in the training loop casts the
        #     heads too (a duplicated fp32 head crashes against bf16 features),
        #   * the optimizer's master parameters and the saved checkpoint come
        #     from ``draft_model.named_parameters()`` / ``state_dict()``, so the
        #     trained head weights are actually persisted and reloaded (a
        #     duplicated head would silently stay at its random init),
        #   * inference (``load_checkpoint``) rebuilds the very same slots.
        missing = [name for name in ("markov_head", "confidence_head") if not hasattr(draft_model, name)]
        if missing:
            raise ValueError(
                "DSpark heads live on the draft model; expected attributes "
                f"{missing} on {type(draft_model).__name__}. Build the trainer module "
                "through vlm_dflash.dspark.factory so the draft model is a "
                "Qwen3DSparkDraftModel."
            )
        mismatches: list[str] = []
        if markov_rank is not None and int(markov_rank) != int(draft_model.markov_rank):
            mismatches.append(f"markov_rank {int(markov_rank)} != model's {int(draft_model.markov_rank)}")
        if markov_head_type is not None and str(markov_head_type).lower() != str(draft_model.markov_head_type):
            mismatches.append(
                f"markov_head_type {str(markov_head_type)!r} != model's {draft_model.markov_head_type!r}"
            )
        if enable_confidence_head is not None and bool(enable_confidence_head) != bool(
            draft_model.enable_confidence_head
        ):
            mismatches.append(
                f"enable_confidence_head {bool(enable_confidence_head)} != model's "
                f"{bool(draft_model.enable_confidence_head)}"
            )
        if confidence_head_with_markov is not None and bool(confidence_head_with_markov) != bool(
            draft_model.confidence_head_with_markov
        ):
            mismatches.append(
                f"confidence_head_with_markov {bool(confidence_head_with_markov)} != model's "
                f"{bool(draft_model.confidence_head_with_markov)}"
            )
        if mismatches:
            raise ValueError(
                "DSpark head knobs disagree with the draft model config (the model is the "
                "single source of truth; both sides must come from algorithm.dspark): "
                + "; ".join(mismatches)
            )
        if float(confidence_head_alpha) < 0:
            raise ValueError(f"confidence_head_alpha must be >= 0, got {confidence_head_alpha}")
        self.markov_rank = int(draft_model.markov_rank)
        self.enable_confidence_head = bool(draft_model.enable_confidence_head)
        self.confidence_head_with_markov = bool(draft_model.confidence_head_with_markov)
        self.confidence_head_alpha = float(confidence_head_alpha)
        # Shared references (the module already owns these as submodules; binding
        # them here is only for attribute access in ``_apply_algorithm_terms``).
        self.markov_head = draft_model.markov_head
        self.confidence_head = draft_model.confidence_head

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
        block_logits = logits.view(bsz, n, bs, -1)
        hidden_blocks = output_hidden.view(bsz, n, bs, -1)
        # Previous token of each position a slot predicts: slot k predicts
        # ``label_indices[..., k]``, and its Markov bias conditions on
        # ``label_indices[..., k] - 1`` (for the first supervised slot this is
        # the anchor token itself: it predicts ``anchor + input_shift``).
        prev_indices = (label_indices - 1).clamp(min=0)
        prev_token_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1), 2, prev_indices
        )
        metrics: dict[str, float] = {}
        prev_emb = None
        if self.markov_head is not None:
            bias = self.markov_head.compute_block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
            )
            block_logits = block_logits + bias
            with torch.no_grad():
                metrics["markov_bias_abs_mean"] = float(bias.detach().abs().mean())
        aux_loss: Optional[torch.Tensor] = None
        if self.confidence_head is not None:
            assert self.markov_head is not None or not self.confidence_head_with_markov
            if self.confidence_head_with_markov:
                prev_emb = self.markov_head.get_prev_embeddings(prev_token_ids)
                conf_features = torch.cat([hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1)
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features)  # [B, N, bs]

            first = self.first_supervised_slot
            with torch.no_grad():
                accepted = (
                    block_logits.detach().argmax(dim=-1)[:, :, first:] == target_ids[:, :, first:]
                ).to(confidence_logits.dtype)
            conf_sup = confidence_logits[:, :, first:]
            mask_sup = block_mask[:, :, first:].to(conf_sup.dtype)
            weights = mask_sup
            if self.loss_decay_gamma is not None:
                depth = torch.exp(
                    -torch.arange(bs - first, device=conf_sup.device, dtype=conf_sup.dtype)
                    / self.loss_decay_gamma
                )
                weights = weights * depth
            bce = F.binary_cross_entropy_with_logits(conf_sup, accepted, reduction="none")
            conf_loss = (bce * weights).sum() / weights.sum().clamp_min(1e-6)
            aux_loss = self.confidence_head_alpha * conf_loss
            with torch.no_grad():
                probs = conf_sup.detach().sigmoid()
                denom = mask_sup.sum().clamp_min(1.0)
                metrics["confidence_bce"] = float(conf_loss.detach())
                metrics["confidence_pred_mean"] = float((probs * mask_sup).sum() / denom)
                metrics["confidence_target_mean"] = float((accepted * mask_sup).sum() / denom)
                metrics["confidence_abs_error"] = float(((probs - accepted).abs() * mask_sup).sum() / denom)
                if self.calibration_sink is not None:
                    # Flatten the block axis: the number of sampled anchors varies
                    # per batch, so concatenating [B, N_i, slots] tensors along
                    # dim 0 would raise a size mismatch.
                    self.calibration_sink.append(
                        (
                            probs.reshape(-1, probs.shape[-1]).float().cpu(),
                            accepted.reshape(-1, accepted.shape[-1]).float().cpu(),
                            mask_sup.reshape(-1, mask_sup.shape[-1]).detach().float().cpu(),
                        )
                    )
        return block_logits.view(bsz, n * bs, -1), aux_loss, metrics or None

    @torch.no_grad()
    def calibration_report(self, loader, device, max_batches: int = 16) -> Optional[dict]:
        """Per-slot confidence calibration on a held-out feature loader.

        Runs the module over up to ``max_batches`` batches with the calibration
        sink enabled and reports BCE, expected calibration error (10 bins),
        AUROC of the confidence signal against the greedy-acceptance indicator,
        and per-slot predicted-vs-observed acceptance. Returns ``None`` when the
        confidence head is disabled.
        """
        if self.confidence_head is None:
            return None
        was_training = self.training
        self.eval()
        self.calibration_sink = []
        try:
            for batch_index, batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                self(
                    input_ids=batch["input_ids"],
                    hidden_states=batch["hidden"],
                    loss_mask=batch["loss_mask"],
                    image_mask=batch["image_mask"],
                    multimodal_position_ids=batch["position3"],
                )
            collected = self.calibration_sink
        finally:
            self.calibration_sink = None
            if was_training:
                self.train()
        if not collected:
            return None
        probs = torch.cat([item[0] for item in collected], dim=0)
        accepted = torch.cat([item[1] for item in collected], dim=0)
        mask = torch.cat([item[2] for item in collected], dim=0).bool()
        p = probs[mask]
        y = accepted[mask]
        if p.numel() == 0:
            return None
        eps = 1e-7
        bce = float(-(y * (p + eps).log() + (1 - y) * (1 - p + eps).log()).mean())
        bins = torch.linspace(0.0, 1.0, 11)
        ece = 0.0
        for low, high in zip(bins[:-1], bins[1:]):
            in_bin = (p >= low) & (p < high if high < 1.0 else p <= high)
            if in_bin.any():
                ece += float(in_bin.float().mean()) * abs(float(p[in_bin].mean()) - float(y[in_bin].mean()))
        positives = y.sum()
        negatives = y.numel() - positives
        if positives > 0 and negatives > 0:
            order = torch.argsort(p)
            ranks = torch.empty_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(1, p.numel() + 1, dtype=torch.float32)
            auroc = float((ranks[y.bool()].sum() - positives * (positives + 1) / 2) / (positives * negatives))
        else:
            auroc = None
        per_slot_pred = probs.mean(dim=0).tolist()
        per_slot_obs = []
        for slot in range(probs.shape[-1]):
            slot_mask = mask[:, slot]
            per_slot_obs.append(
                float(accepted[:, slot][slot_mask].mean()) if slot_mask.any() else None
            )
        return {
            "confidence_bce": bce,
            "confidence_ece_10bin": ece,
            "confidence_auroc": auroc,
            "confidence_pred_std": float(p.std()),
            "confidence_mean": float(p.mean()),
            "accepted_rate": float(y.mean()),
            "supervised_slots": int(probs.shape[-1]),
            "per_slot_pred_mean": per_slot_pred,
            "per_slot_accept_rate": per_slot_obs,
        }
