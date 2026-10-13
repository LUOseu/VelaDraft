"""DFlare trainer module: the shared DFlash block-CE objective plus diagnostics.

DFlare changes the *context pathway* of the DFlash backbone, not the objective:
anchors, block mask, slot/label convention, decay-weighted CE and the frozen
LM head are identical, so the whole training step is inherited from
``DFlashTrainerModule``. This module only adds fusion diagnostics (softmax
entropy per draft layer, trainable flag) so "fusion wired but constant" is
visible in the run's metrics, and keeps the gradient-coverage audit useful for
``layer_fusion_weights`` / the separate context K/V projections.
"""

from __future__ import annotations

import torch

from ..dflash.trainer_core import DFlashTrainerModule

__all__ = ["DFlareTrainerModule"]


class DFlareTrainerModule(DFlashTrainerModule):
    """DFlash training step + per-layer fusion diagnostics."""

    def fusion_metrics(self) -> dict | None:
        model = self.draft_model
        probabilities_fn = getattr(model, "fusion_probabilities", None)
        if probabilities_fn is None:
            return None
        # Prefer the optimizer's fp32 master (attached by the training loop):
        # read from the bf16 live parameter a single AdamW step sits below the
        # quantisation grid, so the logged trajectory freezes at a constant
        # even while the fp32 master keeps moving.
        weights = getattr(model, "_master_fusion_weights", None)
        if weights is None:
            weights = getattr(model, "layer_fusion_weights", None)
        if weights is None:
            return None
        with torch.no_grad():
            logits = weights.detach().float()
            probabilities = torch.softmax(logits, dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            metrics = {
                "fusion_entropy_mean": float(entropy.mean()),
                "fusion_entropy_min": float(entropy.min()),
                "fusion_num_layers": float(probabilities.shape[-1]),
                "fusion_trainable": float(bool(getattr(model, "fusion_trainable", True))),
                "fusion_weight_abs_mean": float(logits.abs().mean()),
            }
            init_logits = getattr(model, "fusion_init_logits", None)
            if init_logits is not None:
                init_probabilities = torch.softmax(init_logits.detach().float(), dim=-1)
                metrics["fusion_prob_l1_drift"] = float((probabilities - init_probabilities).abs().mean())
        return metrics

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
        return logits, None, self.fusion_metrics()

