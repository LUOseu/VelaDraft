"""DFly trainer module: DFlare fusion diagnostics + TreeFlash correction metrics.

The objective is the shared DFlash block-CE (anchors, block mask, slot/label
convention, decay weights and the frozen LM head are unchanged), so the whole
step is inherited. DFly adds exactly one structural term to the *training*
forward: the hidden-states correction between the draft backbone and the frozen
LM head, applied through ``DFlashTrainerModule._correct_draft_hidden`` with
teacher-forced previous tokens (the token right before each supervised slot's
target).
"""

from __future__ import annotations

import torch

from ..dflare.trainer import DFlareTrainerModule

__all__ = ["DFlyTrainerModule"]


class DFlyTrainerModule(DFlareTrainerModule):
    """DFlash step + DFly correction (pre-head) + fusion/correction metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_correction_delta_abs_mean: float | None = None

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
        correction = getattr(self.draft_model, "hidden_correction", None)
        if correction is None:
            self._last_correction_delta_abs_mean = None
            return output_hidden
        previous_embeds = self.embed_tokens(prev_token_ids.reshape(bsz, n * bs))
        corrected = correction(output_hidden, previous_embeds)
        with torch.no_grad():
            self._last_correction_delta_abs_mean = float(
                (corrected - output_hidden).detach().float().abs().mean()
            )
        return corrected

    def _apply_algorithm_terms(self, *, logits, output_hidden, **kwargs):
        metrics = dict(self.fusion_metrics() or {})
        diagnostics = getattr(self.draft_model, "context_diagnostics", None)
        if diagnostics is not None:
            metrics.update(diagnostics())
        metrics["correction_enabled"] = float(
            getattr(self.draft_model, "hidden_correction", None) is not None
        )
        if self._last_correction_delta_abs_mean is not None:
            metrics["correction_delta_abs_mean"] = float(self._last_correction_delta_abs_mean)
        return logits, None, metrics or None

