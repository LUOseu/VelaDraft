"""Forward-KL distillation loss (JetSpec).

Ported from the maintainer implementation
(``nemo_automodel/components/loss/kd_loss.py`` in NVIDIA-NeMo/Automodel,
Apache-2.0; inspected read-only via the pinned local mirror). Semantics kept
identical:

- ``KL(P_target || Q_draft) = sum_v P_v * (log P_v - log Q_v)`` per token, with
  ``P`` the teacher softmax and ``Q`` the student softmax;
- both logits are divided by the temperature ``T`` before the softmax and the
  resulting per-token KL is multiplied by ``T^2`` (Hinton et al., 2015), so the
  gradient scale stays independent of ``T``;
- positions whose label equals ``ignore_index`` are excluded, and the loss is
  the mean over the remaining valid tokens;
- fp32 upcast before softmax; ``chunk_size > 0`` processes the valid tokens in
  chunks so only one chunk's fp32 probability matrix is alive at a time.

The trainer additionally projects teacher *hidden* states to logits chunk-wise
(see ``trainer.py``); this module keeps the pure logits-in / scalar-out
semantics for tests and reuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ForwardKDLoss", "forward_kl_per_token"]

IGNORE_INDEX = -100


def forward_kl_per_token(teacher_log_prob: torch.Tensor, student_log_prob: torch.Tensor) -> torch.Tensor:
    """Per-token ``KL(P || Q)`` from full-vocabulary log probabilities.

    Mirrors ``_forward_kl_from_log_probs`` upstream: tokens with zero teacher
    probability contribute exactly zero (``p * log(p/q) -> 0`` as ``p -> 0``),
    which keeps underflowed entries from producing ``0 * -inf = NaN``.
    """
    teacher_prob = teacher_log_prob.exp()
    log_ratio = teacher_log_prob - student_log_prob
    log_ratio = torch.where(teacher_prob > 0, log_ratio, torch.zeros_like(log_ratio))
    return (teacher_prob * log_ratio).sum(dim=-1)


def _kl_chunked(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Per-token forward KL over ``[tokens, vocab]`` logits, chunked over tokens."""
    num_tokens = teacher_logits.shape[0]
    parts: list[torch.Tensor] = []
    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        teacher_logprob = F.log_softmax(teacher_logits[start:end], dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(student_logits[start:end], dim=-1, dtype=torch.float32)
        parts.append(forward_kl_per_token(teacher_logprob, student_logprob))
    return torch.cat(parts, dim=0)


class ForwardKDLoss(nn.Module):
    """Temperature-scaled forward KL averaged over valid tokens (JetSpec Eq. 8-9).

    Args:
        ignore_index: Label value marking non-supervised positions.
        temperature: Softmax temperature; the loss is ``T^2``-scaled.
        fp32_upcast: Cast logits to float32 before the softmaxes.
        chunk_size: ``> 0`` processes valid tokens in chunks of this size;
            ``0`` (default) materializes the full fp32 probability matrix.
    """

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        chunk_size: int = 0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        if chunk_size < 0:
            raise ValueError(f"chunk_size must be >= 0, got {chunk_size}.")
        self.ignore_index = int(ignore_index)
        self.temperature = float(temperature)
        self.fp32_upcast = bool(fp32_upcast)
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        """Forward KL over the valid positions; a scalar tensor.

        ``student_logits`` / ``teacher_logits`` have shape ``[..., vocab]`` and
        ``labels`` matches their leading dimensions. With ``num_batch_labels``
        the loss is ``sum(kl) / num_batch_labels`` (global normalization);
        otherwise it is the mean over this batch's valid tokens. A batch with no
        valid token returns a zero tensor (the trainer raises ``NoValidAnchors``
        before this point in practice, mirroring upstream).
        """
        valid_mask = (labels != self.ignore_index).reshape(-1)
        teacher_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        student_flat = student_logits.reshape(-1, student_logits.shape[-1])
        if int(valid_mask.sum().item()) == 0:
            return student_logits.new_tensor(0.0)
        t = teacher_flat[valid_mask]
        s = student_flat[valid_mask]
        if self.fp32_upcast:
            t = t.float()
            s = s.float()
        if self.temperature != 1.0:
            t = t.mul(1.0 / self.temperature)
            s = s.mul(1.0 / self.temperature)
        if self.chunk_size > 0:
            kl = _kl_chunked(t, s, self.chunk_size)
        else:
            teacher_logprob = F.log_softmax(t, dim=-1, dtype=torch.float32)
            student_logprob = F.log_softmax(s, dim=-1, dtype=torch.float32)
            kl = forward_kl_per_token(teacher_logprob, student_logprob)
        if self.temperature != 1.0:
            kl = kl * (self.temperature**2)
        if num_batch_labels is not None:
            return kl.sum() / float(num_batch_labels)
        return kl.mean()
