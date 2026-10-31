"""Memory-bounded full-vocabulary CE over supervised block slots only.

This is a memory/time tradeoff, not a new objective or a fused kernel. Keep
the reference CE dtype and reduction order; either checkpoint each chunk or
precompute its hidden gradient when the output head is frozen.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from .decay_loss import DLLMLossOutput


def chunked_block_ce(hidden, head, targets, mask, *, loss_fn, block_size, chunk_size, counter):
    """Return (DLLMLossOutput, prediction IDs) in the original [B, T] layout."""
    bsz, tokens = targets.shape
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    selected = mask.reshape(-1).nonzero(as_tuple=True)[0]
    predictions = torch.full_like(flat_targets, -1)
    nll_parts, pred_parts = [], []

    def score(rows, labels):
        # Includes recomputation, matching the training loop's head counters.
        counter(student_head_calls=1, student_head_rows=rows.shape[0])
        logits = head(rows)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return F.cross_entropy(logits, labels, reduction="none"), logits.argmax(dim=-1)

    for start in range(0, selected.numel(), chunk_size):
        index = selected[start:start + chunk_size]
        rows, labels = flat_hidden[index], flat_targets[index]
        if torch.is_grad_enabled() and rows.requires_grad:
            nll, pred = checkpoint(score, rows, labels, use_reentrant=False)
        else:
            nll, pred = score(rows, labels)
        nll_parts.append(nll)
        pred_parts.append(pred)
    if nll_parts:
        nll_values = torch.cat(nll_parts)
        token_nll = nll_values.new_zeros(bsz * tokens).index_copy(0, selected, nll_values)
        predictions[selected] = torch.cat(pred_parts)
    else:
        # A zero-weight batch remains differentiable without projecting rows.
        token_nll = hidden.new_zeros(bsz * tokens) + hidden.sum() * 0
    predictions = predictions.view(bsz, tokens)
    correct, count = loss_fn._draft_acc_per_pos(predictions.eq(targets), mask, block_size)
    out = loss_fn._reduce(token_nll.view(bsz, tokens), mask, None, block_size, correct, count)
    counter(scored_rows=selected.numel())
    return out, predictions


class _FrozenHeadCE(torch.autograd.Function):
    """First-order scalar CE VJP; keep [B,T,H] gradients instead of [B,T,V].

    Derive each chunk's weighted VJP with ordinary autograd, including the
    reference reducer's dtype and denominator. The only backward operation
    left is multiplication by the upstream scalar (e.g. accumulation scale).
    Moving that scalar past the GEMM can change BF16 rounding; this is not a
    bitwise-equivalence guarantee. Head and supervision weights are frozen.
    """

    @staticmethod
    def forward(ctx, hidden, head, targets, mask, loss_fn, block_size, chunk_size, counter):
        flat = hidden.reshape(-1, hidden.shape[-1])
        labels = targets.reshape(-1)
        selected = mask.reshape(-1).nonzero(as_tuple=True)[0]
        grad_hidden = (torch.zeros_like(hidden, memory_format=torch.contiguous_format)
                       if ctx.needs_input_grad[0] else None)
        predictions = torch.full_like(labels, -1)
        token_nll = None
        coefficients = None
        for start in range(0, selected.numel(), chunk_size):
            index = selected[start:start + chunk_size]
            # The inner graph has one chunk's lifetime and never reaches model
            # parameters. Their gradients flow through this Function's output.
            with torch.enable_grad():
                rows = flat[index].detach().requires_grad_(grad_hidden is not None)
                counter(student_head_calls=1, student_head_rows=rows.shape[0])
                logits = head(rows)
                nll = F.cross_entropy(logits, labels[index], reduction="none")
                if token_nll is None:
                    token_nll = nll.new_zeros(targets.numel())
                    if grad_hidden is not None:
                        # Get d(total_loss)/d(NLL) from the actual reducer, so
                        # BF16 mean normalization and FP32 preserved weights
                        # follow exactly the same cast/order as dense CE.
                        dummy = torch.zeros_like(targets, dtype=nll.dtype, requires_grad=True)
                        reduced = loss_fn._reduce(dummy, mask, None, block_size).total_loss
                        coefficients = torch.autograd.grad(reduced, dummy)[0].reshape(-1)
                if grad_hidden is not None:
                    local_grad = torch.autograd.grad(nll, rows, grad_outputs=coefficients[index])[0]
                    grad_hidden.view_as(flat)[index] = local_grad
            token_nll[index] = nll.detach()
            predictions[index] = logits.detach().argmax(dim=-1)
            del rows, logits, nll
        if token_nll is None:
            token_nll = hidden.new_zeros(targets.numel())
        loss = loss_fn._reduce(token_nll.view_as(targets), mask, None, block_size).total_loss
        if grad_hidden is not None:
            ctx.save_for_backward(grad_hidden)
        predictions = predictions.view_as(targets)
        ctx.mark_non_differentiable(predictions)
        counter(scored_rows=selected.numel())
        return loss, predictions

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _grad_predictions):
        (grad_hidden,) = ctx.saved_tensors
        return grad_hidden * grad_loss, None, None, None, None, None, None, None


def precomputed_block_ce(hidden, head, targets, mask, *, loss_fn, block_size, chunk_size, counter):
    """Frozen unmodified Linear heads only; first-order training gradients."""
    if (type(head) is not torch.nn.Linear or mask.requires_grad
            or any(p.requires_grad or isinstance(p, DTensor) for p in head.parameters())):
        raise ValueError("precomputed CE requires a frozen unsharded Linear head and fixed supervision weights.")
    if not torch.is_grad_enabled():
        return chunked_block_ce(hidden, head, targets, mask, loss_fn=loss_fn, block_size=block_size,
                                chunk_size=chunk_size, counter=counter)
    loss, predictions = _FrozenHeadCE.apply(hidden, head, targets, mask, loss_fn, block_size, chunk_size, counter)
    correct, count = loss_fn._draft_acc_per_pos(predictions.eq(targets), mask, block_size)
    return DLLMLossOutput(loss, loss.detach().clone(), correct, count), predictions
