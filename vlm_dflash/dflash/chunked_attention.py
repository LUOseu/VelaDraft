"""Query-chunked execution of the DFlash block mask.

The reference SDPA path materialises the whole dense additive mask and lets
autograd retain the full attention intermediate. These helpers instead build
one query slice at a time and wrap every slice in activation checkpointing, so
the backward pass recomputes a bounded slice rather than holding every score
matrix for the full ``A * block_size`` query row.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mask import DFlashMaskSpec


class AttentionRuntimeStats:
    """Process-local counters; not model state or hardware profiling.

    Count completed attention calls across layers (not model forwards/updates).
    Internal chunk checkpoint recomputation is excluded. If a caller wraps an
    entire layer in an external checkpoint, a completed replay is another call.
    Failed/incomplete calls and backward kernel dispatches are not measured.
    Reset after warmup and snapshot after the measured updates on each rank.
    """

    count_semantics = (
        "completed attention calls across layers; excludes internal chunk checkpoint "
        "recomputation, failed calls and backward kernels; completed external layer "
        "replays count as calls; not model-forward or optimizer-update counts"
    )
    counters = ("completed_attention_calls", "chunks", "fused_chunks", "fallback_chunks", "sdpa_chunks")

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = dict.fromkeys(self.counters, 0)
        self.backends = set()
        self.reasons = set()

    def record(self, info: dict):
        self.counts["completed_attention_calls"] += 1
        for key in self.counters[1:]:
            self.counts[key] += int(info.get(key, 0))
        backend = info["effective_backend"]
        self.backends.update(("npu_fusion", "sdpa_chunked") if backend == "mixed" else (backend,))
        self.reasons.update(info.get("fallback_reasons", ()))
        if info.get("fallback_reason"):
            self.reasons.add(info["fallback_reason"])

    def merge(self, snapshot: dict):
        for key in self.counters:
            self.counts[key] += snapshot[key]
        self.backends.update(snapshot["effective_backends"])
        self.reasons.update(snapshot["fallback_reasons"])

    def snapshot(self) -> dict:
        return {
            **self.counts,
            "effective_backends": sorted(self.backends),
            "fallback_reasons": sorted(self.reasons),
            "mixed_backend": len(self.backends) > 1,
            "count_semantics": self.count_semantics,
        }


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    return states.repeat_interleave(repeats, dim=1)


def _chunk_ranges(length: int, chunk_size: int):
    for start in range(0, length, chunk_size):
        yield start, min(length, start + chunk_size)


def sdpa_chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    spec: DFlashMaskSpec,
    *,
    dropout_p: float,
    scale: float,
) -> tuple[torch.Tensor, dict]:
    """Run block-masked SDPA without a full dense mask or retained full scores.

    ``query`` is ``[B, Hq, A*L, D]`` and ``key``/``value`` are
    ``[B, Hkv, S+A*L, D]``. Every query attends to the complete K/V sequence;
    chunking changes neither the softmax denominator nor the output.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("sdpa_chunked_attention expects rank-4 Q/K/V tensors.")
    if query.shape[0] != spec.batch_size or query.shape[2] != spec.query_len:
        raise ValueError(
            f"Q shape {tuple(query.shape)} does not match mask spec batch={spec.batch_size}, "
            f"query_len={spec.query_len}."
        )
    if key.shape[-2] != spec.kv_len or value.shape[-2] != spec.kv_len:
        raise ValueError(
            f"K/V lengths {key.shape[-2]}/{value.shape[-2]} do not match mask spec kv_len={spec.kv_len}."
        )
    repeats = int(query.shape[1] // key.shape[1])
    if repeats * key.shape[1] != query.shape[1]:
        raise ValueError(
            f"Query heads ({query.shape[1]}) must be an integer multiple of KV heads ({key.shape[1]})."
        )
    key = _repeat_kv(key, repeats)
    value = _repeat_kv(value, repeats)
    chunk_size = min(spec.query_chunk_size, spec.query_len)

    def run_chunk(q_chunk: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        mask = spec.materialize_additive(start, stop, dtype=q_chunk.dtype)
        return F.scaled_dot_product_attention(
            q_chunk,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )

    outputs = []
    use_checkpoint = torch.is_grad_enabled() and query.requires_grad
    for start, stop in _chunk_ranges(spec.query_len, chunk_size):
        q_chunk = query[:, :, start:stop]
        if use_checkpoint:
            outputs.append(checkpoint(run_chunk, q_chunk, start, stop, use_reentrant=False))
        else:
            outputs.append(run_chunk(q_chunk, start, stop))
    output = torch.cat(outputs, dim=2)
    return output, {
        "requested_backend": spec.backend,
        "effective_backend": "sdpa_chunked",
        "query_chunk_size": chunk_size,
        "chunks": (spec.query_len + chunk_size - 1) // chunk_size,
        "fused_chunks": 0,
        "fallback_chunks": 0,
        "sdpa_chunks": len(outputs),
        "mixed_backend": False,
        "fallback_mode": "not_applicable",
        "fallback_reason": None,
        "fallback_reasons": [],
        "activation_checkpointing": bool(use_checkpoint),
    }


def _npu_fusion_call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    scale: float,
    keep_prob: float,
) -> torch.Tensor:
    try:
        import torch_npu
    except Exception as error:
        raise RuntimeError(
            "attention_backend=npu_fusion needs torch_npu/CANN, but torch_npu could not be imported: "
            f"{type(error).__name__}: {error}."
        ) from error
    fusion = getattr(torch_npu, "npu_fusion_attention", None)
    if not callable(fusion):
        raise RuntimeError(
            "This torch_npu build does not expose torch_npu.npu_fusion_attention; "
            "use attention_backend=sdpa_chunked."
        )
    batch, heads, query_len, head_dim = query.shape
    kv_len = key.shape[-2]
    q_bsh = query.transpose(1, 2).contiguous().reshape(batch, query_len, heads * head_dim)
    k_bsh = key.transpose(1, 2).contiguous().reshape(batch, kv_len, -1)
    v_bsh = value.transpose(1, 2).contiguous().reshape(batch, kv_len, -1)
    result = fusion(
        q_bsh,
        k_bsh,
        v_bsh,
        heads,
        "BSH",
        atten_mask=atten_mask,
        scale=float(scale),
        keep_prob=float(keep_prob),
        sparse_mode=0,
    )
    output = result[0] if isinstance(result, (tuple, list)) else result
    return output.reshape(batch, query_len, heads, head_dim).transpose(1, 2)


def npu_fusion_chunked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    spec: DFlashMaskSpec,
    *,
    dropout_p: float,
    scale: float,
) -> tuple[torch.Tensor, dict]:
    """Opt-in Ascend fused attention over one bounded mask chunk at a time.

    ``npu_fusion_attention`` uses ``True/1 = masked`` while PyTorch SDPA uses
    ``True = attend``. The conversion is explicit in
    :meth:`DFlashMaskSpec.materialize_bool(inverted=True)`. If the kernel
    rejects a shape and the config opted into ``npu_fusion_fallback``, the same
    chunk falls back to the repository's SDPA reference path and the reason is
    returned for the training summary.
    """
    repeats = int(query.shape[1] // key.shape[1])
    if repeats * key.shape[1] != query.shape[1]:
        raise ValueError(
            f"Query heads ({query.shape[1]}) must be an integer multiple of KV heads ({key.shape[1]})."
        )
    key = _repeat_kv(key, repeats)
    value = _repeat_kv(value, repeats)
    chunk_size = min(spec.query_chunk_size, spec.query_len)
    fallback_reasons: list[str] = []
    fused_chunks = 0
    fallback_chunks = 0
    collecting = True

    def run_chunk(q_chunk: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        nonlocal fused_chunks, fallback_chunks
        # Ascend convention: True means masked out. Never pass the additive
        # SDPA -inf mask here.
        mask = spec.materialize_bool(start, stop, inverted=True)
        try:
            result = _npu_fusion_call(
                q_chunk,
                key,
                value,
                mask,
                scale=scale,
                keep_prob=1.0 - float(dropout_p),
            )
            if collecting:
                fused_chunks += 1
            return result
        except RuntimeError as error:
            if not spec.npu_fusion_fallback:
                raise
            if collecting:
                fallback_chunks += 1
                fallback_reasons.append(str(error))
            return F.scaled_dot_product_attention(
                q_chunk,
                key,
                value,
                attn_mask=~mask,
                dropout_p=dropout_p,
                is_causal=False,
                scale=scale,
            )

    outputs = []
    use_checkpoint = torch.is_grad_enabled() and query.requires_grad
    for start, stop in _chunk_ranges(spec.query_len, chunk_size):
        q_chunk = query[:, :, start:stop]
        if use_checkpoint:
            outputs.append(checkpoint(run_chunk, q_chunk, start, stop, use_reentrant=False))
        else:
            outputs.append(run_chunk(q_chunk, start, stop))
    # Freeze forward-only telemetry before checkpoint closures replay in backward.
    collecting = False
    output = torch.cat(outputs, dim=2)
    mixed = fused_chunks > 0 and fallback_chunks > 0
    return output, {
        "requested_backend": spec.backend,
        "effective_backend": "mixed" if mixed else ("npu_fusion" if not fallback_chunks else "sdpa_chunked"),
        "fused_chunks": fused_chunks,
        "fallback_chunks": fallback_chunks,
        "sdpa_chunks": fallback_chunks,
        "mixed_backend": mixed,
        "fallback_mode": "profile" if spec.npu_fusion_fallback else "strict",
        "fallback_reasons": sorted(set(fallback_reasons)),
        "query_chunk_size": chunk_size,
        "chunks": (spec.query_len + chunk_size - 1) // chunk_size,
        "fallback_reason": None if not fallback_reasons else fallback_reasons[0],
        "activation_checkpointing": bool(use_checkpoint),
    }


__all__ = ["AttentionRuntimeStats", "npu_fusion_chunked_attention", "sdpa_chunked_attention"]
