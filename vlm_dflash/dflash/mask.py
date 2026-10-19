# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DFlash sparse-attention masks (SDPA + FlexAttention).

Builds the DFlash block-diagonal attention masks (paper §4.2) so that
multi-anchor DFlash training (up to ~512 anchors per sequence — paper
Appendix A.1) is tractable in memory.

KV layout:  ``[ context (S tokens) | block_0 | block_1 | ... | block_{N-1} ]``
Q  layout:  ``[ block_0 | block_1 | ... | block_{N-1} ]``

Each query in block *b* attends to:

  1. context positions strictly less than ``anchor[b]`` (causal-style prefix)
  2. its own block's noise positions (bidirectional in-block, or causal in-block
     when ``causal=True`` — see below)
  3. nothing else — other blocks are invisible

The context is never queried *from* (the target LM is frozen, we only need
its hidden states), so omitting it from Q halves the attention compute vs.
including context positions in Q.

``causal=False`` (default) is the DFlash mask: in-block attention is
bidirectional. ``causal=True`` is the JetSpec mask: a query at within-block
offset ``i`` attends only to noise positions at offset ``j <= i``, so each
branch follows the target model's autoregressive order (paper §2.2). Offset 0
(the anchor) still attends to itself, so no query row is ever fully masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

# Singleton-cached torch.compile of ``create_block_mask``. The mask_mod closure
# changes per step (new anchor positions), but the create_block_mask machinery
# itself is identical — compiling once and reusing avoids per-step Python
# overhead of evaluating mask_mod across the BlockMask grid.
# ``dynamic=True``: the mask shape depends on the per-batch anchor count
# (``min(num_anchors, valid anchors in the batch)``), so a static compile
# re-specializes (a full recompile, a multi-second stall mid-training) every
# time a batch carries a new anchor count. One dynamic compile covers all.
_compiled_create_block_mask = None


def _get_compiled_create_block_mask():
    """Lazy-initialise a compiled ``create_block_mask`` and cache it."""
    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        from torch.nn.attention.flex_attention import create_block_mask

        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
    return _compiled_create_block_mask


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a dense additive attention mask for the SDPA backend.

    Args:
        anchor_positions: ``[B, N]`` anchor positions per sample (long).
        block_keep_mask:  ``[B, N]`` per-sample valid-anchor mask (bool).
        ctx_len:          context length ``S``.
        block_size:       block size.
        device:           torch device.
        dtype:            dtype for the additive mask (typically the model dtype).
        causal:           When True, make in-block attention causal (JetSpec);
            otherwise it is bidirectional (DFlash).
        ctx_doc_id:       ``[B, S]`` long per-context-token document id (sequence
            packing), or ``None`` to disable the per-document constraint.
        anchor_doc_id:    ``[B, N]`` long document id of each anchor. Required when
            ``ctx_doc_id`` is given; a block then attends only to context tokens in
            its anchor's document.

    Returns:
        ``[B, 1, N*block_size, S + N*block_size]`` float tensor: ``0`` at
        attended positions, ``-inf`` elsewhere.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size

    q_idx = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_idx = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)

    q_block = q_idx // block_size
    anchor_exp = anchor_positions.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    is_ctx = kv_idx < ctx_len
    ctx_visible = is_ctx & (kv_idx < anchor_exp)
    if ctx_doc_id is not None:
        assert anchor_doc_id is not None, "anchor_doc_id is required when ctx_doc_id is given"
        # Packing: a block attends only to context tokens in its anchor's
        # document. Build a per-kv doc id whose noise slots are a sentinel (-1)
        # that never matches an anchor's (>=0) doc id, then compare per block.
        kv_doc = torch.full((B, KV_LEN), -1, dtype=torch.long, device=device)
        kv_doc[:, :ctx_len] = ctx_doc_id
        anchor_doc_exp = anchor_doc_id.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
        ctx_visible = ctx_visible & (kv_doc.view(B, 1, 1, KV_LEN) == anchor_doc_exp)

    is_noise = kv_idx >= ctx_len
    kv_block = (kv_idx - ctx_len) // block_size
    noise_visible = is_noise & (q_block == kv_block)
    if causal:
        # Block-causal in-block attention (JetSpec): offset i sees only offsets
        # j <= i. Offset 0 (anchor) still sees itself, so no row is fully masked.
        q_in_block = q_idx - q_block * block_size
        kv_in_block = (kv_idx - ctx_len) - kv_block * block_size
        noise_visible = noise_visible & (kv_in_block <= q_in_block)

    keep = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    # A padding block (``keep`` False) keeps only its in-block (noise) attention,
    # so its query rows are never fully masked. A fully-masked row makes the dense
    # softmax (the sdpa/eager backends) return NaN, which then contaminates the
    # rest of the sample through the next layer's ``0 * NaN`` value aggregation.
    # The block's output is discarded downstream by the loss block mask anyway.
    bool_mask = (ctx_visible & keep) | noise_visible

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where(bool_mask, zero, neg_inf)


def build_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    num_blocks: int,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
):
    """Return the FlexAttention ``mask_mod`` closure encoding the DFlash mask.

    Factored out of :func:`create_dflash_block_mask` so the doc-aware gating can be
    materialised with ``torch.nn.attention.flex_attention.create_mask`` and tested on
    CPU (FlexAttention's kernel is CUDA-only). The closure uses only tensor ops (no
    Python control flow) so ``torch.compile`` can trace it; see the module docstring
    for the mask semantics and the ``ctx_doc_id`` / ``anchor_doc_id`` packing args.
    """
    N = num_blocks
    doc_aware = ctx_doc_id is not None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        safe_q_block = q_block.clamp(max=N - 1)
        anchor = anchor_positions[b, safe_q_block]

        is_ctx = kv_idx < ctx_len
        ctx_visible = is_ctx & (kv_idx < anchor)
        if doc_aware:
            # Narrow the captured ``Tensor | None`` args for ty; narrowing from
            # the enclosing scope does not reach the closure body.
            assert ctx_doc_id is not None and anchor_doc_id is not None
            # Packing: restrict the context prefix to the anchor's document.
            # Clamp the kv index into range for the ctx lookup; the result only
            # gates ``ctx_visible`` (already False on the noise half).
            kv_doc = ctx_doc_id[b, kv_idx.clamp(max=ctx_len - 1)]
            ctx_visible = ctx_visible & (kv_doc == anchor_doc_id[b, safe_q_block])

        is_noise = kv_idx >= ctx_len
        kv_block = (kv_idx - ctx_len) // block_size
        noise_visible = is_noise & (q_block == kv_block)
        if causal:
            # Block-causal in-block attention (JetSpec): offset i sees only
            # offsets j <= i. Offset 0 (anchor) still sees itself.
            q_in_block = q_idx - q_block * block_size
            kv_in_block = (kv_idx - ctx_len) - kv_block * block_size
            noise_visible = noise_visible & (kv_in_block <= q_in_block)

        keep = block_keep_mask[b, safe_q_block]
        in_bounds = q_block < N
        # Padding blocks keep in-block attention so no query row is fully masked
        # (see create_dflash_sdpa_mask; kept consistent across backends, and the
        # output is dropped by the loss block mask anyway).
        return ((ctx_visible & keep) | noise_visible) & in_bounds

    return dflash_mask_mod


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    use_compile: bool = True,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
) -> "BlockMask":
    """Build a sparse FlexAttention :class:`BlockMask` for DFlash training.

    See module docstring for the mask semantics. The returned ``BlockMask`` is
    consumed directly by transformers' ``flex_attention`` backend when
    ``_attn_implementation="flex_attention"`` is set on the draft model — pass
    it via the ``attention_mask`` kwarg.

    Args:
        anchor_positions: ``[B, N]`` anchor positions (long).
        block_keep_mask:  ``[B, N]`` valid-anchor mask (bool).
        ctx_len:          context length.
        block_size:       block size.
        device:           torch device.
        use_compile:      Cache and reuse a torch.compile'd ``create_block_mask``
            across calls (default True). Set to False when running on PyTorch
            builds that hit Inductor errors during compile.
        causal:           When True, make in-block attention causal (JetSpec);
            otherwise it is bidirectional (DFlash).
        ctx_doc_id:       ``[B, S]`` long per-context-token document id (sequence
            packing), or ``None`` to disable the per-document constraint.
        anchor_doc_id:    ``[B, N]`` long document id of each anchor. Required when
            ``ctx_doc_id`` is given; a block then attends only to context tokens in
            its anchor's document.

    Returns:
        :class:`torch.nn.attention.flex_attention.BlockMask`.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size

    dflash_mask_mod = build_dflash_mask_mod(
        anchor_positions, block_keep_mask, ctx_len, block_size, N, causal, ctx_doc_id, anchor_doc_id
    )

    # ``BLOCK_SIZE`` is left at the default (128). It MUST be a multiple of the
    # underlying flex_attention kernel's BLOCK_M / BLOCK_N (128 on H100); setting
    # it equal to our draft block_size (16) would in theory give finer-grained
    # sparsity but triggers Inductor's "Q and KV block size must be divisible
    # by BLOCK_M and BLOCK_N" lowering error at runtime.
    if use_compile:
        builder = _get_compiled_create_block_mask()
    else:
        from torch.nn.attention.flex_attention import create_block_mask

        builder = create_block_mask
    return builder(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


@dataclass(frozen=True)
class DFlashMaskSpec:
    """Lazily materialisable DFlash block mask for query-chunked attention.

    The normal SDPA backend builds the complete ``[B, 1, A*L, S+A*L]`` additive
    mask before the attention call. At realistic context/anchor counts that
    tensor dominates memory, and a backward pass would retain one softmax
    matrix per query too. This spec keeps only the compact anchor/keep/document
    metadata; :meth:`materialize_additive` / :meth:`materialize_bool` build one
    query slice at a time, and the attention path checkpoints each slice so
    backward recomputes it instead of retaining every slice at once.

    The mask semantics are exactly those of :func:`create_dflash_sdpa_mask` and
    :func:`build_dflash_mask_mod`: the concatenation of all query slices is the
    dense reference mask. ``backend`` records the requested execution path
    (``sdpa_chunked`` or ``npu_fusion``) for diagnostics; it does not alter the
    mask itself.
    """

    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    ctx_len: int
    block_size: int
    causal: bool = False
    ctx_doc_id: torch.Tensor | None = None
    anchor_doc_id: torch.Tensor | None = None
    query_chunk_size: int = 128
    backend: str = "sdpa_chunked"
    npu_fusion_fallback: bool = False

    def __post_init__(self) -> None:
        if self.anchor_positions.ndim != 2 or self.block_keep_mask.shape != self.anchor_positions.shape:
            raise ValueError(
                "DFlashMaskSpec expects anchor_positions [B, N] and block_keep_mask with the same shape; "
                f"got {tuple(self.anchor_positions.shape)} and {tuple(self.block_keep_mask.shape)}."
            )
        if self.ctx_len < 0 or self.block_size <= 0:
            raise ValueError("DFlashMaskSpec requires ctx_len >= 0 and block_size > 0.")
        if self.query_chunk_size <= 0:
            raise ValueError("DFlashMaskSpec.query_chunk_size must be a positive integer.")
        if (self.ctx_doc_id is None) != (self.anchor_doc_id is None):
            raise ValueError("ctx_doc_id and anchor_doc_id must be supplied together.")
        if self.ctx_doc_id is not None and self.ctx_doc_id.shape != (self.anchor_positions.shape[0], self.ctx_len):
            raise ValueError(
                f"ctx_doc_id must have shape {(self.anchor_positions.shape[0], self.ctx_len)}, "
                f"got {tuple(self.ctx_doc_id.shape)}."
            )

    @property
    def batch_size(self) -> int:
        return int(self.anchor_positions.shape[0])

    @property
    def num_blocks(self) -> int:
        return int(self.anchor_positions.shape[1])

    @property
    def query_len(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def kv_len(self) -> int:
        return self.ctx_len + self.query_len

    def _visible(self, query_start: int, query_stop: int) -> torch.Tensor:
        if query_start < 0 or query_stop > self.query_len or query_stop <= query_start:
            raise ValueError(
                f"Invalid query chunk [{query_start}, {query_stop}) for query_len={self.query_len}."
            )
        device = self.anchor_positions.device
        q_idx = torch.arange(query_start, query_stop, device=device).view(1, 1, -1, 1)
        kv_idx = torch.arange(self.kv_len, device=device).view(1, 1, 1, -1)
        q_block = q_idx // self.block_size
        safe_q_block = q_block.clamp(max=self.num_blocks - 1)
        anchor = torch.gather(
            self.anchor_positions.view(self.batch_size, 1, self.num_blocks, 1).expand(-1, 1, -1, -1),
            2,
            safe_q_block.expand(self.batch_size, -1, -1, -1),
        )

        is_ctx = kv_idx < self.ctx_len
        ctx_visible = is_ctx & (kv_idx < anchor)
        if self.ctx_doc_id is not None:
            assert self.anchor_doc_id is not None
            kv_doc = torch.gather(
                self.ctx_doc_id.view(self.batch_size, 1, 1, self.ctx_len).expand(-1, 1, query_stop - query_start, -1),
                3,
                kv_idx.clamp(max=max(self.ctx_len - 1, 0)).expand(self.batch_size, 1, query_stop - query_start, -1),
            )
            anchor_doc = torch.gather(
                self.anchor_doc_id.view(self.batch_size, 1, self.num_blocks, 1).expand(-1, 1, -1, -1),
                2,
                safe_q_block.expand(self.batch_size, -1, -1, -1),
            )
            ctx_visible = ctx_visible & (kv_doc == anchor_doc)

        is_noise = kv_idx >= self.ctx_len
        kv_block = (kv_idx - self.ctx_len) // self.block_size
        noise_visible = is_noise & (q_block == kv_block)
        if self.causal:
            q_in_block = q_idx - q_block * self.block_size
            kv_in_block = (kv_idx - self.ctx_len) - kv_block * self.block_size
            noise_visible = noise_visible & (kv_in_block <= q_in_block)
        keep = torch.gather(
            self.block_keep_mask.view(self.batch_size, 1, self.num_blocks, 1).expand(-1, 1, -1, -1),
            2,
            safe_q_block.expand(self.batch_size, -1, -1, -1),
        )
        return (ctx_visible & keep) | noise_visible

    def materialize_additive(
        self,
        query_start: int,
        query_stop: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Additive mask chunk ``[B, 1, query_stop-query_start, S+A*L]``."""
        dtype = dtype or torch.float32
        visible = self._visible(query_start, query_stop)
        neg_inf = torch.tensor(float("-inf"), device=visible.device, dtype=dtype)
        zero = torch.tensor(0.0, device=visible.device, dtype=dtype)
        return torch.where(visible, zero, neg_inf)

    def materialize_bool(self, query_start: int, query_stop: int, *, inverted: bool = False) -> torch.Tensor:
        """Boolean mask chunk.

        ``inverted=False`` follows the PyTorch SDPA convention (True = attend).
        ``inverted=True`` follows the Ascend fused-attention convention
        (True = mask out); callers must opt in explicitly through this flag.
        """
        visible = self._visible(query_start, query_stop)
        return ~visible if inverted else visible

    def full_additive(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Materialize the complete reference mask (tests / small inputs only)."""
        return self.materialize_additive(0, self.query_len, dtype=dtype)


def estimate_dflash_attention_memory(
    *,
    batch_size: int,
    num_blocks: int,
    block_size: int,
    context_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    query_chunk_size: int | None = None,
) -> dict:
    """Estimate the dominant mask/score tensors for dense and chunked paths.

    Scores/softmax storage is counted in fp32 (the training/backward working
    dtype); masks use ``dtype`` element size. The estimate is intentionally an
    order-of-magnitude allocator guard, not a promise about a fused kernel's
    internal workspace.
    """
    query_len = int(num_blocks) * int(block_size)
    kv_len = int(context_len) + query_len
    mask_bytes = int(batch_size) * query_len * kv_len * torch.empty((), dtype=dtype).element_size()
    score_bytes = int(batch_size) * int(num_heads) * query_len * kv_len * 4
    dense = {"mask_bytes": mask_bytes, "score_bytes": score_bytes, "total_bytes": mask_bytes + score_bytes}
    result = {
        "batch_size": int(batch_size),
        "num_blocks": int(num_blocks),
        "block_size": int(block_size),
        "context_len": int(context_len),
        "query_len": query_len,
        "kv_len": kv_len,
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "dense": {key: value for key, value in dense.items()},
        "dense_total_gb": round(dense["total_bytes"] / 1e9, 4),
        "chunked": None,
    }
    if query_chunk_size is not None:
        chunk = min(query_len, max(1, int(query_chunk_size)))
        chunk_mask = int(batch_size) * chunk * kv_len * torch.empty((), dtype=dtype).element_size()
        chunk_scores = int(batch_size) * int(num_heads) * chunk * kv_len * 4
        result["chunked"] = {
            "query_chunk_size": chunk,
            "mask_bytes": chunk_mask,
            "score_bytes": chunk_scores,
            "total_bytes": chunk_mask + chunk_scores,
        }
        result["chunked_total_gb"] = round((chunk_mask + chunk_scores) / 1e9, 4)
        result["chunks"] = (query_len + chunk - 1) // chunk
    return result
