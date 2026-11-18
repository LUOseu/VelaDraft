"""D1 bucketing: within-logical-batch physical regrouping with preserved weights.

Design constraints (from the D1 brief / shared protocol):
- only the physical microbatch *composition* changes; the logical optimizer batch,
  per-sample anchors/labels/masks, and the ORIGINAL microbatch-mean weighting are
  preserved;
- anchors are frozen once per logical batch (canonical sampling rule) and replayed
  in every physical group, so regrouping cannot change the supervision set;
- per-sample loss weight is ``g_i = 1 / (accumulation * Z_orig_mb(i))`` where
  ``Z_orig_mb(i)`` is the decay-weight sum of the sample's ORIGINAL microbatch.

This module is self-contained (planner + weights + Z bookkeeping). Trainer-core
integration (anchor-table replay hook + scaled-mask loss) is proposed to C0 in
``HANDOFF.md``; nothing here imports the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


#: R2 tensor-probe fit (evidence/tensor_probe_w0_gpu3.json; GPU3 solo, B=8,
#: flex_attention, 7 shapes, |residual| <= 2%):
#:     t ~= 0.00199 * sum(C) + 0.0367 * sum(N * b)   [ms]
#: One anchor block (``A`` = N slots of ``b`` rows in the draft stack / head)
#: therefore costs ``beta*b/alpha ~= 0.294 / 0.0020 ~= 147`` C-token
#: equivalents. This fitted ratio drives ``cost`` sorting and the ``grid``
#: partition choice; it is a scheduling estimate, not a FLOP theorem.
COST_ANCHOR_WEIGHT = 0.0367 * 8 / 0.00199


@dataclass
class SampleStat:
    """Planning-time per-sample facts (all known before any forward)."""

    key: str          # stable sample key (sample_id or digest)
    C: int            # context length (real tokens)
    A: int            # frozen anchor count for this logical batch
    Z: float          # per-sample decay-weight sum under the frozen anchors
    index: int = field(default=0)  # row index within the logical batch


def decay_weights(block_size: int, first_slot: int, gamma: float | None):
    import torch

    slots = block_size - first_slot
    if gamma is None:
        return torch.ones(slots, dtype=torch.float64)
    return torch.exp(-torch.arange(slots, dtype=torch.float64) / gamma)


def sample_z_rows(loss_mask, anchors, keep, block_size: int, first_slot: int,
                  gamma: float | None) -> list[float]:
    """Per-sample Z = sum over frozen anchors/slots of w_k * gathered loss mask.

    Anchors and keep are the frozen canonical table rows ([B, N] tensors, padded).
    loss_mask is the padded [B, S] tensor of the physical batch the anchors were
    sampled in (padding beyond each sample's real length is zero, so it drops out).
    """
    import torch

    w = decay_weights(block_size, first_slot, gamma)
    out: list[float] = []
    B, S = loss_mask.shape
    for row in range(B):
        z = 0.0
        for col in range(anchors.shape[1]):
            if not bool(keep[row, col]):
                continue
            a = int(anchors[row, col])
            for slot in range(first_slot, block_size):
                label = a + slot
                if 0 <= label < S and float(loss_mask[row, label]) > 0.5:
                    z += float(w[slot - first_slot])
        out.append(z)
    return out


def split_evenly(order: Sequence[int], k: int) -> list[list[int]]:
    """Contiguous split of ``order`` into k groups (sizes differ by <= 1)."""
    k = max(1, min(k, len(order)))
    base, extra = divmod(len(order), k)
    out, start = [], 0
    for g in range(k):
        size = base + (1 if g < extra else 0)
        out.append(list(order[start:start + size]))
        start += size
    return [group for group in out if group]


def modeled_cost(stats: Sequence[SampleStat], groups: Sequence[Sequence[int]],
                 cost_weight: float = COST_ANCHOR_WEIGHT) -> float:
    """Max-padded cost proxy sum_g |g| * (maxC_g + w * maxA_g), w = fitted ratio."""
    total = 0.0
    for group in groups:
        if not group:
            continue
        max_c = max(stats[i].C for i in group)
        max_a = max(stats[i].A for i in group)
        total += len(group) * (max_c + cost_weight * max_a)
    return total


def _grid_partition(stats: Sequence[SampleStat], k: int,
                    cost_weight: float) -> list[list[int]]:
    """2D quantile partition: kc C-bands x ka A-bands, best factorization by model.

    Candidate (kc, ka) pairs are the factorizations kc*ka == k plus the
    axis-aligned (1, k) / (k, 1) splits; the winner is chosen by
    :func:`modeled_cost` (so a pool with a constant A degenerates to the
    C-split, a constant C to the A-split, and a mixed pool keeps both axes
    tight). Bands are equal-count quantiles.
    """
    candidates: list[tuple[int, int]] = [(1, k), (k, 1)]
    for kc in range(2, int(k ** 0.5) + 1):
        if k % kc == 0:
            candidates.append((kc, k // kc))
            candidates.append((k // kc, kc))
    seen: set[tuple[int, int]] = set()
    best: tuple[float, list[list[int]]] | None = None
    for kc, ka in candidates:
        if (kc, ka) in seen:
            continue
        seen.add((kc, ka))
        by_c = sorted(range(len(stats)), key=lambda i: (stats[i].C, stats[i].A, stats[i].index))
        groups: list[list[int]] = []
        for band in split_evenly(by_c, kc):
            by_a = sorted(band, key=lambda i: (stats[i].A, stats[i].C, stats[i].index))
            groups.extend(split_evenly(by_a, ka))
        cost = modeled_cost(stats, groups, cost_weight)
        if best is None or cost < best[0]:
            best = (cost, groups)
    return best[1]


def plan_groups(stats: Sequence[SampleStat], mode: str, sub_batch_size: int,
                cost_weight: float = COST_ANCHOR_WEIGHT) -> list[list[int]]:
    """Physical groups (row indices) within one logical batch.

    mode:
      "canonical": positional split into canonical microbatches of sub_batch_size
      "token"    : sort by C (token-only bucketing), split into ~equal groups
      "dual"     : sort by (C, A) lexicographic, split into ~equal groups
                   (the literal dual-axis reading; note this only re-orders
                   within C-ties, so on real pools it often equals "token")
      "cost"     : sort by fitted cost C + cost_weight * A
      "grid"     : 2D quantile partition (see :func:`_grid_partition`)
    Group sizes are as equal as possible and never exceed ``sub_batch_size``
    (except single-sample leftovers), so the caller sees no oversized forward.
    """
    n = len(stats)
    order = list(range(n))
    if mode == "canonical":
        groups = [order[i:i + sub_batch_size] for i in range(0, n, sub_batch_size)]
        return groups
    k = max(1, -(-n // sub_batch_size))
    if mode == "grid":
        groups = _grid_partition(stats, k, cost_weight)
    elif mode == "token":
        order.sort(key=lambda i: (stats[i].C, stats[i].index))
        groups = split_evenly(order, k)
    elif mode == "dual":
        order.sort(key=lambda i: (stats[i].C, stats[i].A, stats[i].index))
        groups = split_evenly(order, k)
    elif mode == "cost":
        order.sort(key=lambda i: (stats[i].C + cost_weight * stats[i].A, stats[i].index))
        groups = split_evenly(order, k)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    # fold zero-anchor samples into the last group (they cost only context
    # padding and would otherwise form a degenerate all-zero forward)
    flat = [i for group in groups for i in group]
    zero = [i for i in flat if stats[i].A == 0]
    if zero and len(groups) > 1:
        keep = [i for i in flat if stats[i].A > 0]
        if keep:
            k = max(1, -(-len(keep) // sub_batch_size))
            groups = split_evenly(keep, k)
            groups[-1] = groups[-1] + zero
        else:
            groups = [zero]
    return groups


def canonical_microbatches(n: int, micro_batch: int) -> list[list[int]]:
    return [list(range(i, min(i + micro_batch, n))) for i in range(0, n, micro_batch)]


def row_weights(keys: Sequence[str], canonical_groups: Sequence[Sequence[int]],
                per_sample_z: Sequence[float], accumulation: int) -> dict[str, float]:
    """g_i = 1 / (accumulation * Z_orig_mb(i)); Z=0 samples get weight 0."""
    z_mb: dict[int, float] = {}
    for mb, members in enumerate(canonical_groups):
        z_mb[mb] = sum(per_sample_z[i] for i in members)
    weights: dict[str, float] = {}
    for mb, members in enumerate(canonical_groups):
        denom = accumulation * z_mb[mb]
        for i in members:
            weights[keys[i]] = (1.0 / denom) if denom > 0 and per_sample_z[i] > 0 else 0.0
    return weights


def estimate_group_cost(stats: Sequence[SampleStat], members: Sequence[int],
                        cost_weight: float = COST_ANCHOR_WEIGHT) -> float:
    """Cheap max-padded cost proxy of one physical group (for reporting/planning)."""
    if not members:
        return 0.0
    max_c = max(stats[i].C for i in members)
    max_a = max(stats[i].A for i in members)
    return len(members) * (max_c + cost_weight * max_a)
