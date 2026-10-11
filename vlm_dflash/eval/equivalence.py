"""Token-exact equivalence between speculative and target-only greedy decoding."""

from __future__ import annotations


def cut_at_first_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Truncate ``ids`` at (and including) the first stop token."""
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[: index + 1]
    return list(ids)


def sequences_equal(spec_ids: list[int], target_ids: list[int], stop_ids: set[int]) -> bool:
    """True when both greedy decodes emit identical tokens up to their stop token."""
    return cut_at_first_stop(spec_ids, stop_ids) == cut_at_first_stop(target_ids, stop_ids)
