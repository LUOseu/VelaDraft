"""Rotary-layout rules shared by the contract and the draft model (no dependencies)."""

from __future__ import annotations


def partial_rotary_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """Head channels that rotate (``head_dim * partial_rotary_factor``, even, in ``(0, head_dim]``).

    A factor of 1.0 is the full head, returned unchecked as before partial rotary existed.
    """
    if float(partial_rotary_factor) == 1.0:
        return int(head_dim)
    rotary_dim = int(int(head_dim) * float(partial_rotary_factor))
    if not 0 < rotary_dim <= int(head_dim) or rotary_dim % 2:
        raise ValueError(
            f"partial_rotary_factor {partial_rotary_factor} must give an even rotary dim in "
            f"(0, {head_dim}], got {rotary_dim}."
        )
    return rotary_dim


def mrope_section_error(sections, rotary_dim: int, interleaved: bool) -> str | None:
    """Why ``sections`` cannot lay out a ``rotary_dim``-wide MRoPE table (``None`` = valid).

    The doubled section sum must equal the rotary dim. For the interleaved
    layout HF ``apply_interleaved_mrope`` writes axis ``a`` (1 = height,
    2 = width) at ``slice(a, 3 * sections[a], 3)`` of the half table, so an axis
    fits while ``a + 3 * (sections[a] - 1) < rotary_dim // 2`` (the draft's
    ``_merge_multiaxis_rotary_embeddings`` writes exactly those channels).
    """
    sections = [int(section) for section in sections]
    if len(sections) != 3 or 2 * sum(sections) != int(rotary_dim):
        return f"MRoPE sections {sections} doubled sum must equal the rotary dim {rotary_dim}."
    half = int(rotary_dim) // 2
    for axis in (1, 2) if interleaved else ():
        if axis + 3 * (sections[axis] - 1) >= half:
            return f"Interleaved MRoPE axis {axis} (section {sections[axis]}) overflows the {half} half-rotary channels."
    return None
