"""Acceptance-length aggregation for speculative runs.

AL definition (recorded in ``eval_result.json``): ``AL = new_tokens /
target_verification_steps``, the bonus-token-inclusive number of tokens produced
per target forward pass. Per-slot acceptance is the per-offset acceptance rate
``accepted(k) / opportunities(k)``, where opportunities are the verification
rounds that drafted at least ``k`` tokens (mirrors the reference
``_aggregate_proposal_offset_acceptance``: a round clipped by the output budget
is excluded from the denominator of offsets it never drafted). Acceptance rate
is accepted draft tokens / drafted tokens.
"""

from __future__ import annotations


def _mean(values, count):
    return sum(values) / count if count else None


def summarize_samples(samples: list[dict], block_size: int) -> dict:
    rounds = sum(sample["verify_steps"] for sample in samples)
    new_tokens = sum(sample["new_tokens"] for sample in samples)
    accepted = sum(sample["accepted_tokens"] for sample in samples)
    drafted = sum(sample["draft_tokens"] for sample in samples)
    events = [event for sample in samples for event in sample["acceptance_events"]]
    per_slot = []
    for slot in range(block_size - 1):
        opportunities = [event for event in events if event["draft_tokens"] >= slot + 1]
        accepted_at_least = sum(1 for event in opportunities if event["accepted_tokens"] >= slot + 1)
        per_slot.append(accepted_at_least / len(opportunities) if opportunities else None)
    count = len(samples)
    return {
        "num_samples": count,
        "verify_steps": rounds,
        "new_tokens": new_tokens,
        "accepted_tokens": accepted,
        "draft_tokens": drafted,
        "mean_al": (new_tokens / rounds) if rounds else None,
        "tokens_per_target_forward": (new_tokens / rounds) if rounds else None,
        "acceptance_rate": (accepted / drafted) if drafted else None,
        "per_slot_acceptance": per_slot,
        "wall_seconds_per_sample": _mean([s.get("wall_seconds", 0.0) for s in samples], count),
        "target_verify_seconds_per_sample": _mean([s.get("target_verify_seconds", 0.0) for s in samples], count),
        "target_prefill_seconds_per_sample": _mean([s.get("target_prefill_seconds", 0.0) for s in samples], count),
        "draft_seconds_per_sample": _mean([s.get("draft_seconds", 0.0) for s in samples], count),
    }
