"""Acceptance-length aggregation for speculative runs.

Two acceptance numbers are reported side by side, because the literature uses
both and they differ by the guaranteed-progress token:

``al_draft_tokens_per_round``
    ``accepted_tokens / verify_steps`` -- mean raw accepted *draft* tokens per
    speculative round (0 when the first drafted token is rejected). This is the
    definition ViSpec's official README/``speed.py`` reports as ``tau``
    (``acceptance_len.append(int(accept_length))`` while
    ``new_token += accept_length + 1``), and the one Medusa/EAGLE tables use.

``al_progress_tokens_per_round``
    ``new_tokens / verify_steps`` -- mean output tokens committed per
    speculative round, i.e. the draft tokens plus the correction/bonus token
    that is emitted from the verifier's own logits. In the single-path greedy
    case this is ``al_draft + 1`` up to EOS/budget truncation.

``mean_al`` is kept as an alias of ``al_progress_tokens_per_round`` so existing
artifacts stay readable; new code should use the explicit names. Never compare
``mean_al`` against a published ``tau`` without checking which of the two the
other side computes (see ``docs/benchmark_metric_alignment.md``).

Per-slot acceptance is the per-offset acceptance rate ``accepted(k) /
opportunities(k)``, where opportunities are the verification rounds that drafted
at least ``k`` tokens (mirrors the reference
``_aggregate_proposal_offset_acceptance``: a round clipped by the output budget
is excluded from the denominator of offsets it never drafted). Acceptance rate
is accepted draft tokens / drafted tokens.
"""

from __future__ import annotations


def _mean(values, count):
    return sum(values) / count if count else None


def summarize_samples(samples: list[dict], block_size: int) -> dict:
    rounds = sum(sample["verify_steps"] for sample in samples)
    target_calls = sum(sample.get("target_forward_calls", 0) for sample in samples)
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
        "speculative_rounds": rounds,
        "target_forward_calls": target_calls,
        "new_tokens": new_tokens,
        "accepted_tokens": accepted,
        "draft_tokens": drafted,
        # AL definition (unchanged): tokens produced per speculative round
        # (bonus token included). ``tokens_per_target_forward`` now divides by
        # the counted target forwards instead of the round count, so the two
        # are reported side by side -- a round is NOT a target forward under
        # sequential verification or a strict-parallel fallback.
        "mean_al": (new_tokens / rounds) if rounds else None,
        # Explicit dual reporting; ``mean_al`` above is the same number as
        # ``al_progress_tokens_per_round`` (legacy alias, kept for old readers).
        "al_progress_tokens_per_round": (new_tokens / rounds) if rounds else None,
        "al_draft_tokens_per_round": (accepted / rounds) if rounds else None,
        "tokens_per_target_forward": (new_tokens / target_calls) if target_calls else None,
        "target_forwards_per_round": (target_calls / rounds) if rounds else None,
        "acceptance_rate": (accepted / drafted) if drafted else None,
        "per_slot_acceptance": per_slot,
        "wall_seconds_per_sample": _mean([s.get("wall_seconds", 0.0) for s in samples], count),
        "target_verify_seconds_per_sample": _mean([s.get("target_verify_seconds", 0.0) for s in samples], count),
        "target_prefill_seconds_per_sample": _mean([s.get("target_prefill_seconds", 0.0) for s in samples], count),
        "draft_seconds_per_sample": _mean([s.get("draft_seconds", 0.0) for s in samples], count),
    }
