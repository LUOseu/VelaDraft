"""Prometheus metric parsing and speculative-decoding counter diffs."""

from __future__ import annotations

import math
import re
from typing import Any, Optional

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

SPEC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
ACTIVE_RUNNING = "vllm:num_requests_running"
ACTIVE_WAITING = "vllm:num_requests_waiting"
GPU_CACHE_USAGE = "vllm:gpu_cache_usage_perc"
GPU_MEMORY_USAGE = "vllm:gpu_memory_usage_perc"


class MetricParseError(ValueError):
    pass


def _strip_total(name: str) -> str:
    return name[:-6] if name.endswith("_total") else name


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse the subset of the Prometheus text format vLLM exposes."""

    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            closing = rest.rfind("}")
            if closing < 0:
                raise MetricParseError(f"unbalanced labels: {raw_line!r}")
            label_text = rest[:closing]
            value_text = rest[closing + 1 :].strip().split()
            labels = {key: value for key, value in _LABEL_RE.findall(label_text)}
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value_text, labels = parts[0], parts[1:], {}
        if not value_text:
            continue
        token = value_text[0]
        if token in ("NaN", "+NaN", "-NaN"):
            value = float("nan")
        elif token in ("Inf", "+Inf"):
            value = float("inf")
        elif token == "-Inf":
            value = float("-inf")
        elif _NUMBER_RE.match(token):
            value = float(token)
        else:
            continue
        name = _strip_total(name.strip())
        metrics.setdefault(name, []).append((labels, value))
    return metrics


def metric_value(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    labels: Optional[dict[str, str]] = None,
) -> float:
    name = _strip_total(name)
    total = 0.0
    for candidate_name in (name, name + "_total"):
        for candidate_labels, value in metrics.get(candidate_name, []):
            if labels is None or all(candidate_labels.get(k) == v for k, v in labels.items()):
                total += value
    return total


def metric_series(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
) -> list[tuple[dict[str, str], float]]:
    name = _strip_total(name)
    return list(metrics.get(name, [])) + list(metrics.get(name + "_total", []))


def spec_counts(metrics: dict[str, list[tuple[dict[str, str], float]]]) -> dict[str, Any]:
    """Return the aggregate speculative counters visible at a scrape."""

    drafts = metric_value(metrics, SPEC_DRAFTS)
    draft_tokens = metric_value(metrics, SPEC_DRAFT_TOKENS)
    accepted = metric_value(metrics, SPEC_ACCEPTED_TOKENS)
    per_pos: dict[str, float] = {}
    for labels, value in metric_series(metrics, SPEC_ACCEPTED_PER_POS):
        position = labels.get("position", "0")
        per_pos[position] = per_pos.get(position, 0.0) + float(value)
    per_position_rate = None
    if drafts > 0 and per_pos:
        per_position_rate = {
            position: per_pos[position] / drafts for position in sorted(per_pos, key=lambda x: int(x))
        }
    return {
        "drafts": int(round(drafts)),
        "draft_tokens": int(round(draft_tokens)),
        "accepted_tokens": int(round(accepted)),
        "per_position_accepted": {k: int(round(v)) for k, v in per_pos.items()},
        "per_position_acceptance_rate": per_position_rate,
        "acceptance_length": (1.0 + accepted / drafts) if drafts > 0 else None,
        "acceptance_rate": (accepted / draft_tokens) if draft_tokens > 0 else None,
    }


def active_requests(metrics: dict[str, list[tuple[dict[str, str], float]]]) -> dict[str, float]:
    return {
        "running": metric_value(metrics, ACTIVE_RUNNING),
        "waiting": metric_value(metrics, ACTIVE_WAITING),
        "total": metric_value(metrics, ACTIVE_RUNNING) + metric_value(metrics, ACTIVE_WAITING),
    }


def diff_metrics(
    before: dict[str, list[tuple[dict[str, str], float]]],
    after: dict[str, list[tuple[dict[str, str], float]]],
) -> dict[str, Any]:
    """Diff monotonic counters and report gauges from the *after* scrape.

    A counter that is absent in either scrape is reported as null rather than
    zero: vLLM omits speculative metrics entirely when SD is disabled.
    """

    def counter(name: str) -> Optional[int]:
        before_series = metric_series(before, name)
        after_series = metric_series(after, name)
        if not before_series and not after_series:
            return None
        delta = metric_value(after, name) - metric_value(before, name)
        # Counter resets are visible as a negative delta; do not hide them.
        return int(round(delta))

    spec = {
        "drafts": counter(SPEC_DRAFTS),
        "draft_tokens": counter(SPEC_DRAFT_TOKENS),
        "accepted_tokens": counter(SPEC_ACCEPTED_TOKENS),
    }
    per_pos: dict[str, int] = {}
    before_pos = {labels.get("position", "0"): value for labels, value in metric_series(before, SPEC_ACCEPTED_PER_POS)}
    after_pos = {labels.get("position", "0"): value for labels, value in metric_series(after, SPEC_ACCEPTED_PER_POS)}
    for position in sorted(set(before_pos) | set(after_pos), key=lambda x: int(x)):
        per_pos[position] = int(round(after_pos.get(position, 0.0) - before_pos.get(position, 0.0)))
    spec["accepted_per_position"] = per_pos
    if spec["drafts"] is not None and spec["drafts"] > 0 and spec["accepted_tokens"] is not None:
        spec["acceptance_length"] = 1.0 + spec["accepted_tokens"] / spec["drafts"]
    else:
        spec["acceptance_length"] = None
    if spec["draft_tokens"] not in (None, 0) and spec["accepted_tokens"] is not None:
        spec["acceptance_rate"] = spec["accepted_tokens"] / spec["draft_tokens"]
    else:
        spec["acceptance_rate"] = None
    return {
        "spec": spec,
        "active_after": active_requests(after),
        "gauges_after": {
            "gpu_cache_usage_perc": metric_value(after, GPU_CACHE_USAGE),
            "gpu_memory_usage_perc": metric_value(after, GPU_MEMORY_USAGE),
        },
    }


def percentile(values: list[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (q in [0, 1]); None for an empty sample."""

    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1))
    return ordered[index]


def summarize_latencies(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean_s": None, "p50_s": None, "p95_s": None, "min_s": None, "max_s": None}
    return {
        "n": len(values),
        "mean_s": sum(values) / len(values),
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
        "min_s": min(values),
        "max_s": max(values),
    }
