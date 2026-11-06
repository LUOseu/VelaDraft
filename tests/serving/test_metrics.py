from __future__ import annotations

from vlm_dflash.serving.metrics import (
    active_requests,
    diff_metrics,
    parse_prometheus,
    percentile,
    spec_counts,
    summarize_latencies,
)

SAMPLE = """
# HELP vllm:spec_decode_num_drafts_total drafts
vllm:spec_decode_num_drafts_total{engine="0"} 10
vllm:spec_decode_num_draft_tokens_total{engine="0"} 40
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 15
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 5
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 2
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_waiting{engine="0"} 1
"""


def test_parse_and_spec_counts():
    metrics = parse_prometheus(SAMPLE)
    counts = spec_counts(metrics)
    assert counts["drafts"] == 10
    assert counts["draft_tokens"] == 40
    assert counts["accepted_tokens"] == 15
    assert counts["acceptance_length"] == 2.5
    assert counts["acceptance_rate"] == 0.375
    assert counts["per_position_acceptance_rate"]["0"] == 0.8
    assert active_requests(metrics)["total"] == 3


def test_diff_metrics():
    before = parse_prometheus("vllm:spec_decode_num_drafts_total 10\nvllm:spec_decode_num_draft_tokens_total 40\nvllm:spec_decode_num_accepted_tokens_total 15\n")
    after = parse_prometheus("vllm:spec_decode_num_drafts_total 13\nvllm:spec_decode_num_draft_tokens_total 52\nvllm:spec_decode_num_accepted_tokens_total 21\n")
    diff = diff_metrics(before, after)
    assert diff["spec"]["drafts"] == 3
    assert diff["spec"]["draft_tokens"] == 12
    assert diff["spec"]["accepted_tokens"] == 6
    assert diff["spec"]["acceptance_length"] == 3.0


def test_diff_absent_spec_is_null():
    diff = diff_metrics(parse_prometheus(""), parse_prometheus(""))
    assert diff["spec"]["drafts"] is None
    assert diff["spec"]["acceptance_length"] is None


def test_percentiles():
    assert percentile([1, 2, 3, 4], 0.5) == 2
    assert percentile([1, 2, 3, 4], 0.95) == 4
    summary = summarize_latencies([1.0, 2.0, 3.0])
    assert summary["p50_s"] == 2.0
    assert summary["max_s"] == 3.0
