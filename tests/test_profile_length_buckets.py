"""Length-bucket helpers of the online-extraction profiler (CPU only).

The per-row profiler mode itself needs the real target; its pure bookkeeping
(edge parsing, bucket assignment, per-bucket medians) is pinned here so a run
on the GPU host starts from reviewed arithmetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_profiler():
    spec = importlib.util.spec_from_file_location(
        "profile_online_extraction", ROOT / "scripts" / "profile_online_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler = _load_profiler()


def test_parse_length_edges_sorts_and_dedupes():
    assert profiler.parse_length_edges("2048, 512,1024,512") == [512, 1024, 2048]
    assert profiler.parse_length_edges("768") == [768]


@pytest.mark.parametrize("raw", ["", "   ", "0,512", "-4", "abc", "512,oops"])
def test_parse_length_edges_rejects_bad_input(raw):
    with pytest.raises(ValueError):
        profiler.parse_length_edges(raw)


def test_assign_length_bucket_edges_are_upper_bounds():
    edges = [512, 1024]
    assert profiler.assign_length_bucket(0, edges) == 0
    assert profiler.assign_length_bucket(511, edges) == 0
    assert profiler.assign_length_bucket(512, edges) == 1  # exactly on the edge -> upper bucket
    assert profiler.assign_length_bucket(1023, edges) == 1
    assert profiler.assign_length_bucket(1024, edges) == 2
    assert profiler.assign_length_bucket(99999, edges) == 2


def test_summarise_length_buckets_medians_and_empty_buckets_skipped():
    rows = [
        {"length": 100, "image_tokens": 64, "render_ms": 10.0, "forward_ms": 100.0},
        {"length": 200, "image_tokens": 256, "render_ms": 20.0, "forward_ms": 200.0},
        {"length": 900, "image_tokens": 512, "render_ms": 30.0, "forward_ms": 300.0},
    ]
    summary = profiler.summarise_length_buckets(rows, [512, 1024, 2048])
    # the 900-token row is in bucket 1 ([512, 1024)); buckets 2 and 3 are
    # empty and skipped
    assert [bucket["bucket"] for bucket in summary] == [0, 1]

    first = summary[0]
    assert first["rows"] == 2
    assert first["length_min"] == 0
    assert first["length_max"] == 512
    assert first["median_length"] == 150
    assert first["median_image_tokens"] == 160  # median of [64, 256]
    assert first["render_ms_per_row"] == 15.0
    assert first["forward_ms_per_row"] == 150.0
    # ms per token: median forward (ms) / median token length
    assert first["forward_ms_per_token"] == pytest.approx(150.0 / 150.0)

    second = summary[1]
    assert second["rows"] == 1
    assert second["length_min"] == 512
    assert second["length_max"] == 1024
    assert second["forward_ms_per_token"] == pytest.approx(300.0 / 900.0, abs=1e-4)


def test_summarise_length_buckets_skips_nothing_when_only_top_bucket_used():
    rows = [{"length": 4096, "image_tokens": 1024, "render_ms": 5.0, "forward_ms": 50.0}]
    summary = profiler.summarise_length_buckets(rows, [512, 1024])
    assert [bucket["bucket"] for bucket in summary] == [2]
    assert summary[0]["length_min"] == 1024
    assert summary[0]["length_max"] is None
