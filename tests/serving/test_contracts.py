from __future__ import annotations

import pytest

from vlm_dflash.serving.contracts import (
    ContractError,
    Commit,
    Proposal,
    SessionState,
    commit_proposal,
    protocol_hash,
    validate_capabilities,
    validate_run_record,
)
from vlm_dflash.serving.records import (
    backend_identity,
    make_run_record,
    model_identity,
    ngram_draft_identity,
    protocol_payload,
    resource_usage,
)


def _target():
    return {
        "model": "/models/qwen",
        "revision": "abc",
        "tokenizer": {"sha256": "t"},
        "processor": {"sha256": "p"},
        "template": {"sha256": "c"},
        "precision": "bfloat16",
        "missing_reasons": [],
    }


def _backend():
    return backend_identity(
        vllm_version="0.22.1",
        code_commit="0" * 40,
        devices=[{"index": 0, "uuid": "GPU-x"}],
    )


def _record(*, draft=True, **overrides):
    target = _target()
    draft_identity = ngram_draft_identity(
        "0.22.1",
        target,
        {"method": "ngram", "num_speculative_tokens": 4},
    ) if draft else None
    kwargs = dict(
        run_id="run",
        request_id="req",
        group_id="group",
        target_identity=target,
        draft_identity=draft_identity,
        backend_identity=_backend(),
        protocol_payload=protocol_payload(
            model_path="/models/qwen",
            mode="ngram" if draft else "ar",
            spec_config=None,
            sampling={"temperature": 0.0},
            pixel_bounds={"min_pixels": 1, "max_pixels": 2},
            max_model_len=8192,
            prefix_caching=False,
            backend_version="0.22.1",
            timing_boundary="test",
        ),
        correctness_status="not_checked",
        measurement_scope={"surface": "test"},
        raw_counts={
            "emitted": 3,
            "proposed": 3 if draft else 0,
            "accepted": 1 if draft else 0,
            "verify_rounds": 1 if draft else 0,
            "target_forward_calls": None,
            "draft_forward_calls": 0,
            "unavailable": {"target_forward_calls": "not exposed"},
        },
        latency={"total_s": 1.0},
        resource_usage_value=resource_usage(
            wall_s=1.0,
            gpu_count=1,
            devices=[{"index": 0, "uuid": "GPU-x"}],
            peak_memory_mib={"gpu0": 1000},
        ),
        missing_reasons={"raw_counts.target_forward_calls": "not exposed"},
    )
    if not draft:
        kwargs["missing_reasons"]["draft_identity"] = "AR baseline"
    kwargs.update(overrides)
    return make_run_record(**kwargs)


def test_valid_record_round_trips():
    record = _record()
    assert validate_run_record(record)["schema_version"] == "veladraft-platform-v1"
    assert record["include_in_paper"] is False
    assert record["protocol_hash"]


def test_null_draft_needs_reason():
    record = _record(draft=False)
    record["missing_reasons"] = {}
    with pytest.raises(ContractError, match="draft_identity"):
        validate_run_record(record)


def test_null_counter_needs_reason():
    record = _record()
    record["raw_counts"]["proposed"] = None
    record["raw_counts"]["unavailable"] = {}
    record["missing_reasons"] = {"raw_counts.target_forward_calls": "not exposed"}
    with pytest.raises(ContractError, match="raw_counts.proposed"):
        validate_run_record(record)


def test_latency_field_must_be_seconds():
    record = _record()
    record["latency"]["tokens_per_s"] = 12.0
    with pytest.raises(ContractError, match="rate"):
        validate_run_record(record)


def test_protocol_hash_is_order_independent():
    left = {"a": 1, "b": [2, 3]}
    right = {"b": [2, 3], "a": 1}
    assert protocol_hash(left) == protocol_hash(right)


def test_session_commit_and_stale_proposal():
    session = SessionState(session_id="s1", target_identity=_target())
    proposal = Proposal(
        request_id="s1",
        based_on_prefix_version=0,
        token_ids=[1, 2, 3],
        source="ngram",
    )
    commit_proposal(
        session,
        proposal,
        Commit(session_id="s1", prefix_version=1, token_ids=[1, 2], accepted=2, rejected=1),
    )
    assert session.prefix_version == 1
    assert session.committed_token_ids == [1, 2]
    with pytest.raises(ContractError, match="stale proposal"):
        commit_proposal(
            session,
            proposal,
            Commit(session_id="s1", prefix_version=2, token_ids=[4], accepted=1),
        )


def test_cross_session_commit_rejected():
    session = SessionState(session_id="s1", target_identity=_target())
    proposal = Proposal(request_id="s1", based_on_prefix_version=0, token_ids=[1], source="ngram")
    with pytest.raises(ContractError, match="cross-session"):
        commit_proposal(
            session,
            proposal,
            Commit(session_id="s2", prefix_version=1, token_ids=[1], accepted=1),
        )


def _capabilities():
    item = {
        "status": "supported",
        "maturity": "cpu_tested",
        "version": "0.22.1",
        "evidence": [{"path": "x"}],
        "limits": [],
    }
    return {
        "schema_version": "veladraft-platform-v1",
        "project_scope": "platform_extension",
        "include_in_paper": False,
        "generated_at": "2026-10-17",
        "vllm_pin": "0.22.1",
        "environment": {},
        "capabilities": {
            key: dict(item)
            for key in (
                "ar_image",
                "ngram_image",
                "dflash_image",
                "stream",
                "cancel",
                "round_trace",
                "return_token_ids",
                "metrics",
            )
        },
    }


def test_capabilities_validation():
    doc = _capabilities()
    assert validate_capabilities(doc)["vllm_pin"] == "0.22.1"
    doc["capabilities"]["stream"]["status"] = "probably"
    with pytest.raises(ContractError):
        validate_capabilities(doc)
