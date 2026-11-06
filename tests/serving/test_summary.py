from __future__ import annotations

import copy

import pytest

from vlm_dflash.serving.summary import build_summary


def _target(model="/models/qwen"):
    return {
        "model": model,
        "revision": "index",
        "weight_content_sha256": "weights",
        "identity_complete": True,
        "config_sha256": "config",
        "architecture": "Qwen2_5_VLForConditionalGeneration",
        "precision": "bfloat16",
        "tokenizer": {"sha256": "tok"},
        "processor": {"sha256": "proc"},
        "template": {"sha256": "tmpl", "effective_source": "chat_template.json"},
        "missing_reasons": [],
    }


def _backend():
    return {
        "backend": "vllm",
        "version": "0.22.1",
        "code_commit": "abc",
        "attention_backend": "FLASH_ATTN",
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "labels_source": "engine_object",
        "applied_evidence": True,
        "device_topology": {
            "gpu_count": 1,
            "devices": [{"uuid": "GPU-x"}],
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        "missing_reasons": [],
    }


def _record(mode, request_id, *, budget=128, prompt="p", image="i"):
    return {
        "request_id": f"{mode}-{request_id}",
        "protocol_hash": f"protocol-{mode}",
        "category": "ocr",
        "prompt_sha256": prompt,
        "image_sha256": image,
        "finish_reason": "stop",
        "terminal_status": "completed",
        "error": None,
        "cancelled": False,
        "token_ids": [1, 2],
        "text": "ok",
        "measurement_scope": {
            "concurrency": 1,
            "session_index": 1,
            "fixture_category": "ocr",
            "max_new_tokens": budget,
            "prefix_caching": False,
            "timing_boundary": "client_request_start_to_last_nonempty_sse_content_chunk",
        },
        "request_control": {"source": "actual_request_body", "max_tokens": budget, "ignore_eos": False},
        "target_identity": _target(),
        "backend_identity": _backend(),
        "latency": {"ttft_s": 0.1, "e2e_s": 1.0, "tpot_s": None, "tpot_unavailable_reason": "no per-token timestamps"},
        "resource_usage": {"gpu_hours": None, "gpu_hours_reason": "exposure", "gpu_hours_scope": "request_exposure", "additive": False},
    }


def _window(mode):
    return {
        "measurement_scope": {"concurrency": 1, "session_index": 1},
        "num_requests": 1,
        "num_errors": 0,
        "latency": {"total_s": 10.0},
    }


def _doc(mode):
    return {
        "meta": {
            "manifest_sha256": "manifest",
            "mode": mode,
            "stream": True,
            "warmup": 4,
            "max_new_tokens": 128,
            "concurrency": "1",
            "sessions": 1,
            "speculative_config": None if mode == "ar" else {"method": "ngram"},
        },
        "windows": [_window(mode)],
        "records": [_record(mode, "c1-s1-nat-ocr-00")],
        "summary": {
            "boundary_results": [{"ok": True}] * 8,
            "nonstream_probes": [{"ok": True}],
            "cancel_probes": [
                {
                    "concurrency": 1,
                    "session": 1,
                    "cancelled": True,
                    "observed_in_flight": True,
                    "cancel_reclaimed_s": 0.1,
                    "error": None,
                    "concurrent_with_healthy_request": False,
                    "internal_abort_evidence": False,
                }
            ],
            "pair_checks": [{"input_submission_ok": True, "output_attribution_verified": False}],
        },
    }


def _correctness():
    return {
        "completion_levels": {
            "AR_OFFLINE_PASS": True,
            "SD_EXECUTED": True,
            "SD_CORRECTNESS_PASS": False,
        }
    }


def test_valid_common_protocol_is_pairable_but_failed_correctness_blocks_speedup():
    summary = build_summary(_correctness(), _doc("ar"), _doc("ngram"))

    assert summary["protocol"]["ok"] is True
    assert summary["pairing"]["pairs"] == 1
    assert summary["pairing"]["claim_eligible"] is True
    assert summary["speedup"]["claim_eligible"] is False
    assert "sd_correctness_pass_not_verified" in summary["speedup"]["reasons"]
    assert summary["speedup"]["by_concurrency"]["1"]["claim_eligible"] is False
    assert summary["speedup"]["by_concurrency"]["1"]["ngram_over_ar"] == 1.0
    assert summary["completion_levels"]["SERVING_LIFECYCLE_PASS"] is False
    assert summary["lifecycle"]["status"] == "pending_internal_evidence"


def test_target_mismatch_is_rejected():
    sd = _doc("ngram")
    sd["records"][0]["target_identity"] = _target("/models/other")

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "target_model_mismatch" in summary["protocol"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


def test_manifest_mismatch_is_rejected():
    sd = _doc("ngram")
    sd["meta"]["manifest_sha256"] = "other"

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "fixture_manifest_sha256_mismatch" in summary["protocol"]["reasons"]


def test_same_path_changed_weight_identity_is_rejected():
    sd = _doc("ngram")
    sd["records"][0]["target_identity"]["weight_content_sha256"] = "changed"

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "target_weight_content_sha256_mismatch" in summary["protocol"]["reasons"]


def test_unverified_backend_labels_are_rejected():
    sd = _doc("ngram")
    sd["records"][0]["backend_identity"]["applied_evidence"] = False

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "backend_labels_not_verified_from_engine_or_runtime_evidence" in summary["protocol"]["reasons"]


def test_inconsistent_identity_across_records_is_rejected():
    sd = _doc("ngram")
    extra = copy.deepcopy(sd["records"][0])
    extra["request_id"] = "ngram-c1-s1-other"
    extra["target_identity"]["weight_content_sha256"] = "changed"
    sd["records"].append(extra)

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "sd_target_identity_inconsistent_across_records" in summary["protocol"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


def test_inconsistent_protocol_hash_is_rejected():
    sd = _doc("ngram")
    extra = copy.deepcopy(sd["records"][0])
    extra["request_id"] = "ngram-c1-s1-other"
    extra["protocol_hash"] = "different"
    sd["records"].append(extra)

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "sd_protocol_hash_inconsistent_across_records" in summary["protocol"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


def test_inconsistent_measurement_scope_across_records_is_rejected():
    sd = _doc("ngram")
    extra = copy.deepcopy(sd["records"][0])
    extra["request_id"] = "ngram-c1-s1-other"
    extra["measurement_scope"]["timing_boundary"] = "other"
    sd["records"].append(extra)

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert "sd_measurement_scope_inconsistent_across_records" in summary["protocol"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


def _set_path(obj, path, value):
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "path,value,reason",
    [
        (("records", 0, "target_identity", "model"), "/models/other", "target_model_mismatch"),
        (("records", 0, "target_identity", "precision"), "float16", "target_precision_mismatch"),
        (("records", 0, "target_identity", "config_sha256"), "other", "target_config_sha256_mismatch"),
        (("records", 0, "target_identity", "architecture"), "Other", "target_architecture_mismatch"),
        (("records", 0, "target_identity", "tokenizer", "sha256"), "other", "target_tokenizer_sha256_mismatch"),
        (("records", 0, "target_identity", "processor", "sha256"), "other", "target_processor_sha256_mismatch"),
        (("records", 0, "target_identity", "template", "sha256"), "other", "target_template_sha256_mismatch"),
        (("records", 0, "target_identity", "template", "effective_source"), "other", "target_template_effective_source_mismatch"),
        (("records", 0, "target_identity", "revision"), "other", "target_revision_mismatch"),
        (("records", 0, "target_identity", "weight_content_sha256"), "other", "target_weight_content_sha256_mismatch"),
        (("records", 0, "target_identity", "identity_complete"), False, "target_content_identity_incomplete"),
        (("records", 0, "backend_identity", "backend"), "other", "backend_backend_mismatch"),
        (("records", 0, "backend_identity", "version"), "other", "backend_version_mismatch"),
        (("records", 0, "backend_identity", "code_commit"), "other", "backend_code_commit_mismatch"),
        (("records", 0, "backend_identity", "attention_backend"), "other", "backend_attention_backend_mismatch"),
        (("records", 0, "backend_identity", "cudagraph_mode"), "other", "backend_cudagraph_mode_mismatch"),
        (("records", 0, "backend_identity", "device_topology", "tensor_parallel_size"), 2, "backend_tensor_parallel_size_mismatch"),
        (("records", 0, "backend_identity", "device_topology", "devices", 0, "uuid"), "GPU-y", "backend_device_uuids_mismatch"),
        (("records", 0, "backend_identity", "applied_evidence"), False, "backend_labels_not_verified_from_engine_or_runtime_evidence"),
        (("records", 0, "backend_identity", "labels_source"), "cli_declared_unverified", "sd_backend_labels_source_untrusted"),
        (("meta", "manifest_sha256"), "other", "fixture_manifest_sha256_mismatch"),
        (("meta", "stream"), False, "measurement_stream_mismatch"),
        (("meta", "warmup"), 5, "measurement_warmup_mismatch"),
        (("meta", "max_new_tokens"), 64, "measurement_max_new_tokens_mismatch"),
        (("records", 0, "measurement_scope", "prefix_caching"), True, "measurement_scope_prefix_caching_mismatch"),
        (("records", 0, "measurement_scope", "timing_boundary"), "other", "measurement_scope_timing_boundary_mismatch"),
    ],
)
def test_each_common_protocol_field_mutation_is_rejected(path, value, reason):
    sd = _doc("ngram")
    _set_path(sd, path, value)

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert reason in summary["protocol"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


@pytest.mark.parametrize("field", ["prompt_sha256", "image_sha256", "category"])
def test_fixture_identity_mutation_breaks_pairing(field):
    sd = _doc("ngram")
    sd["records"][0][field] = "other"

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert any(field in reason for reason in summary["pairing"]["reasons"])
    assert summary["speedup"]["claim_eligible"] is False


@pytest.mark.parametrize("field,value", [("max_tokens", 64), ("ignore_eos", True)])
def test_request_budget_mutation_breaks_pairing(field, value):
    sd = _doc("ngram")
    sd["records"][0]["request_control"][field] = value

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert summary["speedup"]["claim_eligible"] is False
    assert any(field in reason for reason in summary["pairing"]["reasons"])


def test_different_request_set_is_rejected():
    sd = _doc("ngram")
    sd["records"][0]["request_id"] = "ngram-c1-s1-other"

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert summary["pairing"]["reasons"]
    assert summary["speedup"]["claim_eligible"] is False


def test_missing_session_window_is_rejected():
    sd = _doc("ngram")
    sd["windows"] = []
    sd["records"] = []

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert any("missing window" in reason for reason in summary["pairing"]["reasons"])


def test_duplicate_window_is_rejected():
    sd = _doc("ngram")
    sd["windows"].append(copy.deepcopy(sd["windows"][0]))

    summary = build_summary(_correctness(), _doc("ar"), sd)

    assert any("duplicate window" in reason for reason in summary["pairing"]["reasons"])


@pytest.mark.parametrize("level", ["AR_OFFLINE_PASS", "SD_EXECUTED", "SD_CORRECTNESS_PASS"])
@pytest.mark.parametrize("value", [False, None, "false", "true", 1])
def test_speedup_requires_explicit_true_correctness_levels(level, value):
    correctness = _correctness()
    correctness["completion_levels"].update(AR_OFFLINE_PASS=True, SD_EXECUTED=True, SD_CORRECTNESS_PASS=True)
    correctness["completion_levels"][level] = value

    summary = build_summary(correctness, _doc("ar"), _doc("ngram"))

    assert summary["pairing"]["claim_eligible"] is True
    assert summary["speedup"]["claim_eligible"] is False
    assert f"{level.lower()}_not_verified" in summary["speedup"]["reasons"]
    assert summary["completion_levels"][level] is False


def test_verified_correctness_and_protocol_allow_speedup_claim():
    correctness = _correctness()
    correctness["completion_levels"]["SD_CORRECTNESS_PASS"] = True

    summary = build_summary(correctness, _doc("ar"), _doc("ngram"))

    assert summary["speedup"]["claim_eligible"] is True
    assert summary["speedup"]["reasons"] == []


@pytest.mark.parametrize("input_ok", [True, False])
def test_top_level_lifecycle_respects_full_per_mode_checks(input_ok):
    ar, sd = _doc("ar"), _doc("ngram")
    for doc in (ar, sd):
        doc["summary"]["cancel_probes"][0].update(
            concurrent_with_healthy_request=True, internal_abort_evidence=True,
        )
        doc["summary"]["pair_checks"][0].update(
            concurrency=1, session=1, input_submission_ok=input_ok,
            output_attribution_verified=True,
        )

    summary = build_summary(_correctness(), ar, sd)

    assert summary["lifecycle"]["ar"]["full_lifecycle_ok"] is input_ok
    assert summary["completion_levels"]["SERVING_LIFECYCLE_PASS"] is input_ok
    if not input_ok:
        assert "ar_same_text_different_image_input_submission_unverified" in summary["lifecycle"]["reasons"]
