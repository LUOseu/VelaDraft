from __future__ import annotations

import copy
import json
from pathlib import Path

from vlm_dflash.serving.contracts import validate_run_record
from vlm_dflash.serving.profile import PROFILE_ID, PROFILE_VERSION, describe_profile, normalize_record


def _fixture():
    path = Path(__file__).parent / "fixtures" / "wire_profile_record.json"
    return json.loads(path.read_text())


def test_wire_profile_fixture_validates_and_normalizes():
    record = _fixture()
    validate_run_record(record)

    normalized = normalize_record(record)

    assert normalized["profile"] == {"id": PROFILE_ID, "version": PROFILE_VERSION}
    assert normalized["scope"]["request"]["request_id"] == "ar-c1-s1-nat-ocr-00"
    assert normalized["counters"]["request"]["emitted"] == 12
    assert normalized["counters"]["global"]["target_forward_calls"] is None
    assert normalized["counters"]["global"]["reason"]
    assert normalized["latency"]["tpot_s"] is None
    assert normalized["latency"]["tpot_unavailable_reason"]
    assert normalized["resource"]["additive"] is False
    assert normalized["resource"]["gpu_hours"] is None


def test_profile_never_attributes_global_spec_counters_to_requests():
    record = copy.deepcopy(_fixture())
    record["measurement_scope"]["spec_counter_attribution"] = "global_window_only"

    normalized = normalize_record(record)

    assert normalized["counters"]["request"] is None
    assert "no request tag" in normalized["counters"]["request_unavailable_reason"]


def test_profile_documents_scope_and_missing_semantics():
    profile = describe_profile()

    assert set(profile["scope"]) == {"request", "window", "global"}
    assert profile["normalization"]["unknown_counter"].startswith("null")
    assert "request_exposure" in profile["resource"]
