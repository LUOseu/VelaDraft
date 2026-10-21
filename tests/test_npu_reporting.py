"""C10 fail-closed aggregation tests for NPU smoke evidence."""

from __future__ import annotations

from vlm_dflash.npu_reporting import (
    aggregate_phase_results,
    empty_method_record,
    evaluate_method_records,
    update_method_record,
)

METHODS = ["dflash", "dflash2", "dflare", "dfly", "domino", "dspark", "jetspec"]


def _good_records():
    records = []
    for method in METHODS:
        record = empty_method_record(method)
        record.update(
            {
                "steps": 10,
                "nan_seen": False,
                "exit_code": 0,
                "skipped": False,
                "forward": True,
                "backward": True,
                "update": True,
                "resume": True,
                "export": True,
                "method_specific_grad": True,
            }
        )
        records.append(record)
    return records


def test_complete_records_pass():
    verdict = evaluate_method_records(_good_records(), METHODS)
    assert verdict["status"] == "PASSED"
    assert verdict["problems"] == []
    assert all(entry["status"] == "passed" for entry in verdict["per_method"].values())


def test_empty_records_fail():
    verdict = evaluate_method_records([], METHODS)
    assert verdict["status"] == "FAILED"
    assert any("no method records" in problem for problem in verdict["problems"])


def test_missing_method_fails():
    verdict = evaluate_method_records(_good_records()[:-1], METHODS)
    assert verdict["status"] == "FAILED"
    assert any("missing method records" in problem for problem in verdict["problems"])


def test_too_few_steps_fail():
    records = _good_records()
    records[0]["steps"] = 9
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("only 9 optimizer steps" in problem for problem in verdict["problems"])


def test_nonzero_child_exit_fails():
    records = _good_records()
    records[2]["exit_code"] = 4
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("child process exited 4" in problem for problem in verdict["problems"])


def test_any_skipped_record_fails():
    records = _good_records()
    records[3]["skipped"] = True
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("case was skipped" in problem for problem in verdict["problems"])


def test_all_skipped_records_fail():
    records = _good_records()
    for record in records:
        record["skipped"] = True
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("every method record is marked skipped" in problem for problem in verdict["problems"])


def test_nan_loss_fails_even_with_enough_steps():
    records = _good_records()
    records[4]["nan_seen"] = True
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("nan_seen is not explicitly false" in problem for problem in verdict["problems"])


def test_missing_nan_evidence_fails_closed():
    records = _good_records()
    records[5]["nan_seen"] = None
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("nan_seen is not explicitly false" in problem for problem in verdict["problems"])


def test_duplicate_method_records_fail():
    records = _good_records()
    records.append(dict(records[0]))
    verdict = evaluate_method_records(records, METHODS)
    assert verdict["status"] == "FAILED"
    assert any("duplicate method records" in problem for problem in verdict["problems"])


def test_updating_a_failed_method_with_success_keeps_the_failure_exit_code():
    records = {}
    update_method_record(records, "dflash", steps=10, exit_code=4)
    update_method_record(records, "dflash", steps=10, exit_code=0)
    assert records["dflash"]["exit_code"] == 4


def test_phase_aggregate_requires_every_requested_phase_to_pass():
    phases = [
        {"phase": "H0", "status": "NPU_SMOKE_PASSED", "npu_verified": True},
        {"phase": "H1", "status": "FAILED", "npu_verified": True},
        {"phase": "H2", "status": "NOT_RUN", "npu_verified": False},
    ]
    verdict = aggregate_phase_results(phases, requested=["H0", "H1"])
    assert verdict["status"] == "FAILED"
    assert any("H1: status 'FAILED'" in problem for problem in verdict["problems"])
    assert verdict["counts"]["not_run"] == 1


def test_phase_aggregate_passes_only_requested_passed_phases():
    phases = [
        {"phase": "H0", "status": "NPU_SMOKE_PASSED", "npu_verified": True},
        {"phase": "H1", "status": "NPU_SMOKE_PASSED", "npu_verified": True},
        {"phase": "H2", "status": "NOT_RUN", "npu_verified": False},
    ]
    verdict = aggregate_phase_results(phases, requested=["H0", "H1"])
    assert verdict["status"] == "NPU_SMOKE_PASSED"
    assert verdict["counts"]["not_run"] == 1
    assert verdict["phases"][2]["status"] == "NOT_RUN"


def test_phase_aggregate_requires_the_npu_verified_flag():
    phases = [{"phase": "H0", "status": "NPU_SMOKE_PASSED", "npu_verified": False}]
    verdict = aggregate_phase_results(phases, requested=["H0"])
    assert verdict["status"] == "FAILED"
    assert any("npu_verified=False" in problem for problem in verdict["problems"])
