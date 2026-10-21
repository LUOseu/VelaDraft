"""Strict aggregation for NPU-readiness smoke evidence.

A smoke run is only useful if it fails closed: zero records, a missing
registered method, too few optimizer steps, a nonzero child exit, a skipped
case, or a NaN loss must never turn into a green aggregate. The CPU suite and
the on-device smoke runner share these checks so their JSON verdicts mean the
same thing.
"""

from __future__ import annotations

from typing import Iterable, Mapping

METHOD_REQUIRED_FIELDS = (
    "forward",
    "backward",
    "update",
    "resume",
    "export",
)


def empty_method_record(method: str) -> dict:
    """A record with every optional evidence flag explicitly false."""
    return {
        "method": str(method),
        "steps": 0,
        "nan_seen": None,
        "exit_code": 0,
        "skipped": False,
        "forward": False,
        "backward": False,
        "update": False,
        "resume": False,
        "export": False,
        "method_specific_grad": False,
    }


def update_method_record(records: dict[str, dict], method: str, **fields) -> dict:
    """Accumulate evidence for one method without losing earlier test results."""
    record = records.setdefault(str(method), empty_method_record(method))
    for key, value in fields.items():
        if key == "steps":
            record["steps"] = max(int(record.get("steps", 0)), int(value))
        elif key == "nan_seen":
            record["nan_seen"] = bool(value) if record.get("nan_seen") is None else bool(record["nan_seen"] or value)
        elif key == "exit_code":
            previous = int(record.get("exit_code", 0) or 0)
            current = int(value)
            record["exit_code"] = current if previous == 0 else previous
        elif key in ("skipped", "forward", "backward", "update", "resume", "export", "method_specific_grad"):
            record[key] = bool(record.get(key, False) or value)
        else:
            record[key] = value
    return record


def evaluate_method_records(
    records: Iterable[Mapping],
    expected_methods: Iterable[str],
    *,
    min_steps: int = 10,
    require_method_specific_grad: bool = True,
) -> dict:
    """Return a fail-closed verdict for per-method training evidence.

    ``records`` may be the list written by a smoke subprocess or the values of
    the in-process evidence collector. Exactly one record per expected method
    is required, so a misnamed or duplicated case cannot satisfy coverage.
    """
    expected = [str(method) for method in expected_methods]
    expected_set = set(expected)
    problems: list[str] = []
    by_method: dict[str, dict] = {}
    duplicates: list[str] = []
    skipped_records = 0
    record_count = 0

    for raw in records:
        record_count += 1
        record = dict(raw)
        method = str(record.get("method", "")).strip()
        if not method:
            problems.append(f"record #{record_count} has no method name")
            continue
        if method in by_method:
            duplicates.append(method)
            continue
        by_method[method] = record
        if bool(record.get("skipped", False)):
            skipped_records += 1

    if record_count == 0:
        problems.append("no method records were produced")
    if duplicates:
        problems.append(f"duplicate method records: {sorted(set(duplicates))}")
    if skipped_records and skipped_records == record_count:
        problems.append("every method record is marked skipped")

    missing = sorted(expected_set - set(by_method))
    unexpected = sorted(set(by_method) - expected_set)
    if missing:
        problems.append(f"missing method records: {missing}")
    if unexpected:
        problems.append(f"records for unregistered methods: {unexpected}")

    required_checks = list(METHOD_REQUIRED_FIELDS)
    if require_method_specific_grad:
        required_checks.append("method_specific_grad")

    per_method: dict[str, dict] = {}
    for method in expected:
        record = by_method.get(method)
        if record is None:
            per_method[method] = {"status": "missing"}
            continue
        method_problems: list[str] = []
        if bool(record.get("skipped", False)):
            method_problems.append("case was skipped")
        if int(record.get("exit_code", 0) or 0) != 0:
            method_problems.append(f"child process exited {int(record.get('exit_code', 0))}")
        steps = int(record.get("steps", 0) or 0)
        if steps < int(min_steps):
            method_problems.append(f"only {steps} optimizer steps recorded (need >= {min_steps})")
        if record.get("nan_seen") is not False:
            method_problems.append("nan_seen is not explicitly false (missing evidence or NaN loss)")
        for check in required_checks:
            if not bool(record.get(check, False)):
                method_problems.append(f"missing passing check: {check}")
        for problem in method_problems:
            problems.append(f"{method}: {problem}")
        per_method[method] = {
            "status": "passed" if not method_problems else "failed",
            "steps": steps,
            "nan_seen": record.get("nan_seen"),
            "checks": {check: bool(record.get(check, False)) for check in required_checks},
            "problems": method_problems,
            "record": record,
        }

    return {
        "status": "PASSED" if not problems else "FAILED",
        "expected_methods": expected,
        "record_count": record_count,
        "per_method": per_method,
        "problems": problems,
    }


def aggregate_phase_results(phases: Iterable[Mapping], *, requested: Iterable[str]) -> dict:
    """Summarize phase records for the NPU smoke runner.

    Passing requested phases is necessary but never enough to claim H3/H4:
    phases marked ``NOT_RUN``/``NPU_UNVERIFIED`` remain recorded as such in the
    returned payload. The top-level status is ``NPU_SMOKE_PASSED`` only when
    every requested phase passed.
    """
    requested = [str(name) for name in requested]
    records = [dict(phase) for phase in phases]
    by_name = {str(record.get("phase", "")): record for record in records}
    problems: list[str] = []
    for name in requested:
        record = by_name.get(name)
        if record is None:
            problems.append(f"requested phase {name} produced no record")
            continue
        if record.get("status") != "NPU_SMOKE_PASSED" or not bool(record.get("npu_verified", False)):
            problems.append(
                f"{name}: status {record.get('status')!r}, npu_verified={bool(record.get('npu_verified', False))}"
            )
    if not requested:
        problems.append("no phases were requested")

    counts = {
        "planned": len(records),
        "executed": sum(1 for record in records if record.get("status") not in ("NOT_RUN",)),
        "passed": sum(1 for record in records if record.get("status") == "NPU_SMOKE_PASSED"),
        "not_run": sum(1 for record in records if record.get("status") == "NOT_RUN"),
        "unverified": sum(1 for record in records if record.get("status") == "NPU_UNVERIFIED"),
        "failed": sum(1 for record in records if record.get("status") == "FAILED"),
    }
    if any(not bool(record.get("npu_verified", False)) for record in records):
        # This only affects the summary note; per-phase statuses are the
        # authoritative evidence and are never upgraded by this aggregate.
        counts["contains_unverified_or_not_run"] = True
    return {
        "status": "NPU_SMOKE_PASSED" if not problems else "FAILED",
        "requested_phases": requested,
        "counts": counts,
        "problems": problems,
        "phases": records,
    }


__all__ = [
    "METHOD_REQUIRED_FIELDS",
    "aggregate_phase_results",
    "empty_method_record",
    "evaluate_method_records",
    "update_method_record",
]
