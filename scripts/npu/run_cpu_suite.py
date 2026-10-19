#!/usr/bin/env python3
"""Run the NPU-readiness CPU suite and emit a machine-readable verdict.

This is deliberately stricter than a plain pytest invocation: a zero-test run,
any skipped case, a failure, or a missing per-method resume/export case makes
the verdict fail instead of reporting a vacuous PASS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash.npu_reporting import evaluate_method_records
DEFAULT_TARGETS = [
    "tests/test_npu_runtime.py",
    "tests/test_npu_attention.py",
    "tests/test_npu_cpu_suite.py",
    "tests/test_npu_reporting.py",
    "tests/test_npu_presets.py",
    "tests/test_npu_support_matrix.py",
    "tests/test_target_device_map.py",
    "tests/test_qwen35_moe_target.py",
]
REQUIRED_CASES = (
    "run_ten_real_optimizer_updates",
    "specific_modules_receive_nonzero_gradients",
    "six_plus_exact_resume_plus_six",
    "exports_and_reloads",
)


def _methods() -> list[str]:
    from vlm_dflash.algorithms import available_algorithms

    return available_algorithms()


def _parse_junit(path: Path) -> tuple[dict, list[dict]]:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag == "testsuites" and len(root):
        root = root[0]
    summary = {
        "tests": int(root.attrib.get("tests", 0)),
        "failures": int(root.attrib.get("failures", 0)),
        "errors": int(root.attrib.get("errors", 0)),
        "skipped": int(root.attrib.get("skipped", 0)),
        "time_seconds": float(root.attrib.get("time", 0.0)),
    }
    cases = []
    for case in root.iter("testcase"):
        status = "passed"
        if case.find("failure") is not None:
            status = "failed"
        elif case.find("error") is not None:
            status = "error"
        elif case.find("skipped") is not None:
            status = "skipped"
        cases.append(
            {
                "name": case.attrib.get("name", ""),
                "classname": case.attrib.get("classname", ""),
                "status": status,
            }
        )
    return summary, cases


def _coverage(methods: list[str], cases: list[dict]) -> dict:
    by_method = {}
    for method in methods:
        named = [case for case in cases if method in case["name"]]
        by_method[method] = {
            "cases": len(named),
            "requires": {
                required: any(required in case["name"] and case["status"] == "passed" for case in named)
                for required in REQUIRED_CASES
            },
        }
    return by_method


def _verdict(
    summary: dict,
    cases: list[dict],
    methods: list[str],
    evidence: dict,
    *,
    min_steps: int,
) -> tuple[str, list[str], dict]:
    problems = []
    if summary["tests"] == 0:
        problems.append("pytest collected zero tests")
    if summary["failures"] or summary["errors"]:
        problems.append("pytest reported failures/errors")
    if summary["skipped"]:
        problems.append(f"pytest skipped {summary['skipped']} case(s); the NPU CPU suite must be fully executed")
    coverage = _coverage(methods, cases)
    for method, entry in coverage.items():
        missing = [name for name, ok in entry["requires"].items() if not ok]
        if missing:
            problems.append(f"{method}: missing passing cases {missing}")
    evidence_verdict = evaluate_method_records(
        evidence.get("records", []), methods, min_steps=min_steps
    )
    if evidence_verdict["status"] != "PASSED":
        problems.extend(f"evidence: {problem}" for problem in evidence_verdict["problems"])
    return ("CPU_VERIFIED" if not problems else "FAILED"), problems, evidence_verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=str(ROOT / "runs" / "npu" / "cpu_suite.json"))
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--pytest-arg", action="append", default=[], help="extra pytest argument")
    args = parser.parse_args()

    methods = _methods()
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        evidence_path = Path(tmp) / "method_evidence.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *DEFAULT_TARGETS,
            "-q",
            f"--junitxml={junit}",
            *args.pytest_arg,
        ]
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            env={**os.environ, "VLM_DFLASH_NPU_EVIDENCE_JSON": str(evidence_path)},
        )
        if junit.exists():
            summary, cases = _parse_junit(junit)
        else:
            summary, cases = {"tests": 0, "failures": 0, "errors": 1, "skipped": 0, "time_seconds": 0.0}, []
        if evidence_path.exists():
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                evidence = {"records": [], "parse_error": str(error)}
        else:
            evidence = {"records": [], "missing": True}
    status, problems, evidence_verdict = _verdict(
        summary, cases, methods, evidence, min_steps=args.min_steps
    )
    if completed.returncode != 0 and status == "CPU_VERIFIED":
        status = "FAILED"
        problems.append(f"pytest exited {completed.returncode}")
    payload = {
        "format": "vlm-dflash-npu-cpu-suite-v1",
        "status": status,
        "evidence_level": "CPU_VERIFIED" if status == "CPU_VERIFIED" else "NOT_VERIFIED",
        "command": cmd,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_seconds": round(time.time() - started, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": __import__("torch").__version__,
            "torch_npu": __import__("importlib.metadata", fromlist=["version"]).version("torch_npu")
            if _package_present("torch_npu")
            else None,
        },
        "pytest": {**summary, "exit_code": completed.returncode},
        "methods_covered": methods,
        "per_method": _coverage(methods, cases),
        "method_evidence": evidence_verdict,
        "min_steps": args.min_steps,
        "problems": problems,
        "pytest_stdout_tail": completed.stdout[-4000:],
        "pytest_stderr_tail": completed.stderr[-4000:],
        "note": "CPU evidence only; this does not prove any Ascend operator or device path.",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "tests": summary["tests"], "problems": problems, "json": str(out_path)}))
    return 0 if status == "CPU_VERIFIED" else 1


def _package_present(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    raise SystemExit(main())

