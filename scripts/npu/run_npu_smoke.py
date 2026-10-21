#!/usr/bin/env python3
"""Staged Ascend NPU acceptance runner (H0-H4).

Every phase is an isolated subprocess with its own timeout, log and JSON. The
runner writes a manifest and a fail-closed aggregate; a missing NPU exits 2 with
``NPU_UNVERIFIED`` and never falls back to CPU. H3 is a deliberately long
200-step continuity run and is opt-in. H4 needs real extended-model and
multi-device resources, so it is always recorded as ``NOT_RUN`` here (with the
commands to execute later) instead of being faked from tiny synthetic rows.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash import runtime
from vlm_dflash.npu_reporting import aggregate_phase_results

OPERATOR_SMOKE = ROOT / "scripts" / "npu" / "operator_smoke.py"
SEVEN_METHOD_SMOKE = ROOT / "scripts" / "npu" / "seven_method_smoke.py"

DEFAULT_MANIFEST = {
    "format": "vlm-dflash-npu-smoke-manifest-v1",
    "phases": {
        "H0": {
            "enabled": True,
            "description": "environment, tensor allocation, matmul/backward, RNG, peak memory, SDPA",
        },
        "H1": {
            "enabled": True,
            "description": "GQA/BF16 SDPA, chunked mask, draft-module forward/backward parity, optional fused attention",
            "include_fusion": False,
        },
        "H2": {
            "enabled": True,
            "description": "seven registered methods: 10 synthetic-feature optimizer steps, exact resume, export/reload",
            "methods": "all",
            "steps": 10,
        },
        "H3": {
            "enabled": False,
            "description": "200-step DFlash continuity run with a midpoint checkpoint/resume and peak memory",
            "methods": "dflash",
            "steps": 200,
        },
        "H4": {
            "enabled": False,
            "description": "real Qwen3-VL/SmolVLM/Qwen3.6 targets, device-map routing and multi-device HCCL/DDP",
            "status": "NOT_RUN",
            "requires": ["real extended-model checkpoints", "real image-text data", "multiple NPUs for the HCCL checks"],
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_manifest(path: str | None) -> dict:
    if not path:
        return copy.deepcopy(DEFAULT_MANIFEST)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != DEFAULT_MANIFEST["format"]:
        raise SystemExit(
            f"run_npu_smoke: manifest format {payload.get('format')!r} is not "
            f"{DEFAULT_MANIFEST['format']!r}."
        )
    return _deep_merge(DEFAULT_MANIFEST, payload)


def _requested_phases(manifest: dict, cli_phases: str | None) -> list[str]:
    known = list(DEFAULT_MANIFEST["phases"])
    if cli_phases:
        requested = [item.strip().upper() for item in cli_phases.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise SystemExit(f"run_npu_smoke: unknown phases {unknown}; known phases are {known}.")
        for name in known:
            manifest["phases"][name]["enabled"] = name in requested
    return [name for name in known if bool(manifest["phases"][name].get("enabled", False))]


def _npu_smi() -> dict:
    executable = shutil.which("npu-smi")
    if executable is None:
        return {"available": False, "reason": "npu-smi not found on PATH"}
    try:
        completed = subprocess.run([executable, "info"], capture_output=True, text=True, timeout=20)
        return {
            "available": True,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-2000:],
        }
    except Exception as error:
        return {"available": True, "error": f"{type(error).__name__}: {error}"}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _phase_command(name: str, spec: dict, args, phase_json: Path, artifact_dir: Path) -> list[str]:
    python = sys.executable
    if name in ("H0", "H1"):
        command = [python, str(OPERATOR_SMOKE), "--device", args.device, "--out", str(phase_json)]
        if name == "H1":
            command += ["--h1"]
            if bool(spec.get("include_fusion", False)):
                command += ["--include-fusion"]
        return command
    if name in ("H2", "H3"):
        methods = str(spec.get("methods", "all"))
        steps = int(spec["steps"])
        return [
            python,
            str(SEVEN_METHOD_SMOKE),
            "--device",
            args.device,
            "--methods",
            methods,
            "--steps",
            str(steps),
            "--out",
            str(phase_json),
            "--artifact-dir",
            str(artifact_dir / name),
        ]
    raise ValueError(f"no executable command for phase {name}")


def _h4_commands(args) -> list[str]:
    target = args.target or "<local-target-directory>"
    dataset = args.dataset or "<local-jsonl>"
    images = args.images_dir or "<local-image-directory>"
    return [
        f"vlm-dflash train --preset npu-dflash-smoke --target {target} --dataset {dataset} --images-dir {images} --steps 10 --run-dir runs/npu/H4/dflash",
        f"vlm-dflash train --preset npu-dspark-smoke --target {target} --dataset {dataset} --images-dir {images} --steps 10 --run-dir runs/npu/H4/dspark",
        f"vlm-dflash train --preset npu-online-smoke --target {target} --dataset {dataset} --images-dir {images} --mode online --steps 10 --run-dir runs/npu/H4/online",
        f"vlm-dflash train --preset npu-split-device --target {target} --dataset {dataset} --images-dir {images} --mode online --steps 10 --run-dir runs/npu/H4/split-device",
        f"vlm-dflash train --preset npu-target-device-map --target {target} --dataset {dataset} --images-dir {images} --steps 10 --run-dir runs/npu/H4/target-device-map",
    ]


def _run_phase(name: str, spec: dict, args, work_dir: Path) -> dict:
    phase_json = work_dir / "phase_json" / f"{name}.json"
    log_path = work_dir / "logs" / f"{name}.log"
    artifact_dir = work_dir / "artifacts"
    phase_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _phase_command(name, spec, args, phase_json, artifact_dir)
    started = time.time()
    record = {
        "phase": name,
        "description": spec.get("description"),
        "status": "FAILED",
        "npu_verified": False,
        "json": str(phase_json),
        "log": str(log_path),
        "command": command,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=float(args.timeout),
        )
        log_path.write_text(
            "$ " + " ".join(command) + "\n\n[stdout]\n" + completed.stdout + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )
        record["returncode"] = completed.returncode
        record["seconds"] = round(time.time() - started, 3)
        if phase_json.exists():
            payload = json.loads(phase_json.read_text(encoding="utf-8"))
            record["payload_status"] = payload.get("status")
            record["evidence_level"] = payload.get("evidence_level")
            record["npu_verified"] = payload.get("status") == "NPU_SMOKE_PASSED"
            if payload.get("status") == "NPU_UNVERIFIED":
                record["status"] = "NPU_UNVERIFIED"
            elif completed.returncode == 0 and payload.get("status") == "NPU_SMOKE_PASSED":
                record["status"] = "NPU_SMOKE_PASSED"
            else:
                record["status"] = "FAILED"
                record["reason"] = payload.get("error") or f"phase JSON status {payload.get('status')!r}"
        else:
            record["status"] = "FAILED"
            record["reason"] = f"phase produced no JSON (exit {completed.returncode})"
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        log_path.write_text(
            "$ " + " ".join(command) + f"\n\n[timeout after {args.timeout}s]\n[stdout]\n{stdout}\n[stderr]\n{stderr}",
            encoding="utf-8",
        )
        record.update(
            {
                "status": "FAILED",
                "reason": f"timeout after {args.timeout}s",
                "seconds": round(time.time() - started, 3),
            }
        )
    except Exception as error:
        record.update(
            {
                "status": "FAILED",
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()[-4000:],
                "seconds": round(time.time() - started, 3),
            }
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--phases", default=None, help="comma-separated H0,H1,H2,H3,H4; default uses the manifest")
    parser.add_argument("--manifest", default=None, help="optional JSON manifest overlaying the built-in one")
    parser.add_argument("--steps", type=int, default=None, help="override H2 optimizer steps (default: manifest or 10)")
    parser.add_argument("--h3-steps", type=int, default=None, help="override H3 continuity steps (default: manifest or 200)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="per-phase timeout in seconds")
    parser.add_argument("--target", default=None, help="local target path used in H4 command templates")
    parser.add_argument("--dataset", default=None, help="local JSONL used in H4 command templates")
    parser.add_argument("--images-dir", default=None, help="local image root used in H4 command templates")
    parser.add_argument("--work-dir", default=str(ROOT / "runs" / "npu" / "smoke"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser().resolve() if args.out else work_dir / "run_npu_smoke.json"
    manifest = _load_manifest(args.manifest)
    for name, steps in (("H2", args.steps), ("H3", args.h3_steps)):
        if steps is not None:
            if steps < 10:
                parser.error(f"{name} requires at least 10 optimizer steps")
            manifest["phases"][name]["steps"] = steps
    requested = _requested_phases(manifest, args.phases)
    started = time.time()
    manifest["requested_phases"] = requested
    manifest["device"] = args.device
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = {
        "format": "vlm-dflash-npu-smoke-run-v1",
        "requested_device": args.device,
        "requested_phases": requested,
        "manifest": str(manifest_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": runtime.environment_report().get("torch"),
            "repo_commit": _git_commit(),
            "runtime": runtime.environment_report(),
            "npu_smi": _npu_smi(),
        },
        "phases": [],
    }

    try:
        device = runtime.ensure_runtime_device(args.device, context="run_npu_smoke.device")
    except Exception as error:
        phases = []
        for name in requested:
            phases.append(
                {
                    "phase": name,
                    "description": manifest["phases"][name].get("description"),
                    "status": "NOT_RUN",
                    "npu_verified": False,
                    "reason": "no usable NPU; the phase was not started",
                }
            )
        payload.update(
            {
                "status": "NPU_UNVERIFIED",
                "evidence_level": "NPU_UNVERIFIED",
                "error": f"{type(error).__name__}: {error}",
                "phases": phases,
                "counts": aggregate_phase_results(phases, requested=[])["counts"],
                "duration_seconds": round(time.time() - started, 3),
                "note": "No NPU is available; no phase ran and no CPU fallback was attempted.",
                "h4_commands_not_run": _h4_commands(args),
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "NPU_UNVERIFIED", "error": payload["error"], "json": str(out_path)}))
        return 2

    payload["resolved_device"] = str(device)
    payload["device_name"] = runtime.device_name(device)
    phases = []
    executable_requested = []
    for name in requested:
        spec = manifest["phases"][name]
        if name == "H4" or spec.get("status") == "NOT_RUN":
            phases.append(
                {
                    "phase": name,
                    "description": spec.get("description"),
                    "status": "NOT_RUN",
                    "npu_verified": False,
                    "reason": "requires real extended-model and multi-device resources; not executed by this runner",
                    "requires": spec.get("requires", []),
                    "commands": _h4_commands(args),
                }
            )
            continue
        executable_requested.append(name)
        record = _run_phase(name, spec, args, work_dir)
        phases.append(record)
        print(json.dumps({"phase": name, "status": record["status"], "seconds": record.get("seconds")}))

    if executable_requested:
        verdict = aggregate_phase_results(phases, requested=requested)
        status = verdict["status"]
        problems = verdict["problems"]
    else:
        verdict = aggregate_phase_results(phases, requested=[])
        status = "NPU_UNVERIFIED"
        problems = ["no executable phase was requested; H4 alone cannot be verified by this runner"]
    if status == "NPU_SMOKE_PASSED":
        exit_code = 0
        evidence_level = "NPU_SMOKE_PASSED"
    elif status == "NPU_UNVERIFIED":
        exit_code = 2
        evidence_level = "NPU_UNVERIFIED"
    else:
        exit_code = 4
        evidence_level = "NOT_VERIFIED"

    payload.update(
        {
            "status": status,
            "evidence_level": evidence_level,
            "phases": phases,
            "executable_requested_phases": executable_requested,
            "counts": verdict["counts"],
            "problems": problems,
            "duration_seconds": round(time.time() - started, 3),
            "artifacts": str(work_dir / "artifacts"),
            "logs": str(work_dir / "logs"),
            "note": (
                "Only executed phase statuses are evidence. H3/H4 not run remain NOT_RUN; "
                "synthetic H2 features are not a real-teacher validation."
            ),
            "h4_commands_not_run": _h4_commands(args),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "problems": problems, "json": str(out_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
