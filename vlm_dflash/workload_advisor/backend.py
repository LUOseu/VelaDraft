"""Backend readiness, capability declaration and GPU lease/ledger helpers.

GPU work is coordinated by an advisory flock per device UUID:
``/tmp/veladraft_gpu_locks/<UUID>.lock``. The lock only coordinates jobs that
follow the same convention; it never replaces occupancy checks or permission.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import EVIDENCE_STATES, PROJECT_SCOPE, SCHEMA_VERSION

LOCK_DIR = Path(os.environ.get("VELADRAFT_GPU_LOCK_DIR", "/tmp/veladraft_gpu_locks"))

CAPABILITIES = {
    "inputs": {
        "text": True,
        "image": True,
        "video": False,
        "audio": False,
        "vla_action": False,
        "tool_calls": False,
        "multi_turn_state": False,
    },
    "generation": {
        "ar": True,
        "sd_sequential": True,
        "sd_parallel": False,
        "streaming": False,
        "cancel": False,
    },
    "visibility": {
        "per_token_timestamps": False,
        "proposal_commit_events": False,
        "chunk_callbacks": False,
    },
    "metrics": {
        "client_e2e": True,
        "ttft": False,
        "itl": False,
        "gpu_peak_memory": True,
        "gpu_hours": True,
        "target_visual_tokens": True,
    },
}


class GpuBusy(RuntimeError):
    def __init__(self, uuid: str):
        super().__init__(f"GPU {uuid} is locked by another veladraft job (lock={LOCK_DIR / (uuid + '.lock')}).")
        self.uuid = uuid


@dataclass
class GpuLease:
    uuids: list[str]
    lock_dir: Path = LOCK_DIR
    _handles: list[Any] = field(default_factory=list)
    acquired_at: float | None = None
    released_at: float | None = None

    def acquire(self) -> "GpuLease":
        if not self.uuids:
            raise ValueError("GpuLease requires at least one GPU UUID.")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        for uuid in sorted(self.uuids):
            handle = open(self.lock_dir / f"{uuid}.lock", "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                self.release()
                raise GpuBusy(uuid) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} acquired={time.time():.6f}\n")
            handle.flush()
            self._handles.append(handle)
        self.acquired_at = time.time()
        return self

    def release(self) -> None:
        for handle in reversed(self._handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self._handles.clear()
        self.released_at = time.time()

    def __enter__(self) -> "GpuLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    @property
    def allocated_hours(self) -> float:
        if self.acquired_at is None:
            return 0.0
        end = self.released_at or time.time()
        return max(0.0, end - self.acquired_at) * len(self.uuids) / 3600.0


@dataclass
class GpuLedger:
    budget_gpu_hours: float
    lease: GpuLease
    intervals: list[dict] = field(default_factory=list)
    measured_started_at: float | None = None
    measured_released_at: float | None = None

    def start_measured(self) -> None:
        self.measured_started_at = time.time()

    def stop_measured(self) -> None:
        self.measured_released_at = time.time()

    def record_interval(self, purpose: str) -> None:
        if self.lease.acquired_at is None:
            return
        end = self.lease.released_at or time.time()
        self.intervals.append(
            {
                "uuids": sorted(self.lease.uuids),
                "start_unix_s": self.lease.acquired_at,
                "end_unix_s": end,
                "seconds": end - self.lease.acquired_at,
                "purpose": purpose,
            }
        )

    @property
    def all_allocated_gpu_hours(self) -> float:
        return self.lease.allocated_hours

    @property
    def measured_run_gpu_hours(self) -> float:
        if self.measured_started_at is None:
            return 0.0
        end = self.measured_released_at or time.time()
        return max(0.0, end - self.measured_started_at) * len(self.lease.uuids) / 3600.0

    def check_budget(self) -> dict:
        used = self.all_allocated_gpu_hours
        return {
            "budget_gpu_hours": self.budget_gpu_hours,
            "all_allocated_gpu_hours": used,
            "remaining_gpu_hours": max(0.0, self.budget_gpu_hours - used),
            "exceeded": used >= self.budget_gpu_hours,
        }


def _run(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def discover_gpus() -> list[dict]:
    """Return device UUIDs/names from nvidia-smi or torch (empty if none)."""
    smi = shutil.which("nvidia-smi")
    if smi:
        output = _run([smi, "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"])
        if output:
            devices = []
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2:
                    devices.append({"uuid": parts[0], "name": parts[1], "source": "nvidia-smi"})
            if devices:
                return devices
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            devices = []
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                uuid = getattr(properties, "uuid", None)
                devices.append(
                    {
                        "uuid": str(uuid) if uuid else f"GPU-{index}",
                        "name": properties.name,
                        "source": "torch",
                    }
                )
            return devices
    except Exception:
        pass
    return []


def detect_environment() -> dict:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "devices": discover_gpus(),
    }
    try:
        import torch  # noqa: PLC0415

        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        env["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import transformers  # noqa: PLC0415

        env["transformers"] = {"version": transformers.__version__}
    except Exception as exc:
        env["transformers"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return env


def capabilities_document(evidence_path: str | None = None, state: str = "implemented") -> dict:
    if state not in EVIDENCE_STATES:
        raise ValueError(f"evidence state must be one of {EVIDENCE_STATES}")
    return {
        "schema_version": SCHEMA_VERSION,
        "project_scope": PROJECT_SCOPE,
        "include_in_paper": False,
        "backend": {"name": "hf-transformers", "version": "local", "code_commit": None},
        "capabilities": CAPABILITIES,
        "evidence": {
            "state": state,
            "path": evidence_path,
            "note": "streaming, cancel, per-token timestamps and proposal/commit traces are not exposed by the HF eval path.",
        },
    }


def readiness(manifest: dict) -> dict:
    env = detect_environment()
    blockers: list[dict] = []
    host_blockers: list[dict] = []
    candidate_blockers: list[dict] = []
    warnings: list[dict] = []
    torch_info = env.get("torch") or {}
    if not torch_info.get("version"):
        host_blockers.append({"code": "torch_missing", "detail": torch_info.get("error", "torch is not importable")})
    elif not torch_info.get("cuda_available"):
        host_blockers.append({"code": "cuda_unavailable", "detail": "torch reports no CUDA device on this host"})
    if not env.get("devices"):
        host_blockers.append({"code": "no_gpu", "detail": "nvidia-smi/torch found no GPU"})
    target = manifest.get("target") or {}
    model_path = target.get("model_path")
    if model_path and not Path(model_path).expanduser().exists():
        host_blockers.append({"code": "target_missing", "detail": f"target model_path not found: {model_path}"})
    elif not model_path:
        warnings.append({"code": "target_path_unset", "detail": "target.model_path is not set; HF cache may resolve it."})
    for candidate in manifest.get("draft_candidates") or []:
        if candidate.get("method") != "sd":
            continue
        checkpoint = candidate.get("checkpoint")
        if checkpoint and not Path(str(checkpoint)).expanduser().exists():
            candidate_blockers.append(
                {
                    "code": "checkpoint_missing",
                    "candidate_id": candidate.get("candidate_id"),
                    "detail": f"checkpoint not found: {checkpoint}",
                }
            )
    if manifest.get("concurrency") not in (1, None):
        host_blockers.append({"code": "concurrency_unsupported", "detail": "first-round backend is serial only"})
    blockers = host_blockers + candidate_blockers
    state = "blocked" if host_blockers else ("partial" if candidate_blockers else "ready")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "environment": env,
        "blockers": blockers,
        "host_blockers": host_blockers,
        "candidate_blockers": candidate_blockers,
        "warnings": warnings,
        "capabilities": CAPABILITIES,
    }
