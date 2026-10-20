"""Backend readiness, capability declaration and GPU lease/ledger helpers.

GPU work is coordinated by an advisory flock per device UUID:
``/tmp/veladraft_gpu_locks/<UUID>.lock``. The lock only coordinates jobs that
follow the same convention; it never replaces occupancy checks or permission.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import re
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
    def __init__(self, uuid: str, *, lock_dir: Path | None = None, detail: str | None = None):
        lock_dir = lock_dir or LOCK_DIR
        message = f"GPU {uuid} is locked by another veladraft job (lock={lock_dir / (uuid + '.lock')})."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)
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
                raise GpuBusy(uuid, lock_dir=self.lock_dir) from exc
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

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.budget_gpu_hours)) or float(self.budget_gpu_hours) <= 0:
            raise ValueError("budget_gpu_hours must be a positive finite number.")

    def start_measured(self) -> None:
        self.measured_started_at = time.time()

    def stop_measured(self) -> None:
        self.measured_released_at = time.time()

    def record_interval(
        self,
        purpose: str,
        *,
        start_unix_s: float | None = None,
        end_unix_s: float | None = None,
    ) -> None:
        if self.lease.acquired_at is None:
            return
        start = self.lease.acquired_at if start_unix_s is None else float(start_unix_s)
        end = self.lease.released_at or time.time() if end_unix_s is None else float(end_unix_s)
        if end < start:
            return
        self.intervals.append(
            {
                "uuids": sorted(self.lease.uuids),
                "start_unix_s": start,
                "end_unix_s": end,
                "seconds": end - start,
                "gpu_hours": (end - start) * len(self.lease.uuids) / 3600.0,
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

    def checkpoint(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "budget_gpu_hours": self.budget_gpu_hours,
            "all_allocated_gpu_hours": self.all_allocated_gpu_hours,
            "measured_run_gpu_hours": self.measured_run_gpu_hours,
            "uuids": sorted(self.lease.uuids),
            "checkpoint_unix_s": time.time(),
            "intervals": list(self.intervals),
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


def resolve_device_uuid(
    device: str,
    devices: list[dict],
    *,
    cuda_visible_devices: str | None = None,
) -> dict:
    """Resolve ``cuda:N`` through ``CUDA_VISIBLE_DEVICES`` to one physical UUID.

    ``nvidia-smi`` reports physical devices even when a process sees a remapped
    subset.  Locking/ledgering every visible physical device while only using
    one of them overcounts GPU-hours and can block an unrelated job; this helper
    maps the logical index to the UUID CUDA will actually use.
    """
    match = re.fullmatch(r"cuda(?::(\d+))?", str(device or "").strip())
    if match is None:
        raise ValueError(f"device must be 'cuda' or 'cuda:N', got {device!r}.")
    index = int(match.group(1) or 0)
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cvd = "" if cuda_visible_devices is None else str(cuda_visible_devices).strip()
    if cvd:
        entries = [entry.strip() for entry in cvd.split(",") if entry.strip()]
        if index >= len(entries):
            raise ValueError(
                f"device cuda:{index} is outside CUDA_VISIBLE_DEVICES={cvd!r} ({len(entries)} device(s))."
            )
        entry = entries[index]
        if entry.isdigit():
            physical_index = int(entry)
            if physical_index >= len(devices):
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES entry {entry!r} does not name a discovered physical GPU."
                )
            selected = dict(devices[physical_index])
        else:
            selected = next((dict(item) for item in devices if str(item.get("uuid")) == entry), None)
            if selected is None:
                selected = {"uuid": entry, "name": None, "source": "CUDA_VISIBLE_DEVICES"}
    else:
        if index >= len(devices):
            raise ValueError(f"device cuda:{index} is not present; discovered {len(devices)} GPU(s).")
        selected = dict(devices[index])
    selected.setdefault("source", "torch")
    selected["logical_device"] = f"cuda:{index}"
    selected["cuda_visible_devices"] = cvd or None
    return selected


def check_external_occupancy(uuid: str) -> dict:
    """Best-effort read-only check for non-VelaDraft processes on a GPU.

    The flock only coordinates jobs that opt into the convention, so a lock can
    be free while another user's process is active.  We never signal or evict
    that process; the caller must decline to start timed work.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"status": "unavailable", "processes": [], "busy": False}
    output = _run(
        [
            smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=10.0,
    )
    if output is None:
        return {"status": "unavailable", "processes": [], "busy": False}
    processes = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        if parts[0] != uuid:
            continue
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]) if parts[1].isdigit() else parts[1],
                "process_name": parts[2] if len(parts) > 2 else None,
                "used_memory_mib": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
            }
        )
    return {"status": "checked", "processes": processes, "busy": bool(processes)}


_SYMBOLIC_CHAT_TEMPLATES = {"hf_default", "qwen2_5_vl", "qwen3_vl", "smolvlm"}
_SUPPORTED_STRUCTURES = {"qwen2_5_vl", "qwen3_vl", "smolvlm"}
_SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}
_SUPPORTED_ATTENTION = {"sdpa", "eager", "flash_attention_2"}


def _execution_issue(code: str, detail: str, path: str) -> dict:
    return {"code": code, "detail": detail, "path": path}


def validate_execution_contract(manifest: dict) -> list[dict]:
    """Reject manifest settings the local sequential HF backend cannot honor."""
    issues: list[dict] = []
    backend = manifest.get("backend") or {}
    target = manifest.get("target") or {}
    decode = manifest.get("decode") or {}
    name = str(backend.get("name") or "")
    if name not in {"hf-transformers", "hf_transformers"}:
        issues.append(
            _execution_issue(
                "unsupported_backend",
                f"the measurement backend implements local hf-transformers, not {name!r}",
                "backend.name",
            )
        )
    if manifest.get("measurement_scope") != "local_client":
        issues.append(
            _execution_issue(
                "unsupported_measurement_scope",
                "this backend records local_client timing only; server_endpoint/storage/aggregate_only are not supported",
                "measurement_scope",
            )
        )
    if manifest.get("concurrency") != 1:
        issues.append(
            _execution_issue("unsupported_concurrency", "the HF eval path is serial (concurrency=1)", "concurrency")
        )
    endpoint = manifest.get("server_endpoint") or backend.get("server_endpoint")
    if endpoint:
        issues.append(
            _execution_issue(
                "unsupported_server_endpoint",
                "server_endpoint cannot be measured by the local HF backend",
                "server_endpoint",
            )
        )
    for owner, prefix in ((backend, "backend"), (manifest, "")):
        for key in ("streaming", "stream", "cancel"):
            if owner.get(key):
                issues.append(
                    _execution_issue(
                        "unsupported_streaming",
                        f"{key} is not exposed by the HF eval path",
                        f"{prefix + '.' if prefix else ''}{key}",
                    )
                )
        if owner.get("parallel") or owner.get("parallel_verification"):
            issues.append(
                _execution_issue(
                    "unsupported_parallel",
                    "parallel verification/serving is not supported by this measurement path",
                    f"{prefix + '.' if prefix else ''}parallel",
                )
            )
    if backend.get("type") not in (None, "hf", "local", "hf-transformers"):
        issues.append(
            _execution_issue(
                "unsupported_backend_type",
                f"backend.type={backend.get('type')!r} is not the local HF path",
                "backend.type",
            )
        )
    verification = decode.get("verification", "sequential")
    if verification != "sequential":
        issues.append(
            _execution_issue(
                "unsupported_verification",
                f"decode.verification={verification!r}; only sequential token-exact verification is implemented",
                "decode.verification",
            )
        )
    eos_policy = decode.get("eos_policy", "natural_eos_first_stop")
    if eos_policy not in ("natural_eos_first_stop", "stop_ids_only"):
        issues.append(
            _execution_issue(
                "unsupported_eos_policy",
                f"decode.eos_policy={eos_policy!r} is not implemented by the HF generate path",
                "decode.eos_policy",
            )
        )
    verify_backend = decode.get("verify_backend", "parity")
    if verify_backend not in ("parity", "reference"):
        issues.append(
            _execution_issue(
                "unsupported_verify_backend",
                f"decode.verify_backend={verify_backend!r}; expected 'parity' or 'reference'",
                "decode.verify_backend",
            )
        )
    dtype = str(target.get("dtype") or "")
    if dtype not in _SUPPORTED_DTYPES:
        issues.append(
            _execution_issue("unsupported_dtype", f"target.dtype={dtype!r} is not supported", "target.dtype")
        )
    attn = str(target.get("attn_implementation") or "sdpa")
    if attn not in _SUPPORTED_ATTENTION:
        issues.append(
            _execution_issue(
                "unsupported_attention",
                f"target.attn_implementation={attn!r} is not supported",
                "target.attn_implementation",
            )
        )
    structure = str(target.get("structure") or "")
    if structure and structure not in _SUPPORTED_STRUCTURES:
        issues.append(
            _execution_issue(
                "unsupported_structure",
                f"target.structure={structure!r} is not a supported local HF family",
                "target.structure",
            )
        )
    templates = {
        "target.chat_template": target.get("chat_template"),
        "processor.chat_template": (manifest.get("processor") or {}).get("chat_template"),
    }
    for path, template in templates.items():
        if template in (None, ""):
            continue
        text = str(template)
        is_jinja = "{{" in text or "{%" in text
        if not is_jinja and text not in _SYMBOLIC_CHAT_TEMPLATES:
            issues.append(
                _execution_issue(
                    "unsupported_chat_template",
                    f"{path}={text!r} is neither a pinned Jinja template nor a supported family name",
                    path,
                )
            )
    vision = target.get("vision") or {}
    resolution = manifest.get("resolution") or {}
    bounds = {
        "target.vision.min_pixels": vision.get("min_pixels", resolution.get("min_pixels")),
        "target.vision.max_pixels": vision.get("max_pixels", resolution.get("max_pixels")),
    }
    for path, value in bounds.items():
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            issues.append(_execution_issue("invalid_pixel_bound", f"{path} must be a positive integer", path))
    min_pixels = bounds["target.vision.min_pixels"]
    max_pixels = bounds["target.vision.max_pixels"]
    if isinstance(min_pixels, int) and isinstance(max_pixels, int) and min_pixels > max_pixels:
        issues.append(
            _execution_issue(
                "invalid_pixel_bounds",
                f"min_pixels {min_pixels} exceeds max_pixels {max_pixels}",
                "target.vision",
            )
        )
    return issues


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


def readiness(manifest: dict, *, manifest_dir: str | Path | None = None) -> dict:
    def local_path(value: str) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and manifest_dir is not None:
            path = Path(manifest_dir) / path
        return path

    env = detect_environment()
    blockers: list[dict] = []
    host_blockers: list[dict] = []
    candidate_blockers: list[dict] = []
    contract_blockers = validate_execution_contract(manifest)
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
    if model_path and not local_path(model_path).exists():
        host_blockers.append({"code": "target_missing", "detail": f"target model_path not found: {model_path}"})
    elif not model_path:
        warnings.append({"code": "target_path_unset", "detail": "target.model_path is not set; HF cache may resolve it."})
    for candidate in manifest.get("draft_candidates") or []:
        if candidate.get("method") != "sd":
            continue
        checkpoint = candidate.get("checkpoint")
        if checkpoint and not local_path(checkpoint).exists():
            candidate_blockers.append(
                {
                    "code": "checkpoint_missing",
                    "candidate_id": candidate.get("candidate_id"),
                    "detail": f"checkpoint not found: {checkpoint}",
                }
            )
    blockers = host_blockers + candidate_blockers + contract_blockers
    state = "blocked" if host_blockers or contract_blockers else ("partial" if candidate_blockers else "ready")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": state,
        "environment": env,
        "blockers": blockers,
        "host_blockers": host_blockers,
        "candidate_blockers": candidate_blockers,
        "contract_blockers": contract_blockers,
        "warnings": warnings,
        "capabilities": CAPABILITIES,
    }
