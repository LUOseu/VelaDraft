"""Backend-neutral runtime helpers for CPU, CUDA and Ascend NPU.

The project keeps CUDA as the default accelerator, but every training-reachable
device operation goes through this module. Importing :mod:`vlm_dflash` never
imports ``torch_npu``; an NPU is initialised only after a caller explicitly
resolves an ``npu`` device.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

_DEVICE_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?::(?P<index>\d+))?$")


class DeviceResolutionError(RuntimeError):
    """Raised when an explicitly requested device cannot be used."""


def _split_device(value: str | torch.device) -> tuple[str, int | None]:
    if isinstance(value, torch.device):
        return value.type, value.index
    match = _DEVICE_RE.match(str(value).strip())
    if match is None:
        raise DeviceResolutionError(f"Unrecognised device string {value!r}; expected e.g. cpu, cuda:0 or npu:0.")
    index = match.group("index")
    return match.group("type").lower(), None if index is None else int(index)


def device_type(value: str | torch.device) -> str:
    """Return the device type without requiring that backend to be installed."""
    return _split_device(value)[0]


def _torch_npu_module():
    """Import and validate ``torch_npu``, with an actionable failure."""
    try:
        importlib.import_module("torch_npu")
    except Exception as error:  # ImportError, OSError (CANN libraries), ...
        raise DeviceResolutionError(
            "An NPU device was requested, but torch_npu could not be imported. "
            "Install the TorchNPU wheel matching this PyTorch/CANN/Python stack and source the "
            f"CANN environment before running (import error: {type(error).__name__}: {error})."
        ) from error
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise DeviceResolutionError(
            "torch_npu imported successfully but did not register torch.npu. Check that the TorchNPU "
            "package matches this PyTorch build."
        )
    return npu


def npu_available() -> bool:
    """Whether a working ``torch.npu`` backend is importable."""
    try:
        return bool(_torch_npu_module().is_available())
    except DeviceResolutionError:
        return False


def cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(
    value: str | torch.device | None,
    *,
    default: str = "cuda:0",
    context: str = "device",
    require_available: bool = True,
) -> torch.device:
    """Resolve a configured device, initialising ``torch_npu`` for NPU requests.

    An explicit ``npu`` request never falls back to CPU/CUDA. Missing packages,
    CANN libraries, out-of-range indices and unavailable devices raise
    :class:`DeviceResolutionError` with the actionable cause.
    """
    raw = default if value is None or str(value).strip() == "" else value
    kind, index = _split_device(raw)
    if kind == "cpu":
        return torch.device("cpu")
    if kind == "cuda":
        if require_available and not cuda_available():
            raise DeviceResolutionError(
                f"{context}={raw!r} requested CUDA, but torch.cuda.is_available() is False. "
                "Use an available CUDA device, or explicitly select cpu/npu."
            )
        count = torch.cuda.device_count() if cuda_available() else 0
        resolved_index = 0 if index is None else index
        if require_available and resolved_index >= count:
            raise DeviceResolutionError(
                f"{context}={raw!r} selected CUDA index {resolved_index}, but only {count} CUDA device(s) exist."
            )
        if require_available:
            torch.cuda.set_device(resolved_index)
        return torch.device("cuda", resolved_index)
    if kind == "npu":
        npu = _torch_npu_module()
        if require_available and not bool(npu.is_available()):
            raise DeviceResolutionError(
                f"{context}={raw!r} requested an Ascend NPU, but torch.npu.is_available() is False. "
                "Check the driver, CANN installation, container device mounts and NPU visibility."
            )
        count = int(npu.device_count()) if bool(npu.is_available()) else 0
        resolved_index = 0 if index is None else index
        if require_available and resolved_index >= count:
            raise DeviceResolutionError(
                f"{context}={raw!r} selected NPU index {resolved_index}, but only {count} NPU device(s) exist."
            )
        try:
            device = torch.device("npu", resolved_index)
        except (RuntimeError, TypeError) as error:
            raise DeviceResolutionError(
                "torch_npu is importable but this PyTorch build did not register the 'npu' device type "
                f"({type(error).__name__}: {error}). Install the torch_npu wheel built for this exact "
                "PyTorch version; no CPU/CUDA fallback is performed."
            ) from error
        if require_available:
            npu.set_device(device)
        return device
    raise DeviceResolutionError(
        f"{context}={raw!r} uses unsupported device type {kind!r}; supported types are cpu, cuda and npu."
    )


def ensure_runtime_device(value: str | torch.device, *, context: str = "device") -> torch.device:
    """Resolve a device and verify the requested accelerator is present."""
    return resolve_device(value, context=context, require_available=True)


def accelerator_devices(model: torch.nn.Module) -> list[torch.device]:
    """Every CUDA/NPU device that currently owns a parameter or persistent buffer.

    A device-mapped target can keep a small module's parameters on one device
    and its buffers on another; timing and peak-memory accounting must include
    both. Non-registered references (e.g. ``lm_head`` in the drafter) are not
    discoverable here and are handled by their call sites.
    """
    devices = {
        tensor.device
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type in ("cuda", "npu")
    }
    return sorted(devices, key=lambda item: (item.type, -1 if item.index is None else item.index))


def synchronize(device: str | torch.device) -> None:
    """Synchronize a CUDA/NPU device; CPU is a no-op."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        _torch_npu_module().synchronize(device)


def synchronize_devices(devices: Iterable[str | torch.device]) -> None:
    for device in sorted({torch.device(item) for item in devices}, key=lambda item: (item.type, item.index or 0)):
        synchronize(device)


def _api_for(device: str | torch.device):
    kind = device_type(device)
    if kind == "cuda":
        # Return the namespace even when no GPU is visible: unit tests and
        # diagnostics monkeypatch torch.cuda on CPU hosts. The wrappers below
        # translate the resulting runtime error into an "unavailable" result.
        return torch.cuda
    if kind == "npu":
        try:
            return _torch_npu_module() if npu_available() else None
        except DeviceResolutionError:
            return None
    return None


def reset_peak_memory_stats(device: str | torch.device) -> bool:
    api = _api_for(device)
    function = getattr(api, "reset_peak_memory_stats", None) if api is not None else None
    if function is None:
        return False
    try:
        function(device)
    except Exception:
        return False
    return True


def max_memory_allocated(device: str | torch.device) -> int | None:
    api = _api_for(device)
    function = getattr(api, "max_memory_allocated", None) if api is not None else None
    if function is None:
        return None
    try:
        return int(function(device))
    except Exception:
        return None


def memory_allocated(device: str | torch.device) -> int | None:
    api = _api_for(device)
    function = getattr(api, "memory_allocated", None) if api is not None else None
    if function is None:
        return None
    try:
        return int(function(device))
    except Exception:
        return None


def device_memory_info(device: str | torch.device) -> tuple[int, int] | None:
    api = _api_for(device)
    function = getattr(api, "mem_get_info", None) if api is not None else None
    if function is None:
        return None
    try:
        free, total = function(device)
    except Exception:
        return None
    return int(free), int(total)


def device_name(device: str | torch.device) -> str | None:
    api = _api_for(device)
    function = getattr(api, "get_device_name", None) if api is not None else None
    if function is None:
        return None
    try:
        return str(function(device))
    except Exception:
        return None


@dataclass
class TimingRun:
    """One timed region using accelerator events, or a synchronized wall clock."""

    device: torch.device
    start_event: Any = None
    end_event: Any = None
    started: float = 0.0
    stopped: float | None = None

    @classmethod
    def start(cls, device: str | torch.device) -> "TimingRun":
        resolved = torch.device(device)
        run = cls(device=resolved)
        api = _api_for(resolved)
        event_cls = getattr(api, "Event", None) if api is not None else None
        if event_cls is not None:
            try:
                with api.device(resolved):
                    run.start_event = event_cls(enable_timing=True)
                    run.start_event.record()
                return run
            except Exception:
                run.start_event = None
        synchronize(resolved)
        run.started = time.perf_counter()
        return run

    def stop(self) -> None:
        if self.stopped is not None:
            return
        if self.start_event is not None:
            api = _api_for(self.device)
            event_cls = getattr(api, "Event", None) if api is not None else None
            if event_cls is not None:
                with api.device(self.device):
                    self.end_event = event_cls(enable_timing=True)
                    self.end_event.record()
                self.stopped = 0.0
                return
        synchronize(self.device)
        self.stopped = time.perf_counter()

    def elapsed_seconds(self) -> float:
        if self.stopped is None:
            self.stop()
        if self.start_event is not None and self.end_event is not None:
            synchronize(self.device)
            return float(self.start_event.elapsed_time(self.end_event)) / 1000.0
        assert self.stopped is not None
        return max(0.0, self.stopped - self.started)


def seed_all(seed: int, device: str | torch.device | None = None) -> None:
    """Seed Python, NumPy, torch and the selected accelerator."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    kind = device_type(device) if device is not None else None
    if cuda_available() and kind in (None, "cuda"):
        torch.cuda.manual_seed_all(seed)
    if npu_available() and kind in (None, "npu"):
        _torch_npu_module().manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Serialize Python/NumPy/torch and every available accelerator RNG state."""
    state: dict[str, Any] = {
        "version": 1,
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda_available() else [],
        "npu": [],
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    if npu_available():
        npu = _torch_npu_module()
        getter = getattr(npu, "get_rng_state_all", None)
        if getter is not None:
            state["npu"] = getter()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a state from :func:`capture_rng_state` or a legacy checkpoint."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    torch_state = state.get("torch", state.get("torch_rng_state"))
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_states = state.get("cuda", state.get("cuda_rng_states"))
    if cuda_states and cuda_available():
        torch.cuda.set_rng_state_all(cuda_states)
    npu_states = state.get("npu", state.get("npu_rng_states"))
    if npu_states and npu_available():
        setter = getattr(_torch_npu_module(), "set_rng_state_all", None)
        if setter is not None:
            setter(npu_states)


def pin_memory_supported(device: str | torch.device) -> bool:
    """PyTorch pinned host memory is a CUDA feature; NPU/CPU keep it off."""
    return device_type(device) == "cuda" and cuda_available()


def compile_supported(device: str | torch.device) -> bool:
    """Whether the repository opts into ``torch.compile`` on this backend."""
    return device_type(device) != "npu"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def device_report(value: str | torch.device, *, context: str = "device") -> dict:
    """JSON-serializable information for one configured device."""
    kind = device_type(value)
    index = _split_device(value)[1]
    resolved_index = 0 if index is None else index
    report: dict[str, Any] = {
        "requested": str(value),
        "type": kind,
        "index": resolved_index,
    }
    try:
        device = resolve_device(value, context=context)
    except DeviceResolutionError as error:
        report.update({"available": False, "error": str(error)})
        return report
    report["available"] = True
    report["resolved"] = str(device)
    report["name"] = device_name(device)
    memory = device_memory_info(device)
    if memory is not None:
        report["free_gb"] = round(memory[0] / 1e9, 2)
        report["total_gb"] = round(memory[1] / 1e9, 2)
    return report


def environment_report() -> dict:
    """Runtime versions and accelerator availability (no device selection)."""
    report: dict[str, Any] = {
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_npu": package_version("torch_npu"),
        "cann": os.environ.get("ASCEND_HOME_PATH") or os.environ.get("CANN_HOME"),
        "python": sys.version.split()[0],
    }
    cuda_count = torch.cuda.device_count() if cuda_available() else 0
    report["cuda"] = {
        "available": cuda_available(),
        "device_count": cuda_count,
        "devices": [
            {
                "index": index,
                "name": device_name(torch.device("cuda", index)),
                "memory": device_memory_info(torch.device("cuda", index)),
            }
            for index in range(cuda_count)
        ],
    }
    npu_count = 0
    npu_error = None
    try:
        npu = _torch_npu_module()
        if bool(npu.is_available()):
            npu_count = int(npu.device_count())
    except DeviceResolutionError as error:
        npu_error = str(error)
    report["npu"] = {
        "available": npu_available(),
        "device_count": npu_count,
        "devices": [
            {
                "index": index,
                "name": device_name(torch.device("npu", index)),
                "memory": device_memory_info(torch.device("npu", index)),
            }
            for index in range(npu_count)
        ],
        "import_error": npu_error,
    }
    return report


__all__ = [
    "DeviceResolutionError",
    "TimingRun",
    "accelerator_devices",
    "capture_rng_state",
    "cuda_available",
    "device_memory_info",
    "device_name",
    "device_report",
    "device_type",
    "ensure_runtime_device",
    "environment_report",
    "max_memory_allocated",
    "memory_allocated",
    "npu_available",
    "package_version",
    "pin_memory_supported",
    "compile_supported",
    "reset_peak_memory_stats",
    "resolve_device",
    "restore_rng_state",
    "seed_all",
    "synchronize",
    "synchronize_devices",
]
