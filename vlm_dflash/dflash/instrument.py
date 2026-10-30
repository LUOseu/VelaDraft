"""Training-loop measurement: per-update fields, synchronized windows and segments, memory peaks.

Two kinds of timing, kept apart on purpose:

* **Per-update host fields** (``update_s``, ``data_s``, ``compute_host_s``,
  ``optim_host_s``) are CPU ``perf_counter`` deltas taken without extra device
  synchronization. The loop already blocks once per update when it reads the
  loss, and the online source synchronizes around every target forward, so they
  track real time closely, but they are host-side numbers, not GPU latency.
* **Windows and segments** synchronize every participating device at their
  boundaries only, so their wall time is the authoritative measurement:
  ``timing_windows`` (explicit ``[start_step, stop_step]`` pairs, e.g. 50
  warmup + 500 measured updates) and fixed-size ``timing_segment_every``
  segments over a whole run.

Time spent in validation and checkpoint writes ("aux") is measured with
synchronization and excluded from both, and reported separately. Memory peaks
(allocated and reserved, per device) are read and reset at every boundary and
folded into the run, window and segment maxima, so a reset by another
component (the online source resets its device's peak per phase) cannot hide
a peak.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import runtime
from ..utils import append_jsonl

GIB = float(1 << 30)


def _peak_reserved(device) -> int | None:
    if torch.device(device).type == "cuda":
        return int(torch.cuda.max_memory_reserved(device))
    return None


def participating_devices(device, source=None) -> list[torch.device]:
    """The training device followed by the source's other target devices."""
    device = torch.device(device)
    devices = [device]
    for extra in getattr(source, "target_devices", None) or []:
        if torch.device(extra) != device:
            devices.append(torch.device(extra))
    return devices


class LoopInstrument:
    def __init__(self, train_cfg: dict, device, source=None, out_dir: Path | None = None):
        self.device = torch.device(device)
        self.source = source
        self.devices = participating_devices(self.device, source)
        self.windows = {}
        for pair in train_cfg.get("timing_windows") or []:
            start, stop = int(pair[0]), int(pair[1])
            if stop <= start:
                raise ValueError(f"train.timing_windows entry {pair!r} must have stop > start.")
            self.windows.setdefault(start, []).append(stop)
        self.segment_every = int(train_cfg.get("timing_segment_every") or 0)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.open_windows: list[dict] = []
        self.segment: dict | None = None
        self.run_peak = {str(d): {"allocated": 0, "reserved": 0} for d in self.devices}
        self.totals = {"validation_s": 0.0, "checkpoint_s": 0.0, "training_s": 0.0, "updates": 0}
        self.window_results: list[dict] = []
        self._reset_update()
        self.last_t = None

    # -- helpers -------------------------------------------------------------
    def sync(self) -> None:
        runtime.synchronize_devices(self.devices)

    def _target_stats(self) -> dict:
        entry = (getattr(self.source, "stats", None) or {}).get("train") or {}
        return {
            "target_prepare_s": float(entry.get("prepare_seconds", 0.0)),
            "target_forward_s": float(entry.get("forward_seconds", 0.0)),
            "target_forwards": int(entry.get("target_forwards", 0)),
        }

    def _fold_peaks(self, reset: bool = True) -> dict:
        """Read every device's peak since the last reset, fold into open maxima, then reset."""
        reading = {}
        for device in self.devices:
            allocated = runtime.max_memory_allocated(device)
            reserved = _peak_reserved(device)
            reading[str(device)] = {"allocated": int(allocated or 0), "reserved": int(reserved or 0)}
            if reset:
                # On CUDA this resets the reserved peak as well.
                runtime.reset_peak_memory_stats(device)
        holders = [self.run_peak] + [w["peak"] for w in self.open_windows]
        if self.segment is not None:
            holders.append(self.segment["peak"])
        for holder in holders:
            for name, value in reading.items():
                slot = holder.setdefault(name, {"allocated": 0, "reserved": 0})
                slot["allocated"] = max(slot["allocated"], value["allocated"])
                slot["reserved"] = max(slot["reserved"], value["reserved"])
        return reading

    @staticmethod
    def _peak_gib(peak: dict) -> dict:
        return {
            name: {"allocated_gib": value["allocated"] / GIB, "reserved_gib": value["reserved"] / GIB,
                   "allocated_bytes": value["allocated"], "reserved_bytes": value["reserved"]}
            for name, value in peak.items()
        }

    def _write(self, name: str, record: dict) -> None:
        if self.out_dir is not None:
            append_jsonl(self.out_dir / name, record)

    def _reset_update(self) -> None:
        self.workload: dict = {}
        self.data_s = 0.0
        self.compute_s = 0.0
        self.optim_s = 0.0
        self.aux_in_update = 0.0
        self.samples = 0

    # -- loop hooks ----------------------------------------------------------
    def start(self, step: int) -> None:
        self.sync()
        self._fold_peaks()
        self.last_t = time.perf_counter()
        self.started = self.last_t
        self.boundary(step)

    def add_workload(self, counts: dict | None) -> None:
        for key, value in (counts or {}).items():
            self.workload[key] = self.workload.get(key, 0) + int(value)

    def aux(self, kind: str, fn, *args, **kwargs):
        """Run validation/checkpoint work outside every timed interval."""
        self.sync()
        self._fold_peaks()
        began = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.sync()
            # Validation memory is not training memory: drop its peaks.
            for device in self.devices:
                runtime.reset_peak_memory_stats(device)
            spent = time.perf_counter() - began
            self.totals[f"{kind}_s"] = self.totals.get(f"{kind}_s", 0.0) + spent
            self.aux_in_update += spent
            for window in self.open_windows:
                window["aux_s"] += spent
            if self.segment is not None:
                self.segment["aux_s"] += spent

    def finish_update(self, step: int) -> dict:
        """Fields for the update that just completed (``step`` updates done)."""
        now = time.perf_counter()
        interval = now - self.last_t
        update_s = interval - self.aux_in_update
        self.last_t = now
        self.totals["training_s"] += update_s
        self.totals["updates"] += 1
        fields = {
            "t_host_s": now - self.started,
            "update_s": update_s,
            "aux_s": self.aux_in_update,
            "data_s": self.data_s,
            "compute_host_s": self.compute_s,
            "optim_host_s": self.optim_s,
            "samples": self.samples,
            "mem_peak_allocated_gib": (runtime.max_memory_allocated(self.device) or 0) / GIB,
        }
        fields.update({f"w_{key}": value for key, value in self.workload.items()})
        for holder in self.open_windows + ([self.segment] if self.segment is not None else []):
            holder["updates"] += 1
            holder["samples"] += self.samples
            for key, value in self.workload.items():
                holder["workload"][key] = holder["workload"].get(key, 0) + value
        self._reset_update()
        return fields

    def boundary(self, step: int) -> None:
        """Called after every update (and once at start): close/open windows and segments."""
        closing = [w for w in self.open_windows if w["stop_step"] == step]
        segment_due = self.segment_every and step % self.segment_every == 0
        opening = self.windows.get(step, [])
        if not (closing or segment_due or opening or self.segment is None and self.segment_every):
            return
        self.sync()
        now = time.perf_counter()
        self._fold_peaks()
        target = self._target_stats()
        for window in closing:
            self.open_windows.remove(window)
            record = self._close(window, now, target, step)
            self.window_results.append(record)
            self._write("timing_windows.jsonl", record)
        if self.segment is not None and segment_due:
            record = self._close(self.segment, now, target, step)
            self._write("timing_segments.jsonl", record)
            self.segment = None
        if self.segment_every and self.segment is None:
            self.segment = self._open("segment", step, now, target)
        for stop in opening:
            window = self._open("window", step, now, target)
            window["stop_step"] = stop
            self.open_windows.append(window)

    def _open(self, kind: str, step: int, now: float, target: dict) -> dict:
        return {
            "kind": kind,
            "start_step": step,
            "t0": now,
            "aux_s": 0.0,
            "updates": 0,
            "samples": 0,
            "workload": {},
            "target0": target,
            "peak": {str(d): {"allocated": 0, "reserved": 0} for d in self.devices},
        }

    def _close(self, holder: dict, now: float, target: dict, step: int) -> dict:
        wall = now - holder["t0"] - holder["aux_s"]
        record = {
            "kind": holder["kind"],
            "start_step": holder["start_step"],
            "stop_step": step,
            "updates": holder["updates"],
            "samples": holder["samples"],
            "wall_s": wall,
            "aux_s": holder["aux_s"],
            "update_time_ms": 1000.0 * wall / holder["updates"] if holder["updates"] else None,
            "samples_per_s": holder["samples"] / wall if wall > 0 else None,
            "effective_positions_per_s": (
                holder["workload"].get("effective_positions", 0) / wall if wall > 0 else None
            ),
            "workload": holder["workload"],
            "target": {key: target[key] - holder["target0"][key] for key in target},
            "peak": self._peak_gib(holder["peak"]),
            "devices": [str(d) for d in self.devices],
            "synchronized": True,
        }
        return record

    def summary(self) -> dict:
        self.sync()
        self._fold_peaks(reset=False)
        return {
            **self.totals,
            "run_peak": self._peak_gib(self.run_peak),
            "devices": [str(d) for d in self.devices],
            "windows": self.window_results,
        }
