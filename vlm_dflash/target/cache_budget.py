"""Adaptive byte-budget controller for the online feature cache (opt-in).

The fixed cache (``train.online_feature_cache_gb``) retains extracted records
under a byte cap in :class:`~vlm_dflash.target.online.OnlineFeatureSource`.
This module adds the *adaptive* budget on top, behind
``train.online_feature_cache_adaptive``: while the cache is enabled, the
retained payload is bounded by the job's own memory headroom, so under
pressure the cache yields host RAM back (rows are re-extracted later) instead
of competing with the trainer for the last gigabytes.

Design (D4 brief, 2026-10-14; review fixes 2026-10-15):

- ``budget = min(configured cap, job share, headroom + retained - reserve)``;
  the reserve is ``max(reserve_bytes, reserve_fraction * limit)`` so the cache
  is never the allocator that trips an OOM. The sensor's headroom already
  excludes the bytes the cache itself retains (they count as the job's usage),
  so the current payload is added back before subtracting the reserve: the
  cache only yields bytes for the safety margin that is actually missing, and
  it is not charged twice for its own residency (which would shrink it on
  every observation and oscillate under a constant external load). A negative
  headroom (the job is over its limit right now) trims the shortfall on the
  first observation.
- Shrink fast: the payload is trimmed immediately when the allowed headroom
  drops below it (the caller trims after every ``observe``), with a small
  tolerance and a minimum interval between readings to stay quiet under noise.
- Grow slowly: growth is watermark-gated (only while the *allowed* headroom is
  above ``min(high_watermark_bytes, bound)``; the gate is capped by the
  effective bound so a cap below the watermark can still grow back) and
  stepwise (a fraction of the current budget per adjustment, with a minimum
  step).
- The sensor resolves the process's own cgroup from ``/proc/self/cgroup`` and
  walks the ancestor chain for the tightest limit (the mount root is not
  necessarily the job under systemd/Slurm/container nesting); a finite cgroup
  reading is combined with the host ``MemAvailable`` so machine-wide pressure
  still reaches the controller.
- No sensor available (no cgroup limit and no readable ``/proc/meminfo``) ->
  the budget keeps its current value; at startup that is the configured cap,
  i.e. exactly the fixed-budget policy. A budget of zero is supported: the
  cache keeps accounting and stores nothing until there is room again.
- The budget covers the cached *record payload* only; renders, queues,
  staging and the trainer stay outside it. Sensor reads are throttled and no
  garbage collection ever runs from here.

The controller is stdlib-only, CPU-testable with an injected ``sensor`` and
``clock``, and never persists state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

__all__ = ["MemorySample", "read_memory_sample", "BudgetParams", "CacheBudgetController"]

# cgroup v1 reports a little under 2**63 for "no limit".
_NO_LIMIT = 1 << 60


@dataclass(frozen=True)
class MemorySample:
    """One memory observation for the job (cgroup-scoped when a limit exists).

    ``available_bytes`` is the headroom the job may still consume before
    hitting its own limit (cgroup limit minus current usage) or the system's
    ``MemAvailable`` when no limit applies; ``None`` means no source could be
    read. It is *negative* when the job is currently over its cgroup limit
    (the next observation then trims the shortfall). When both a finite cgroup
    limit and a readable ``MemAvailable`` exist, ``available_bytes`` is the
    smaller of the two so a machine-wide squeeze still reaches the controller;
    ``host_available_bytes`` keeps the host reading for observability.
    """

    source: str  # "cgroup-v2" | "cgroup-v1" | "meminfo" | "unavailable"
    limit_bytes: Optional[int]
    used_bytes: Optional[int]
    available_bytes: Optional[int]
    host_available_bytes: Optional[int] = None


def _parse_proc_self_cgroup(proc_cgroup: Path) -> dict[str, str]:
    """Controller name -> cgroup path for this process (``""`` = the v2 line)."""
    memberships: dict[str, str] = {}
    for line in proc_cgroup.read_text(encoding="ascii").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        if hierarchy == "0" and not controllers:
            memberships[""] = relative  # cgroup v2 unified hierarchy
            continue
        for name in controllers.split(","):
            if name:
                memberships[name] = relative
    return memberships


def _cgroup_chain(mount: Path, relative: str) -> list[Path]:
    """The process cgroup ``mount/<relative>`` and its ancestors, leaf first."""
    directories = [mount]
    segments = [segment for segment in relative.split("/") if segment]
    for depth in range(1, len(segments) + 1):
        directories.append(mount.joinpath(*segments[:depth]))
    directories.reverse()
    return directories


def _read_meminfo(meminfo: Path) -> dict[str, int]:
    """Readable ``key -> bytes`` fields of ``/proc/meminfo`` ({} when unreadable)."""
    info: dict[str, int] = {}
    try:
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if rest.strip():
                info[key.strip()] = int(rest.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return info


def _read_cgroup_v2(mount: Path, relative: str) -> Optional[MemorySample]:
    """Tightest finite ``memory.max`` on the chain from ``mount/<relative>`` up."""
    binding: Optional[tuple[int, int, int]] = None  # (headroom, limit, used)
    for directory in _cgroup_chain(mount, relative):
        try:
            raw_limit = (directory / "memory.max").read_text(encoding="ascii").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            used = int((directory / "memory.current").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        headroom = limit - used
        if binding is None or headroom < binding[0]:
            binding = (headroom, limit, used)
    if binding is None:
        return None
    headroom, limit, used = binding
    return MemorySample("cgroup-v2", limit, used, headroom)


def _read_cgroup_v1(mount: Path, relative: str) -> Optional[MemorySample]:
    """Tightest finite ``memory.limit_in_bytes`` on the chain, leaf first."""
    binding: Optional[tuple[int, int, int]] = None
    for directory in _cgroup_chain(mount, relative):
        try:
            limit = int((directory / "memory.limit_in_bytes").read_text(encoding="ascii").strip())
            if limit >= _NO_LIMIT:
                continue
            used = int((directory / "memory.usage_in_bytes").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        headroom = limit - used
        if binding is None or headroom < binding[0]:
            binding = (headroom, limit, used)
    if binding is None:
        return None
    headroom, limit, used = binding
    return MemorySample("cgroup-v1", limit, used, headroom)


def _with_host_available(sample: MemorySample, meminfo: Path) -> MemorySample:
    """Combine a cgroup reading with the host ``MemAvailable`` (smaller wins)."""
    host = _read_meminfo(meminfo).get("MemAvailable")
    if host is None or sample.available_bytes is None:
        return sample
    return MemorySample(
        sample.source,
        sample.limit_bytes,
        sample.used_bytes,
        min(sample.available_bytes, host),
        host,
    )


def read_memory_sample(
    *,
    cgroup_v2: Path = Path("/sys/fs/cgroup"),
    cgroup_v1: Path = Path("/sys/fs/cgroup/memory"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    meminfo: Path = Path("/proc/meminfo"),
) -> MemorySample:
    """Best-effort read of the job's memory headroom.

    Resolves the process's own cgroup from ``/proc/self/cgroup`` and walks the
    hierarchy up to the mount root, so limits set by systemd/Slurm/container
    nesting are honoured (a finite limit at any ancestor binds its whole
    subtree). The cgroup reading (v2, then v1) is combined with the host
    ``MemAvailable`` when readable, so machine-wide pressure still reaches the
    controller. Returns ``MemorySample("unavailable", ...)`` when none of the
    sources can be read; callers then keep the fixed policy.
    """
    try:
        memberships: Optional[dict[str, str]] = _parse_proc_self_cgroup(proc_cgroup)
    except OSError:
        memberships = None  # no procfs: fall back to the mount roots
    if memberships is None or "" in memberships:
        relative_v2 = "/" if memberships is None else memberships[""]
        sample = _read_cgroup_v2(cgroup_v2, relative_v2)
        if sample is not None:
            return _with_host_available(sample, meminfo)
    if memberships is None or "memory" in memberships:
        relative_v1 = "/" if memberships is None else memberships["memory"]
        sample = _read_cgroup_v1(cgroup_v1, relative_v1)
        if sample is not None:
            return _with_host_available(sample, meminfo)
    info = _read_meminfo(meminfo)
    available = info.get("MemAvailable")
    if available is not None:
        return MemorySample("meminfo", info.get("MemTotal"), None, available, available)
    return MemorySample("unavailable", None, None, None)


@dataclass(frozen=True)
class BudgetParams:
    """Controller constants. Deliberately not part of any config/digest surface."""

    reserve_bytes: int = 1 << 30
    reserve_fraction: float = 0.10
    low_watermark_bytes: int = 768 << 20
    high_watermark_bytes: int = 4 << 30
    shrink_tolerance_bytes: int = 256 << 20
    grow_fraction: float = 0.125
    grow_min_bytes: int = 32 << 20
    min_interval_s: float = 0.5


class CacheBudgetController:
    """Keeps the cache payload inside the job's memory headroom (see module docstring).

    ``observe(retained_bytes)`` returns the effective payload budget in bytes
    and is meant to be called from cache lookups/stores; ``retained_bytes`` is
    the payload the caller's cache currently holds (required for the coupling
    above — the sensor's headroom already excludes those bytes). The sensor is
    read at most once per ``params.min_interval_s``; within that window the
    last budget is returned. The caller is responsible for trimming the store
    to the returned budget (``OnlineFeatureSource`` does this on every
    lookup/store).
    """

    def __init__(
        self,
        cap_bytes: int,
        *,
        job_share_bytes: int | None = None,
        params: BudgetParams = BudgetParams(),
        sensor: Callable[[], MemorySample] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cap_bytes = max(0, int(cap_bytes))
        self._share_bytes = None if job_share_bytes is None else max(0, int(job_share_bytes))
        self._params = params
        self._sensor = sensor if sensor is not None else (lambda: read_memory_sample())
        self._clock = clock if clock is not None else time.monotonic
        self._bound_bytes = (
            self._cap_bytes if self._share_bytes is None else min(self._cap_bytes, self._share_bytes)
        )
        self._budget_bytes = self._bound_bytes
        self._last_sample: MemorySample | None = None
        self._last_allowed_bytes: int | None = None
        self._last_reserve_bytes: int | None = None
        self._last_retained_bytes = 0
        self._last_observe_at: float | None = None
        self.adjustments = 0
        self.shrinks = 0
        self.grows = 0
        # Size the first budget from the current pressure so a job that starts
        # constrained never builds the full cache first.
        self.observe()

    @property
    def budget_bytes(self) -> int:
        """Current effective payload budget (bytes)."""
        return self._budget_bytes

    @property
    def cap_bytes(self) -> int:
        """Configured fixed cap (``train.online_feature_cache_gb``); the hard ceiling."""
        return self._cap_bytes

    def observe(self, retained_bytes: int = 0) -> int:
        """Return the effective payload budget, refreshing the pressure reading if due.

        ``retained_bytes`` is the current cached payload. The default (0)
        models an empty cache; callers that own a store must pass their
        current bytes so the cache is not charged twice for its own residency.
        """
        now = self._clock()
        if self._last_observe_at is not None and (now - self._last_observe_at) < self._params.min_interval_s:
            return self._budget_bytes
        self._last_observe_at = now
        self._last_retained_bytes = max(0, int(retained_bytes))
        self._last_sample = self._sensor()
        self._adjust()
        return self._budget_bytes

    def stats(self) -> dict:
        """Serializable snapshot for ``train_summary.json`` and lane reports."""
        sample = self._last_sample
        return {
            "cap_bytes": self._cap_bytes,
            "job_share_bytes": self._share_bytes,
            "budget_bytes": self._budget_bytes,
            "allowed_bytes": self._last_allowed_bytes,
            "reserve_bytes": self._last_reserve_bytes,
            "retained_bytes": self._last_retained_bytes,
            "adjustments": self.adjustments,
            "shrinks": self.shrinks,
            "grows": self.grows,
            "sensor": {
                "source": None if sample is None else sample.source,
                "limit_bytes": None if sample is None else sample.limit_bytes,
                "used_bytes": None if sample is None else sample.used_bytes,
                "available_bytes": None if sample is None else sample.available_bytes,
                "host_available_bytes": None if sample is None else sample.host_available_bytes,
            },
            "note": (
                "adaptive payload budget: min(cap, job share, headroom + retained - reserve), "
                "where headroom already excludes the cache's own retained bytes and is the "
                "smaller of the process cgroup's headroom and host MemAvailable (negative when "
                "the job is over its limit); shrinks are immediate (debounced by min "
                "interval/tolerance), growth is watermark gated (the gate is capped by the "
                "effective bound, so caps below the watermark can still regrow) and stepwise; "
                "sensor unavailable -> the budget keeps its current value (the fixed cap at "
                "startup)."
            ),
        }

    # -- policy ---------------------------------------------------------------
    def _adjust(self) -> None:
        sample = self._last_sample
        if sample is None or sample.available_bytes is None:
            # No sensor: keep the current value (at startup: the configured cap,
            # i.e. the fixed-budget policy).
            return
        allowed, reserve = self._allowed_bytes(sample, self._last_retained_bytes)
        self._last_allowed_bytes = allowed
        self._last_reserve_bytes = reserve
        if allowed < self._budget_bytes and (
            allowed <= self._params.low_watermark_bytes
            or self._budget_bytes - allowed >= self._params.shrink_tolerance_bytes
        ):
            self._set_budget(allowed)  # shrink fast, keeping the reserve intact
            return
        # The growth gate is capped by the effective bound: a bound below the
        # watermark (e.g. a 0.87 GiB working set with the default 4 GiB
        # watermark) still regrows once the headroom covers the whole bound.
        grow_gate = min(self._params.high_watermark_bytes, self._bound_bytes)
        if allowed >= grow_gate and allowed > self._budget_bytes:
            step = max(self._params.grow_min_bytes, int(self._budget_bytes * self._params.grow_fraction))
            self._set_budget(min(self._bound_bytes, allowed, self._budget_bytes + step))

    def _allowed_bytes(self, sample: MemorySample, retained_bytes: int) -> tuple[int, int]:
        # ``sample.available_bytes`` excludes the bytes this cache already
        # retains (they are part of the job's usage), so add them back: the
        # budget bounds the cache's *total* payload and only the missing
        # safety margin is evicted. A negative headroom (over the limit)
        # trims the shortfall plus the reserve.
        available = int(sample.available_bytes)
        retained = max(0, int(retained_bytes))
        limit = sample.limit_bytes
        reserve = max(
            self._params.reserve_bytes,
            int(self._params.reserve_fraction * (limit if limit else max(0, available))),
        )
        allowed = max(0, available + retained - reserve)
        return min(allowed, self._bound_bytes), reserve

    def _set_budget(self, value: int) -> None:
        value = max(0, int(value))
        if value == self._budget_bytes:
            return
        if value < self._budget_bytes:
            self.shrinks += 1
        else:
            self.grows += 1
        self.adjustments += 1
        self._budget_bytes = value
