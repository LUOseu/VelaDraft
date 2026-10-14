"""Unit tests for the adaptive cache-budget controller (train.online_feature_cache_adaptive).

CPU-only and torch-free: the controller is exercised with injected sensors and
a fake clock, mirroring how ``OnlineFeatureSource`` drives it (observe, then
trim the LRU store to the returned budget).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.target.cache_budget import (
    BudgetParams,
    CacheBudgetController,
    MemorySample,
    read_memory_sample,
)

MB = 1 << 20
GB = 1 << 30


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSensor:
    def __init__(self, sample: MemorySample) -> None:
        self.sample = sample
        self.reads = 0

    def __call__(self) -> MemorySample:
        self.reads += 1
        return self.sample


def _sample(available_gb: float, limit_gb: float | None = None) -> MemorySample:
    return MemorySample(
        "meminfo" if limit_gb is None else "cgroup-v2",
        None if limit_gb is None else int(limit_gb * GB),
        None,
        int(available_gb * GB),
    )


def test_no_sensor_keeps_the_fixed_cap():
    sensor = FakeSensor(MemorySample("unavailable", None, None, None))
    controller = CacheBudgetController(2 * GB, sensor=sensor, clock=FakeClock())
    assert controller.budget_bytes == 2 * GB
    assert controller.observe() == 2 * GB
    assert controller.adjustments == 0
    assert controller.stats()["sensor"]["source"] == "unavailable"


def test_sensor_becoming_available_right_sizes_the_fixed_start():
    sensor = FakeSensor(MemorySample("unavailable", None, None, None))
    clock = FakeClock()
    controller = CacheBudgetController(8 * GB, sensor=sensor, clock=clock)
    assert controller.budget_bytes == 8 * GB
    clock.advance(1.0)
    # available 2 GiB with a 1 GiB reserve -> allowed 1 GiB.
    sensor.sample = _sample(available_gb=2.0)
    assert controller.observe() == 1 * GB
    assert controller.stats()["shrinks"] == 1


def test_headroom_below_the_reserve_shrinks_to_zero():
    # cgroup 2 GiB limit, 1.6 GiB available; reserve = max(1 GiB, 10% of 2 GiB).
    sensor = FakeSensor(_sample(available_gb=1.6, limit_gb=2.0))
    clock = FakeClock()
    controller = CacheBudgetController(4 * GB, sensor=sensor, clock=clock)
    assert controller.budget_bytes == int(1.6 * GB) - GB
    clock.advance(1.0)
    # 0.9 GiB available is below the reserve -> nothing may be retained.
    sensor.sample = _sample(available_gb=0.9, limit_gb=2.0)
    assert controller.observe() == 0
    assert controller.stats()["shrinks"] == 2


def test_cgroup_reserve_uses_the_larger_of_bytes_and_fraction():
    sensor = FakeSensor(MemorySample("cgroup-v2", 8 * GB, 2 * GB, 6 * GB))
    controller = CacheBudgetController(16 * GB, sensor=sensor, clock=FakeClock())
    # reserve = max(1 GiB, 10% of 8 GiB) = 1 GiB -> allowed 5 GiB.
    assert controller.stats()["allowed_bytes"] == 5 * GB
    assert controller.budget_bytes == 5 * GB


def test_recovery_grows_stepwise_only_above_the_high_watermark():
    params = BudgetParams()
    sensor = FakeSensor(_sample(available_gb=2.0))
    clock = FakeClock()
    controller = CacheBudgetController(4 * GB, sensor=sensor, clock=clock)
    assert controller.budget_bytes == 1 * GB  # first observe right-sizes down
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=4.0)  # allowed 3 GiB < 4 GiB watermark -> hold
    assert controller.observe() == 1 * GB
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=8.0)  # allowed 7 GiB -> one step (+25% of 1 GiB)
    assert controller.observe() == 1 * GB + max(32 * MB, int(1 * GB * params.grow_fraction))
    for _ in range(64):
        clock.advance(1.0)
        controller.observe()
    assert controller.budget_bytes == 4 * GB  # clamped at the cap
    assert controller.stats()["grows"] > 0


def test_small_cap_below_the_watermark_regrows_after_a_shrink():
    # Regression: with the pre-fix gate (plain high watermark) a cap below the
    # 4 GiB watermark could shrink but never grow back, because the *allowed*
    # headroom is clamped to the bound. The gate is now min(watermark, bound),
    # so the full bound fitting in the headroom is enough to regrow.
    params = BudgetParams()
    sensor = FakeSensor(_sample(available_gb=8.0))  # allowed 7 GiB -> stays at the cap
    clock = FakeClock()
    controller = CacheBudgetController(1 * GB, sensor=sensor, clock=clock)
    assert controller.budget_bytes == 1 * GB
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=2.0)  # allowed == budget -> hold
    assert controller.observe() == 1 * GB
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=1.5)  # allowed 0.5 GiB -> shrink
    assert controller.observe() == int(0.5 * GB)
    assert controller.stats()["shrinks"] == 1
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=8.0)  # headroom covers the bound -> grow
    shrunk = int(0.5 * GB)
    assert controller.observe() == shrunk + max(params.grow_min_bytes, int(shrunk * params.grow_fraction))
    for _ in range(64):
        clock.advance(1.0)
        controller.observe()
    assert controller.budget_bytes == 1 * GB  # clamped at the cap
    assert controller.stats()["grows"] > 0


def test_small_dips_inside_the_tolerance_do_not_shrink():
    sensor = FakeSensor(_sample(available_gb=8.0))  # allowed 7 GiB -> stays at the cap
    clock = FakeClock()
    controller = CacheBudgetController(4 * GB, sensor=sensor, clock=clock)
    assert controller.budget_bytes == 4 * GB and controller.adjustments == 0
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=4.9)  # 0.1 GiB below budget < 0.25 GiB tolerance
    controller.observe()
    assert controller.budget_bytes == 4 * GB and controller.adjustments == 0
    clock.advance(1.0)
    sensor.sample = _sample(available_gb=4.5)  # 0.5 GiB below budget -> shrink
    assert controller.observe() == int(4.5 * GB) - GB
    assert controller.stats()["shrinks"] == 1


def test_min_interval_debounces_sensor_reads():
    sensor = FakeSensor(_sample(available_gb=8.0))
    clock = FakeClock()
    controller = CacheBudgetController(2 * GB, sensor=sensor, clock=clock)
    assert sensor.reads == 1
    for _ in range(4):
        clock.advance(0.1)
        controller.observe()
    assert sensor.reads == 1  # all within the 0.5 s interval
    clock.advance(0.2)
    controller.observe()
    assert sensor.reads == 2


def test_job_share_caps_the_effective_budget():
    sensor = FakeSensor(_sample(available_gb=64.0))
    controller = CacheBudgetController(32 * GB, job_share_bytes=512 * MB, sensor=sensor, clock=FakeClock())
    assert controller.budget_bytes == 512 * MB
    assert controller.stats()["job_share_bytes"] == 512 * MB
    capped = CacheBudgetController(256 * MB, job_share_bytes=1 * GB, sensor=sensor, clock=FakeClock())
    assert capped.budget_bytes == 256 * MB


def test_read_memory_sample_prefers_cgroup_v2(tmp_path):
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text(f"{4 * GB}\n", encoding="ascii")
    (cg2 / "memory.current").write_text(f"{1 * GB}\n", encoding="ascii")
    sample = read_memory_sample(cgroup_v2=cg2, cgroup_v1=tmp_path / "nope1", meminfo=tmp_path / "nope2")
    assert sample == MemorySample("cgroup-v2", 4 * GB, 1 * GB, 3 * GB)


def test_read_memory_sample_uses_cgroup_v1_when_v2_is_absent(tmp_path):
    cg1 = tmp_path / "cg1"
    cg1.mkdir()
    (cg1 / "memory.limit_in_bytes").write_text(f"{2 * GB}\n", encoding="ascii")
    (cg1 / "memory.usage_in_bytes").write_text(f"{GB // 2}\n", encoding="ascii")
    sample = read_memory_sample(cgroup_v2=tmp_path / "nope", cgroup_v1=cg1, meminfo=tmp_path / "no")
    assert sample == MemorySample("cgroup-v1", 2 * GB, GB // 2, 2 * GB - GB // 2)


def test_read_memory_sample_falls_back_to_meminfo_when_unlimited(tmp_path):
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text("max\n", encoding="ascii")
    (cg2 / "memory.current").write_text("12345\n", encoding="ascii")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16299460 kB\nMemAvailable: 8000000 kB\n", encoding="ascii")
    sample = read_memory_sample(cgroup_v2=cg2, cgroup_v1=tmp_path / "nope", meminfo=meminfo)
    assert sample.source == "meminfo"
    assert sample.available_bytes == 8000000 * 1024


def test_read_memory_sample_unavailable_when_nothing_is_readable(tmp_path):
    sample = read_memory_sample(cgroup_v2=tmp_path / "a", cgroup_v1=tmp_path / "b", meminfo=tmp_path / "c")
    assert sample.source == "unavailable"
    assert sample.available_bytes is None
