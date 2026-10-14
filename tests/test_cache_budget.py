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


class FakeMemoryEnv:
    """Closed-loop memory model: used = other usage + the cache's own payload.

    A real cgroup probe already excludes the cache's retained bytes from
    ``available``; this couples the fake reading to the retained payload the
    same way, so the controller cannot shrink the cache by charging its own
    residency against its own budget (review finding: the old
    ``available - reserve`` formula oscillated under a constant external load).
    """

    def __init__(self, limit_bytes: int, other_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        self.other_bytes = other_bytes
        self.retained = 0

    def sample(self) -> MemorySample:
        used = self.other_bytes + self.retained
        return MemorySample("cgroup-v2", self.limit_bytes, used, self.limit_bytes - used)


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


def test_closed_loop_does_not_shrink_the_cache_against_its_own_residency():
    # Review finding: 16 GiB limit, 4 GiB fixed non-cache usage, the cache
    # already holds 8 GiB. The old ``available - reserve`` formula treated the
    # 4 GiB of remaining headroom as the cache's whole cap (2.4 GiB), evicted
    # 5.6 GiB of resident data and then oscillated while the external load
    # stayed constant. With the retained payload added back, the cache keeps
    # its 8 GiB and the budget settles at the safe cap.
    env = FakeMemoryEnv(16 * GB, other_bytes=4 * GB)
    clock = FakeClock()
    controller = CacheBudgetController(16 * GB, sensor=env.sample, clock=clock)
    reserve = int(0.10 * 16 * GB)
    expected = 12 * GB - reserve
    budgets = []
    for _ in range(10):  # stores arrive 1 GiB at a time up to the reviewer's 8 GiB
        env.retained = min(8 * GB, env.retained + 1 * GB)
        clock.advance(1.0)
        budget = controller.observe(retained_bytes=env.retained)
        env.retained = min(env.retained, budget)  # the caller trims to the budget
        budgets.append(budget)
    assert env.retained == 8 * GB  # no eviction of the resident cache
    assert set(budgets[-3:]) == {expected}  # settled, no oscillation
    assert controller.adjustments == 1  # only the startup right-size: no per-cycle churn
    assert env.limit_bytes - env.other_bytes - env.retained >= reserve  # free headroom kept
    assert controller.stats()["retained_bytes"] == 8 * GB


def test_closed_loop_eviction_yields_exactly_the_missing_safety_margin():
    env = FakeMemoryEnv(16 * GB, other_bytes=4 * GB)
    clock = FakeClock()
    controller = CacheBudgetController(16 * GB, sensor=env.sample, clock=clock)
    reserve = int(0.10 * 16 * GB)
    expected = 12 * GB - reserve
    for _ in range(16):  # fill the cache to its budget
        env.retained = min(expected, env.retained + 1 * GB)
        clock.advance(1.0)
        env.retained = min(env.retained, controller.observe(retained_bytes=env.retained))
    assert env.retained == expected
    assert env.limit_bytes - env.other_bytes - env.retained == reserve  # exactly the reserve
    # External pressure rises by 800 MiB: the cache yields exactly that much.
    missing = 800 * MB
    env.other_bytes += missing
    clock.advance(1.0)
    budget = controller.observe(retained_bytes=env.retained)
    evicted = env.retained - min(env.retained, budget)
    env.retained -= evicted
    assert evicted == missing
    assert env.retained == expected - missing
    assert env.limit_bytes - env.other_bytes - env.retained == reserve
    # And the loop is at a fixed point: no further adjustments or evictions.
    adjustments = controller.adjustments
    for _ in range(5):
        clock.advance(1.0)
        budget = controller.observe(retained_bytes=env.retained)
        evicted = env.retained - min(env.retained, budget)
        env.retained -= evicted
        assert evicted == 0
    assert controller.adjustments == adjustments


def test_read_memory_sample_prefers_cgroup_v2(tmp_path):
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text(f"{4 * GB}\n", encoding="ascii")
    (cg2 / "memory.current").write_text(f"{1 * GB}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "nope1", proc_cgroup=proc, meminfo=tmp_path / "nope2"
    )
    assert sample == MemorySample("cgroup-v2", 4 * GB, 1 * GB, 3 * GB)


def test_read_memory_sample_uses_cgroup_v1_when_v2_is_absent(tmp_path):
    cg1 = tmp_path / "cg1"
    cg1.mkdir()
    (cg1 / "memory.limit_in_bytes").write_text(f"{2 * GB}\n", encoding="ascii")
    (cg1 / "memory.usage_in_bytes").write_text(f"{GB // 2}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("7:memory:/\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=tmp_path / "nope", cgroup_v1=cg1, proc_cgroup=proc, meminfo=tmp_path / "no"
    )
    assert sample == MemorySample("cgroup-v1", 2 * GB, GB // 2, 2 * GB - GB // 2)


def test_read_memory_sample_falls_back_to_meminfo_when_unlimited(tmp_path):
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text("max\n", encoding="ascii")
    (cg2 / "memory.current").write_text("12345\n", encoding="ascii")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16299460 kB\nMemAvailable: 8000000 kB\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "nope", proc_cgroup=proc, meminfo=meminfo
    )
    assert sample.source == "meminfo"
    assert sample.available_bytes == 8000000 * 1024


def test_read_memory_sample_unavailable_when_nothing_is_readable(tmp_path):
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=tmp_path / "a", cgroup_v1=tmp_path / "b", proc_cgroup=proc, meminfo=tmp_path / "c"
    )
    assert sample.source == "unavailable"
    assert sample.available_bytes is None


def test_read_memory_sample_resolves_the_process_cgroup_path(tmp_path):
    # Review finding (P1): the mount root is not the job under systemd / Slurm /
    # container nesting. The old probe read /sys/fs/cgroup directly, saw the
    # root's "max"/absent limit and fell back to the host's 64 GiB
    # MemAvailable while the job actually had 0.5 GiB left.
    cg2 = tmp_path / "cg2"
    job = cg2 / "system.slice" / "d4-job.scope"
    job.mkdir(parents=True)
    (cg2 / "memory.max").write_text("max\n", encoding="ascii")
    (cg2 / "memory.current").write_text(f"{3 * GB}\n", encoding="ascii")
    (job / "memory.max").write_text(f"{GB // 2}\n", encoding="ascii")
    (job / "memory.current").write_text(f"{GB // 8}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/system.slice/d4-job.scope\n", encoding="ascii")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal: {128 * GB // 1024} kB\nMemAvailable: {64 * GB // 1024} kB\n", encoding="ascii"
    )
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "nope", proc_cgroup=proc, meminfo=meminfo
    )
    assert sample.source == "cgroup-v2"
    assert sample.limit_bytes == GB // 2
    assert sample.used_bytes == GB // 8
    assert sample.available_bytes == GB // 2 - GB // 8  # the job's headroom, not 64 GiB
    assert sample.host_available_bytes == 64 * GB


def test_read_memory_sample_honours_ancestor_limits(tmp_path):
    # The job's own cgroup may be unlimited while a parent (a Slurm job or a
    # container scope) carries the binding limit.
    cg2 = tmp_path / "cg2"
    leaf = cg2 / "job" / "leaf"
    leaf.mkdir(parents=True)
    (cg2 / "memory.max").write_text("max\n", encoding="ascii")
    (cg2 / "job" / "memory.max").write_text(f"{2 * GB}\n", encoding="ascii")
    (cg2 / "job" / "memory.current").write_text(f"{int(1.5 * GB)}\n", encoding="ascii")
    (leaf / "memory.max").write_text("max\n", encoding="ascii")
    (leaf / "memory.current").write_text(f"{GB // 4}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/job/leaf\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "no", proc_cgroup=proc, meminfo=tmp_path / "no"
    )
    assert sample.source == "cgroup-v2"
    assert (sample.limit_bytes, sample.used_bytes) == (2 * GB, int(1.5 * GB))
    assert sample.available_bytes == 2 * GB - int(1.5 * GB)


def test_read_memory_sample_uses_proc_membership_for_cgroup_v1(tmp_path):
    cg1 = tmp_path / "cg1"
    leaf = cg1 / "job"
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text(f"{GB}\n", encoding="ascii")
    (leaf / "memory.usage_in_bytes").write_text(f"{GB // 4}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("7:memory:/job\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=tmp_path / "no", cgroup_v1=cg1, proc_cgroup=proc, meminfo=tmp_path / "no"
    )
    assert sample == MemorySample("cgroup-v1", GB, GB // 4, 3 * GB // 4)


def test_read_memory_sample_combines_cgroup_with_host_memavailable(tmp_path):
    # A finite cgroup reading must not mask a machine-wide squeeze.
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text(f"{8 * GB}\n", encoding="ascii")
    (cg2 / "memory.current").write_text(f"{GB}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal: {64 * GB // 1024} kB\nMemAvailable: {2 * GB // 1024} kB\n", encoding="ascii"
    )
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "no", proc_cgroup=proc, meminfo=meminfo
    )
    assert sample.source == "cgroup-v2"
    assert sample.available_bytes == 2 * GB  # host headroom wins when smaller
    assert sample.host_available_bytes == 2 * GB
    assert sample.limit_bytes == 8 * GB  # cgroup identity kept


def test_read_memory_sample_reports_over_limit_headroom_as_negative(tmp_path):
    cg2 = tmp_path / "cg2"
    cg2.mkdir()
    (cg2 / "memory.max").write_text(f"{GB}\n", encoding="ascii")
    (cg2 / "memory.current").write_text(f"{GB + GB // 4}\n", encoding="ascii")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/\n", encoding="ascii")
    sample = read_memory_sample(
        cgroup_v2=cg2, cgroup_v1=tmp_path / "no", proc_cgroup=proc, meminfo=tmp_path / "no"
    )
    assert sample.available_bytes == -(GB // 4)  # over the limit: trim the shortfall
