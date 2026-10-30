"""Failure paths of the training-systems campaign scheduler (scripts/q3vl8b_trainopt)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "q3vl8b_trainopt"))
import scheduler  # noqa: E402


def _job(tmp_path, name, **extra):
    return {"name": name, "config": "unused.json", "run_dir": str(tmp_path / name), "gpus": 1, **extra}


def test_failing_pre_hook_fails_only_its_job(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "FLOOR", "")
    job = _job(tmp_path, "a", pre=[f"seed_from_init:{tmp_path / 'missing_init'}"])
    sched = scheduler.Scheduler([job], [0, 1], tmp_path / "state")
    sched.launch(job, [0])
    assert sched.state["a"]["status"] == "failed" and sched.state["a"]["rc"] == 98
    assert sched.free == [0, 1] and not sched.running


def test_restart_refuses_a_job_that_is_still_running(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "jobs_state.json").write_text('{"a": {"status": "running", "pid": 4242}}')
    monkeypatch.setattr(scheduler, "launcher_alive", lambda pid: pid == 4242)
    with pytest.raises(SystemExit, match="still running"):
        scheduler.Scheduler([_job(tmp_path, "a")], [0], state)
    monkeypatch.setattr(scheduler, "launcher_alive", lambda pid: False)
    assert scheduler.Scheduler([_job(tmp_path, "a")], [0], state).state["a"]["status"] == "pending"


def test_cleanup_hooks_run_on_failure_and_after_waits_end_on_blocked(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setitem(scheduler.HOOKS, "note", lambda job, arg: ran.append((job["name"], arg)))
    failed = _job(tmp_path, "rec", on_fail=["note:fail"], post_always=["note:always"], post=["note:post"])
    sched = scheduler.Scheduler([failed], [0], tmp_path / "state")
    sched.free.remove(0)
    sched.running["rec"] = (SimpleNamespace(poll=lambda: 3), [0], 0.0)
    sched.reap()
    assert ran == [("rec", "fail"), ("rec", "always")] and sched.state["rec"]["status"] == "failed"
    sched.state["arm"] = {"status": "blocked"}
    assert sched.ready(_job(tmp_path, "next", after=["arm"]))


def test_floor_release_reads_only_the_current_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "FLOOR", "")
    job = _job(tmp_path, "a")
    run_dir = Path(job["run_dir"])
    run_dir.mkdir()
    (run_dir / "stdout.log").write_text(f"attempt 1\n{scheduler.TRAIN_LOOP_MARK} 2500\n")
    sched = scheduler.Scheduler([job], [0], tmp_path / "state")
    dropped = []
    sched.floors = {0: object()}
    monkeypatch.setattr(sched, "floor_off", dropped.append)
    sched.log_offsets["a"] = (run_dir / "stdout.log").stat().st_size
    sched.running["a"] = (None, [0], 0.0)
    sched.release_floors()
    assert dropped == []
    with (run_dir / "stdout.log").open("a") as fh:
        fh.write(f"attempt 2\n{scheduler.TRAIN_LOOP_MARK} 2500\n")
    sched.release_floors()
    assert dropped == [0]
