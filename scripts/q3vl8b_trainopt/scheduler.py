"""GPU-slot scheduler for one node of the Qwen3-VL-8B training-systems campaign.

    python3 scheduler.py --jobs jobs.json --state-dir <campaign>/raw/scheduler --gpus 0,1,2,3,4,5,6,7

``jobs.json`` is a list of ``{name, config, run_dir, gpus, pin, deps, after, wait_files, pre, post,
post_always, on_fail, extra_args}``; ``post`` hooks run after exit code 0, ``on_fail`` hooks after
any other, ``post_always`` hooks after both.
A job starts once every dependency finished with exit code 0 and enough GPUs are free
(two-GPU jobs get an aligned pair). When ``FLOOR_SCRIPT`` is set, a GPU without a job
carries that site process (``FLOOR_PY FLOOR_SCRIPT --gpu <g> --util 0.45``) until a job on
it reaches its training loop. Every job's GPU interval goes to ``gpu_time_ledger.jsonl``
and its state to ``jobs_state.json`` (a restarted scheduler skips finished jobs).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

# Node paths from the environment (TRAINOPT_ENV, sourced by launch_campaign.sh).
FLOOR_PY = os.environ.get("FLOOR_PY", "python3")
FLOOR = os.environ.get("FLOOR_SCRIPT", "")
LOCAL = Path(os.environ.get("VELA_LOCAL", "/tmp/trainopt"))
HERE = Path(__file__).resolve().parent
CORES_PER_GPU = 30
# Printed by train_dflash right before the training loop (after load, digest and checks).
TRAIN_LOOP_MARK = "train: save cadence"


def now() -> float:
    return time.time()


def say(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def pre_seed_from_init(job: dict, arg: str) -> None:
    """Start a run from the method's frozen step-0 state as an exact resume."""
    init_train = Path(arg) / "train"
    train = Path(job["run_dir"]) / "train"
    link(init_train / "init_state.pt", train / "ckpt_last.pt")
    link(init_train / "ckpt_init.pt", train / "ckpt_init.pt")
    # The exact-resume preflight reads the run's frozen contract (same target, same method).
    link(Path(arg) / "contract.json", Path(job["run_dir"]) / "contract.json")
    for name in ("visual_path_check.json", "swap_pair.pt"):
        source = Path(arg) / "online" / name
        if source.exists():
            link(source, Path(job["run_dir"]) / "online" / name)


def post_freeze_init(job: dict, _arg: str) -> None:
    train = Path(job["run_dir"]) / "train"
    link(train / "ckpt_last.pt", train / "init_state.pt")


def post_drop_recording(_job: dict, arg: str) -> None:
    """Free a replayed feature recording (node-local tmpfs)."""
    shutil.rmtree(LOCAL / "replay" / arg, ignore_errors=True)


HOOKS = {"seed_from_init": pre_seed_from_init, "freeze_init": post_freeze_init, "drop_recording": post_drop_recording}


def launcher_alive(pid: int | None) -> bool:
    try:
        return pid is not None and b"launch_run.sh" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def run_hooks(job: dict, key: str) -> None:
    for spec in job.get(key) or []:
        name, _, arg = spec.partition(":")
        HOOKS[name](job, arg)


class Scheduler:
    def __init__(self, jobs: list[dict], gpus: list[int], state_dir: Path):
        self.jobs = jobs
        self.gpus = gpus
        self.state_dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = state_dir / "jobs_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        for job in jobs:
            entry = self.state.setdefault(job["name"], {"status": "pending"})
            if entry["status"] == "running":
                # Jobs run in their own session and can outlive a killed scheduler.
                if launcher_alive(entry.get("pid")):
                    raise SystemExit(f"{job['name']} (pid {entry['pid']}) from a previous scheduler is still "
                                     "running; wait for it or stop it before restarting")
                entry["status"] = "pending"
        self.free = list(gpus)
        self.running: dict[str, tuple[subprocess.Popen, list[int], float]] = {}
        self.log_offsets: dict[str, int] = {}
        self.floors: dict[int, subprocess.Popen] = {}
        self.kill_stray_floors()

    def kill_stray_floors(self) -> None:
        """Remove floors a previous scheduler left on this pool's GPUs."""
        if not FLOOR:
            return
        name = Path(FLOOR).name
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5 and parts[2].endswith(name) and parts[3] == "--gpu" and int(parts[4]) in self.gpus:
                os.kill(int(parts[0]), signal.SIGTERM)

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def ledger(self, record: dict) -> None:
        with open(self.state_dir / "gpu_time_ledger.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- floors ------------------------------------------------------------
    def floor_on(self, gpu: int) -> None:
        if not FLOOR or (gpu in self.floors and self.floors[gpu].poll() is None):
            return
        log = open(LOCAL / f"floor_gpu{gpu}.log", "a")
        self.floors[gpu] = subprocess.Popen(
            [str(FLOOR_PY), str(FLOOR), "--gpu", str(gpu), "--util", "0.45"],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self.ledger({"event": "floor_on", "gpu": gpu, "time": now()})

    def floor_off(self, gpu: int) -> None:
        proc = self.floors.pop(gpu, None)
        if proc is None or proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        self.ledger({"event": "floor_off", "gpu": gpu, "time": now()})

    # -- jobs ----------------------------------------------------------------
    def allocate(self, count: int, pin: list[int] | None = None) -> list[int] | None:
        if pin:
            return list(pin) if all(gpu in self.free for gpu in pin) else None
        if count == 1:
            return [self.free[0]] if self.free else None
        for start in range(0, len(self.gpus) - count + 1, count):
            block = self.gpus[start : start + count]
            if all(gpu in self.free for gpu in block):
                return block
        return None

    def launch(self, job: dict, gpus: list[int]) -> None:
        # The GPUs' floors stay up through the job's startup (target load, data
        # digest, compile) and are removed once the job reaches its training loop.
        for gpu in gpus:
            self.free.remove(gpu)
            self.floor_on(gpu)
        try:
            run_hooks(job, "pre")
        except Exception as exc:
            say(f"PRE HOOK FAILED {job['name']}: {exc!r}")
            self.free.extend(gpus)
            self.free.sort()
            self.state[job["name"]].update(status="failed", rc=98, ended=now())
            self.save()
            return
        cpus = ",".join(f"{g * CORES_PER_GPU}-{g * CORES_PER_GPU + CORES_PER_GPU - 1}" for g in gpus)
        env = dict(os.environ, CPUS=cpus, THREADS=str(job.get("threads", 8)))
        cmd = ["bash", str(HERE / "launch_run.sh"), job["config"], job["run_dir"], ",".join(map(str, gpus))]
        cmd += list(job.get("extra_args") or [])
        Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
        stdout = Path(job["run_dir"]) / "stdout.log"
        self.log_offsets[job["name"]] = stdout.stat().st_size if stdout.exists() else 0
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
        started = now()
        self.running[job["name"]] = (proc, gpus, started)
        self.state[job["name"]].update(status="running", gpus=gpus, started=started, pid=proc.pid)
        self.save()
        say(f"START {job['name']} gpus={gpus}")

    def release_floors(self) -> None:
        """Drop a running job's startup floors right before its training loop starts."""
        for name, (_, gpus, _) in self.running.items():
            if not any(gpu in self.floors for gpu in gpus):
                continue
            job = next(job for job in self.jobs if job["name"] == name)
            log = Path(job["run_dir"]) / "stdout.log"
            try:
                # launch_run.sh appends every attempt to the same log: read this attempt only.
                with log.open("rb") as fh:
                    fh.seek(self.log_offsets.get(name, 0))
                    started_loop = TRAIN_LOOP_MARK in fh.read().decode(errors="replace")
            except OSError:
                started_loop = False
            if started_loop:
                for gpu in gpus:
                    self.floor_off(gpu)

    def reap(self) -> None:
        for name, (proc, gpus, started) in list(self.running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            ended = now()
            del self.running[name]
            job = next(job for job in self.jobs if job["name"] == name)
            if rc == 0:
                try:
                    run_hooks(job, "post")
                except Exception as exc:  # a failed hook fails the job
                    say(f"POST HOOK FAILED {name}: {exc!r}")
                    rc = 99
            try:
                if rc:
                    run_hooks(job, "on_fail")
                run_hooks(job, "post_always")
            except Exception as exc:
                say(f"CLEANUP HOOK FAILED {name}: {exc!r}")
            self.state[name].update(status="done" if rc == 0 else "failed", rc=rc, ended=ended)
            self.save()
            for gpu in gpus:
                self.ledger({"event": "job", "job": name, "gpu": gpu, "start": started, "end": ended,
                             "seconds": ended - started, "rc": rc})
            self.free.extend(gpus)
            self.free.sort()
            say(f"END {name} rc={rc} after {ended - started:.0f}s")

    def blocked(self, job: dict) -> bool:
        return any(self.state.get(dep, {}).get("status") == "failed" for dep in job.get("deps") or [])

    def ready(self, job: dict) -> bool:
        # ``wait_files``: snapshots a fork needs (written hours into a core run).
        # ``after``: run once those jobs ended, whatever their exit code (a capacity
        # point that runs out of memory is a result, not a failure of the chain).
        return (
            all(self.state.get(dep, {}).get("status") == "done" for dep in job.get("deps") or [])
            and all(self.state.get(dep, {}).get("status") in ("done", "failed", "blocked")
                    for dep in job.get("after") or [])
            and all(Path(path).exists() for path in job.get("wait_files") or [])
        )

    def reload(self, path: Path) -> None:
        """Pick up jobs appended to the jobs file while the scheduler runs."""
        mtime = path.stat().st_mtime
        if mtime == getattr(self, "_jobs_mtime", None):
            return
        self._jobs_mtime = mtime
        known = {job["name"] for job in self.jobs}
        for job in json.loads(path.read_text()):
            if job["name"] not in known:
                self.jobs.append(job)
                self.state.setdefault(job["name"], {"status": "pending"})
                say(f"ADDED {job['name']}")

    def run(self, jobs_path: Path | None = None) -> None:
        while True:
            if jobs_path is not None:
                self.reload(jobs_path)
            self.reap()
            self.release_floors()
            pending = [job for job in self.jobs if self.state[job["name"]]["status"] == "pending"]
            for job in pending:
                if self.blocked(job):
                    self.state[job["name"]]["status"] = "blocked"
                    self.save()
                    say(f"BLOCKED {job['name']}")
            for job in pending:
                if self.state[job["name"]]["status"] != "pending" or not self.ready(job):
                    continue
                gpus = self.allocate(int(job.get("gpus", 1)), job.get("pin"))
                if gpus is not None:
                    self.launch(job, gpus)
            for gpu in self.free:
                self.floor_on(gpu)
            if not self.running and not any(self.state[job["name"]]["status"] == "pending" for job in self.jobs):
                break
            time.sleep(10)
        # Floors stay up on exit; the next phase's scheduler replaces them.
        say("SCHEDULER_DONE " + json.dumps({job["name"]: self.state[job["name"]]["status"] for job in self.jobs}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(args.jobs.read_text())
    Scheduler(jobs, [int(g) for g in args.gpus.split(",")], args.state_dir).run(args.jobs)


if __name__ == "__main__":
    main()
