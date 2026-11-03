"""One GPU lane: claim jobs and run their stages, each in a fresh engine process, on this GPU only.

A job is one method under one protocol. Its timed stages alternate AR and SD on the same card
(AR0, SD0, SD1, AR1, AR2, SD2 for three repeats), so every AR/SD pair shares hardware and a natural
SD job ends with an untimed ``stats`` pass for native acceptance counters. The untimed AR
``reference`` (history for multi-turn sessions) runs once per run before any timed stage.
Claims are atomic directories under ``claims/``; ``done/`` and ``failed/`` hold job outcomes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import read_json, save_json, stage_names
from research.qwen36_a100_seven_eval.stage import missing_topics, owner_of

STAGE_ATTEMPTS = 2


def job_list(manifest: dict) -> list[tuple[str, str]]:
    jobs = [
        (m, p) for m in manifest["methods"] for p in manifest["protocol"]["protocols"]
    ]
    if manifest["protocol"].get("ar_default", True):
        jobs += [("ar_default", p) for p in manifest["protocol"]["protocols"]]
    return jobs


def job_stages(manifest: dict, method: str, protocol: str) -> list[str]:
    stages = stage_names(manifest["protocol"]["repeats"])
    if method == "ar_default":
        return [s for s in stages if s.startswith("ar")]
    if protocol == "natural" and manifest["protocol"].get("acceptance_stats", True):
        stages.append("stats")
    return stages


def stage_done(
    run: Path, sessions: list[dict], method: str, protocol: str, stage: str
) -> bool:
    proto = "natural" if stage in ("reference", "stats") else protocol
    return not missing_topics(run, sessions, owner_of(method, stage), proto, stage)


def job_name(method: str, protocol: str) -> str:
    return f"{method}__{protocol}"


def run_stage(
    run: Path,
    gpu: int,
    method: str,
    protocol: str,
    stage: str,
    log: Path,
    site_hook: str | None,
) -> int:
    if site_hook:
        subprocess.run(shlex.split(site_hook) + [str(gpu)], check=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        LANE_GPU=str(gpu),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VELADRAFT_GPU_OWNERSHIP=os.environ.get("VELADRAFT_GPU_OWNERSHIP", "pid"),
        VELADRAFT_GPU_MAX_OWN_APPS="2",
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "research.qwen36_a100_seven_eval.stage",
        "--run-dir",
        str(run),
        "--method",
        method,
        "--protocol",
        protocol,
        "--stage",
        stage,
    ]
    with log.open("a") as stream:
        stream.write(
            f"\n=== {time.strftime('%F %T')} gpu={gpu} {method} {protocol} {stage}\n"
        )
        stream.flush()
        return subprocess.run(
            cmd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            pass_fds=(int(env["Q36_LAUNCHER_LOCK_FD"]),)
            if "Q36_LAUNCHER_LOCK_FD" in env
            else (),
        ).returncode


def ensure_reference(
    run: Path,
    gpu: int,
    sessions: list[dict],
    log: Path,
    site_hook: str | None,
    poll: int,
) -> bool:
    """Exactly one lane builds the reference; the others wait for it."""
    claim = run / "claims" / "_reference"
    while not stage_done(run, sessions, "ar", "natural", "reference"):
        try:
            claim.mkdir(parents=True)
        except FileExistsError:
            if (run / "failed" / "_reference").exists():
                return False
            time.sleep(poll)
            continue
        (claim / "owner").write_text(f"gpu {gpu} {time.strftime('%F %T')}\n")
        for _ in range(STAGE_ATTEMPTS):
            if run_stage(run, gpu, "ar", "natural", "reference", log, site_hook) == 0:
                break
        if not stage_done(run, sessions, "ar", "natural", "reference"):
            (run / "failed").mkdir(exist_ok=True)
            (run / "failed" / "_reference").write_text(
                "reference stage failed twice; see lane logs\n"
            )
            return False
    return True


def lane(run: Path, gpu: int, poll: int = 30) -> int:
    manifest = read_json(run / "manifest.json")
    sessions = read_json(run / "sessions.json")
    site_hook = manifest.get("site", {}).get("pre_stage_hook")
    log = run / "logs" / f"lane_gpu{gpu}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not ensure_reference(run, gpu, sessions, log, site_hook, poll):
        return 2
    for method, protocol in job_list(manifest):
        name = job_name(method, protocol)
        if (run / "done" / name).exists() or (run / "failed" / name).exists():
            continue
        binding = read_json(run / "bindings" / f"{name}.json")
        if binding and binding["index"] != gpu:
            continue
        try:
            (run / "claims" / name).mkdir(parents=True)
        except FileExistsError:
            continue
        (run / "claims" / name / "owner").write_text(
            f"gpu {gpu} {time.strftime('%F %T')}\n"
        )
        failed = None
        for stage in job_stages(manifest, method, protocol):
            for _ in range(STAGE_ATTEMPTS):
                if stage_done(run, sessions, method, protocol, stage):
                    break
                run_stage(run, gpu, method, protocol, stage, log, site_hook)
            if not stage_done(run, sessions, method, protocol, stage):
                failed = stage
                break
        outcome = run / ("failed" if failed else "done") / name
        outcome.parent.mkdir(exist_ok=True)
        save_json(
            outcome,
            {"gpu": gpu, "finished": time.strftime("%F %T"), "failed_stage": failed},
        )
    with log.open("a") as stream:
        stream.write(
            f"=== {time.strftime('%F %T')} lane gpu={gpu} found no more jobs\n"
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args(argv)
    return lane(args.run_dir, args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
