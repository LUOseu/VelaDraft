"""Job lists of the campaign phases (list order is the scheduling priority).

    python3 jobs.py <phase> <campaign_dir> > jobs_<phase>.json

* ``prep``: the two seed-42 initial states, the numerical check (one real update
  from the same state, every arm and stage) and the layout pilots (20 warmup +
  50 measured updates, Reference/Optimized, single-GPU and split layout).
* ``core``: the four fixed_config runs (200K x 3 epochs), started as exact
  resumes of the method's frozen step-0 state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from make_configs import WINDOW_ARMS, WINDOW_SNAPSHOTS, snapshot

STAGES = {"dflash2": ["single_stage"], "dfly": ["stage1", "stage2"]}


def prep(camp: Path) -> list[dict]:
    cfg, work = camp / "configs", camp / "work"
    jobs = [
        {"name": f"init_{m}", "config": str(cfg / "init" / f"{m}.json"), "run_dir": str(work / f"init_{m}_s42"),
         "gpus": 1, "post": ["freeze_init"]}
        for m in STAGES
    ]
    for m, stages in STAGES.items():
        for stage in stages:
            for arm in ("ref", "opt"):
                name = f"numcheck_{m}_{stage}_{arm}"
                jobs.append({"name": name, "config": str(cfg / "numcheck" / f"{m}_{stage}_{arm}.json"),
                             "run_dir": str(work / name), "gpus": 1, "deps": [f"init_{m}"]})
    for layout, gpus in (("single", 1), ("split", 2)):
        for m, stages in STAGES.items():
            for stage in stages:
                for arm in ("ref", "opt"):
                    name = f"pilot_{m}_{stage}_{arm}_{layout}"
                    jobs.append({"name": name, "config": str(cfg / "pilot" / f"{m}_{stage}_{arm}_{layout}.json"),
                                 "run_dir": str(work / name), "gpus": gpus, "deps": [f"init_{m}"]})
    return jobs


def core(camp: Path) -> list[dict]:
    cfg, work = camp / "configs", camp / "work"
    jobs = []
    for m in STAGES:
        for arm in ("ref", "opt"):
            name = f"fixed_config_{m}_s42_{arm}"
            jobs.append({"name": name, "config": str(cfg / "formal" / "fixed_config" / f"{m}_s42_{arm}.json"),
                         "run_dir": str(work / name), "gpus": 1, "extra_args": ["--resume"],
                         "pre": [f"seed_from_init:{work / f'init_{m}_s42'}"]})
    return jobs


def _lane(gpu: int, items: list[tuple[str, str, list[str]]], camp: Path) -> list[dict]:
    """Jobs pinned to one GPU and run one after another (``after``), whatever each exit code."""
    cfg, work = camp / "configs", camp / "work"
    jobs, previous = [], None
    for name, rel, wait in items:
        job = {"name": name, "config": str(cfg / rel), "run_dir": str(work / name), "gpus": 1, "pin": [gpu],
               "wait_files": wait}
        if previous:
            job["after"] = [previous]
        jobs.append(job)
        previous = name
    return jobs


def _snap(camp: Path, method: str, step: int) -> str:
    return str(snapshot(camp / "work", method, step))


def main_phase(camp: Path) -> list[dict]:
    """Core runs on GPUs 0-3 and the four system lanes on GPUs 4-7 (contract table 3.2)."""
    jobs = core(camp)
    for gpu, job in zip(range(4), jobs):
        job["pin"] = [gpu]
    # The intermediate ablation arms' numerical checks (prep ran Reference and Optimized).
    numchecks = {
        4: [(f"numcheck_dflash2_single_stage_{arm}", f"numcheck/dflash2_single_stage_{arm}.json", [])
            for arm in ("d1", "ref_repeat")],
        5: [(f"numcheck_dfly_{stage}_{arm}", f"numcheck/dfly_{stage}_{arm}.json", [])
            for stage in ("stage1", "stage2") for arm in ("compaction", "ref_repeat")],
    }
    probes = {
        4: numchecks[4] + [(f"probe_dflash2_opt_chunk{c}", f"probe/dflash2_opt_chunk{c}.json", [])
                           for c in (256, 512, 1024)],
        5: numchecks[5] + [(f"probe_dflash2_single_stage_{a}_mb{mb}", f"probe/dflash2_single_stage_{a}_mb{mb}.json", [])
                           for a in ("ref", "opt") for mb in (4, 8)],
        6: [(f"probe_dfly_stage1_{a}_mb{mb}", f"probe/dfly_stage1_{a}_mb{mb}.json", []) for a in ("ref", "opt")
            for mb in (4, 8)],
        7: [(f"probe_dfly_stage2_{a}_mb{mb}", f"probe/dfly_stage2_{a}_mb{mb}.json", []) for a in ("ref", "opt")
            for mb in (4, 8)],
    }
    windows, scaling = {}, {}
    for (method, stage), steps in WINDOW_SNAPSHOTS.items():
        for pair, step in enumerate(steps, 1):
            arms = WINDOW_ARMS[(method, stage)]
            # Alternate the ON/OFF order from pair to pair on the same GPU.
            order = arms if pair % 2 else list(reversed(arms))
            windows.setdefault(method, []).extend(
                (f"window_{method}_{stage}_p{pair}_{arm}", f"window/{method}_{stage}_p{pair}_{arm}.json",
                 [_snap(camp, method, step)]) for arm in order
            )
            # Drafter-only scope right after the online windows of the same snapshot.
            windows[method].append((f"replay_{method}_{stage}_p{pair}_record",
                                    f"replay/{method}_{stage}_p{pair}_record.json", [_snap(camp, method, step)]))
            windows[method].extend(
                (f"replay_{method}_{stage}_p{pair}_{arm}", f"replay/{method}_{stage}_p{pair}_{arm}.json", [])
                for arm in order
            )
            scaling.setdefault(method, []).extend(
                (f"scaling_{method}_{stage}_p{pair}_{arm}_mb{mb}",
                 f"scaling/{method}_{stage}_p{pair}_{arm}_mb{mb}.json", [_snap(camp, method, step)])
                for mb in (2, 4, 8) for arm in (("ref", "opt") if pair % 2 else ("opt", "ref"))
            )
    lanes = {4: probes[4] + windows["dflash2"], 5: probes[5] + windows["dfly"],
             6: probes[6] + scaling["dflash2"], 7: probes[7] + scaling["dfly"]}
    for gpu, items in lanes.items():
        jobs += _lane(gpu, items, camp)
    # Replay arms need their snapshot's complete recording. The last replay arm frees the
    # recording (tens of GB of node-local memory) whatever its exit code, and a failed
    # recording frees itself, since its replay arms are then blocked.
    last_arm: dict[str, dict] = {}
    for job in jobs:
        if job["name"].startswith("replay_") and not job["name"].endswith("_record"):
            prefix = job["name"].rsplit("_", 1)[0]
            job["deps"] = [prefix + "_record"]
            last_arm[prefix] = job
    for prefix, job in last_arm.items():
        drop = [f"drop_recording:{prefix[len('replay_'):]}"]
        job["post_always"] = drop
        next(j for j in jobs if j["name"] == prefix + "_record")["on_fail"] = drop
    return jobs


if __name__ == "__main__":
    phase, camp = sys.argv[1], Path(sys.argv[2])
    print(json.dumps({"prep": prep, "core": core, "main": main_phase}[phase](camp), indent=1))
