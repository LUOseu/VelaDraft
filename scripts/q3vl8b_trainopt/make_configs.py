"""Frozen configs of the Qwen3-VL-8B training-systems campaign (DFlash2 / DFly x Reference / Optimized).

Every config starts from the effective config of the matching row of the
Qwen3-VL-8B 200K matrix (same target, corpus, drafter geometry, objective and
optimizer) and changes only what the campaign contract fixes:

* layout: frozen target and drafter on one GPU (``single``) or split over two
  (``split``: drafter cuda:0, target cuda:1);
* execution switches per arm (never part of the objective or plan digest);
* micro-batch geometry at a constant global batch of 8;
* measurement: timing segments, full-state snapshots, dev/test validation;
* purpose-specific control (init state, pilot window, fork).

    python scripts/q3vl8b_trainopt/make_configs.py --old-task <200K matrix task dir> \
        --devtest <shared_data/devtest_regen> --out <campaign>/configs
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ARMS = {
    "dflash2": {
        "ref": {
            "dflash2_teacher_projection_reuse": False,
            "dflash2_scoring_row_compaction": False,
            "dflash2_fused_teacher_kl": False,
        },
        "d1": {
            "dflash2_teacher_projection_reuse": True,
            "dflash2_scoring_row_compaction": False,
            "dflash2_fused_teacher_kl": False,
        },
        "opt": {
            "dflash2_teacher_projection_reuse": True,
            "dflash2_scoring_row_compaction": True,
            "dflash2_fused_teacher_kl": True,
        },
    },
    "dfly": {
        "ref": {"recipe_row_compaction": False, "recipe_compile_tv_kl": False},
        "compaction": {"recipe_row_compaction": True, "recipe_compile_tv_kl": False},
        "opt": {"recipe_row_compaction": True, "recipe_compile_tv_kl": True},
    },
}
SNAPSHOT_STEPS = {
    "dflash2": [10000, 30000, 60000],
    "dfly": [2500, 10000, 15000, 18750, 30000, 45000, 60000],
}
GLOBAL_BATCH = 8
TRANSITION_STEP = 18750
# Paired windows (section 8.2) fork the Reference run's snapshots: three per method and stage.
WINDOW_SNAPSHOTS = {
    ("dflash2", "single_stage"): [10000, 30000, 60000],
    ("dfly", "stage1"): [2500, 10000, 15000],
    ("dfly", "stage2"): [30000, 45000, 60000],
}
WINDOW_ARMS = {
    ("dflash2", "single_stage"): ["ref", "d1", "opt"],
    ("dfly", "stage1"): ["ref", "compaction", "opt"],
    ("dfly", "stage2"): ["ref", "compaction"],  # e2e-TV never calls the compiled TV/KL path
}
SCALING_MICRO_BATCHES = [2, 4, 8]
WINDOW = {"warmup": 50, "measured": 500}
SCALING = {"warmup": 20, "measured": 200}
PILOT = {"warmup": 20, "measured": 50}
DRAFTER = {"warmup": 20, "measured": 200}  # drafter-only scope on recorded teacher features


def replay_dir(local: Path, method: str, stage: str, pair: int) -> Path:
    return local / "replay" / f"{method}_{stage}_p{pair}"


def snapshot(work: Path, method: str, step: int) -> Path:
    return work / f"fixed_config_{method}_s42_ref" / "train" / f"state_step{step:07d}.pt"


def bounded(start: int, protocol: dict, fork: Path, *, override: bool = False) -> dict:
    """Train overrides of a bounded segment: fork at ``start``, warmup, measured window, stop."""
    warm, measured = protocol["warmup"], protocol["measured"]
    out = {
        "fork_state": str(fork),
        "stop_at_step": start + warm + measured,
        "timing_windows": [[start + warm, start + warm + measured]],
        "save_every_steps": 0,
        "save_final_state": False,
    }
    if override and start:
        out["fork_step_override"] = start
    return out


def validation_block(devtest: Path, method: str) -> dict:
    return {
        "enabled": True,
        "rows": str(devtest / "regen" / "target_data.jsonl"),
        "meta": str(devtest / "data" / "meta.json"),
        "splits": {"dev": str(devtest / "data" / "split_dev.jsonl"), "test": str(devtest / "data" / "split_test.jsonl")},
        "every_steps": 2500,
        "extra_steps": [TRANSITION_STEP] if method == "dfly" else [],
        "at_start": True,
        "at_epoch_end": True,
        "batch_size": 2,
        "seed": 20261030,
    }


def init_state(work: Path, method: str) -> Path:
    """The immutable copy of a method's seed-42 step-0 state (written by launch_campaign.sh)."""
    return work / f"init_{method}_s42" / "train" / "init_state.pt"


def build(base: dict, method: str, arm: str, *, name: str, purpose: str, layout: str = "single",
          micro_batch: int = 2, devtest: Path | None = None, algorithm: dict | None = None,
          **train_overrides) -> dict:
    cfg = copy.deepcopy(base)
    cfg["drafter"].update(ARMS[method][arm])
    if algorithm:
        cfg["algorithm"][method].update(algorithm)
    cfg["target"]["device"] = "cuda:0" if layout == "single" else "cuda:1"
    train = cfg["train"]
    train["device"] = "cuda:0"
    train["micro_batch_size"] = int(micro_batch)
    train["grad_accumulation_steps"] = GLOBAL_BATCH // int(micro_batch)
    train.pop("stage_timing", None)
    train["milestone_snapshots"] = False
    train["save_every_steps"] = 1250
    train["timing_segment_every"] = 250
    train["overfit_sanity"]["enabled"] = purpose == "init"
    if purpose in ("core", "best"):
        train["snapshot_steps"] = SNAPSHOT_STEPS[method]
        cfg["validation"] = validation_block(devtest, method)
    if purpose == "init":
        train["stop_at_step"] = 0
    if purpose in ("pilot", "numcheck", "window", "scaling"):
        # Forks reuse the init run's visual-path evidence instead of recomputing it.
        cfg["capture"]["visual_path_check"] = False
    train.update(train_overrides)
    cfg["run"] = {"name": name, "dir": name}
    cfg["campaign"] = {"method": method, "arm": arm, "purpose": purpose, "layout": layout,
                       "micro_batch_size": int(micro_batch), "switches": ARMS[method][arm]}
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-task", required=True, type=Path)
    parser.add_argument("--devtest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path, help="campaign run directories (work/<run name>)")
    parser.add_argument("--local", required=True, type=Path, help="node-local scratch (feature recordings)")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bases = {method: json.loads((args.old_task / "runs" / method / "config.json").read_text()) for method in ARMS}
    for method, base in bases.items():
        if base["algorithm"]["name"] != method:
            raise SystemExit(f"{method}: base config is algorithm {base['algorithm']['name']!r}")
    written = []

    def write(rel: str, cfg: dict) -> None:
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        written.append(rel)

    for method, base in bases.items():
        # One seed-42 initial state per method; both arms start from it.
        write(f"init/{method}.json", build(base, method, "ref", name=f"init_{method}_s42", purpose="init"))
        for arm in ("ref", "opt"):
            write(
                f"formal/fixed_config/{method}_s42_{arm}.json",
                build(base, method, arm, name=f"fixed_config_{method}_s42_{arm}", purpose="core", devtest=args.devtest),
            )
        # Layout pilots: 20 warmup + 50 measured updates per arm, stage and layout.
        stages = {"dflash2": {"single_stage": 0}, "dfly": {"stage1": 0, "stage2": TRANSITION_STEP}}[method]
        for stage, start in stages.items():
            for arm in ARMS[method]:
                for layout in ("single", "split"):
                    overrides = {"stop_at_step": start + 70, "timing_windows": [[start + 20, start + 70]],
                                 "save_every_steps": 0, "save_final_state": False}
                    overrides["fork_state"] = str(init_state(args.work, method))
                    if start:
                        overrides["fork_step_override"] = start
                    write(
                        f"pilot/{method}_{stage}_{arm}_{layout}.json",
                        build(base, method, arm, name=f"pilot_{method}_{stage}_{arm}_{layout}", purpose="pilot",
                              layout=layout, **overrides),
                    )
        # Numerical check: one real update from the same state, every arm (section 10).
        for stage, start in stages.items():
            # ``ref_repeat`` reruns Reference: the run-to-run noise floor of one bf16 update.
            for arm in [*ARMS[method], "ref_repeat"]:
                overrides = {"stop_at_step": start + 1, "save_every_steps": 0,
                             "fork_state": str(init_state(args.work, method))}
                if start:
                    overrides["fork_step_override"] = start
                write(
                    f"numcheck/{method}_{stage}_{arm}.json",
                    build(base, method, "ref" if arm == "ref_repeat" else arm, name=f"numcheck_{method}_{stage}_{arm}",
                          purpose="numcheck", **overrides),
                )
        # Early probes from the initial state (no snapshot needed): D2 chunk memory
        # pilot and micro-batch capacity per arm and stage.
        for stage, start in stages.items():
            for arm in ("ref", "opt"):
                for mb in (4, 8):
                    write(
                        f"probe/{method}_{stage}_{arm}_mb{mb}.json",
                        build(base, method, arm, name=f"probe_{method}_{stage}_{arm}_mb{mb}", purpose="pilot",
                              micro_batch=mb, **bounded(start, PILOT, init_state(args.work, method), override=True)),
                    )
        if method == "dflash2":
            for chunk in (256, 512, 1024):
                write(
                    f"probe/dflash2_opt_chunk{chunk}.json",
                    build(base, method, "opt", name=f"probe_dflash2_opt_chunk{chunk}", purpose="pilot",
                          algorithm={"kd_chunk_size": chunk}, **bounded(0, PILOT, init_state(args.work, method))),
                )
    for (method, stage), steps in WINDOW_SNAPSHOTS.items():
        base = bases[method]
        for pair, step in enumerate(steps, 1):
            fork = snapshot(args.work, method, step)
            for arm in WINDOW_ARMS[(method, stage)]:
                write(
                    f"window/{method}_{stage}_p{pair}_{arm}.json",
                    build(base, method, arm, name=f"window_{method}_{stage}_p{pair}_{arm}", purpose="window",
                          **bounded(step, WINDOW, fork)),
                )
            # Drafter training update scope: one recording per snapshot, replayed by every arm.
            record_dir = replay_dir(args.local, method, stage, pair)
            write(
                f"replay/{method}_{stage}_p{pair}_record.json",
                build(base, method, "ref", name=f"replay_{method}_{stage}_p{pair}_record", purpose="window",
                      feature_replay={"mode": "record", "dir": str(record_dir),
                                      "microbatches": (DRAFTER["warmup"] + DRAFTER["measured"]) * GLOBAL_BATCH // 2},
                      **bounded(step, DRAFTER, fork)),
            )
            for arm in WINDOW_ARMS[(method, stage)]:
                write(
                    f"replay/{method}_{stage}_p{pair}_{arm}.json",
                    build(base, method, arm, name=f"replay_{method}_{stage}_p{pair}_{arm}", purpose="window",
                          feature_replay={"mode": "replay", "dir": str(record_dir)}, **bounded(step, DRAFTER, fork)),
                )
            for arm in ("ref", "opt"):
                for mb in SCALING_MICRO_BATCHES:
                    write(
                        f"scaling/{method}_{stage}_p{pair}_{arm}_mb{mb}.json",
                        build(base, method, arm, name=f"scaling_{method}_{stage}_p{pair}_{arm}_mb{mb}",
                              purpose="scaling", micro_batch=mb, **bounded(step, SCALING, fork)),
                    )
    (out / "configs_index.json").write_text(json.dumps(sorted(written), indent=1) + "\n")
    print(f"wrote {len(written)} configs to {out}")


if __name__ == "__main__":
    main()
