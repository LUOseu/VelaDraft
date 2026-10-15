#!/usr/bin/env python3
"""Config-driven A100 experiment matrix runner (VelaDraft convergence round).

Subcommands
    dry-run    validate the matrix (algorithms, configs, data paths), print the
               resolved per-entry command plan + rough GPU-hour estimate; writes
               ``<runs_root>/matrix_plan.json``. No GPU is touched.
    run        execute pending stages of every entry across a GPU pool (one
               serial worker per GPU, >=2 GPUs by default guarded via nvidia-smi).
    resume     same as run, but train legs with an existing ``ckpt_last.pt``
               continue with ``--resume`` instead of restarting.
    summarize  aggregate train/eval/benchmark artifacts into summary.{json,csv,md}.

Idempotence: state is appended to ``<runs_root>/matrix_state.jsonl``; a stage is
skipped when its marker file exists AND the state has a successful record for it.
Failed entries never block other entries.

Isolation: the runner refuses to start when the matrix' train store resolves to
the val split path, and benchmark workloads are declared in ``isolation`` only
(public test sets, never the training slice).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

STAGE_MARKERS = {
    "train": ("train/train_summary.json",),
    "eval": ("eval/eval_result.json",),
    "benchmark": None,  # per-workload reports, handled specially
}


# --------------------------------------------------------------------------- #
# matrix loading
# --------------------------------------------------------------------------- #
def expand(value, env):
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    if isinstance(value, str):
        def sub(match):
            name = match.group(1)
            if name not in env:
                raise KeyError(f"environment variable {name} is not set (needed for matrix paths)")
            return env[name]
        return re.sub(r"\$\{([A-Z0-9_]+)\}", sub, value)
    return value


def load_matrix(path: Path, cli_env: dict[str, str]) -> dict:
    raw = json.loads(path.read_text())
    matrix = expand(raw, cli_env)
    for key in ("repo", "python", "runs_root", "controlled_config", "drafters", "budget"):
        if key not in matrix:
            raise SystemExit(f"matrix is missing required key {key!r}")
    is_ = matrix.get("isolation", {})
    if is_.get("train_store") and is_.get("val_split") and \
            os.path.realpath(is_["train_store"]) == os.path.realpath(is_["val_split"]):
        raise SystemExit("isolation violation: train_store == val_split path")
    return matrix


def build_entries(matrix: dict) -> list[dict]:
    steps = int(matrix["budget"]["steps"])
    seeds = [int(s) for s in matrix["budget"]["seeds"]]
    tracks = matrix.get("tracks", ["controlled"])
    regroup_sets = matrix["regroup_sets"]
    entries = []
    for drafter, info in matrix["drafters"].items():
        for track in tracks:
            cfg = info["native_config"] if track == "native" else matrix["controlled_config"]
            modes = ["off", "grid2"] if info.get("regroup_support") else ["off"]
            for seed in seeds:
                for mode in modes:
                    entries.append({
                        "id": f"{drafter}__{track}__s{seed}__rg{mode}",
                        "drafter": drafter,
                        "track": track,
                        "seed": seed,
                        "regroup": mode,
                        "config": cfg,
                        "steps": steps,
                        "run_dir": str(Path(matrix["runs_root"]) / f"{drafter}__{track}__s{seed}__rg{mode}"),
                        "sets": [f"train.seed={seed}", *regroup_sets[mode]],
                        "adaptation_required": bool(info.get("regroup_adaptation", False)) and mode != "off",
                    })
    entries.sort(key=lambda e: e["id"])
    return entries


def stage_commands(matrix: dict, entry: dict, stage: str, resume_train: bool) -> list[list[str]]:
    py, repo = matrix["python"], matrix["repo"]
    run_dir, cfg = entry["run_dir"], entry["config"]
    if stage == "train":
        cmd = [py, "scripts/train_dflash.py", "--config", cfg, "--run-dir", run_dir,
               "--steps", str(entry["steps"]), "--skip-sanity"]
        if resume_train and (Path(run_dir) / "train" / "ckpt_last.pt").exists():
            cmd.append("--resume")
        for item in entry["sets"]:
            cmd += ["--set", item]
        return [cmd]
    if stage == "eval":
        ev = matrix.get("eval", {})
        return [[py, "scripts/eval_dflash.py", "--config", cfg, "--run-dir", run_dir,
                 "--num-prompts", str(ev.get("num_prompts", 64)),
                 "--set", f"eval.max_new_tokens={ev.get('max_new_tokens', 128)}"]]
    if stage == "benchmark":
        return [[py, "scripts/benchmark.py", "--run-dir", run_dir, "--workload", w,
                 "--name", f"benchmark_{Path(run_dir).name}"]
                for w in matrix.get("isolation", {}).get("test_workloads", [])]
    raise ValueError(stage)


def stage_done(matrix: dict, entry: dict, stage: str) -> bool:
    run_dir = Path(entry["run_dir"])
    if stage == "train":
        return (run_dir / "train" / "train_summary.json").exists()
    if stage == "eval":
        return (run_dir / "eval" / "eval_result.json").exists()
    if stage == "benchmark":
        reports = Path(matrix["repo"]) / "benchmarks" / "reports"
        return all((reports / f"benchmark_{run_dir.name}_{w}.json").exists()
                   or (reports / f"benchmark_{run_dir.name}.json").exists()
                   for w in matrix.get("isolation", {}).get("test_workloads", []))
    raise ValueError(stage)


# --------------------------------------------------------------------------- #
# device guard + execution
# --------------------------------------------------------------------------- #
def nvidia_query(gpu: int) -> tuple[int, int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu),
             "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        util, used, free = (int(v.strip()) for v in out.split(","))
        return util, used, free
    except Exception:
        return None


def home_free_gb(path: str) -> int:
    try:
        out = subprocess.run(["df", "-BG", "--output=avail", path], capture_output=True, text=True, timeout=20)
        return int(re.sub(r"[^0-9]", "", out.stdout.splitlines()[-1]) or 0)
    except Exception:
        return 0


def guard(gpu: int, min_free_mb: int, min_disk_gb: int, timeout_s: int, log) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        q = nvidia_query(gpu)
        disk = home_free_gb(os.path.expanduser("~"))
        if q is not None:
            util, used, free = q
            if util <= 5 and (used <= 200 or free >= min_free_mb) and free >= min_free_mb and disk >= min_disk_gb:
                return True
            log(f"gpu{gpu} guard wait util={util}% used={used}MiB free={free}MiB disk={disk}G")
        time.sleep(30)
    return False


def run_entry(matrix: dict, entry: dict, gpu: int, statef, state_lock, resume_train: bool,
              guard_timeout: int, log) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = matrix["repo"]
    env.update({k: str(v) for k, v in matrix.get("env", {}).items()})
    min_disk = int(matrix.get("guard", {}).get("min_disk_gb", 15))
    min_mem = int(matrix.get("guard", {}).get("min_free_mb", 40960))
    if entry.get("adaptation_required"):
        log(f"{entry['id']}: NOTE adaptation_required - regroup on a non-primary drafter; "
            f"run the smoke gates in RUNBOOK.md before trusting results")
    for stage in matrix.get("stages", ["train", "eval", "benchmark"]):
        if stage_done(matrix, entry, stage):
            continue
        if not guard(gpu, min_mem, min_disk, guard_timeout, log):
            record(statef, state_lock, entry, stage, "guard_timeout", None, gpu)
            log(f"{entry['id']} {stage}: guard timeout on gpu{gpu}")
            return
        for cmd in stage_commands(matrix, entry, stage, resume_train):
            started = now()
            log(f"{entry['id']} {stage}: {' '.join(cmd)}")
            rc = subprocess.run(cmd, cwd=matrix["repo"], env=env).returncode
            record(statef, state_lock, entry, stage, "ok" if rc == 0 else f"rc={rc}", started, gpu)
            if rc != 0:
                log(f"{entry['id']} {stage}: FAILED rc={rc}; entry parked (rerun with resume)")
                return
    log(f"{entry['id']}: all stages complete on gpu{gpu}")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(statef, lock, entry: dict, stage: str, status: str, started, gpu: int) -> None:
    row = {"ts": now(), "entry": entry["id"], "stage": stage, "status": status,
           "started": started, "gpu": gpu, "git_sha": os.environ.get("D1_GIT_SHA")}
    with lock:
        statef.write(json.dumps(row) + "\n")
        statef.flush()


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #
def cmd_dry_run(matrix: dict, args) -> int:
    import importlib.util
    sys.path.insert(0, matrix["repo"])
    problems, warnings = [], []
    spec = importlib.util.find_spec("vlm_dflash.algorithms")
    if spec is None:
        problems.append("vlm_dflash package not importable with this python/PYTHONPATH")
    else:
        from vlm_dflash.algorithms import available_algorithms
        algs = set(available_algorithms())
        for d in matrix["drafters"]:
            if d not in algs:
                problems.append(f"drafter {d!r} not in registry {sorted(algs)}")
    for p in [matrix["controlled_config"], *(info["native_config"] for info in matrix["drafters"].values())]:
        if not (Path(matrix["repo"]) / p).exists():
            warnings.append(f"config missing: {p}")
    for key in ("train_store", "val_split"):
        val = matrix.get("isolation", {}).get(key)
        if val and not Path(val).exists():
            warnings.append(f"data path missing: {key}={val}")
    if not Path(matrix["python"]).exists():
        warnings.append(f"python missing: {matrix['python']}")

    entries = build_entries(matrix)
    cm = matrix.get("cost_model", {})
    plan, total_h = [], 0.0
    for e in entries:
        sps = cm.get("sec_per_step", {}).get(e["drafter"], cm.get("sec_per_step", {}).get("default", 1.3))
        hours = e["steps"] * sps / 3600 + (cm.get("eval_minutes", 12) + cm.get("benchmark_minutes", 10)) / 60
        total_h += hours
        plan.append({"id": e["id"], "stages": matrix.get("stages"), "est_gpu_hours": round(hours, 2),
                     "commands": [c for stage in matrix.get("stages", []) for c in stage_commands(matrix, e, stage, False)],
                     "adaptation_required": e["adaptation_required"]})
    out = Path(matrix["runs_root"]) / "matrix_plan.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"matrix": args.matrix, "entries": plan,
                                   "est_total_gpu_hours": round(total_h, 1)}, indent=2) + "\n")
    except OSError as exc:
        warnings.append(f"could not write plan to {out}: {exc}")
    print(f"entries: {len(entries)}  est total GPU-hours: {total_h:.1f}")
    for row in plan:
        flag = " [adaptation]" if row["adaptation_required"] else ""
        print(f"  {row['id']}  est={row['est_gpu_hours']}h{flag}")
        if args.verbose:
            for c in row["commands"]:
                print("     ", " ".join(c))
    for w in warnings:
        print(f"WARN  {w}")
    for p in problems:
        print(f"ERROR {p}")
    if problems or (warnings and args.strict):
        return 1
    return 0


def cmd_run(matrix: dict, args, resume_mode: bool) -> int:
    entries = build_entries(matrix)
    if args.only:
        rx = re.compile(args.only)
        entries = [e for e in entries if rx.search(e["id"])]
    if args.limit:
        entries = entries[: args.limit]
    gpus = [int(g) for g in args.gpus.split(",")]
    state_path = Path(matrix["runs_root"]) / "matrix_state.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    log_lock = threading.Lock()

    def log(msg):
        with log_lock:
            print(f"{now()} {msg}", flush=True)

    with open(state_path, "a", encoding="utf-8") as statef:
        buckets = {g: [] for g in gpus}
        for i, e in enumerate(entries):
            buckets[gpus[i % len(gpus)]].append(e)

        def worker(gpu: int, queue: list[dict]):
            for e in queue:
                run_entry(matrix, e, gpu, statef, lock, resume_mode, args.guard_timeout, log)

        threads = [threading.Thread(target=worker, args=(g, q), daemon=False) for g, q in buckets.items() if q]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    log(f"matrix run done ({len(entries)} entries)")
    return 0


def cmd_summarize(matrix: dict, args) -> int:
    entries = build_entries(matrix)
    rows = []
    for e in entries:
        run_dir = Path(e["run_dir"])
        row = {"entry": e["id"], "drafter": e["drafter"], "track": e["track"], "seed": e["seed"],
               "regroup": e["regroup"], "train_wall_s": None, "peak_mem_gb": None,
               "mean_al": None, "acceptance_rate": None, "token_exact": None, "bench": {}}
        ts = run_dir / "train" / "train_summary.json"
        if ts.exists():
            s = json.loads(ts.read_text())
            row["train_wall_s"] = s.get("wall_seconds")
            row["peak_mem_gb"] = s.get("peak_gpu_mem_gb")
        er = run_dir / "eval" / "eval_result.json"
        if er.exists():
            r = json.loads(er.read_text())
            trained = {k: v for k, v in r.get("runs", {}).items() if "init" not in k}
            pick = trained.get("ckpt_final.pt") or (list(trained.values())[-1] if trained else None)
            if pick:
                row["mean_al"] = pick.get("mean_al")
                row["acceptance_rate"] = pick.get("acceptance_rate")
            row["token_exact"] = r.get("equivalence_check", {}).get("token_exact")
        reports = Path(matrix["repo"]) / "benchmarks" / "reports"
        for w in matrix.get("isolation", {}).get("test_workloads", []):
            f = reports / f"benchmark_{run_dir.name}_{w}.json"
            if not f.exists():
                f = reports / f"benchmark_{run_dir.name}.json"
            if f.exists():
                try:
                    rep = json.loads(f.read_text())
                    row["bench"][w] = (rep.get("workloads", {}).get(w, {}) or {}).get("mean_al")
                except json.JSONDecodeError:
                    pass
        rows.append(row)
    out_json = Path(args.out_dir) / "summary.json"
    out_csv = Path(args.out_dir) / "summary.csv"
    out_md = Path(args.out_dir) / "summary.md"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    bench_keys = matrix.get("isolation", {}).get("test_workloads", [])
    with open(out_csv, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["entry", "drafter", "track", "seed", "regroup", "train_wall_s", "peak_mem_gb",
                       "mean_al", "acceptance_rate", "token_exact", *bench_keys])
        for r in rows:
            wcsv.writerow([r["entry"], r["drafter"], r["track"], r["seed"], r["regroup"], r["train_wall_s"],
                           r["peak_mem_gb"], r["mean_al"], r["acceptance_rate"], r["token_exact"],
                           *[r["bench"].get(w) for w in bench_keys]])
    lines = ["# A100 matrix summary", "", f"entries: {len(rows)}", "",
             "| entry | AL | acc | wall_s | peak_GB | token_exact | " + " | ".join(bench_keys) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(bench_keys)]
    for r in rows:
        lines.append("| {e} | {al} | {ac} | {w} | {m} | {te} | {b} |".format(
            e=r["entry"], al=fmt(r["mean_al"]), ac=fmt(r["acceptance_rate"], 4), w=fmt(r["train_wall_s"], 0),
            m=fmt(r["peak_mem_gb"], 1), te=r["token_exact"],
            b=" | ".join(fmt(r["bench"].get(w)) for w in bench_keys)))
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}, {out_csv}, {out_md}")
    return 0


def fmt(v, digits=3):
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["dry-run", "run", "resume", "summarize"])
    ap.add_argument("--matrix", default=str(Path(__file__).resolve().parent / "matrix.json"))
    ap.add_argument("--repo", default=None, help="override VELADRAFT_REPO")
    ap.add_argument("--python", default=None, help="override VELADRAFT_PYTHON")
    ap.add_argument("--runs-root", default=None, help="override VELADRAFT_RUNS_ROOT")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--only", default=None, help="regex filter on entry id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--guard-timeout", type=int, default=3600)
    ap.add_argument("--strict", action="store_true", help="dry-run: warnings fail")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default=".", help="summarize output directory")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.repo:
        env["VELADRAFT_REPO"] = args.repo
    if args.python:
        env["VELADRAFT_PYTHON"] = args.python
    if args.runs_root:
        env["VELADRAFT_RUNS_ROOT"] = args.runs_root
    env.setdefault("VELADRAFT_REPO", str(Path(__file__).resolve().parents[3]))
    env.setdefault("VELADRAFT_PYTHON", sys.executable)
    env.setdefault("VELADRAFT_RUNS_ROOT", str(Path(env["VELADRAFT_REPO"]) / "runs" / "a100"))
    matrix = load_matrix(Path(args.matrix), env)
    if args.command == "dry-run":
        return cmd_dry_run(matrix, args)
    if args.command == "run":
        return cmd_run(matrix, args, resume_mode=False)
    if args.command == "resume":
        return cmd_run(matrix, args, resume_mode=True)
    return cmd_summarize(matrix, args)


if __name__ == "__main__":
    raise SystemExit(main())
