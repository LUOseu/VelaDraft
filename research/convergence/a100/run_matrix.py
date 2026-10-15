#!/usr/bin/env python3
"""Config-driven A100 experiment matrix runner (VelaDraft convergence round).

Subcommands
    dry-run    validate the matrix and print the resolved per-entry command plan
               + rough GPU-hour estimate; writes ``<runs_root>/matrix_plan.json``.
               Validates *effective* per-drafter configs: each entry pins a config
               whose algorithm resolves to its drafter (controlled = shared
               recipe + the drafter's algorithm block + declared structural
               overrides; native = the drafter's own config), and runs the
               algorithm's own ``validate_config``. No GPU is touched.
    run        execute pending stages of every entry across a GPU pool (one
               serial worker per GPU). Each entry first runs a ``prepare``
               stage: bind the declared train store / val split into the run
               dir (symlinks + normalized val meta.json) and run the preflight
               (algorithm/config/contract/store/val checks) that writes
               ``<run_dir>/preflight.json``.
    resume     same as run, but train legs with an existing ``ckpt_last.pt``
               continue with ``--resume`` instead of restarting.
    summarize  aggregate train/eval/benchmark artifacts into summary.{json,csv,md}.

Idempotence: state is appended to ``<runs_root>/matrix_state.jsonl``; a stage is
skipped only when its marker artifact exists AND parses AND carries the fields
the stage is supposed to produce (benchmark is checked per workload:
``benchmark_<entry>_<workload>.json`` with that workload ``measured``).

Isolation: the runner refuses to start when the matrix' train store resolves to
the val split path, and benchmark workloads are declared in ``isolation`` only
(public test sets, never the training slice).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

BENCHMARK_FORMAT = "vlm-dflash-benchmark-v1"
PREPARE_STAGE = "prepare"
DEFAULT_STAGES = ("train", "eval", "benchmark")


# --------------------------------------------------------------------------- #
# matrix loading + effective configs
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


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted!r}: {key!r} is not a mapping")
    node[keys[-1]] = value


def build_entries(matrix: dict) -> list[dict]:
    steps = int(matrix["budget"]["steps"])
    seeds = [int(s) for s in matrix["budget"]["seeds"]]
    tracks = matrix.get("tracks", ["controlled"])
    regroup_sets = matrix["regroup_sets"]
    repo = Path(matrix["repo"])
    runs_root = Path(matrix["runs_root"])
    entries = []
    for drafter, info in matrix["drafters"].items():
        for track in tracks:
            source = info["native_config"] if track == "native" else matrix["controlled_config"]
            modes = ["off", "grid2"] if info.get("regroup_support") else ["off"]
            for seed in seeds:
                for mode in modes:
                    entry_id = f"{drafter}__{track}__s{seed}__rg{mode}"
                    run_dir = str(runs_root / entry_id)
                    entries.append({
                        "id": entry_id,
                        "drafter": drafter,
                        "track": track,
                        "seed": seed,
                        "regroup": mode,
                        "config": str(runs_root / "_configs" / f"{entry_id}.yaml"),
                        "config_source": str(repo / source),
                        "config_sha256": None,
                        "steps": steps,
                        "run_dir": run_dir,
                        "sets": [f"train.seed={seed}", *regroup_sets[mode]],
                        "adaptation_required": bool(info.get("regroup_adaptation", False)) and mode != "off",
                    })
    entries.sort(key=lambda e: e["id"])
    return entries


def effective_config_doc(matrix: dict, entry: dict):
    """Effective config for one entry: shared/native recipe + pinned per-algorithm wiring.

    Controlled track: the shared recipe base with the drafter's own ``algorithm``
    block (from its native config) plus the matrix-declared structural overrides
    (``controlled_overrides``) the algorithm requires. Native track: the
    drafter's own config. In both cases ``run.name``/``run.dir`` are pinned to
    the entry so train/eval/benchmark all consume one identical file.
    """
    doc = yaml.safe_load(Path(entry["config_source"]).read_text())
    if entry["track"] == "controlled":
        info = matrix["drafters"][entry["drafter"]]
        native = yaml.safe_load(Path(matrix["repo"]).joinpath(info["native_config"]).read_text())
        if "algorithm" in native:
            doc["algorithm"] = native["algorithm"]
        for dotted, value in (info.get("controlled_overrides") or {}).items():
            set_dotted(doc, dotted, value)
    doc.setdefault("run", {})["name"] = Path(entry["run_dir"]).name
    doc["run"]["dir"] = str(entry["run_dir"])
    return doc


def materialize_configs(matrix: dict, entries: list[dict]) -> list[str]:
    """Write the pinned effective config per entry; returns problems (never raises)."""
    problems = []
    for entry in entries:
        try:
            text = yaml.safe_dump(effective_config_doc(matrix, entry), sort_keys=False, allow_unicode=True)
            out = Path(entry["config"])
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.read_text() != text:
                out.write_text(text)
            entry["config_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{entry['id']}: effective config failed: {exc}")
    return problems


def validate_config_doc(cfg: dict) -> str | None:
    """Return an error string when the config does not resolve/validate, else None."""
    from vlm_dflash.algorithms import algorithm_name, available_algorithms, get_algorithm
    name = algorithm_name(cfg.get("algorithm"))
    available = available_algorithms()
    if name not in available:
        return f"unknown algorithm {name!r} (registry: {sorted(available)})"
    try:
        get_algorithm(cfg.get("algorithm")).validate_config(cfg)
    except Exception as exc:  # noqa: BLE001
        return f"algorithm.validate_config({name}) failed: {exc}"
    return None


def config_sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# data binding + preflight
# --------------------------------------------------------------------------- #
def resolve_val_media(val_split: Path) -> Path | None:
    """Directory the val split's images live in (staging source for media_dir)."""
    val_dir = val_split.parent
    meta = val_dir / "meta.json"
    if meta.exists():
        try:
            recorded = json.loads(meta.read_text()).get("media_dir")
        except (json.JSONDecodeError, OSError):
            recorded = None
        if recorded:
            candidate = Path(recorded)
            if not candidate.is_absolute():
                candidate = (meta.parent / candidate).resolve()
            return candidate
    images = val_dir / "images"
    if images.is_dir():
        return images.resolve()
    return None


def _ensure_link(link: Path, target: Path, kind: str) -> None:
    if link.is_symlink():
        current = Path(os.path.realpath(link))
        if current != target:
            raise RuntimeError(f"refusing to rebind {link}: points at {current}, expected {target}")
        return
    if link.exists():
        if kind == "features" and not (link / "manifest.json").exists():
            raise RuntimeError(f"{link} exists but is not a feature store (no manifest.json)")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def bind_entry_data(matrix: dict, entry: dict, log) -> None:
    """Bind the declared train store / val split into the run dir (idempotent)."""
    run_dir = Path(entry["run_dir"])
    isolation = matrix.get("isolation", {})
    store = isolation.get("train_store")
    val = isolation.get("val_split")
    if store:
        target = Path(store).resolve()
        if not target.is_dir():
            raise RuntimeError(f"train_store missing or not a directory: {target}")
        if not (target / "manifest.json").exists() or not (target / "index.jsonl").exists():
            raise RuntimeError(f"train_store is not a feature store (manifest.json/index.jsonl): {target}")
        _ensure_link(run_dir / "features", target, "features")
        log(f"{entry['id']} prepare: features -> {target}")
    if val:
        target = Path(val).resolve()
        if not target.is_file():
            raise RuntimeError(f"val_split missing: {target}")
        _ensure_link(run_dir / "data" / "val.jsonl", target, "val")
        media = resolve_val_media(target)
        if media is None:
            raise RuntimeError(
                f"val split media unresolved: neither {target.parent / 'meta.json'} (media_dir) "
                f"nor {target.parent / 'images'} exists")
        meta_path = run_dir / "data" / "meta.json"
        wanted = {"media_dir": str(media), "source": "jsonl"}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                raise RuntimeError(f"existing {meta_path} unreadable: {exc}") from exc
            existing_media = existing.get("media_dir")
            resolved = existing_media
            if resolved is not None and not Path(resolved).is_absolute():
                resolved = str((meta_path.parent / resolved).resolve())
            if resolved != str(media):
                raise RuntimeError(
                    f"existing {meta_path} media_dir resolves to {resolved}, expected {media}; "
                    "remove the stale file to re-stage")
        else:
            meta_path.write_text(json.dumps(wanted, indent=2) + "\n")
        log(f"{entry['id']} prepare: val -> {target} (media {media})")


def preflight_command(matrix: dict, entry: dict) -> list[str]:
    cmd = [matrix["python"], "research/convergence/a100/preflight_entry.py",
           "--repo", matrix["repo"], "--config", entry["config"], "--run-dir", entry["run_dir"],
           "--expect-algorithm", entry["drafter"]]
    isolation = matrix.get("isolation", {})
    if isolation.get("train_store"):
        cmd += ["--store", isolation["train_store"]]
    if isolation.get("val_split"):
        cmd += ["--val", isolation["val_split"],
                "--min-prompts", str(int(matrix.get("eval", {}).get("num_prompts", 64)))]
    return cmd


def prepare_done(matrix: dict, entry: dict) -> bool:
    path = Path(entry["run_dir"]) / "preflight.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if doc.get("status") != "ok" or doc.get("algorithm") != entry["drafter"]:
        return False
    if doc.get("config_sha256") != config_sha256(entry["config"]):
        return False
    isolation = matrix.get("isolation", {})
    for key in ("store", "val"):
        declared = isolation.get("train_store" if key == "store" else "val_split")
        recorded = (doc.get(key) or {}).get("path")
        if declared and recorded != str(Path(declared).resolve()):
            return False
    return True


# --------------------------------------------------------------------------- #
# commands + completion checks
# --------------------------------------------------------------------------- #
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
        return [[py, "scripts/benchmark.py", "--run-dir", run_dir, "--config", cfg,
                 "--workloads", str(bench_manifests_dir(matrix)),
                 "--out-dir", str(bench_reports_dir(matrix)),
                 "--workload", workload, "--name", benchmark_stem(entry, workload)]
                for workload in matrix.get("isolation", {}).get("test_workloads", [])]
    raise ValueError(stage)


def bench_reports_dir(matrix: dict) -> Path:
    configured = (matrix.get("benchmark") or {}).get("reports_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(matrix["repo"]) / path
    return Path(matrix["repo"]) / "benchmarks" / "reports"


def bench_manifests_dir(matrix: dict) -> Path:
    configured = (matrix.get("benchmark") or {}).get("manifests_dir")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(matrix["repo"]) / path
    return Path(matrix["repo"]) / "benchmarks" / "manifests"


def benchmark_stem(entry: dict, workload: str) -> str:
    return f"benchmark_{Path(entry['run_dir']).name}_{workload}"


def benchmark_report_path(matrix: dict, entry: dict, workload: str) -> Path:
    return bench_reports_dir(matrix) / f"{benchmark_stem(entry, workload)}.json"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def stage_done(matrix: dict, entry: dict, stage: str) -> bool:
    run_dir = Path(entry["run_dir"])
    if stage == PREPARE_STAGE:
        return prepare_done(matrix, entry)
    if stage == "train":
        doc = _load_json(run_dir / "train" / "train_summary.json")
        return isinstance(doc, dict) and {"steps", "final_loss", "wall_seconds"} <= set(doc)
    if stage == "eval":
        doc = _load_json(run_dir / "eval" / "eval_result.json")
        return isinstance(doc, dict) and bool(doc.get("runs")) and isinstance(doc.get("equivalence_check"), dict)
    if stage == "benchmark":
        for workload in matrix.get("isolation", {}).get("test_workloads", []):
            doc = _load_json(benchmark_report_path(matrix, entry, workload))
            if not isinstance(doc, dict):
                return False
            record = (doc.get("workloads") or {}).get(workload)
            if not isinstance(record, dict):
                return False
            status = record.get("status")
            if status == "measured":
                if not isinstance(record.get("al"), (int, float)):
                    return False
            elif status != "planned":
                return False
        return True
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
    for stage in [PREPARE_STAGE, *matrix.get("stages", list(DEFAULT_STAGES))]:
        if stage_done(matrix, entry, stage):
            continue
        started = now()
        if stage == PREPARE_STAGE:
            try:
                bind_entry_data(matrix, entry, log)
            except Exception as exc:  # noqa: BLE001
                record(statef, state_lock, entry, stage, "prepare_failed", started, gpu)
                log(f"{entry['id']} prepare: FAILED {exc}")
                return
            cmd = preflight_command(matrix, entry)
            log(f"{entry['id']} prepare: {' '.join(cmd)}")
            rc = subprocess.run(cmd, cwd=matrix["repo"], env=env).returncode
            record(statef, state_lock, entry, stage, "ok" if rc == 0 else f"rc={rc}", started, gpu)
            if rc != 0:
                log(f"{entry['id']} prepare: FAILED rc={rc}; entry parked (rerun with resume after fixing)")
                return
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


def repo_sha(repo: str) -> str | None:
    """Code SHA for the state log: D1_GIT_SHA if set, else the repo checkout's HEAD."""
    sha = os.environ.get("D1_GIT_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


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
    sys.path.insert(0, matrix["repo"])
    problems, warnings = [], []
    try:
        from vlm_dflash.algorithms import available_algorithms
    except Exception as exc:  # noqa: BLE001
        problems.append(f"vlm_dflash algorithms not importable with this python/PYTHONPATH: {exc}")
    else:
        algs = set(available_algorithms())
        for drafter in matrix["drafters"]:
            if drafter not in algs:
                problems.append(f"drafter {drafter!r} not in registry {sorted(algs)}")
    for path in [matrix["controlled_config"], *(info["native_config"] for info in matrix["drafters"].values())]:
        if not (Path(matrix["repo"]) / path).exists():
            warnings.append(f"config missing: {path}")
    isolation = matrix.get("isolation", {})
    store, val = isolation.get("train_store"), isolation.get("val_split")
    if store:
        store_path = Path(store)
        if not store_path.exists():
            warnings.append(f"data path missing: train_store={store_path}")
        elif not (store_path / "manifest.json").exists() or not (store_path / "index.jsonl").exists():
            warnings.append(f"train_store does not look like a feature store: {store_path}")
    if val:
        val_path = Path(val)
        if not val_path.exists():
            warnings.append(f"data path missing: val_split={val_path}")
        else:
            media = resolve_val_media(val_path)
            if media is None:
                warnings.append(f"val media unresolved (need {val_path.parent/'meta.json'} or {val_path.parent/'images'}): {val_path}")
            else:
                print(f"val media: {media}")
    if not Path(matrix["python"]).exists():
        warnings.append(f"python missing: {matrix['python']}")

    entries = build_entries(matrix)
    problems += materialize_configs(matrix, entries)
    seen: dict[str, str | None] = {}
    for entry in entries:
        digest = entry["config_sha256"] or entry["config"]
        if digest not in seen:
            try:
                cfg = yaml.safe_load(Path(entry["config"]).read_text())
            except Exception as exc:  # noqa: BLE001
                seen[digest] = f"effective config unreadable: {exc}"
            else:
                seen[digest] = validate_config_doc(cfg)
        if seen[digest]:
            problems.append(f"{entry['id']}: {seen[digest]}")
    cost = matrix.get("cost_model", {})
    plan, total_h = [], 0.0
    stages = [PREPARE_STAGE, *matrix.get("stages", list(DEFAULT_STAGES))]
    for entry in entries:
        sps = cost.get("sec_per_step", {}).get(entry["drafter"], cost.get("sec_per_step", {}).get("default", 1.3))
        hours = entry["steps"] * sps / 3600 + (cost.get("eval_minutes", 12) + cost.get("benchmark_minutes", 10)) / 60
        total_h += hours
        plan.append({"id": entry["id"], "stages": stages, "est_gpu_hours": round(hours, 2),
                     "config": entry["config"], "config_source": entry["config_source"],
                     "config_sha256": entry["config_sha256"],
                     "commands": [c for stage in matrix.get("stages", list(DEFAULT_STAGES))
                                  for c in stage_commands(matrix, entry, stage, False)],
                     "adaptation_required": entry["adaptation_required"]})
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
        print(f"  {row['id']}  est={row['est_gpu_hours']}h  config={Path(row['config']).name}{flag}")
        if args.verbose:
            for cmd in row["commands"]:
                print("     ", " ".join(cmd))
    for warning in warnings:
        print(f"WARN  {warning}")
    for problem in problems:
        print(f"ERROR {problem}")
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
    problems = materialize_configs(matrix, entries)
    for problem in problems:
        print(f"ERROR {problem}")
    if problems:
        return 1
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
        for i, entry in enumerate(entries):
            buckets[gpus[i % len(gpus)]].append(entry)

        def worker(gpu: int, queue: list[dict]):
            for entry in queue:
                run_entry(matrix, entry, gpu, statef, lock, resume_mode, args.guard_timeout, log)

        threads = [threading.Thread(target=worker, args=(g, q), daemon=False) for g, q in buckets.items() if q]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    log(f"matrix run done ({len(entries)} entries)")
    return 0


def cmd_summarize(matrix: dict, args) -> int:
    entries = build_entries(matrix)
    rows = []
    for entry in entries:
        run_dir = Path(entry["run_dir"])
        row = {"entry": entry["id"], "drafter": entry["drafter"], "track": entry["track"], "seed": entry["seed"],
               "regroup": entry["regroup"], "train_wall_s": None, "peak_mem_gb": None,
               "mean_al": None, "acceptance_rate": None, "token_exact": None, "bench": {}, "bench_token_exact": {}}
        summary = _load_json(run_dir / "train" / "train_summary.json")
        if isinstance(summary, dict):
            row["train_wall_s"] = summary.get("wall_seconds")
            row["peak_mem_gb"] = summary.get("peak_gpu_mem_gb")
        result = _load_json(run_dir / "eval" / "eval_result.json")
        if isinstance(result, dict):
            trained = {name: run for name, run in (result.get("runs") or {}).items() if "init" not in name}
            pick = trained.get("ckpt_final.pt") or (list(trained.values())[-1] if trained else None)
            if pick:
                row["mean_al"] = pick.get("mean_al")
                row["acceptance_rate"] = pick.get("acceptance_rate")
            row["token_exact"] = (result.get("equivalence_check") or {}).get("token_exact")
        for workload in matrix.get("isolation", {}).get("test_workloads", []):
            report = _load_json(benchmark_report_path(matrix, entry, workload))
            if not isinstance(report, dict):
                continue
            record = (report.get("workloads") or {}).get(workload) or {}
            if record.get("status") == "measured":
                row["bench"][workload] = record.get("al")
                row["bench_token_exact"][workload] = record.get("token_exact")
        rows.append(row)
    out_json = Path(args.out_dir) / "summary.json"
    out_csv = Path(args.out_dir) / "summary.csv"
    out_md = Path(args.out_dir) / "summary.md"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    bench_keys = matrix.get("isolation", {}).get("test_workloads", [])
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entry", "drafter", "track", "seed", "regroup", "train_wall_s", "peak_mem_gb",
                         "mean_al", "acceptance_rate", "token_exact", *bench_keys])
        for row in rows:
            writer.writerow([row["entry"], row["drafter"], row["track"], row["seed"], row["regroup"],
                             row["train_wall_s"], row["peak_mem_gb"], row["mean_al"], row["acceptance_rate"],
                             row["token_exact"], *[row["bench"].get(w) for w in bench_keys]])
    lines = ["# A100 matrix summary", "", f"entries: {len(rows)}", "",
             "| entry | AL | acc | wall_s | peak_GB | token_exact | " + " | ".join(bench_keys) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(bench_keys)]
    for row in rows:
        lines.append("| {e} | {al} | {ac} | {w} | {m} | {te} | {b} |".format(
            e=row["entry"], al=fmt(row["mean_al"]), ac=fmt(row["acceptance_rate"], 4),
            w=fmt(row["train_wall_s"], 0), m=fmt(row["peak_mem_gb"], 1), te=row["token_exact"],
            b=" | ".join(fmt(row["bench"].get(w)) for w in bench_keys)))
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
    os.environ.setdefault("D1_GIT_SHA", repo_sha(matrix["repo"]) or "")
    if args.command == "dry-run":
        return cmd_dry_run(matrix, args)
    if args.command == "run":
        return cmd_run(matrix, args, resume_mode=False)
    if args.command == "resume":
        return cmd_run(matrix, args, resume_mode=True)
    return cmd_summarize(matrix, args)


if __name__ == "__main__":
    raise SystemExit(main())
