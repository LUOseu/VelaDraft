"""Rebuild every campaign table from ``raw/`` (contract section 12); nothing is filled by hand.

    python3 recompute_results.py <campaign_dir>

Inputs are the exported records only (``raw/formal``, ``raw/windows``, ``raw/pilot``,
ledgers), so the tables can be recomputed from the published data package. Window wall
times are the synchronized measurements; per-update host fields are only summed into
``training_wall_s`` (the loop's own update time, validation/checkpoint excluded).
Missing values are written empty (N/A).
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

UPDATES_PER_EPOCH = 25000
# The arm that is the Optimized system in each window stage. DFly stage 2 trains with
# e2e-TV, which never calls the fused KL path, so Optimized reduces to compaction there.
OPTIMIZED_ARM = {("dfly", "stage2"): "compaction"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})


def run_meta(run: Path) -> dict:
    config = read_json(run / "config.json") or {}
    train = config.get("train") or {}
    camp = config.get("campaign") or {}
    stdout = (run / "stdout.log").read_text(errors="replace") if (run / "stdout.log").exists() else ""
    commits = re.findall(r"^commit=([0-9a-f]{40})", stdout, re.MULTILINE)
    starts = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] START", stdout, re.MULTILINE)
    exits = re.findall(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] EXIT_CODE=(\d+)", stdout, re.MULTILINE)
    return {
        "method": camp.get("method"),
        "arm": camp.get("arm"),
        "layout": camp.get("layout"),
        "gpu_count": 1 if camp.get("layout") == "single" else 2,
        "microbatch": train.get("micro_batch_size"),
        "accumulation": train.get("grad_accumulation_steps"),
        "source_revision": commits[-1] if commits else None,
        "source_revisions": sorted(set(commits)),
        "attempts": len(starts),
        "exit_codes": [int(code) for _, code in exits],
        "oom": "OutOfMemoryError" in stdout or "CUDA out of memory" in stdout,
        "switches": camp.get("switches"),
    }


def window_of(run: Path) -> dict | None:
    rows = read_jsonl(run / "timing_windows.jsonl")
    return rows[-1] if rows else None


def peak(window_or_summary: dict, key: str) -> float | None:
    peaks = (window_or_summary or {}).get("peak") or (window_or_summary or {}).get("run_peak")
    if not peaks:
        return None
    return max(value[key] for value in peaks.values())


def formal_tables(raw: Path, ledgers: list[dict], paired: dict) -> tuple[list, list, list, list, list]:
    main, epochs, validation, stage_rows, workload = [], [], [], [], []
    for run in sorted((raw / "formal").glob("*/*/*/*")):
        track, method, seed, arm = run.parts[-4:]
        meta = run_meta(run)
        updates = read_jsonl(run / "updates.jsonl.gz")
        summary = read_json(run / "run_summary.json") or {}
        instrument = summary.get("instrument") or {}
        vals = read_jsonl(run / "validation.jsonl")
        name = f"{track}_{method}_{seed}_{arm}"
        # Retries of a run are separate scheduler jobs (``<name>_attemptN``) on the same run directory.
        gpu_seconds = sum(r["seconds"] for r in ledgers if r.get("event") == "job"
                          and re.fullmatch(re.escape(name) + r"(_attempt\d+)?", r.get("job", "")))
        wall = sum(u.get("update_s", 0.0) for u in updates)
        samples = sum(u.get("samples", 0) for u in updates)
        positions = sum(u.get("w_effective_positions", 0) for u in updates)
        dev_final = [v for v in vals if v["split"] == "dev" and "final" in v.get("reasons", [])]
        test_final = [v for v in vals if v["split"] == "test"]
        first = sorted(u.get("update_s", 0.0) for u in updates[:200])
        compile_s = (updates[0]["update_s"] - statistics.median(first)) if len(first) > 10 else None
        pair = paired.get((method, "all", arm)) or {}
        main.append({
            "track": track, "method": method, "seed": seed, "arm": arm,
            "target_revision": summary.get("target_revision"), "source_revision": meta["source_revision"],
            "config_id": name, "layout": meta["layout"], "gpu_count": meta["gpu_count"],
            "microbatch": meta["microbatch"], "accumulation": meta["accumulation"],
            "completed_updates": updates[-1]["step"] if updates else 0, "consumed_samples": samples,
            "training_wall_s": wall, "run_elapsed_s": gpu_seconds / max(meta["gpu_count"], 1) or None,
            "compile_s_estimate": compile_s, "validation_s": instrument.get("validation_s"),
            "checkpoint_s": instrument.get("checkpoint_s"), "gpu_hours": gpu_seconds / 3600 or None,
            "samples_per_s": samples / wall if wall else None,
            "effective_positions_per_s": positions / wall if wall else None,
            "max_device_peak_allocated_gib": peak(instrument, "allocated_gib"),
            "max_device_peak_reserved_gib": peak(instrument, "reserved_gib"),
            "final_dev_objective": dev_final[-1]["objective_mean"] if dev_final else None,
            "final_test_objective": test_final[-1]["objective_mean"] if test_final else None,
            "paired_training_speedup": pair.get("speedup"),
            "paired_time_reduction_pct": pair.get("time_reduction_pct"),
            "paired_memory_reduction_pct": pair.get("memory_reduction_pct"),
            "status": "complete" if updates and updates[-1]["step"] >= 75000 else "partial",
            "attempts": meta["attempts"],
        })
        by_epoch = defaultdict(list)
        for u in updates:
            by_epoch[(u["step"] - 1) // UPDATES_PER_EPOCH].append(u)
        for epoch, rows in sorted(by_epoch.items()):
            ewall = sum(r.get("update_s", 0.0) for r in rows)
            dev = [v for v in vals if v["split"] == "dev" and v["step"] == rows[-1]["step"]]
            epochs.append({"config_id": name, "track": track, "method": method, "arm": arm, "epoch": epoch + 1,
                           "updates": len(rows), "training_wall_s": ewall,
                           "samples": sum(r.get("samples", 0) for r in rows),
                           "dev_objective": dev[0]["objective_mean"] if dev else None,
                           "complete": len(rows) == UPDATES_PER_EPOCH})
        for v in vals:
            validation.append({"config_id": name, "track": track, "method": method, "arm": arm,
                               "split": v["split"], "step": v["step"], "recipe_stage": v.get("recipe_stage"),
                               "objective_mean": v["objective_mean"], "objective_weighted": v.get("objective_weighted"),
                               "accuracy": v["accuracy"], "accept_len": v["accept_len"],
                               "supervised_positions": v["supervised_positions"], "rows": v["rows"],
                               "wall_s": v["wall_s"], "reasons": "+".join(v.get("reasons", [])),
                               **{f"component_{k}": val for k, val in (v.get("components_mean") or {}).items()}})
        by_stage = defaultdict(list)
        for u in updates:
            by_stage[u.get("recipe_stage")].append(u)
        for stage, rows in sorted(by_stage.items(), key=lambda item: str(item[0])):
            stage_rows.append({"config_id": name, "method": method, "arm": arm, "recipe_stage": stage,
                               "updates": len(rows), "training_wall_s": sum(r.get("update_s", 0.0) for r in rows),
                               "data_s": sum(r.get("data_s", 0.0) for r in rows),
                               "compute_host_s": sum(r.get("compute_host_s", 0.0) for r in rows),
                               "optim_host_s": sum(r.get("optim_host_s", 0.0) for r in rows),
                               "timing": "host per-update sums"})
        totals = defaultdict(int)
        for u in updates:
            for key, value in u.items():
                if key.startswith("w_"):
                    totals[key[2:]] += value
        workload.append({"config_id": name, "scope": "formal", **totals})
    return main, epochs, validation, stage_rows, workload


def window_tables(raw: Path) -> tuple[list, list, list, list, dict, list]:
    online, drafter, scaling, workload = [], [], [], []
    for run in sorted((raw / "windows").glob("*/*/*/*/*")):
        method, stage, experiment, pair, arm = run.parts[-5:]
        window = window_of(run)
        meta = run_meta(run)
        row = {"method": method, "stage": stage, "experiment": experiment, "pair_id": pair, "arm": arm,
               "timing_scope": "drafter" if experiment == "drafter_replay" else "online",
               "layout": meta["layout"], "microbatch": meta["microbatch"], "accumulation": meta["accumulation"],
               "status": "ok" if window else ("oom" if meta["oom"] else "missing"),
               "source_revision": meta["source_revision"]}
        if window:
            row.update({"window_start": window["start_step"], "window_stop": window["stop_step"],
                        "updates": window["updates"], "wall_s": window["wall_s"],
                        "update_time_ms": window["update_time_ms"], "samples_per_s": window["samples_per_s"],
                        "effective_positions_per_s": window["effective_positions_per_s"],
                        "max_device_peak_allocated_gib": peak(window, "allocated_gib"),
                        "max_device_peak_reserved_gib": peak(window, "reserved_gib"),
                        "aux_s": window["aux_s"]})
            workload.append({"config_id": f"{method}_{stage}_{experiment}_{pair}_{arm}", "scope": experiment,
                             **window["workload"]})
        {"ablation": online, "drafter_replay": drafter, "batch_scaling": scaling}[experiment].append(row)
    ablation, paired = [], {}
    for rows, scope in ((online, "online"), (drafter, "drafter")):
        groups = defaultdict(dict)
        for row in rows:
            if row["status"] == "ok" and row["arm"] != "record":
                groups[(row["method"], row["stage"])].setdefault(row["arm"], []).append(row)
        for (method, stage), arms in groups.items():
            ref = arms.get("ref") or []
            ref_pairs = {r["pair_id"]: r for r in ref}
            for arm, arm_rows in arms.items():
                common = [r for r in arm_rows if r["pair_id"] in ref_pairs]
                ref_wall = sum(ref_pairs[r["pair_id"]]["wall_s"] for r in common)
                wall = sum(r["wall_s"] for r in common)
                ref_peak = max((ref_pairs[r["pair_id"]]["max_device_peak_allocated_gib"] for r in common), default=None)
                arm_peak = max((r["max_device_peak_allocated_gib"] for r in common), default=None)
                entry = {"method": method, "stage": stage, "timing_scope": scope, "arm": arm, "pairs": len(common),
                         "reference_wall_s": ref_wall, "wall_s": wall,
                         "speedup": ref_wall / wall if wall else None,
                         "time_reduction_pct": 100 * (1 - wall / ref_wall) if ref_wall else None,
                         "per_pair_speedup": ";".join(f"{ref_pairs[r['pair_id']]['wall_s'] / r['wall_s']:.4f}"
                                                      for r in common),
                         "max_peak_allocated_gib": arm_peak,
                         "memory_reduction_pct": 100 * (1 - arm_peak / ref_peak) if ref_peak and arm_peak else None}
                ablation.append(entry)
                if scope == "online" and arm == OPTIMIZED_ARM.get((method, stage), "opt"):
                    paired[(method, stage, "opt")] = entry
    # Whole-run pairing per method: combine its stages' windows.
    for method in {key[0] for key in paired}:
        parts = [entry for key, entry in paired.items() if key[0] == method]
        ref_wall = sum(p["reference_wall_s"] for p in parts)
        wall = sum(p["wall_s"] for p in parts)
        peaks = [p["memory_reduction_pct"] for p in parts if p["memory_reduction_pct"] is not None]
        paired[(method, "all", "opt")] = {"speedup": ref_wall / wall if wall else None,
                                          "time_reduction_pct": 100 * (1 - wall / ref_wall) if ref_wall else None,
                                          "memory_reduction_pct": min(peaks) if peaks else None}
    return online, drafter, scaling, ablation, paired, workload


def node_table(ledgers: list[dict]) -> list[dict]:
    jobs = [r for r in ledgers if r.get("event") == "job"]
    if not jobs:
        return []
    start, end = min(r["start"] for r in jobs), max(r["end"] for r in jobs)
    per_gpu = defaultdict(lambda: defaultdict(float))
    for r in jobs:
        category = r["job"].split("_")[0]
        per_gpu[r["gpu"]][category] += r["seconds"]
    rows = []
    for gpu, categories in sorted(per_gpu.items()):
        rows.append({"gpu": gpu, "campaign_start": start, "campaign_end": end, "span_s": end - start,
                     "occupied_s": sum(categories.values()),
                     "occupancy": sum(categories.values()) / (end - start) if end > start else None,
                     **{f"{key}_s": value for key, value in sorted(categories.items())}})
    return rows


def main() -> None:
    camp = Path(sys.argv[1])
    raw, tables = camp / "raw", camp / "tables"
    ledgers = [row for path in sorted(raw.glob("*_gpu_time_ledger.jsonl")) for row in read_jsonl(path)]
    online, drafter, scaling, ablation, paired, window_workload = window_tables(raw)
    main_rows, epochs, validation, stages, formal_workload = formal_tables(raw, ledgers, paired)
    write_csv(tables / "formal_training_main.csv", main_rows)
    write_csv(tables / "formal_epoch_results.csv", epochs)
    write_csv(tables / "efficiency_online.csv", online)
    write_csv(tables / "efficiency_drafter.csv", drafter)
    write_csv(tables / "ablation.csv", ablation)
    write_csv(tables / "batch_scaling.csv", scaling)
    write_csv(tables / "stage_timings.csv", stages)
    write_csv(tables / "workload_counts.csv", formal_workload + window_workload)
    write_csv(tables / "training_validation.csv", validation)
    write_csv(tables / "node_efficiency.csv", node_table(ledgers))
    print(json.dumps({"formal": len(main_rows), "windows": len(online), "drafter": len(drafter),
                      "scaling": len(scaling), "validation_rows": len(validation)}))


if __name__ == "__main__":
    main()
