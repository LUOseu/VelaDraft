"""Run a trained drafter against fixed workload manifests, one report out.

Every workload pins its prompt source, prompt count and output budget; the
harness records the full protocol next to the numbers so two reports are only
compared when they were produced the same way:

- greedy decoding (temperature 0) everywhere;
- the target-only greedy output is generated once per workload and every
  speculative run is compared token-exact against it;
- acceptance numbers are measured with ``verification=sequential`` by default.
  ``parallel``/``strict_parallel`` are throughput experiments and are marked
  ``experimental_runtime`` in the report instead of the correctness table.

Workloads whose public datasets still need downloading/preprocessing are
declared ``status: planned`` with the protocol they will pin; the harness skips
them with a note rather than fabricating numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..algorithms import get_algorithm
from ..eval.run import (
    _stops,
    _target_head_is_tied,
    load_eval_rows,
    preflight_checkpoints,
    prepare_prompts,
    run_checkpoints,
)
from ..target.load import load_target
from ..utils import ensure_dir, log, now_iso, write_json

BENCHMARK_FORMAT = "vlm-dflash-benchmark-v1"
_REQUIRED_KEYS = ("name", "description", "status", "prompt_source", "num_prompts", "max_new_tokens")
_VALID_STATUS = {"validated", "scaffold", "planned"}


def validate_workload(workload: dict, source: Path) -> None:
    missing = [key for key in _REQUIRED_KEYS if key not in workload]
    if missing:
        raise ValueError(f"{source}: workload manifest is missing keys {missing}.")
    if workload["status"] not in _VALID_STATUS:
        raise ValueError(
            f"{source}: status must be one of {sorted(_VALID_STATUS)}, got {workload['status']!r}."
        )
    if workload["status"] != "planned":
        prompt_source = workload.get("prompt_source")
        if not isinstance(prompt_source, dict) or "type" not in prompt_source:
            raise ValueError(
                f"{source}: non-planned workloads need a prompt_source object with a 'type' "
                "(run_val or sharegpt_jsonl)."
            )
    if int(workload["num_prompts"]) < 1 or int(workload["max_new_tokens"]) < 1:
        raise ValueError(f"{source}: num_prompts and max_new_tokens must be >= 1.")


def load_workloads(manifests_dir: str | Path) -> list[dict]:
    """Load + validate every ``*.json`` workload manifest (sorted by name)."""
    manifests_dir = Path(manifests_dir)
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Workload manifest directory not found: {manifests_dir}")
    workloads = []
    for path in sorted(manifests_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            workload = json.load(handle)
        if not isinstance(workload, dict):
            raise ValueError(f"{path}: workload manifest must be a JSON object.")
        validate_workload(workload, path)
        workloads.append(workload)
    if not workloads:
        raise FileNotFoundError(f"No workload manifests (*.json) found in {manifests_dir}.")
    return workloads


def select_workloads(workloads: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return workloads
    by_name = {workload["name"]: workload for workload in workloads}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown workload(s) {unknown}; available: {sorted(by_name)}.")
    return [by_name[name] for name in names]


def _markdown(report: dict) -> str:
    lines = [
        f"# Benchmark report — {report['algorithm']} — {report['run_dir']}",
        "",
        f"- created: {report['created']}",
        f"- protocol: temperature={report['protocol']['temperature']}, "
        f"verification={report['protocol']['verification']}, checkpoint={report['protocol']['checkpoint']}",
        "",
        "| workload | prompts | max_new | AL | acc rate | slot 1..k | token exact | wall/s | target-only/s | speedup |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for name, entry in report["workloads"].items():
        if entry.get("status") != "measured":
            lines.append(f"| {name} | - | - | - | - | - | (planned) | - | - | - |")
            continue
        slots = ", ".join(
            "-" if value is None else f"{value:.3f}" for value in entry["per_slot_acceptance"]
        )
        lines.append(
            f"| {name} | {entry['num_prompts']} | {entry['max_new_tokens']} | {entry['al']:.3f} | "
            f"{entry['acceptance_rate']:.3f} | {slots} | {entry['token_exact']} | "
            f"{entry['wall_seconds_per_sample']:.2f} | {entry['target_only_wall_seconds_per_sample']:.2f} | "
            f"{entry['speedup_vs_target_only']:.2f} |"
        )
    if report.get("experimental_runtime"):
        lines += [
            "",
            f"> verification={report['protocol']['verification']}: **experimental runtime numbers, "
            "not bitwise-lossless; do not mix into correctness tables.**",
        ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    cfg: dict,
    run_dir: str | Path,
    contract: dict,
    workloads: list[dict],
    *,
    out_dir: str | Path,
    num_prompts: int | None = None,
    max_new_tokens: int | None = None,
    verification: str = "sequential",
    strict_parallel_margin: float | None = None,
    checkpoint: str = "ckpt_final.pt",
    name: str | None = None,
) -> dict:
    """Measure every runnable workload with one target load; write JSON + MD."""
    run_dir = Path(run_dir)
    out_dir = ensure_dir(Path(out_dir))
    algorithm = get_algorithm(cfg.get("algorithm"))
    if verification == "strict_parallel" and strict_parallel_margin is None:
        strict_parallel_margin = float(cfg.get("eval", {}).get("strict_parallel_margin", 2.0))

    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    target_tied_head = _target_head_is_tied(model)
    stop_ids = set(_stops(model))
    block_size = int(cfg["drafter"]["block_size"])
    draft_dtype = getattr(torch, str(cfg.get("target", {}).get("dtype", "bfloat16")), torch.bfloat16)
    preflight_checkpoints(run_dir, [checkpoint], contract, target_tied_head, algorithm)

    results: dict[str, dict] = {}
    for workload in workloads:
        workload_name = workload["name"]
        if workload["status"] == "planned":
            results[workload_name] = {"status": "planned", "note": workload.get("note")}
            log(f"bench[{workload_name}]: planned workload skipped ({workload.get('note', 'no note')})")
            continue
        rows, media_dir = load_eval_rows(workload.get("prompt_source"), run_dir)
        count = min(int(num_prompts or workload["num_prompts"]), len(rows))
        budget = int(max_new_tokens or workload["max_new_tokens"])
        rows = rows[:count]
        label = f"bench[{workload_name}]"
        prompts = prepare_prompts(
            model, processor, rows, media_dir, max_new_tokens=budget, device=device, label=label
        )
        all_results, equivalence_runs, _per_sample, _diag = run_checkpoints(
            model,
            run_dir,
            cfg,
            contract,
            prompts,
            algorithm=algorithm,
            checkpoint_names=[checkpoint],
            verification=verification,
            strict_parallel_margin=strict_parallel_margin,
            verify_backend="parity",
            block_size=block_size,
            draft_dtype=draft_dtype,
            max_new_tokens=budget,
            parallel_diagnostic_prompts=0,
            stop_ids=stop_ids,
            label=label,
        )
        summary = all_results[checkpoint]
        equivalence = equivalence_runs[checkpoint]
        target_only_wall = sum(entry["wall_seconds"] for entry in prompts) / max(1, len(prompts))
        results[workload_name] = {
            "status": "measured",
            "description": workload["description"],
            "num_prompts": count,
            "max_new_tokens": budget,
            "al": summary["mean_al"],
            "acceptance_rate": summary["acceptance_rate"],
            "per_slot_acceptance": summary["per_slot_acceptance"],
            "token_exact": equivalence["token_exact"],
            "failed_ids": equivalence["failed_ids"],
            "wall_seconds_per_sample": summary["wall_seconds_per_sample"],
            "target_only_wall_seconds_per_sample": target_only_wall,
            "speedup_vs_target_only": summary.get("speedup_vs_target_only"),
        }
        log(
            f"{label}: AL={summary['mean_al']:.3f} exact={equivalence['token_exact']} "
            f"speedup={summary.get('speedup_vs_target_only')}"
        )

    protocol = {
        "temperature": 0.0,
        "verification": verification,
        "strict_parallel_margin": strict_parallel_margin,
        "checkpoint": checkpoint,
        "prompt_overrides": {"num_prompts": num_prompts, "max_new_tokens": max_new_tokens},
    }
    report = {
        "format": BENCHMARK_FORMAT,
        "created": now_iso(),
        "algorithm": algorithm.name,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "experimental_runtime": verification != "sequential",
        "workloads": results,
    }
    stem = name or f"benchmark_{run_dir.name}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    write_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    log(f"benchmark: wrote {json_path} and {md_path}")
    return report
