#!/usr/bin/env python3
"""Extract the frozen canonical zoo results from recorded run artifacts.

Reads ``runs/<run>/eval/eval_result*.json``, ``train/train_summary.json`` and
the export safetensors headers, and writes a small machine-readable summary
(JSON on stdout). The committed ``benchmarks/results/zoo_canonical.{json,csv}``
were produced with this script on the reference machine; re-run it there (with
the runs/ tree present) to refresh:

    python scripts/extract_zoo_results.py > benchmarks/results/zoo_canonical.json
    python scripts/extract_zoo_results.py --csv > benchmarks/results/zoo_canonical.csv

Paths are relative to the repo root (override with ``VLM_DFLASH_ROOT``).
"""

from __future__ import annotations

import csv
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(os.environ.get("VLM_DFLASH_ROOT", Path(__file__).resolve().parents[1]))

RUNS = [
    # run dir, algorithm, eval file, zoo export name, target family, [export path]
    # The export path defaults to ``assets/zoo/<name>``; runs whose export lives
    # in the run directory (``runs/<run>/export``) pass it explicitly.
    ("qwen25vl3b_2k", "dflash", "eval/eval_result.json", "qwen25vl-3b-dflash", "qwen25_vl"),
    ("qwen25vl3b_dspark_2k", "dspark", "eval/eval_result.json", "qwen25vl-3b-dspark", "qwen25_vl"),
    ("qwen25vl3b_domino_2k", "domino", "eval/eval_result.json", "qwen25vl-3b-domino", "qwen25_vl"),
    (
        "qwen25vl3b_jetspec_2k",
        "jetspec",
        "eval/eval_result_jetspec_2k_causalmask.json",
        "qwen25vl-3b-jetspec",
        "qwen25_vl",
    ),
    ("qwen25vl3b_dflash2_2k", "dflash2", "eval/eval_result.json", "qwen25vl-3b-dflash2-exp", "qwen25_vl"),
    ("qwen3vl8b_2k", "dflash", "eval/eval_result_2k.json", "qwen3vl-8b-dflash", "qwen3_vl"),
    # DFlare / DFly canonical pair + ablations + layer-rich variants (2026-10-13).
    # One shared 5-layer superset store; the matched runs consume [1, 17, 33].
    ("qwen25vl3b_dflare_2k", "dflare", "eval/eval_result.json", "qwen25vl-3b-dflare", "qwen25_vl", "runs/qwen25vl3b_dflare_2k/export"),
    ("qwen25vl3b_dfly_2k", "dfly", "eval/eval_result.json", "qwen25vl-3b-dfly", "qwen25_vl", "runs/qwen25vl3b_dfly_2k/export"),
    ("qwen25vl3b_dflare_2k_layers", "dflare", "eval/eval_result.json", "qwen25vl-3b-dflare-layers", "qwen25_vl", "runs/qwen25vl3b_dflare_2k_layers/export"),
    ("qwen25vl3b_dfly_2k_layers", "dfly", "eval/eval_result.json", "qwen25vl-3b-dfly-layers", "qwen25_vl", "runs/qwen25vl3b_dfly_2k_layers/export"),
    ("qwen25vl3b_dflare_2k_fixed_fusion", "dflare", "eval/eval_result.json", "qwen25vl-3b-dflare-fixed-fusion", "qwen25_vl", "runs/qwen25vl3b_dflare_2k_fixed_fusion/export"),
    ("qwen25vl3b_dflare_2k_shared_kv", "dflare", "eval/eval_result.json", "qwen25vl-3b-dflare-shared-kv", "qwen25_vl", "runs/qwen25vl3b_dflare_2k_shared_kv/export"),
    ("qwen25vl3b_dfly_2k_no_correction", "dfly", "eval/eval_result.json", "qwen25vl-3b-dfly-no-correction", "qwen25_vl", "runs/qwen25vl3b_dfly_2k_no_correction/export"),
    ("qwen3vl8b_reduced_dflare", "dflare", "eval/eval_result.json", "qwen3vl-8b-reduced-dflare", "qwen3_vl", "runs/qwen3vl8b_reduced_dflare/export"),
    ("qwen3vl8b_reduced_dfly", "dfly", "eval/eval_result.json", "qwen3vl-8b-reduced-dfly", "qwen3_vl", "runs/qwen3vl8b_reduced_dfly/export"),
]


def count_params(path: Path) -> int:
    """Parameter count from a safetensors header (no tensor data is read)."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        numel = 1
        for dim in meta["shape"]:
            numel *= int(dim)
        total += numel
    return total


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def extract() -> dict:
    out: dict = {
        "schema": "vlm-dflash-canonical-zoo-results-v1",
        "source": "recorded run artifacts (eval/eval_result*.json, train/train_summary.json, exports)",
        "runs": [],
    }
    for entry_spec in RUNS:
        run, algo, eval_file, zoo, family = entry_spec[:5]
        export_path = entry_spec[5] if len(entry_spec) > 5 else f"assets/zoo/{zoo}"
        run_dir = ROOT / "runs" / run
        if not run_dir.exists():
            print(f"skip missing run: {run}", file=sys.stderr)
            continue
        ev = json.load(open(run_dir / eval_file))
        eq = ev["equivalence_check"]
        entry: dict = {
            "run": run,
            "target_family": family,
            "algorithm": algo,
            "eval_result_file": f"runs/{run}/{eval_file}",
            "verification": eq.get("verification"),
            "prompts": None,
            "max_new_tokens": eq.get("max_new_tokens"),
        }
        for ckpt, role in (("ckpt_final.pt", "trained"), ("ckpt_init.pt", "random_init")):
            if ckpt not in ev["runs"]:
                continue
            record = ev["runs"][ckpt]
            prefix = "" if role == "trained" else "random_init_"
            entry[prefix + "al"] = record["mean_al"]
            entry[prefix + "acceptance_rate"] = record["acceptance_rate"]
            entry[prefix + "per_slot_acceptance"] = record["per_slot_acceptance"]
            entry[prefix + "tokens_per_target_forward"] = record["tokens_per_target_forward"]
            entry[prefix + "wall_seconds_per_sample"] = record["wall_seconds_per_sample"]
            entry[prefix + "speedup_vs_target_only"] = record.get("speedup_vs_target_only")
            entry["prompts"] = record["num_samples"]
            eqr = eq["runs"].get(ckpt, {})
            entry[prefix + "token_exact"] = (
                f"{eqr.get('num_prompts', 0) - len(eqr.get('failed_ids', []))}/{eqr.get('num_prompts', '?')}"
            )
        ts_path = run_dir / "train" / "train_summary.json"
        if ts_path.exists():
            ts = json.load(open(ts_path))
            entry["train"] = {
                k: ts.get(k)
                for k in ("steps", "final_loss", "first10_mean_loss", "last10_mean_loss", "peak_gpu_mem_gb", "wall_seconds", "nan_seen")
            }
        feats = run_dir / "features"
        if feats.exists():
            entry["features_dir"] = str(feats.resolve())
            entry["features_size_bytes"] = dir_size(feats)
            manifest_path = feats / "manifest.json"
            if manifest_path.exists():
                man = json.load(open(manifest_path))
                entry["features"] = {
                    k: man.get(k)
                    for k in ("format", "num_samples", "seq_length", "storage_dtype", "stores_final_hidden")
                }
        weights = ROOT / export_path / "model.safetensors"
        if weights.exists():
            entry["zoo_export"] = zoo
            entry["export_path"] = export_path
            entry["trainable_params"] = count_params(weights)
        out["runs"].append(entry)
    return out


CSV_COLUMNS = [
    "run",
    "target_family",
    "algorithm",
    "al_trained",
    "al_random_init",
    "token_exact",
    "acceptance_rate",
    "tokens_per_target_forward",
    "wall_s_per_sample",
    "speedup_vs_target_only",
    "train_steps",
    "final_loss",
    "peak_gpu_mem_gb",
    "train_wall_s",
    "trainable_params",
    "features_size_bytes",
    "prompts",
    "max_new_tokens",
    "verification",
]


def csv_row(entry: dict) -> dict:
    """Flatten one JSON entry into the committed CSV column layout."""
    train = entry.get("train") or {}
    row = {
        "run": entry.get("run"),
        "target_family": entry.get("target_family"),
        "algorithm": entry.get("algorithm"),
        "al_trained": entry.get("al"),
        "al_random_init": entry.get("random_init_al"),
        "token_exact": entry.get("token_exact"),
        "acceptance_rate": entry.get("acceptance_rate"),
        "tokens_per_target_forward": entry.get("tokens_per_target_forward"),
        "wall_s_per_sample": entry.get("wall_seconds_per_sample"),
        "speedup_vs_target_only": entry.get("speedup_vs_target_only"),
        "train_steps": train.get("steps"),
        "final_loss": train.get("final_loss"),
        "peak_gpu_mem_gb": train.get("peak_gpu_mem_gb"),
        "train_wall_s": train.get("wall_seconds"),
        "trainable_params": entry.get("trainable_params"),
        "features_size_bytes": entry.get("features_size_bytes"),
        "prompts": entry.get("prompts"),
        "max_new_tokens": entry.get("max_new_tokens"),
        "verification": entry.get("verification"),
    }
    return {key: ("" if row.get(key) is None else row[key]) for key in CSV_COLUMNS}


if __name__ == "__main__":
    result = extract()
    if "--csv" in sys.argv:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for entry in result["runs"]:
            writer.writerow(csv_row(entry))
    else:
        print(json.dumps(result, indent=2))
