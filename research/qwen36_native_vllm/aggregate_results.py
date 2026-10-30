"""Aggregate archived native-eval runs into the per-method x per-task tables (no plots).

  python aggregate_results.py --runs <archive dir with <name>/result.json + <name>/result.evidence>
                              [--al <dir with <name>/result.json from stats-enabled supplements>]
                              --out <dir>

Writes:
  summary_by_task.csv   method x task (+ ALL): paired E2E speedup sum(AR)/sum(SD), session-cluster
                        95% CI, AR/SD latency p50/p90/p99, mean output tokens, exact-output share,
                        acceptance (when a stats-enabled supplement exists), peak GPU memory
  requests.csv          every timed physical request (method, repeat, arm, id, turn, topic, ...)
  summary.json          the same tables plus provenance (result sha256, plan, implementation scope)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from common import METHODS as METHOD_ORDER, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_dflash.eval.native import bootstrap_ci  # noqa: E402

TASKS = [("general vqa", "General VQA"), ("text vqa", "Text VQA"), ("image captioning", "Image Captioning"),
         ("chart understanding", "Chart VQA"), ("complex reasoning pro", "Complex Reasoning"),
         ("multi-turn conversation", "Multi-turn Conversation")]
TASKS_AND_ALL = TASKS + [(None, "ALL")]
TASK_LABELS = [label for _, label in TASKS_AND_ALL]


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    low = int(index)
    frac = index - low
    return values[low] * (1 - frac) + values[min(low + 1, len(values) - 1)] * frac


def load_run(run_dir: Path):
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    evidence = run_dir / "result.evidence"
    plan = json.loads((evidence / "plan.json").read_text())
    sessions = {s["id"]: s for s in plan["sessions"]}
    rows = []
    for stage in sorted(evidence.glob("[as][rd]_r*.json")):
        report = json.loads(stage.read_text())
        arm, repeat = report["stage"].split("_r")
        for row in report["rows"]:
            if row.get("phase") != "timed":
                continue
            rows.append(dict(arm=arm, repeat=int(repeat), id=row["id"], turn=row["turn"], topic=sessions[row["id"]].get("topic"),
                             category=sessions[row["id"]].get("category"), wall_seconds=row["wall_seconds"],
                             output_tokens=len(row["tokens"]), prompt_tokens=len(row["prompt_token_ids"]),
                             finish_reason=row["finish_reason"], tokens=row["tokens"],
                             counters=row.get("official_counters")))
    return result, plan, rows, result_path


def per_session(group):
    walls = {}
    for r in group:
        walls.setdefault(r["id"], []).append(r["wall_seconds"])
    return walls


def summarize(method, result, rows, al_rows):
    by_key = {(r["arm"], r["repeat"], r["id"], r["turn"]): r for r in rows}
    peak = (result.get("resource_evidence") or {}).get("peak_memory_used_mib")
    peak_mib = max(peak.values()) if peak else None
    out = []
    for topic, label in TASKS_AND_ALL:
        selected = [r for r in rows if topic is None or r["topic"] == topic]
        ar = [r for r in selected if r["arm"] == "ar"]
        sd = [r for r in selected if r["arm"] == "sd"]
        if not ar or len(ar) != len(sd):
            continue
        ar_s, sd_s = per_session(ar), per_session(sd)
        low, high = bootstrap_ci(ar_s, sd_s)
        ar_seconds = sum(r["wall_seconds"] for r in ar)
        sd_seconds = sum(r["wall_seconds"] for r in sd)
        exact = sum(by_key[("ar", r["repeat"], r["id"], r["turn"])]["tokens"] == r["tokens"] for r in sd)
        row = {"method": method, "task": label, "sessions": len(ar_s), "timed_pairs": len(sd),
               "speedup_sum_ar_over_sd": ar_seconds / sd_seconds,
               "speedup_ci95_low": low, "speedup_ci95_high": high,
               "ar_seconds": ar_seconds, "sd_seconds": sd_seconds}
        for arm, group in (("ar", ar), ("sd", sd)):
            walls = [r["wall_seconds"] for r in group]
            row.update({f"{arm}_latency_p50": quantile(walls, .5), f"{arm}_latency_p90": quantile(walls, .9),
                        f"{arm}_latency_p99": quantile(walls, .99),
                        f"{arm}_mean_output_tokens": sum(r["output_tokens"] for r in group) / len(group),
                        f"{arm}_tokens_per_second": sum(r["output_tokens"] for r in group) / sum(walls),
                        f"{arm}_length_capped": sum(r["finish_reason"] == "length" for r in group)})
        row["exact_output_pairs_share"] = exact / len(sd)
        counters = [r["counters"] for r in al_rows if (topic is None or r["topic"] == topic)
                    and r["arm"] == "sd" and (r.get("counters") or {}).get("status") == "complete"]
        if counters:
            rounds = sum(c["rounds"] for c in counters)
            accepted = sum(c["accepted"] for c in counters)
            proposed = sum(c["proposed"] for c in counters)
            row.update({"acceptance_length": 1 + accepted / rounds if rounds else None,
                        "accepted_draft_tokens_per_round": accepted / rounds if rounds else None,
                        "accepted_draft_fraction": accepted / proposed if proposed else None,
                        "acceptance_source": "separate stats-enabled supplement (1 repeat), vLLM convention 1+accepted/drafts"})
        row["peak_gpu_memory_mib"] = peak_mib
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--al", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, requests, provenance = [], [], {}
    for run_dir in sorted(p for p in args.runs.iterdir() if (p / "result.json").exists()):
        result, plan, rows, result_path = load_run(run_dir)
        method = result.get("method") or plan["method"]
        if result.get("status") != "complete":
            provenance[run_dir.name] = {"status": result.get("status"), "error": result.get("error")}
            continue
        al_rows = []
        if args.al and (args.al / run_dir.name / "result.json").exists():
            _, _, al_rows, _ = load_run(args.al / run_dir.name)
        summary += summarize(method, result, rows, al_rows)
        for r in rows:
            requests.append({"method": method, **{k: v for k, v in r.items() if k not in ("tokens", "counters")}})
        provenance[run_dir.name] = {"method": method, "status": "complete", "result_sha256": sha256(result_path),
                                    "implementation_scope": result.get("implementation_scope"),
                                    "speedup_sum_ar_over_sd": result.get("speedup_sum_ar_over_sd"),
                                    "session_cluster_CI95": result.get("session_cluster_CI95"),
                                    "repeats": result.get("repeats"), "sessions": result.get("sessions"),
                                    "timed_pairs": result.get("timed_pairs"), "paper_ready": result.get("paper_ready")}
    summary.sort(key=lambda r: (METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99,
                                TASK_LABELS.index(r["task"])))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("summary_by_task.csv", summary), ("requests.csv", requests)):
        if table:
            fields = list(dict.fromkeys(k for row in table for k in row))
            with (args.out / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(table)
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "runs": provenance}, indent=2,
                                                      ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(provenance), "summary_rows": len(summary), "requests": len(requests)}))


if __name__ == "__main__":
    main()
