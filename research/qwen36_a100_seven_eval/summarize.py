"""Recompute every reported number from the raw shards (no evaluator state is trusted).

Per method and protocol, over complete AR/SD pairs on the same card:
- request-level E2E speedup = sum(AR wall seconds) / sum(SD wall seconds);
- per-token ratio = (sum AR time / sum AR tokens) / (sum SD time / sum SD tokens), a separate metric;
- paired bootstrap 95% CI resampling whole MMSpec sessions (all turns and repeats stay together);
- native acceptance from the stats pass (accepted draft tokens per verification round, no bonus;
  the bonus-inclusive length is reported separately and labelled);
- token-exact rate (SD vs AR of the same repeat), output lengths and finish-reason shares.
Per topic, pooled over all requests, and a labelled macro mean of the per-topic speedups.
Task quality is ``pending``: MMSpec test ships no references.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import CATEGORIES, read_json, save_json

BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 2000, 42


def read_rows(run: Path, owner: str, protocol: str, stage_prefix: str) -> list[dict]:
    rows = []
    for path in sorted(
        (Path(run) / "raw" / "mmspec" / owner).glob(
            f"{protocol}__{stage_prefix}*.jsonl"
        )
    ):
        rows += [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return rows


def keyed(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        repeat = r["stage"].rsplit("_r", 1)[-1] if "_r" in r["stage"] else "0"
        key = (r["id"], r["turn"], repeat)
        if key in out:
            raise ValueError(f"duplicate result: {key}")
        out[key] = r
    return out


def bootstrap(
    ar: dict, sd: dict, samples: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED
) -> list[float]:
    """95% CI of sum(ar)/sum(sd), resampling session clusters; ar/sd map session -> list of seconds."""
    keys = sorted(ar)
    if set(keys) != set(sd) or not keys:
        raise ValueError("paired clusters differ")
    a = [sum(ar[k]) for k in keys]
    b = [sum(sd[k]) for k in keys]
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        pick = [rng.randrange(len(keys)) for _ in keys]
        values.append(sum(a[i] for i in pick) / sum(b[i] for i in pick))
    values.sort()

    def quantile(q):
        index = (len(values) - 1) * q
        low = int(index)
        frac = index - low
        return values[low] * (1 - frac) + values[min(low + 1, len(values) - 1)] * frac

    return [quantile(0.025), quantile(0.975)]


def pair_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    ar_time = sum(a["wall_seconds"] for a, _ in pairs)
    sd_time = sum(s["wall_seconds"] for _, s in pairs)
    ar_tok = sum(a["output_tokens"] for a, _ in pairs)
    sd_tok = sum(s["output_tokens"] for _, s in pairs)
    ar_c, sd_c = defaultdict(list), defaultdict(list)
    for a, s in pairs:
        ar_c[a["id"]].append(a["wall_seconds"])
        sd_c[s["id"]].append(s["wall_seconds"])
    return {
        "requests": len(pairs),
        "sessions": len(ar_c),
        "ar_wall_seconds": ar_time,
        "sd_wall_seconds": sd_time,
        "speedup_e2e": ar_time / sd_time if sd_time else None,
        "speedup_e2e_ci95": bootstrap(ar_c, sd_c) if len(ar_c) > 1 else None,
        "ar_seconds_per_token": ar_time / ar_tok if ar_tok else None,
        "sd_seconds_per_token": sd_time / sd_tok if sd_tok else None,
        "per_token_ratio": (ar_time / ar_tok) / (sd_time / sd_tok)
        if ar_tok and sd_tok
        else None,
        "token_exact_rate": sum(a["tokens"] == s["tokens"] for a, s in pairs)
        / len(pairs)
        if pairs
        else None,
        "ar_output_tokens": ar_tok,
        "sd_output_tokens": sd_tok,
        "sd_finish": dict(Counter(s["finish_reason"] for _, s in pairs)),
        "ar_finish": dict(Counter(a["finish_reason"] for a, _ in pairs)),
        "sd_latency_seconds": percentiles([s["wall_seconds"] for _, s in pairs]),
        "ar_latency_seconds": percentiles([a["wall_seconds"] for a, _ in pairs]),
    }


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]
    return {
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": v[-1],
        "mean": sum(v) / len(v),
    }


def acceptance(rows: list[dict]) -> dict:
    counters = [r.get("official_counters") or {} for r in rows]
    ok = [c for c in counters if c.get("status") == "complete"]
    if not ok:
        return {"status": "unavailable", "requests": len(rows)}
    accepted = sum(c["accepted"] for c in ok)
    proposed = sum(c["proposed"] for c in ok)
    rounds = sum(c["rounds"] for c in ok)
    return {
        "status": "complete" if len(ok) == len(rows) else "partial",
        "requests": len(ok),
        "missing": len(rows) - len(ok),
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "verification_rounds": rounds,
        "accepted_per_round": accepted / rounds if rounds else None,
        "acceptance_rate": accepted / proposed if proposed else None,
        "tokens_per_round_with_bonus_upper_bound": (accepted + rounds) / rounds
        if rounds
        else None,
        "definitions": {
            "accepted_per_round": "accepted draft tokens / verification rounds (no bonus)",
            "tokens_per_round_with_bonus_upper_bound": "(accepted + rounds) / rounds; upper bound, bonus emission is not measured",
        },
    }


def method_summary(run: Path, method: str, protocol: str) -> dict:
    ar = keyed(read_rows(run, method, protocol, "ar_r"))
    sd = keyed(read_rows(run, method, protocol, "sd_r"))
    common = sorted(set(ar) & set(sd))
    out = {
        "method": method,
        "protocol": protocol,
        "unpaired_ar": len(set(ar) - set(sd)),
        "unpaired_sd": len(set(sd) - set(ar)),
    }
    if not common:
        return dict(out, status="pending")
    pairs = [(ar[k], sd[k]) for k in common]
    if any(not a.get("gpu_id") or a["gpu_id"] != b.get("gpu_id") for a, b in pairs):
        raise ValueError("AR/SD pair does not identify the same physical GPU")
    out["pooled"] = pair_metrics(pairs)
    out["per_topic"] = {
        t: pair_metrics([p for p in pairs if p[0]["topic"] == t])
        for t in CATEGORIES
        if any(p[0]["topic"] == t for p in pairs)
    }
    speedups = [m["speedup_e2e"] for m in out["per_topic"].values() if m["speedup_e2e"]]
    out["macro_mean_topic_speedup"] = (
        sum(speedups) / len(speedups) if speedups else None
    )
    if protocol == "natural":
        out["acceptance"] = acceptance(read_rows(run, method, "natural", "stats"))
    expected = expected_keys(run)
    out["expected_requests"] = len(expected)
    out["missing_pairs"] = len(expected - set(common))
    out["status"] = "complete" if set(ar) == set(sd) == expected else "partial"
    if protocol == "natural" and read_json(run / "manifest.json")["protocol"].get(
        "acceptance_stats", True
    ):
        stats = keyed(read_rows(run, method, protocol, "stats"))
        if (
            set(stats) != expected_keys(run, repeats=1)
            or out["acceptance"]["status"] != "complete"
        ):
            out["status"] = "partial"
    # Stock AR has a separate lane and is reported independently, without a paired speedup claim.
    out["quality"] = {
        "status": "pending",
        "reason": "MMSpec test ships no reference answers or official scorer",
    }
    return out


def expected_keys(run: Path, repeats: int | None = None) -> set:
    if repeats is None:
        repeats = read_json(run / "manifest.json")["protocol"]["repeats"]
    return {
        (s["id"], t, str(r))
        for s in read_json(run / "sessions.json")
        for t in range(len(s["questions"]))
        for r in range(repeats)
    }


def summarize(run: Path) -> dict:
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    result = {
        "run_id": manifest["run_id"],
        "suite": manifest["suite"],
        "formal": manifest["data"]["formal"],
        "draft_length": manifest["protocol"]["draft_length"],
        "methods": {},
        "ar_default": {},
    }
    for protocol in manifest["protocol"]["protocols"]:
        rows = read_rows(run, "ar_default", protocol, "ar_r")
        result["ar_default"][protocol] = (
            {
                "requests": len(rows),
                "wall_seconds": sum(r["wall_seconds"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
            }
            if rows
            else None
        )
        for method in manifest["methods"]:
            result["methods"][f"{method}__{protocol}"] = method_summary(
                run, method, protocol
            )
    statuses = [m["status"] for m in result["methods"].values()]
    result["status"] = (
        "complete"
        if statuses and all(s == "complete" for s in statuses)
        else ("partial" if any(s != "pending" for s in statuses) else "running")
    )
    if manifest["protocol"].get("ar_default", True) and any(
        set(keyed(read_rows(run, "ar_default", p, "ar_r"))) != expected_keys(run)
        for p in manifest["protocol"]["protocols"]
    ):
        result["status"] = "partial"
    result["failed_jobs"] = (
        sorted(p.name for p in (run / "failed").glob("*"))
        if (run / "failed").exists()
        else []
    )
    if result["failed_jobs"]:
        result["status"] = "partial"
    result["formal"] = result["formal"] and result["status"] == "complete"
    return result


def write_csv(result: dict, path: Path) -> None:
    fields = [
        "method",
        "protocol",
        "status",
        "requests",
        "sessions",
        "speedup_e2e",
        "ci_low",
        "ci_high",
        "per_token_ratio",
        "token_exact_rate",
        "accepted_per_round",
        "acceptance_rate",
        "macro_mean_topic_speedup",
    ]
    temporary = Path(path).with_name(Path(path).name + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for m in result["methods"].values():
            pooled = m.get("pooled") or {}
            ci = pooled.get("speedup_e2e_ci95") or [None, None]
            acc = m.get("acceptance") or {}
            writer.writerow(
                {
                    "method": m["method"],
                    "protocol": m["protocol"],
                    "status": m["status"],
                    "requests": pooled.get("requests"),
                    "sessions": pooled.get("sessions"),
                    "speedup_e2e": pooled.get("speedup_e2e"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "per_token_ratio": pooled.get("per_token_ratio"),
                    "token_exact_rate": pooled.get("token_exact_rate"),
                    "accepted_per_round": acc.get("accepted_per_round"),
                    "acceptance_rate": acc.get("acceptance_rate"),
                    "macro_mean_topic_speedup": m.get("macro_mean_topic_speedup"),
                }
            )

    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.run_dir)
    save_json(args.run_dir / "summary.json", result)
    write_csv(result, args.run_dir / "summary.csv")
    print(json.dumps({k: v for k, v in result.items() if k != "methods"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
