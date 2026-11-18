#!/usr/bin/env python3
"""Harvest + numeric-gate report for the lane's paired A/B runs.

Default pairs (label, off-leg, on-leg):
  s2-pair1: s2/a1 vs s2/b1   s2-pair2: s2/a2 vs s2/b2   (256-slice, cache 0 vs 8 GiB)
  s3:       s3/a_off vs s3/b_on                        (canonical 2K, cache 0 vs 16 GiB)

Reads summaries/logs/hashes under $LANE/logs (override with LANE=...).
Prints a compact report for REPORT.md; exits nonzero if a numeric gate fails.
"""
import json
import os
import sys

LANE = os.environ.get("LANE", "/home/lyh/vlm-systems")
LOGS = os.path.join(LANE, "logs")


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        return {"__error__": str(exc)}


def load_log(path):
    rows = []
    if not os.path.exists(path):
        return None, "missing"
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, None


def pick(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def leg(label, tag, short):
    s = load_json(os.path.join(LOGS, f"{tag}_summary.json"))
    rss_kb = None
    tpath = os.path.join(LOGS, f"{tag}.time")
    if os.path.exists(tpath):
        with open(tpath) as th:
            for line in th:
                if "Maximum resident set size" in line:
                    rss_kb = int(line.split(":")[1].strip())
    log, err = load_log(os.path.join(LOGS, f"{tag}_train_log.jsonl"))
    fs = s.get("feature_source") or {}
    fc = fs.get("feature_cache") or {}
    out = {
        "label": label,
        "tag": tag,
        "short": short,
        "summary_error": s.get("__error__"),
        "wall": s.get("wall_seconds"),
        "steps": s.get("steps"),
        "final_loss": s.get("final_loss"),
        "first10": s.get("first10_mean_loss"),
        "last10": s.get("last10_mean_loss"),
        "peak_gpu_gb": s.get("peak_gpu_mem_gb"),
        "extract_wall": fs.get("extraction_wall_seconds_total"),
        "extract_prepare": fs.get("extraction_prepare_seconds_total"),
        "extract_forward": fs.get("extraction_forward_seconds_total"),
        "forwards": fs.get("target_forwards_total"),
        "samples": fs.get("target_samples_total"),
        "fast_forward": fs.get("fast_forward_items_total"),
        "phases": sorted((fs.get("extractions") or {}).keys()),
        "cache": {k: fc.get(k) for k in
                  ("enabled", "hits", "misses", "stores", "evictions", "peak_bytes",
                   "current_bytes", "capacity_bytes", "skipped_too_large")} if fc else None,
        "log_rows": len(log) if log else 0,
        "host_rss_gb": (rss_kb / 1048576) if rss_kb else None,
    }
    if log:
        out["loss_seq"] = [r.get("loss") for r in log]
        out["step_seq"] = [r.get("step") for r in log]
    return out


def fmt(v, nd=1, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}{suffix}"
    return str(v)


def compare(pair_name, a, b, hashes):
    print(f"== {pair_name}: {a['label']} vs {b['label']}")
    print(f"  wall_s      A={fmt(a['wall'])}  B={fmt(b['wall'])}  "
          f"delta={fmt((b['wall'] or 0) - (a['wall'] or 0))} "
          f"({fmt(100 * ((b['wall'] or 0) - (a['wall'] or 0)) / a['wall']) if a['wall'] else 'n/a'}%)")
    print(f"  extract_s   A={fmt(a['extract_wall'])}  B={fmt(b['extract_wall'])}  "
          f"saved={fmt((a['extract_wall'] or 0) - (b['extract_wall'] or 0))}s")
    print(f"  prep/fwd_s  A={fmt(a['extract_prepare'])}/{fmt(a['extract_forward'])}  "
          f"B={fmt(b['extract_prepare'])}/{fmt(b['extract_forward'])}")
    print(f"  forwards    A={a['forwards']}  B={b['forwards']}   samples A={a['samples']} B={b['samples']}")
    print(f"  loss        A_final={a['final_loss']} B_final={b['final_loss']}  steps A={a['steps']} B={b['steps']}")
    if b.get("cache"):
        print(f"  cacheB      {b['cache']}")
    ok = True
    if a.get("loss_seq") and b.get("loss_seq"):
        same = a["loss_seq"] == b["loss_seq"] and a.get("step_seq") == b.get("step_seq")
        print(f"  GATE loss-seq identical: {same}  (rows A={a['log_rows']} B={b['log_rows']})")
        ok &= same
    else:
        print("  GATE loss-seq: cannot compare (missing logs)")
        ok = False
    ha = [h for p, h in hashes.items()
          if f"/{a['short']}/" in p and p.endswith("ckpt_final.pt")]
    hb = [h for p, h in hashes.items()
          if f"/{b['short']}/" in p and p.endswith("ckpt_final.pt")]
    ck_same = len(ha) == 1 and len(hb) == 1 and ha[0] == hb[0]
    print(f"  GATE ckpt_final sha256 identical: {ck_same}  ({ha[:1]} vs {hb[:1]})")
    ok &= ck_same
    print()
    return ok


def read_hashes(name):
    out = {}
    path = os.path.join(LOGS, name)
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    out[parts[1]] = parts[0]
    return out


def main():
    s2h = read_hashes("s2_ckpt_hashes.txt")
    s3h = read_hashes("s3_ckpt_hashes.txt")
    pairs = [
        ("s2-pair1", leg("s2_a1_off", "s2_a1", "a1"), leg("s2_b1_on", "s2_b1", "b1"), s2h),
        ("s2-pair2", leg("s2_a2_off", "s2_a2", "a2"), leg("s2_b2_on", "s2_b2", "b2"), s2h),
        ("s2-pair3", leg("s2_a3_off", "s2_a3", "a3"), leg("s2_b3_on", "s2_b3", "b3"), s2h),
        ("s3", leg("s3_a_off", "s3_a_off", "a_off"), leg("s3_b_on", "s3_b_on", "b_on"), s3h),
    ]
    allok = True
    compared = 0
    for name, a, b, hashes in pairs:
        if a.get("summary_error") and b.get("summary_error"):
            print(f"== {name}: both summaries missing ({a['summary_error']}); skipped\n")
            continue
        compared += 1
        for x in (a, b):
            print(f"{x['tag']}: wall={fmt(x['wall'])} steps={x['steps']} final={x['final_loss']} "
                  f"extract={fmt(x['extract_wall'])} fwd={x['forwards']} samples={x['samples']} "
                  f"rss={fmt(x['host_rss_gb'], 2, 'G')} cache={x['cache'] if x['cache'] else '-'} "
                  f"phases={x['phases']}")
        print()
        allok &= compare(name, a, b, hashes)
    if compared == 0:
        print("NO DATA (no leg summaries present)")
        return 1
    print("ALL GATES PASS" if allok else "GATES NOT ALL PASSING / incomplete")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
