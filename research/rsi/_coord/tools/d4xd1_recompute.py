#!/usr/bin/env python3
"""C0 independent recompute for the D4xD1 2x2 (lease L-W3-D4xD1).

Contract: research/rsi/_coord/proposals/d4xd1-001-c0.md (F1-F5) + D1 addendum
(research/rsi-d1:research/rsi/d1/proposals/d4xd1-001-d1-addendum.md) for
field semantics (call k <-> train_log row k+1; timed = calls 1..100).

Read-only: never writes into the D1 run tree. Usage:
  python3 d4xd1_recompute.py --root /home/lyh/rsi-d1/runs \
      [--status /home/lyh/rsi-d1/logs/d4xd1_status.log] [--out report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as stats
from pathlib import Path

CELLS = {"w3": 192, "mix": 335}
LEGS = [(c, ca, bu) for c in CELLS for ca in ("off", "on") for bu in ("off", "on")]
LOSS_BAND = 7.5e-3
GRAD_TOL = 6.0e-3
UPDATE_TOL = 1.0e-7
PEAK_RATIO = 0.75


def load_jsonl(path: Path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def leg_name(cell, cache, bucket):
    return f"d4xd1_{cell}_cache{cache}_bucket{bucket}"


def read_leg(root: Path, cell, cache, bucket):
    d = root / leg_name(cell, cache, bucket)
    out = {"dir": str(d), "exists": d.exists()}
    if not out["exists"]:
        return out
    tl = load_jsonl(d / "train" / "train_log.jsonl")
    out["loss_by_step"] = {r["step"]: r["loss"] for r in tl}
    ts = d / "train" / "train_summary.json"
    out["summary"] = json.loads(ts.read_text()) if ts.exists() else None
    steps = load_jsonl(d / "d1_capture" / "steps.jsonl")
    out["wall_by_call"] = {r["step"]: r.get("wall_s") for r in steps}
    out["anchors"] = [(r.get("call"), r.get("anchors_sha1"), r.get("keep_sha1"))
                      for r in load_jsonl(d / "d1_capture" / "anchors.jsonl")]
    parity = None
    for cand in (d / "parity_vs_ref.json", d / "d1_capture" / "parity_vs_ref.json"):
        if cand.exists():
            parity = json.loads(cand.read_text())
            break
    out["parity"] = parity
    return out


def med_walls(leg, lo=1, hi=100):
    walls = [leg["wall_by_call"].get(k) for k in range(lo, hi + 1)]
    walls = [w for w in walls if isinstance(w, (int, float))]
    return stats.median(walls) if walls else None


def loss_rel(leg, ref):
    rel, worst = {}, None
    for s in range(2, 102):
        lv, rv = leg["loss_by_step"].get(s), ref["loss_by_step"].get(s)
        if lv is None or rv in (None, 0):
            continue
        r = abs(lv - rv) / abs(rv)
        rel[s] = r
        if worst is None or r > rel[worst]:
            worst = s
    return rel, worst, (rel[worst] if worst else None)


def anchor_digest(leg):
    payload = json.dumps(leg["anchors"], sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lyh/rsi-d1/runs")
    ap.add_argument("--status", default="/home/lyh/rsi-d1/logs/d4xd1_status.log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    legs = {leg_name(*k): read_leg(root, *k) for k in LEGS}
    report = {"root": str(root), "legs": {}, "cells": {}, "status": {}}

    done = [n for n, L in legs.items() if L.get("summary")]
    print(f"legs with train_summary: {len(done)}/8")
    for n in sorted(legs):
        L = legs[n]
        if not L["exists"]:
            print(f"  [pending] {n}")
        elif L["summary"] is None:
            print(f"  [running/no-summary] {n}")
        else:
            s = L["summary"]
            fc = s.get("feature_source", {}).get("feature_cache", {})
            print(f"  [done] {n}: steps={s.get('steps')} final_loss={s.get('final_loss')} "
                  f"peak={s.get('peak_gpu_mem_gb'):.2f}GB wall={s.get('wall_seconds'):.1f}s "
                  f"nan={s.get('nan_seen')} cache(h/m/s)={fc.get('hits')}/{fc.get('misses')}/{fc.get('stores')}")

    # status log rc parse
    rc = {}
    sp = Path(args.status)
    if sp.exists():
        for line in sp.read_text().splitlines():
            parts = line.split()
            if line.startswith("end") and len(parts) >= 4:
                rc[parts[1]] = parts[2]
    report["status"]["rc"] = rc
    print("\nstatus rc:", {k: v for k, v in rc.items()})

    for cell, rows in CELLS.items():
        cell_key = f"{cell}"
        out = {"rows": rows, "F1": {}, "F2": {}, "F3": None, "F4": {}, "F5": {},
               "parity": {}, "anchor_invariant": None}
        report["cells"][cell_key] = out
        ref = legs[leg_name(cell, "off", "off")]
        if ref.get("summary") is None:
            print(f"\n[{cell}] ref leg not finished; skip")
            continue
        print(f"\n[{cell}] ref median step = {med_walls(ref):.4f}s | median(2..100) = {med_walls(ref,2,100):.4f}s")

        # ---- F1 ----
        for ca in ("off", "on"):
            for bu in ("off", "on"):
                n = leg_name(cell, ca, bu)
                L = legs[n]
                if L.get("summary") is None:
                    out["F1"][f"{ca}x{bu}"] = "pending"
                    continue
                rel, worst, w = loss_rel(L, ref)
                verdict = "PASS" if (w is not None and w <= LOSS_BAND) else "FAIL"
                out["F1"][f"{ca}x{bu}"] = {"max_rel": w, "worst_step": worst, "verdict": verdict}
                print(f"  F1 {ca}x{bu}: max loss_rel={w:.3e} @step {worst} ({verdict}; band {LOSS_BAND:.1e})")

        # ---- F2 / F3 ----
        for ca in ("off", "on"):
            for bu in ("off", "on"):
                n = leg_name(cell, ca, bu)
                L = legs[n]
                if L.get("summary") is None:
                    continue
                fc = L["summary"].get("feature_source", {}).get("feature_cache", {})
                h, m, s_, e = fc.get("hits"), fc.get("misses"), fc.get("stores"), fc.get("evictions")
                if ca == "on":
                    ok = (h or 0) > 0 and (s_ or 0) > 0 and (h or 0) >= 0.5 * rows
                else:
                    ok = (h or 0) == 0 and (s_ or 0) == 0
                out["F2"][f"{ca}x{bu}"] = {"hits": h, "misses": m, "stores": s_, "evictions": e,
                                            "verdict": "PASS" if ok else "FAIL"}
                print(f"  F2 {ca}x{bu}: hits={h} misses={m} stores={s_} evict={e} -> "
                      f"{'PASS' if ok else 'FAIL'}")
        p_on_off = None; p_on_on = None
        if legs[leg_name(cell, "on", "off")].get("summary"):
            p_on_off = legs[leg_name(cell, "on", "off")]["summary"].get("peak_gpu_mem_gb")
        if legs[leg_name(cell, "on", "on")].get("summary"):
            p_on_on = legs[leg_name(cell, "on", "on")]["summary"].get("peak_gpu_mem_gb")
        if p_on_off and p_on_on:
            ratio = p_on_on / p_on_off
            out["F3"] = {"peak_on_on": p_on_on, "peak_on_off": p_on_off, "ratio": ratio,
                         "verdict": "PASS" if ratio <= PEAK_RATIO else "FAIL"}
            print(f"  F3 peak(on,on)/peak(on,off) = {p_on_on:.2f}/{p_on_off:.2f} = {ratio:.3f} "
                  f"({'PASS' if ratio <= PEAK_RATIO else 'FAIL'}; gate <= {PEAK_RATIO})")
        else:
            print("  F3 pending (cache_on legs not finished)")

        # ---- F4 ----
        for ca in ("off", "on"):
            base = legs[leg_name(cell, ca, "off")]
            regr = legs[leg_name(cell, ca, "on")]
            if base.get("summary") is None or regr.get("summary") is None:
                out["F4"][ca] = "pending"; print(f"  F4 cache={ca}: pending"); continue
            mb, mr = med_walls(base), med_walls(regr)
            ratio = mb / mr if mr else None
            verdict = "PASS" if (ratio is not None and ratio >= 1.0) else "FAIL"
            out["F4"][ca] = {"median_canon": mb, "median_regroup": mr, "ratio": ratio, "verdict": verdict}
            print(f"  F4 cache={ca}: median canon={mb:.4f}s regroup={mr:.4f}s ratio={ratio:.4f} ({verdict})")
            if ca == "on" and out["F4"].get("off") not in (None, "pending"):
                pass

        # ---- F5 (descriptive) ----
        for ca in ("off", "on"):
            for bu in ("off", "on"):
                L = legs[leg_name(cell, ca, bu)]
                if L.get("summary") is not None:
                    out["F5"].setdefault(f"median_{ca}x{bu}", med_walls(L))
        print(f"  F5 medians: {out['F5']}")

        # ---- parity ----
        for n in [leg_name(cell, ca, bu) for ca in ("off", "on") for bu in ("off", "on")]:
            L = legs[n]
            if L.get("parity") is None:
                out["parity"][n] = "pending"
                continue
            par = L["parity"].get("parity", {})
            entry = {}
            for kind, tol in (("grads", GRAD_TOL), ("updated", UPDATE_TOL)):
                for idx in ("0", "50", "100"):
                    cellp = par.get(kind, {}).get(idx)
                    if cellp is None:
                        entry[f"{kind}_{idx}"] = None
                    else:
                        w = cellp.get("worst_rel")
                        entry[f"{kind}_{idx}"] = {"worst_rel": w, "n": cellp.get("n_tensors"),
                                                   "ok": (w is not None and w <= tol)}
            out["parity"][n] = entry
            if n != leg_name(cell, "off", "off"):
                worst_g = max([v["worst_rel"] for k, v in entry.items() if k.startswith("grads") and v], default=None)
                worst_u = max([v["worst_rel"] for k, v in entry.items() if k.startswith("updated") and v], default=None)
                print(f"  parity {n.split('_',1)[1]}: worst grad_rel={worst_g} update_rel={worst_u}")

        # ---- anchor invariant (frozen table = first 101 calls; trailing diag calls reported) ----
        FROZEN_CALLS = 101
        names = [leg_name(cell, ca, bu) for ca in ("off", "on") for bu in ("off", "on")]
        seqs = {n: legs[n]["anchors"] for n in names if legs[n]["exists"]}
        if len(seqs) == 4:
            mlen = min(len(v) for v in seqs.values())
            lengths = {n: len(v) for n, v in seqs.items()}
            if mlen < FROZEN_CALLS:
                out["anchor_invariant"] = {"full_lengths": lengths, "prefix_len": mlen,
                                            "equal": None, "note": f"< {FROZEN_CALLS} rows present; pending"}
                print(f"  anchor invariant PENDING (min len {mlen} < {FROZEN_CALLS}) lengths="
                      f"{ {n.split('_',1)[1]: len(v) for n, v in seqs.items()} }")
            else:
                base = None; ok = True
                for n, v in seqs.items():
                    pref = v[:FROZEN_CALLS]
                    if base is None:
                        base = pref
                    elif pref != base:
                        ok = False
                tails = {}
                for n, v in seqs.items():
                    if len(v) > FROZEN_CALLS:
                        payload = json.dumps(v[FROZEN_CALLS:], sort_keys=True).encode()
                        tails[n] = {"len": len(v) - FROZEN_CALLS,
                                    "sha1_16": hashlib.sha1(payload).hexdigest()[:16]}
                out["anchor_invariant"] = {"full_lengths": lengths, "prefix_len": FROZEN_CALLS,
                                            "equal": ok, "trailing": tails,
                                            "digests": {n: anchor_digest(legs[n]) for n in seqs}}
                print(f"  anchor invariant (frozen {FROZEN_CALLS} calls): {'EQUAL' if ok else 'MISMATCH'} "
                      f"lengths={ {n.split('_',1)[1]: len(v) for n, v in seqs.items()} } "
                      f"trailing={ {k.split('_',1)[1]: v for k, v in tails.items()} }")
        else:
            print(f"  anchor invariant pending ({len(seqs)}/4 legs present)")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")
    else:
        print("\n(report not saved; use --out to persist JSON)")


if __name__ == "__main__":
    main()
