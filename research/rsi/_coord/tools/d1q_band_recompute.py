#!/usr/bin/env python3
"""C0 independent recompute for d1-006 (L-W3-D1Q) — held-out quality band/paired verdict.

Contract: research/rsi/d1/proposals/d1-006-d1-addendum.md §3 (frozen) + C0 02:45 orchestration:
  band_AL  = |AL_c(42)  - AL_c(43)|    (ckpt_final evals of the two canonical legs)
  band_acc = |acc_c(42) - acc_c(43)|
  PASS  <=> for s in {42,43}: |AL_r(s) - AL_c(s)| <= band_AL AND |acc_r(s) - acc_c(s)| <= band_acc
  a seed whose canonical or regroup leg did not converge (train_summary.loss_decreased false)
  -> that seed is 'undetermined' (never噪声当 PASS).

Read-only; stdlib only. Raw inputs are the server run dirs (mirrored locally):
  <root>/<leg>/{eval/eval_result*.json, train/train_summary.json, train/train_log.jsonl}
Usage:
  python3 d1q_band_recompute.py --root /tmp/c0_d1q [--out report.json]
  python3 d1q_band_recompute.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

LEGS = {
    "c42": "d1q_canon_s42",
    "c43": "d1q_canon_s43",
    "r42": "d1q_regroup_s42",
    "r43": "d1q_regroup_s43",
}
FINAL_METRICS = ("mean_al", "acceptance_rate", "per_slot_acceptance",
                 "num_samples", "new_tokens", "verify_steps", "draft_tokens")


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_leg(d: Path) -> dict:
    out: dict = {"dir": str(d), "exists": d.exists(), "files_sha256": {}}
    if not out["exists"]:
        return out
    ev_dir = d / "eval"
    ev = None
    for cand in [ev_dir / "eval_result.json", *sorted(ev_dir.glob("eval_result*.json"))]:
        if cand.exists():
            ev = _load(cand)
            out["eval_file"] = str(cand)
            out["files_sha256"][str(cand.relative_to(d))] = _sha256(cand)
            break
    if ev is not None:
        runs = ev.get("runs", {})
        out["eval_runs_keys"] = list(runs.keys())
        for want, tag in (("final", "final"), ("init", "init")):
            picked = next((r for k, r in runs.items() if tag in k.lower()), None)
            out[want] = ({m: picked.get(m) for m in FINAL_METRICS} if picked else None)
        eq = ev.get("equivalence_check", {})
        out["token_exact"] = eq.get("token_exact")
        out["settings"] = ev.get("settings", {})
    ts = d / "train" / "train_summary.json"
    if ts.exists():
        s = _load(ts)
        out["train"] = {m: s.get(m) for m in
                        ("steps", "final_loss", "first10_mean_loss", "last10_mean_loss",
                         "loss_decreased", "nan_seen", "wall_seconds")}
        out["files_sha256"][str(ts.relative_to(d))] = _sha256(ts)
    tl = d / "train" / "train_log.jsonl"
    if tl.exists():
        rows = [json.loads(l) for l in tl.read_text(encoding="utf-8").splitlines() if l.strip()]
        out["train_rows"] = len(rows)
        if rows:
            out["train_last_step"] = rows[-1].get("step")
        out["files_sha256"][str(tl.relative_to(d))] = _sha256(tl)
    return out


def compute(root: Path) -> dict:
    legs = {k: read_leg(root / name) for k, name in LEGS.items()}
    report: dict = {"root": str(root), "legs": legs, "pending": [], "verdict": None}
    for k, L in legs.items():
        if not L["exists"] or "eval_file" not in L or L.get("final") is None:
            report["pending"].append(k)

    def _f(k, m):
        v = legs[k]["final"].get(m)
        if v is None:
            raise SystemExit(f"leg {k}: final.{m} is None (cannot compute frozen verdict)")
        return float(v)

    if all(legs[k].get("final") is not None for k in ("c42", "c43")):
        band_al = abs(_f("c42", "mean_al") - _f("c43", "mean_al"))
        band_acc = abs(_f("c42", "acceptance_rate") - _f("c43", "acceptance_rate"))
        report["band"] = {
            "al": band_al, "acc": band_acc,
            "al_c42": _f("c42", "mean_al"), "al_c43": _f("c43", "mean_al"),
            "acc_c42": _f("c42", "acceptance_rate"), "acc_c43": _f("c43", "acceptance_rate"),
        }

    if report["pending"]:
        report["verdict"] = "pending: " + ",".join(report["pending"])
        return report

    compare = {}
    ok_all, undet = True, []
    for s in (42, 43):
        ck, rk = f"c{s}", f"r{s}"
        conv = all(legs[k].get("train", {}).get("loss_decreased") is not False for k in (ck, rk))
        item: dict = {"converged": conv}
        if conv:
            dal = abs(_f(rk, "mean_al") - _f(ck, "mean_al"))
            dacc = abs(_f(rk, "acceptance_rate") - _f(ck, "acceptance_rate"))
            item.update({
                "al_canon": _f(ck, "mean_al"), "al_regroup": _f(rk, "mean_al"),
                "d_al": dal, "acc_canon": _f(ck, "acceptance_rate"),
                "acc_regroup": _f(rk, "acceptance_rate"), "d_acc": dacc,
                "pass": bool(dal <= band_al and dacc <= band_acc),
            })
            ok_all = ok_all and item["pass"]
        else:
            undet.append(s)
        compare[f"seed{s}"] = item
    report["compare"] = compare
    report["verdict"] = ("undetermined (seed " + ",".join(map(str, undet)) + " not converged)"
                         if undet else ("PASS" if ok_all else "FAIL"))
    return report


def _write_synth_leg(root: Path, name: str, al: float, acc: float, converged: bool = True):
    d = root / name
    (d / "eval").mkdir(parents=True, exist_ok=True)
    (d / "train").mkdir(parents=True, exist_ok=True)
    (d / "eval" / "eval_result.json").write_text(json.dumps({
        "runs": {
            "ckpt_final.pt": {"mean_al": al, "acceptance_rate": acc, "per_slot_acceptance": [0.5],
                              "num_samples": 64, "new_tokens": 100, "verify_steps": 80, "draft_tokens": 120},
            "ckpt_init.pt": {"mean_al": 1.0, "acceptance_rate": 0.1},
        },
        "equivalence_check": {"token_exact": True},
        "settings": {"num_prompts": 64, "max_new_tokens": 128},
    }), encoding="utf-8")
    (d / "train" / "train_summary.json").write_text(json.dumps(
        {"steps": 200, "final_loss": 5.0, "loss_decreased": converged, "nan_seen": False}), encoding="utf-8")
    (d / "train" / "train_log.jsonl").write_text(
        "\n".join(json.dumps({"step": i, "loss": 5.0}) for i in range(1, 201)) + "\n", encoding="utf-8")


def _selftest() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_synth_leg(root, LEGS["c42"], 1.20, 0.50)
        _write_synth_leg(root, LEGS["c43"], 1.22, 0.51)
        _write_synth_leg(root, LEGS["r42"], 1.21, 0.505)
        _write_synth_leg(root, LEGS["r43"], 1.215, 0.505)
        r = compute(root)
        assert r["verdict"] == "PASS", r["verdict"]
        assert abs(r["band"]["al"] - 0.02) < 1e-12, r["band"]
        _write_synth_leg(root, LEGS["r43"], 1.30, 0.505)
        assert compute(root)["verdict"] == "FAIL"
        _write_synth_leg(root, LEGS["r43"], 1.215, 0.505)
        _write_synth_leg(root, LEGS["c43"], 1.22, 0.51, converged=False)
        assert compute(root)["verdict"].startswith("undetermined")
        (root / LEGS["r42"] / "eval" / "eval_result.json").unlink()
        assert compute(root)["verdict"].startswith("pending")
    print("selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/c0_d1q")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    report = compute(Path(args.root))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print("verdict:", report["verdict"])
    if report.get("band"):
        print(f"band: AL {report['band']['al']:.4f} / acc {report['band']['acc']:.4f}")
    for k in ("seed42", "seed43"):
        if report.get("compare", {}).get(k):
            print(k, json.dumps(report["compare"][k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
