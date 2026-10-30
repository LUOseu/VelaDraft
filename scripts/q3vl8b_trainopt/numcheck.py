"""Numerical check (contract section 10): every arm against Reference after one real update.

    python numcheck.py <campaign_dir>

For each method and stage, the ``numcheck_*`` runs forked the same seed-42 state and
ran exactly one optimizer update on the same real batch. Compares the logged loss
components, supervision counts and gradient norm, and the fp32 master update: for
every parameter tensor, ``||opt_after - ref_after|| / ||ref_after - init||`` (how far
the Optimized update lands from the Reference update, relative to the update itself).
Writes ``checks/numerical_check.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from make_configs import TRANSITION_STEP

STAGES = {"dflash2": {"single_stage": 0}, "dfly": {"stage1": 0, "stage2": TRANSITION_STEP}}
ARMS = {"dflash2": ["ref", "d1", "opt", "ref_repeat"], "dfly": ["ref", "compaction", "opt", "ref_repeat"]}
COUNT_KEYS = ("valid_tokens", "w_effective_positions", "w_supervised_rows", "w_kept_rows", "w_samples")
# BF16 drafter, fp32 masters: loss components agree to bf16 rounding of the logits.
LOSS_RTOL = 2e-2


def first_row(run: Path) -> dict:
    return json.loads((run / "train" / "train_log.jsonl").read_text().splitlines()[0])


def masters(run: Path) -> dict:
    return torch.load(run / "train" / "ckpt_last.pt", map_location="cpu", weights_only=False)["master"]


def main() -> None:
    camp = Path(sys.argv[1])
    work = camp / "work"
    report = {"tolerances": {"loss_rtol": LOSS_RTOL}, "checks": []}
    ok_all = True
    for method, stages in STAGES.items():
        init = torch.load(work / f"init_{method}_s42" / "train" / "init_state.pt", map_location="cpu",
                          weights_only=False)["master"]
        for stage in stages:
            runs = {arm: work / f"numcheck_{method}_{stage}_{arm}" for arm in ARMS[method]}
            ref_row = first_row(runs["ref"])
            ref_master = masters(runs["ref"])
            for arm in ARMS[method][1:]:
                if not (runs[arm] / "train" / "ckpt_last.pt").exists():
                    report["checks"].append({"method": method, "stage": stage, "arm": arm, "passed": None,
                                             "status": "not_run"})
                    print(f"{method} {stage} {arm}: not run yet")
                    continue
                row = first_row(runs[arm])
                master = masters(runs[arm])
                loss_keys = sorted(
                    key for key in ref_row
                    if key in row and isinstance(ref_row[key], float) and not key.startswith(("t_", "w_", "mem_"))
                    and key not in ("update_s", "aux_s", "data_s", "compute_host_s", "optim_host_s", "lr")
                )
                losses = {key: {"ref": ref_row[key], arm: row[key],
                                "rel_diff": abs(row[key] - ref_row[key]) / max(abs(ref_row[key]), 1e-12)}
                          for key in loss_keys}
                counts = {key: {"ref": ref_row.get(key), arm: row.get(key)} for key in COUNT_KEYS if key in ref_row}
                ratios = {}
                for name, ref_after in ref_master.items():
                    step = (ref_after - init[name]).float().norm()
                    gap = (master[name] - ref_after).float().norm()
                    ratios[name] = float(gap / step) if step > 0 else float(gap)
                worst = max(ratios.items(), key=lambda item: item[1])
                counts_equal = all(value["ref"] == value[arm] for value in counts.values())
                losses_close = all(entry["rel_diff"] <= LOSS_RTOL for key, entry in losses.items()
                                   if key in ("loss", "unary_kl", "selector_ce") or key.startswith("recipe_"))
                passed = counts_equal and losses_close
                ok_all &= passed
                report["checks"].append({
                    "method": method, "stage": stage, "arm": arm, "passed": passed,
                    "losses": losses, "counts": counts, "counts_equal": counts_equal,
                    "grad_norm": {"ref": ref_row.get("grad_norm"), arm: row.get("grad_norm")},
                    "update_gap_over_update": {"median": float(torch.tensor(list(ratios.values())).median()),
                                               "max": worst[1], "max_tensor": worst[0]},
                    "head_rows": {key: {"ref": ref_row.get(key), arm: row.get(key)}
                                  for key in ("w_teacher_head_rows", "w_student_head_rows", "w_scored_rows")},
                })
                print(f"{method} {stage} {arm}: passed={passed} loss {ref_row['loss']:.6f} vs {row['loss']:.6f} "
                      f"update gap median {report['checks'][-1]['update_gap_over_update']['median']:.2e}")
    report["passed"] = ok_all
    report["complete"] = all(check.get("status") != "not_run" for check in report["checks"])
    out = camp / "checks"
    out.mkdir(exist_ok=True)
    (out / "numerical_check.json").write_text(json.dumps(report, indent=1))
    print("NUMCHECK_PASSED" if ok_all else "NUMCHECK_FAILED")


if __name__ == "__main__":
    main()
