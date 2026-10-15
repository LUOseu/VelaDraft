"""Validity-gate regressions for research/convergence/analyze_quality.py.

The frozen prereg (PREREGISTRATION.md section 5) requires leg validity checks
before any statistic: a construction with four seeds, one prompt each,
token_exact=false and no train summaries must NOT produce PASS. Invalid or
incomplete evidence must yield HOLD with reasons.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "research" / "convergence" / "analyze_quality.py"


def _write_leg(root: Path, name: str, per_prompt_al: dict[str, float], *,
               token_exact: bool = True, train_summary: bool = True,
               aggregate_al: float | None = None) -> Path:
    leg = root / name
    (leg / "train").mkdir(parents=True, exist_ok=True)
    (leg / "eval").mkdir(parents=True, exist_ok=True)
    if train_summary:
        (leg / "train" / "train_summary.json").write_text(json.dumps({
            "loss_decreased": True, "nan_seen": False, "skipped_no_anchor_batches": 0,
            "final_loss": 0.001,
        }))
    ids = sorted(per_prompt_al)
    runs = {"ckpt_final.pt": {
        "mean_al": aggregate_al if aggregate_al is not None else sum(per_prompt_al.values()) / len(ids),
        "acceptance_rate": 0.02, "per_slot_acceptance": [0.1, 0.05],
    }}
    if not ids:
        runs = {}
    (leg / "eval" / "eval_result.json").write_text(json.dumps({
        "runs": runs,
        "equivalence_check": {
            "num_prompts": len(ids), "token_exact": token_exact,
            "runs": {"ckpt_final.pt": {"num_prompts": len(ids), "token_exact": token_exact,
                                       "failed_ids": []}},
        },
    }))
    rows = []
    for prompt_id in ids:
        al = per_prompt_al[prompt_id]
        rows.append({
            "checkpoint": "ckpt_final.pt", "id": prompt_id,
            "verify_steps": 100, "new_tokens": int(round(al * 100)),
            "accepted_tokens": 5, "draft_tokens": 100,
        })
    (leg / "eval" / "eval_samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return leg


def _spec(tmp: Path, arms: dict, *, prompts_per_leg: int | None = 2,
          registered: dict | None = None) -> Path:
    data = {
        "reference_arm": "canonical", "arms": arms,
        "gate_al_rel": -0.01, "gate_acc_abs": -0.005,
        "bootstrap": 2000, "bootstrap_seed": 20261015,
    }
    if prompts_per_leg is not None:
        data["prompts_per_leg"] = prompts_per_leg
    if registered is not None:
        data["registered"] = registered
    path = tmp / "quality_data.json"
    path.write_text(json.dumps(data))
    return path


def _run(tmp: Path, spec: Path) -> dict:
    out = tmp / "verdict.json"
    proc = subprocess.run([sys.executable, str(ANALYZER), "--data", str(spec), "--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_invalid_construction_holds_not_pass(tmp_path):
    """Reviewer repro: 4 seeds x 1 prompt, token_exact=false, no train summary -> HOLD."""
    arms = {}
    for arm in ("canonical", "grid2"):
        arms[arm] = {}
        for seed in (42, 43, 44, 45):
            leg = _write_leg(tmp_path, f"{arm}_s{seed}", {"p0": 1.0},
                             token_exact=False, train_summary=False)
            arms[arm][str(seed)] = str(leg)
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=64))
    block = verdict["arms"]["grid2"]
    assert block["verdict"] == "HOLD"
    joined = " ".join(block["hold_reasons"])
    assert "token_exact" in joined and "train_summary" in joined and "prompts=" in joined


def test_registered_statistic_uses_prompt_means(tmp_path):
    """R_s = mean_p(dAL)/mean_p(AL_canon); the pooled aggregate must be ignored."""
    arms = {}
    for arm, al in (("canonical", {"p0": 1.0, "p1": 2.0}), ("grid2", {"p0": 1.5, "p1": 2.5})):
        arms[arm] = {"42": str(_write_leg(tmp_path, f"{arm}_s42", al, aggregate_al=9.9))}
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=2))
    block = verdict["arms"]["grid2"]
    assert block["verdict"] in ("PASS", "FAIL")
    row = block["seeds"][0]
    assert abs(row["rel"] - (0.5 / 1.5)) < 1e-9          # registered formula
    assert abs(row["canonical_al"] - 1.5) < 1e-9          # prompt mean, not 9.9
    assert abs(block["mean_rel_al_drop"] - (0.5 / 1.5)) < 1e-9
    assert "mean_p(AL_canon)" in verdict["statistic"]


def test_prompt_id_mismatch_holds_instead_of_intersecting(tmp_path):
    arms = {}
    canon = {"p0": 1.0, "p1": 2.0}
    arm = {"p0": 1.1, "p2": 2.2}  # same count, different ids: no silent intersection
    arms["canonical"] = {"42": str(_write_leg(tmp_path, "canonical_s42", canon))}
    arms["grid2"] = {"42": str(_write_leg(tmp_path, "grid2_s42", arm))}
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=None))
    block = verdict["arms"]["grid2"]
    assert block["verdict"] == "HOLD"
    assert any("prompt-id sets differ" in reason for reason in block["hold_reasons"])


def test_short_leg_holds_on_prompt_count(tmp_path):
    arms = {}
    arms["canonical"] = {"42": str(_write_leg(tmp_path, "canonical_s42", {"p0": 1.0, "p1": 2.0}))}
    arms["grid2"] = {"42": str(_write_leg(tmp_path, "grid2_s42", {"p0": 1.1}))}
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=2))
    block = verdict["arms"]["grid2"]
    assert block["verdict"] == "HOLD"
    assert any("prompts=1" in reason for reason in block["hold_reasons"])


def test_missing_registered_seed_holds(tmp_path):
    arms = {}
    for arm in ("canonical", "grid2"):
        arms[arm] = {"42": str(_write_leg(tmp_path, f"{arm}_s42", {"p0": 1.0, "p1": 2.0}))}
    spec = _spec(tmp_path, arms, prompts_per_leg=2,
                 registered={"canonical": [42, 43], "grid2": [42, 43]})
    verdict = _run(tmp_path, spec)
    assert verdict["arms"]["grid2"]["verdict"] == "HOLD"
    assert any("registered seed legs missing" in reason for reason in verdict["invalid_evidence"])


def test_clear_regression_fails(tmp_path):
    arms = {}
    arms["canonical"] = {"42": str(_write_leg(tmp_path, "canonical_s42", {"p0": 2.0, "p1": 2.0}))}
    arms["grid2"] = {"42": str(_write_leg(tmp_path, "grid2_s42", {"p0": 1.0, "p1": 1.0}))}
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=2))
    assert verdict["arms"]["grid2"]["verdict"] == "FAIL"


def test_identical_arms_pass(tmp_path):
    arms = {}
    for arm in ("canonical", "grid2"):
        arms[arm] = {"42": str(_write_leg(tmp_path, f"{arm}_s42", {"p0": 1.0, "p1": 2.0}))}
    verdict = _run(tmp_path, _spec(tmp_path, arms, prompts_per_leg=2))
    block = verdict["arms"]["grid2"]
    assert block["verdict"] == "PASS"
    assert abs(block["mean_rel_al_drop"]) < 1e-12
