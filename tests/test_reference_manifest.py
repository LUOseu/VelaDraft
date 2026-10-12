"""Reference manifest: frozen DFlash baselines + refactor regression gates."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "tests" / "reference_dflash" / "manifest.json"
BLOCK_SIZE = 8


def _load() -> dict:
    with MANIFEST.open() as handle:
        return json.load(handle)


def test_manifest_shape_and_baselines():
    data = _load()
    assert data["format"] == "vlm-dflash-reference-manifest-v1"
    for key in (
        "metric_definitions",
        "baselines",
        "checkpoint_semantics",
        "feature_semantics",
        "regression_gates",
        "remeasurements",
    ):
        assert key in data, f"manifest missing {key}"
    assert data["baselines"], "no frozen baselines recorded"
    for row in data["baselines"]:
        for key in (
            "name",
            "algorithm",
            "target",
            "run_dir",
            "prompts",
            "max_new_tokens",
            "al",
            "al_random_init_drafter",
            "per_slot_acceptance",
            "token_exact",
        ):
            assert key in row, f"{row.get('name')}: missing {key}"
        assert row["algorithm"] == "dflash"
        assert row["run_dir"].startswith("runs/")
        assert row["token_exact"] is True, f"{row['name']}: frozen gate row must be token-exact"
        assert row["al"] > 1.0, f"{row['name']}: trained AL should beat the random-init floor"
        assert row["al_random_init_drafter"] < 1.1
        slots = row["per_slot_acceptance"]
        assert len(slots) == BLOCK_SIZE - 1
        assert all(0.0 <= value <= 1.0 for value in slots)
        assert slots == sorted(slots, reverse=True), f"{row['name']}: acceptance decays with depth"


def test_remeasurements_are_recorded():
    data = _load()
    assert data["remeasurements"], "no post-refactor re-measurements recorded"
    for entry in data["remeasurements"]:
        for key in ("date", "tag", "file", "prompts", "max_new_tokens"):
            assert key in entry, f"{entry.get('tag')}: missing {key}"
        assert entry["file"].startswith("runs/")
        assert entry["prompts"] > 0
        assert entry["max_new_tokens"] > 0


def test_regression_gates_include_token_exact():
    data = _load()
    gates = " ".join(data["regression_gates"]).lower()
    assert "token_exact" in gates
    assert "al" in gates
