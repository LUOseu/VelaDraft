"""Benchmark harness: manifest schema + prompt-source resolution (no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.benchmark.harness import load_workloads, select_workloads, validate_workload
from vlm_dflash.eval.run import load_eval_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = REPO_ROOT / "benchmarks" / "manifests"


def test_repo_manifests_load_and_validate():
    workloads = load_workloads(MANIFESTS)
    names = {workload["name"] for workload in workloads}
    assert {"caption_general", "textvqa", "docvqa", "chartqa"} <= names
    for workload in workloads:
        assert workload["status"] in {"validated", "scaffold", "planned"}
        if workload["status"] == "planned":
            assert workload["prompt_source"] is None
        else:
            assert isinstance(workload["prompt_source"], dict)


def test_validate_workload_rejects_bad_manifests():
    base = {
        "name": "x",
        "description": "d",
        "status": "scaffold",
        "prompt_source": {"type": "run_val"},
        "num_prompts": 8,
        "max_new_tokens": 16,
    }
    validate_workload(base, Path("manifest.json"))

    with pytest.raises(ValueError, match="missing keys"):
        validate_workload({key: value for key, value in base.items() if key != "name"}, Path("m.json"))
    with pytest.raises(ValueError, match="status"):
        validate_workload({**base, "status": "unknown"}, Path("m.json"))
    with pytest.raises(ValueError, match="prompt_source"):
        validate_workload({**base, "prompt_source": None}, Path("m.json"))
    with pytest.raises(ValueError, match=">= 1"):
        validate_workload({**base, "num_prompts": 0}, Path("m.json"))


def test_select_workloads_unknown_name_fails():
    workloads = load_workloads(MANIFESTS)
    assert [w["name"] for w in select_workloads(workloads, ["caption_general"])] == ["caption_general"]
    with pytest.raises(ValueError, match="Unknown workload"):
        select_workloads(workloads, ["nope"])


def test_load_eval_rows_sharegpt_jsonl(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    media = tmp_path / "images"
    media.mkdir()
    (data / "prompts.jsonl").write_text(
        json.dumps(
            {
                "id": "a",
                "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}],
                "images": ["a.jpg"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data / "meta.json").write_text(json.dumps({"media_dir": str(media)}), encoding="utf-8")

    rows, media_dir = load_eval_rows(
        {"type": "sharegpt_jsonl", "path": str(data / "prompts.jsonl")}, tmp_path
    )
    assert [row["id"] for row in rows] == ["a"]
    assert media_dir == media

    rows, _ = load_eval_rows(None, tmp_path, default_split="prompts.jsonl")
    assert [row["id"] for row in rows] == ["a"]

    with pytest.raises(ValueError, match="Unknown prompt_source"):
        load_eval_rows({"type": "weird"}, tmp_path)


def test_load_eval_rows_relative_path_resolves_against_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fixed.jsonl").write_text(
        json.dumps({"id": "b", "conversations": [], "images": ["b.jpg"]}) + "\n", encoding="utf-8"
    )
    (run_dir / "fixed_meta.json").write_text(json.dumps({"media_dir": "images"}), encoding="utf-8")
    rows, media_dir = load_eval_rows(
        {"type": "sharegpt_jsonl", "path": "fixed.jsonl", "meta": "fixed_meta.json"}, run_dir
    )
    assert [row["id"] for row in rows] == ["b"]
    assert media_dir == (run_dir / "images").resolve()
