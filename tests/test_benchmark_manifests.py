"""Benchmark harness: manifest schema + prompt-source resolution (no model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.benchmark.adapters import (
    export_chartqa,
    export_docvqa,
    pinned_prompt_manifest,
    select_indices,
)
from vlm_dflash.benchmark.harness import (
    load_workloads,
    resolve_prompt_source,
    select_workloads,
    validate_workload,
)
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


def test_resolve_prompt_source_repo_relative():
    source = resolve_prompt_source(
        {"type": "sharegpt_jsonl", "path": "benchmarks/data/textvqa/textvqa_v1.jsonl", "meta": "benchmarks/data/textvqa/meta.json"}
    )
    assert Path(source["path"]).is_absolute()
    assert str(source["path"]).endswith("benchmarks/data/textvqa/textvqa_v1.jsonl")
    assert source["path"].startswith(str(REPO_ROOT))
    # run_val sources have no paths and pass through untouched
    assert resolve_prompt_source({"type": "run_val"}) == {"type": "run_val"}
    assert resolve_prompt_source(None) is None


def test_resolve_prompt_source_env_default_and_override(monkeypatch):
    manifest_source = {
        "type": "sharegpt_jsonl",
        "path": "${VLM_DFLASH_BENCH_DATA:-benchmarks/data}/textvqa/textvqa_v1.jsonl",
        "meta": "${VLM_DFLASH_BENCH_DATA:-benchmarks/data}/textvqa/meta.json",
    }
    monkeypatch.delenv("VLM_DFLASH_BENCH_DATA", raising=False)
    default = resolve_prompt_source(manifest_source)
    assert default["path"] == str(REPO_ROOT / "benchmarks" / "data" / "textvqa" / "textvqa_v1.jsonl")
    monkeypatch.setenv("VLM_DFLASH_BENCH_DATA", "/mnt/bench")
    overridden = resolve_prompt_source(manifest_source)
    assert overridden["path"] == "/mnt/bench/textvqa/textvqa_v1.jsonl"


def test_textvqa_selection_is_deterministic_and_bounded():
    first = select_indices(5000, 128, seed=42)
    assert first == select_indices(5000, 128, seed=42)
    assert first != select_indices(5000, 128, seed=43)
    assert len(first) == len(set(first)) == 128
    assert first == sorted(first)
    assert min(first) >= 0 and max(first) < 5000
    with pytest.raises(ValueError, match="rows"):
        select_indices(16, 32, seed=42)


def test_image_suffix_fails_loud_on_unknown_magic():
    from vlm_dflash.benchmark.adapters import _image_suffix

    assert _image_suffix(b"\x89PNG\r\n\x1a\nxxx") == ".png"
    assert _image_suffix(b"\xff\xd8\xffxxx") == ".jpg"
    with pytest.raises(ValueError, match="image bytes"):
        _image_suffix(b"GIF89a")


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
_PNG = b"\x89PNG\r\n\x1a\n" + b"fixture-png-bytes"
_JPG = b"\xff\xd8\xff" + b"fixture-jpg-bytes"


def _write_table(path: Path, rows: dict[str, list]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows), path)


def test_export_chartqa_from_fixture(tmp_path):
    source = tmp_path / "chartqa-src"
    _write_table(
        source / "data" / "test-00000-of-00001.parquet",
        {
            "type": ["human_test", "human_test", "augmented_test", "human_test", "augmented_test", "human_test"],
            "question": [f"question {i}?" for i in range(6)],
            "answer": [str(i) for i in range(6)],
            "image": [
                {"bytes": _PNG + bytes([i]), "path": f"chart{i % 2}.png"} for i in range(6)
            ],
        },
    )
    out = tmp_path / "out"
    manifest = export_chartqa(source, out, num_prompts=4, seed=42)
    assert manifest["dataset"] == "lmms-lab/ChartQA"
    assert manifest["num_rows_in_split"] == 6 and manifest["num_prompts"] == 4
    prompts = [json.loads(line) for line in (out / "chartqa_v1.jsonl").read_text().splitlines()]
    assert len(prompts) == 4
    ids = [row["id"] for row in prompts]
    assert ids == sorted(ids)
    assert all(row["images"][0].endswith(".png") for row in prompts)
    assert all(len(row["conversations"]) == 2 for row in prompts)
    # Deterministic re-run over the same shard reproduces the same subset/hashes.
    rerun = export_chartqa(source, tmp_path / "out2", num_prompts=4, seed=42)
    assert rerun["rows_sha256"] == manifest["rows_sha256"]
    pinned = pinned_prompt_manifest(manifest)
    assert {"id", "source_index", "question_sha256", "image_sha256"} <= set(pinned[0])
    assert "image" not in pinned[0]
    with pytest.raises(FileNotFoundError, match="ChartQA"):
        export_chartqa(tmp_path / "missing", out, num_prompts=4, seed=42)


def test_export_docvqa_from_fixture(tmp_path):
    source = tmp_path / "docvqa-src"
    _write_table(
        source / "DocVQA" / "validation-00000-of-00006.parquet",
        {
            "questionId": ["101", "205", "57", "999"],
            "question": [f"doc question {i}?" for i in range(4)],
            "question_types": [["figure/diagram"], ["others"], ["layout"], ["figure/diagram"]],
            "image": [
                {"bytes": _PNG + bytes([i]), "path": f"page{i}.png"} for i in range(3)
            ]
            + [{"bytes": _JPG, "path": "page3.jpg"}],
            "docId": [7, 4, 9, 2],
            "ucsf_document_id": [f"doc{i}" for i in range(4)],
            "ucsf_document_page_no": [str(i + 1) for i in range(4)],
            "answers": [["a"], ["b"], ["c"], ["d"]],
            "data_split": ["val"] * 4,
        },
    )
    out = tmp_path / "out"
    manifest = export_docvqa(source, out, num_prompts=3, seed=42)
    assert manifest["config"] == "DocVQA" and manifest["split"] == "validation"
    assert manifest["num_rows_in_split"] == 4 and manifest["num_prompts"] == 3
    prompts = [json.loads(line) for line in (out / "docvqa_v1.jsonl").read_text().splitlines()]
    ids = [row["id"] for row in prompts]
    # Sorted by numeric questionId (57 < 101 < 999), not by shard order.
    assert ids == ["docvqa-000057", "docvqa-000101", "docvqa-000999"]
    suffixes = {row["id"]: row["images"][0].rsplit(".", 1)[1] for row in prompts}
    assert suffixes["docvqa-000057"] == "png"
    assert suffixes["docvqa-000999"] == "jpg"
    rerun = export_docvqa(source, tmp_path / "out2", num_prompts=3, seed=42)
    assert rerun["rows_sha256"] == manifest["rows_sha256"]
    pinned = pinned_prompt_manifest(manifest)
    assert {"id", "question_id", "doc_id", "question_sha256", "image_sha256"} <= set(pinned[0])
    with pytest.raises(FileNotFoundError, match="DocVQA"):
        export_docvqa(tmp_path / "missing", out, num_prompts=3, seed=42)
