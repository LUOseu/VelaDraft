"""Regressions for research/convergence/a100/run_matrix.py (PR #9 review fixes).

Covers: per-drafter effective configs (controlled track must not run dflash for
every algorithm), data binding, per-workload benchmark completion, and the
summarize field mapping (benchmark reports store ``al``, not ``mean_al``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER_DIR = REPO / "research" / "convergence" / "a100"


def load_runner():
    spec = importlib.util.spec_from_file_location("conv_run_matrix", RUNNER_DIR / "run_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner():
    return load_runner()


def mini_matrix(tmp_path: Path, workloads=("w1", "w2")) -> dict:
    return {
        "repo": str(REPO), "python": sys.executable, "runs_root": str(tmp_path / "runs"),
        "controlled_config": "configs/qwen2_5_vl_3b_2k.yaml",
        "drafters": {"dflash": {"native_config": "configs/qwen2_5_vl_3b_2k.yaml",
                                "regroup_support": False, "controlled_overrides": {}}},
        "budget": {"steps": 2, "seeds": [42]},
        "tracks": ["controlled"],
        "regroup_sets": {"off": [], "grid2": []},
        "stages": ["train", "eval", "benchmark"],
        "eval": {"num_prompts": 4, "max_new_tokens": 8},
        "isolation": {"test_workloads": list(workloads)},
        "benchmark": {"reports_dir": str(tmp_path / "reports"), "manifests_dir": str(tmp_path / "manifests")},
    }


def test_controlled_configs_resolve_per_algorithm(tmp_path, runner):
    env = {
        "VELADRAFT_REPO": str(REPO), "VELADRAFT_PYTHON": sys.executable,
        "VELADRAFT_RUNS_ROOT": str(tmp_path / "runs"),
        "VELADRAFT_TARGET_MODEL": "dummy", "VELADRAFT_DATA_ROOT": str(tmp_path / "data"),
    }
    matrix = runner.load_matrix(RUNNER_DIR / "matrix.json", env)
    entries = runner.build_entries(matrix)
    assert runner.materialize_configs(matrix, entries) == []
    sys.path.insert(0, str(REPO))
    from vlm_dflash.algorithms import algorithm_name

    import yaml
    checked = set()
    for entry in entries:
        assert Path(entry["config"]).exists()
        assert entry["config_sha256"]
        key = (entry["drafter"], entry["track"])
        if key in checked:
            continue
        checked.add(key)
        cfg = yaml.safe_load(Path(entry["config"]).read_text())
        assert algorithm_name(cfg.get("algorithm")) == entry["drafter"], key
        assert runner.validate_config_doc(cfg) is None, key
    assert ("jetspec", "controlled") in checked  # pins the structural overrides path
    jetspec = yaml.safe_load(Path(next(e for e in entries if e["id"].startswith("jetspec__controlled"))["config"]).read_text())
    assert jetspec["capture"]["store_final_hidden"] is True
    assert jetspec["drafter"]["loss_decay_gamma"] is None
    dflare = yaml.safe_load(Path(next(e for e in entries if e["id"].startswith("dflare__controlled"))["config"]).read_text())
    assert dflare["drafter"]["capture_layer_ids"] == [1, 17, 33]


def test_stage_commands_pin_config_and_unique_benchmark_names(tmp_path, runner):
    matrix = mini_matrix(tmp_path)
    entry = runner.build_entries(matrix)[0]
    train_cmd = runner.stage_commands(matrix, entry, "train", resume_train=False)[0]
    eval_cmd = runner.stage_commands(matrix, entry, "eval", resume_train=False)[0]
    bench_cmds = runner.stage_commands(matrix, entry, "benchmark", resume_train=False)
    assert entry["config"] in train_cmd and entry["config"] in eval_cmd
    names = [cmd[cmd.index("--name") + 1] for cmd in bench_cmds]
    assert len(names) == len(set(names)) == 2
    assert all("--config" in cmd for cmd in bench_cmds)
    reports = [runner.benchmark_report_path(matrix, entry, w) for w in ("w1", "w2")]
    assert len(set(reports)) == 2


def test_stage_done_requires_complete_artifacts(tmp_path, runner):
    matrix = runner.load_matrix_for_test = mini_matrix(tmp_path)
    entry = runner.build_entries(matrix)[0]
    run_dir = Path(entry["run_dir"])
    # benchmark: nothing yet
    assert not runner.stage_done(matrix, entry, "benchmark")
    reports = Path(matrix["benchmark"]["reports_dir"])
    reports.mkdir(parents=True)
    report_path = runner.benchmark_report_path(matrix, entry, "w1")
    report_path.write_text(json.dumps({"format": "vlm-dflash-benchmark-v1",
                                       "workloads": {"w1": {"status": "measured", "al": 2.1}}}))
    assert not runner.stage_done(matrix, entry, "benchmark")  # w2 still missing
    # a legacy common-name file must not satisfy the per-workload check
    (reports / f"benchmark_{run_dir.name}.json").write_text(json.dumps(
        {"workloads": {"w1": {"status": "measured", "al": 9.9}, "w2": {"status": "measured", "al": 9.9}}}))
    assert not runner.stage_done(matrix, entry, "benchmark")
    runner.benchmark_report_path(matrix, entry, "w2").write_text(json.dumps(
        {"workloads": {"w2": {"status": "planned"}}}))
    assert runner.stage_done(matrix, entry, "benchmark")
    # empty/corrupt report flips back to not done
    report_path.write_text("{}")
    assert not runner.stage_done(matrix, entry, "benchmark")
    # train + eval parse checks
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "train" / "train_summary.json").write_text("{}")
    assert not runner.stage_done(matrix, entry, "train")
    (run_dir / "train" / "train_summary.json").write_text(json.dumps(
        {"steps": 2, "final_loss": 0.01, "wall_seconds": 3.0}))
    assert runner.stage_done(matrix, entry, "train")
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "eval" / "eval_result.json").write_text("{}")
    assert not runner.stage_done(matrix, entry, "eval")
    (run_dir / "eval" / "eval_result.json").write_text(json.dumps(
        {"runs": {"ckpt_final.pt": {"mean_al": 1.0}}, "equivalence_check": {"token_exact": True}}))
    assert runner.stage_done(matrix, entry, "eval")


def test_summarize_reads_benchmark_al_field(tmp_path, runner):
    matrix = mini_matrix(tmp_path)
    entry = runner.build_entries(matrix)[0]
    run_dir = Path(entry["run_dir"])
    (run_dir / "train").mkdir(parents=True)
    (run_dir / "train" / "train_summary.json").write_text(json.dumps(
        {"steps": 2, "final_loss": 0.01, "wall_seconds": 3.5, "peak_gpu_mem_gb": 9.0}))
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "eval" / "eval_result.json").write_text(json.dumps(
        {"runs": {"ckpt_final.pt": {"mean_al": 1.25, "acceptance_rate": 0.02}},
         "equivalence_check": {"token_exact": True}}))
    reports = Path(matrix["benchmark"]["reports_dir"])
    reports.mkdir(parents=True)
    runner.benchmark_report_path(matrix, entry, "w1").write_text(json.dumps(
        {"workloads": {"w1": {"status": "measured", "al": 2.5, "token_exact": True}}}))
    runner.benchmark_report_path(matrix, entry, "w2").write_text(json.dumps(
        {"workloads": {"w2": {"status": "measured", "al": 3.5, "token_exact": True}}}))
    out_dir = tmp_path / "summary"
    runner.cmd_summarize(matrix, SimpleNamespace(out_dir=str(out_dir)))
    rows = json.loads((out_dir / "summary.json").read_text())
    assert rows[0]["bench"] == {"w1": 2.5, "w2": 3.5}
    csv_text = (out_dir / "summary.csv").read_text()
    assert "2.5" in csv_text and "3.5" in csv_text
    assert ",2.5,3.5" in csv_text  # benchmark AL columns filled from the reports


def test_bind_entry_data_symlinks_and_media(tmp_path, runner):
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}")
    (store / "index.jsonl").write_text("")
    val_dir = tmp_path / "val"
    (val_dir / "images").mkdir(parents=True)
    (val_dir / "val.jsonl").write_text("")
    (val_dir / "meta.json").write_text(json.dumps({"media_dir": "images"}))
    matrix = mini_matrix(tmp_path)
    matrix["isolation"]["train_store"] = str(store)
    matrix["isolation"]["val_split"] = str(val_dir / "val.jsonl")
    entry = runner.build_entries(matrix)[0]
    run_dir = Path(entry["run_dir"])
    runner.bind_entry_data(matrix, entry, lambda msg: None)
    assert (run_dir / "features").is_symlink()
    assert (run_dir / "features").resolve() == store.resolve()
    assert (run_dir / "data" / "val.jsonl").resolve() == (val_dir / "val.jsonl").resolve()
    meta = json.loads((run_dir / "data" / "meta.json").read_text())
    assert meta["media_dir"] == str((val_dir / "images").resolve())
    runner.bind_entry_data(matrix, entry, lambda msg: None)  # idempotent
    # conflicting pre-existing symlink is refused
    other = tmp_path / "other_store"
    other.mkdir()
    (run_dir / "features").unlink()
    (run_dir / "features").symlink_to(other)
    with pytest.raises(RuntimeError, match="refusing to rebind"):
        runner.bind_entry_data(matrix, entry, lambda msg: None)


def test_prepare_done_requires_matching_config_hash(tmp_path, runner):
    matrix = mini_matrix(tmp_path)
    entry = runner.build_entries(matrix)[0]
    assert runner.materialize_configs(matrix, [entry]) == []
    run_dir = Path(entry["run_dir"])
    run_dir.mkdir(parents=True)
    doc = {"status": "ok", "algorithm": entry["drafter"], "config_sha256": "deadbeef"}
    (run_dir / "preflight.json").write_text(json.dumps(doc))
    assert not runner.prepare_done(matrix, entry)
    doc["config_sha256"] = entry["config_sha256"]
    (run_dir / "preflight.json").write_text(json.dumps(doc))
    assert runner.prepare_done(matrix, entry)
