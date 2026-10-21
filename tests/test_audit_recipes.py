"""CPU assertions for the hardware recipe harness; no NPU evidence."""
import copy
import pytest
import torch


def test_recipe_audit_rejects_default_objective():
    from scripts.npu.audit_recipes import RecipeObserver
    from tests.test_dspark import _tiny_dspark_module
    with pytest.raises(AssertionError, match="upstream_dspark"):
        RecipeObserver(_tiny_dspark_module(), "dspark")


def test_observer_requires_teacher_supervision_and_method_gradients():
    from scripts.npu.audit_recipes import RecipeObserver
    from tests.test_dspark import _tiny_dspark_module, _dspark_recipe, _tiny_batch, _teacher_hidden
    module = _tiny_dspark_module(train_recipe=_dspark_recipe())
    ids, mask, hidden = _tiny_batch()
    with RecipeObserver(module, "dspark") as observer:
        output = module(input_ids=ids, hidden_states=hidden, loss_mask=mask, teacher_hidden=_teacher_hidden(ids))
        with pytest.raises(AssertionError, match="gradient"):
            observer.validate({0})
        output.loss.backward()
        observer.validate({0})
        assert observer.rows[0]["teacher_rows"] > 0
        assert observer.rows[0]["valid_tokens"] > 0
        assert observer.rows[0]["stage"]["confidence_coef"] == 1.0


def test_resume_equality_detects_optimizer_or_rng_change():
    from scripts.npu.audit_recipes import assert_state_equal
    state = {"optimizer": {"state": {0: {"exp_avg": torch.ones(3)}}}, "rng": torch.tensor([3, 4])}
    assert_state_equal(state, copy.deepcopy(state))
    bad = copy.deepcopy(state)
    bad["optimizer"]["state"][0]["exp_avg"][0] = 0
    with pytest.raises(AssertionError, match="exp_avg"):
        assert_state_equal(state, bad)


def test_formal_geometry_gate_rejects_smoke_width():
    from scripts.npu.audit_recipes import validate_formal_config
    from types import SimpleNamespace
    cfg = {"train": {"micro_batch_size": 1, "grad_accumulation_steps": 1},
           "drafter": {"block_size": 8, "num_anchors": 256}}
    geometry = SimpleNamespace(hidden_size=2048, intermediate_size=11008, num_hidden_layers=3)
    validate_formal_config(cfg, geometry)
    geometry.intermediate_size = 1024
    with pytest.raises(AssertionError, match="geometry"):
        validate_formal_config(cfg, geometry)


def test_production_loop_stage_boundary_checkpoint_reload(tmp_path, monkeypatch):
    from scripts.npu import audit_recipes as audit
    from tests.test_dfly import _tiny_module, _tiny_batch
    from vlm_dflash.objectives import resolve_recipe

    def build(bundle, cfg, device, seed=42):
        module, _ = _tiny_module(seed=seed)
        module.train_recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly")
        module._recipe_stage_index = 0
        return module, module.draft_model.config
    monkeypatch.setattr(audit, "build_module", build)
    monkeypatch.setattr(audit, "validate_formal_config", lambda cfg, geometry: None)
    ids, mask, hidden = _tiny_batch(batch_size=1)
    record = {"input_ids": ids[0], "loss_mask": mask[0], "hidden": hidden[0],
              "final_hidden": torch.randn(48, 64), "position3": torch.arange(48).repeat(3, 1),
              "image_span": (0, 0), "sample_id": "cpu-fixture"}
    bundle = {"records": [record] * 12, "contract": {"tokens": {"image_token_id": 999999}}}
    cfg = {"train": {"micro_batch_size": 1, "grad_accumulation_steps": 1, "num_epochs": 1,
                     "draft_dtype": "float32", "lr": 1e-3, "log_every_steps": 0}}
    result = {}
    audit.run_method("dfly", bundle, cfg, torch.device("cpu"), tmp_path, result)
    assert result["status"] == "pass"
    assert [row["step"] for row in result["checkpoint_reload_steps"]] == [5, 6]
    assert result["continuous_observation"]["observations"][5]["stage_index"] == 0
    assert result["continuous_observation"]["observations"][6]["stage_index"] == 1


def test_cli_writes_failed_result_for_missing_bundle(tmp_path):
    import json
    from scripts.npu.audit_recipes import main
    out = tmp_path / "report"
    assert main(["--device", "cpu", "--bundle", str(tmp_path / "missing.pt"), "--out", str(out)]) == 1
    report = json.loads((out / "results.json").read_text())
    assert report["status"] == "failed"
    assert "FileNotFoundError" in report["error"]
    # Avoid leaking the CLI's requested deterministic setting to unrelated tests.
    torch.use_deterministic_algorithms(False)


def test_dspark_production_loop_marks_success(tmp_path, monkeypatch):
    from scripts.npu import audit_recipes as audit
    from tests.test_dspark import _tiny_dspark_module, _dspark_recipe, _tiny_batch, _teacher_hidden

    def build(bundle, cfg, device, seed=42):
        torch.manual_seed(seed)
        module = _tiny_dspark_module(train_recipe=_dspark_recipe())
        return module, module.draft_model.config
    monkeypatch.setattr(audit, "build_module", build)
    monkeypatch.setattr(audit, "validate_formal_config", lambda cfg, geometry: None)
    ids, mask, hidden = _tiny_batch()
    record = {"input_ids": ids[0], "loss_mask": mask[0], "hidden": hidden[0],
              "final_hidden": _teacher_hidden(ids)[0], "position3": torch.arange(ids.shape[1]).repeat(3, 1),
              "image_span": (0, 0), "sample_id": "cpu-fixture"}
    bundle = {"records": [record] * 10, "contract": {"tokens": {"image_token_id": 999999}}}
    cfg = {"train": {"micro_batch_size": 1, "grad_accumulation_steps": 1, "num_epochs": 1,
                     "draft_dtype": "float32", "lr": 1e-3, "log_every_steps": 0}}
    result = {}
    audit.run_method("dspark", bundle, cfg, torch.device("cpu"), tmp_path, result)
    assert result["status"] == "pass"
    assert result["horizon"] == 10


def test_provenance_accepts_deployed_source_without_git(tmp_path):
    import hashlib
    from scripts.npu.audit_recipes import source_provenance
    source = tmp_path / 'vlm_dflash' / 'example.py'
    source.parent.mkdir()
    source.write_text('value = 1\n')
    result = source_provenance(tmp_path)
    assert result['commit'] is None
    assert result['source_hashes']['vlm_dflash/example.py'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert 'no .git' in result['git_metadata_note']
