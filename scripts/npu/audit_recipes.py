#!/usr/bin/env python3
"""Formal-objective acceptance and same-device resume at the DFly stage boundary.

Natural capture-v2 features, production training loop, fixed 10/12-step horizons.
This is diagnostic microbatch-one offline training, not a throughput benchmark.
DFly reloads production checkpoints at committed steps 5 and 6, preserving the
12-step scheduler horizon. No default-objective or smaller-model fallback exists.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.npu.audit_common import build_module, load_bundle
from vlm_dflash import runtime
from vlm_dflash.dflash.train import run_training_loop
from vlm_dflash.objectives import TeacherLogitsStream, resolve_recipe

RECIPES = {"dspark": "upstream_dspark", "dfly": "upstream_two_stage"}
CONFIGS = {"dspark": "dspark/qwen2_5_vl_3b_2k_recipe.yaml", "dfly": "dfly/qwen2_5_vl_3b_2k_recipe_matched.yaml"}
HEADS = {"dspark": ("markov_head", "confidence_head"),
         "dfly": ("layer_fusion_weights", "hidden_correction.down_proj")}



def source_provenance(root):
    """Git metadata is optional in deployed trees; content hashes remain exact."""
    root = Path(root)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                              text=True, capture_output=True)
    paths = list((root / 'vlm_dflash').rglob('*.py'))
    paths += list((root / 'scripts' / 'npu').glob('audit*.py'))
    return {'commit': revision.stdout.strip() if revision.returncode == 0 else None,
            'git_metadata_note': None if revision.returncode == 0 else
                'Deployed source tree has no .git metadata; source hashes are authoritative.',
            'source_hashes': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(paths)}}


def assert_state_equal(left, right, path="state"):
    """Exact recursive comparison, including optimizer, CPU/NPU RNG and NumPy RNG."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu()), path
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_state_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_state_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


def validate_formal_config(cfg, geometry):
    assert (geometry.hidden_size, geometry.intermediate_size, geometry.num_hidden_layers) == (2048, 11008, 3), "formal model geometry mismatch"
    assert (cfg["drafter"]["block_size"], cfg["drafter"]["num_anchors"]) == (8, 256), "formal block/anchors mismatch"
    assert (cfg["train"]["micro_batch_size"], cfg["train"]["grad_accumulation_steps"]) == (1, 1), "diagnostic microbatch must be one"


class RecipeObserver:
    """Observe actual teacher projections, recipe loss outputs and head gradients.

    Hooks only read detached values and do not alter the objective or gradients.
    Teacher calls are instrumented only while this single-process context lives.
    """
    def __init__(self, module, method):
        self.module, self.method = module, method
        expected = resolve_recipe({"algorithm": {"train_recipe": RECIPES[method]}}, method)
        actual = module.train_recipe
        assert actual is not None and actual.as_dict() == expected.as_dict(), f"expected exact {RECIPES[method]} recipe and coefficients"
        self.recipe = actual.as_dict()
        self.rows, self.handles = [], []
        self.teacher_calls = self.teacher_rows = 0
        self.gradient_norms = {}
        names = [name for name, _ in module.draft_model.named_parameters()]
        for group in HEADS[method]:
            assert any(group in name for name in names), f"missing method head: {group}"

    def __enter__(self):
        self.original_logits = TeacherLogitsStream.logits
        observer = self
        def logits(stream, start, stop):
            result = observer.original_logits(stream, start, stop)
            observer.teacher_calls += 1
            observer.teacher_rows += int(result.shape[0])
            return result
        TeacherLogitsStream.logits = logits
        self.handles.append(self.module.register_forward_pre_hook(self._before))
        self.handles.append(self.module.register_forward_hook(self._after))
        for name, parameter in self.module.draft_model.named_parameters():
            for group in HEADS[self.method]:
                if group in name and parameter.requires_grad:
                    self.handles.append(parameter.register_hook(self._gradient_hook(group)))
        return self

    def _before(self, module, inputs):
        self.teacher_calls = self.teacher_rows = 0

    def _after(self, module, inputs, metrics):
        stage_index = int(module._recipe_stage_index)
        extras = metrics.extra_metrics or {}
        assert int(extras.get("recipe_stage", -1)) == stage_index, "actual loss did not execute requested recipe stage"
        stage = module.train_recipe.stages[stage_index]
        components = {key: float(value) for key, value in extras.items()}
        expected_terms = ((stage.ce_coef, "ce_loss"), (stage.lk_coef, "lk_loss"),
                          (stage.e2e_tv_coef, "e2e_tv_loss"), (stage.confidence_coef, "confidence_bce"))
        for coef, key in expected_terms:
            if coef > 0:
                assert key in components and math.isfinite(components[key]), f"missing/nonfinite actual loss term {key}"
        row = {"call": len(self.rows) + 1, "stage_index": stage_index, "stage": stage.as_dict(),
               "loss": float(metrics.loss.detach()), "valid_tokens": int(metrics.valid_tokens),
               "loss_weight": float(metrics.loss_weight), "valid_blocks": int(metrics.valid_blocks),
               "teacher_calls": self.teacher_calls, "teacher_rows": self.teacher_rows, "components": components}
        assert math.isfinite(row["loss"]), "nonfinite loss"
        assert row["valid_tokens"] > 0 and row["loss_weight"] > 0, "no effective supervision"
        assert row["teacher_calls"] > 0 and row["teacher_rows"] > 0, "teacher distribution was not consumed"
        self.rows.append(row)

    def _gradient_hook(self, group):
        def hook(gradient):
            stage = int(self.module._recipe_stage_index)
            norm = float(gradient.detach().float().norm())
            assert math.isfinite(norm), f"nonfinite gradient: {group}"
            self.gradient_norms.setdefault(stage, {})[group] = max(norm, self.gradient_norms.get(stage, {}).get(group, 0.0))
        return hook

    def validate(self, expected_stages):
        assert {row["stage_index"] for row in self.rows} == set(expected_stages), "stage coverage mismatch"
        for stage in expected_stages:
            for group in HEADS[self.method]:
                assert self.gradient_norms.get(stage, {}).get(group, 0) > 0, f"missing nonzero gradient at stage {stage}: {group}"

    def report(self):
        return {"recipe": self.recipe, "observations": self.rows, "max_head_gradient_norm_by_stage": self.gradient_norms}

    def __exit__(self, *args):
        TeacherLogitsStream.logits = self.original_logits
        for handle in self.handles:
            handle.remove()


def diagnostic_config(method, config_dir, bundle, device):
    path = Path(config_dir) / CONFIGS[method]
    cfg = yaml.safe_load(path.read_text())
    cfg["target"]["model"] = bundle["contract"]["target"]["model_path"]
    cfg["target"]["device"] = str(device)
    cfg["drafter"].update(attention_backend="sdpa_chunked", attention_chunk_size=128,
                           capture_layer_ids=list(bundle["manifest"]["capture_layer_ids"]))
    cfg["train"].update(micro_batch_size=1, grad_accumulation_steps=1, num_epochs=12,
                        num_workers=0, pin_memory=False, non_blocking=False,
                        device=str(device), draft_dtype="bfloat16", log_every_steps=1,
                        feature_mode="offline", overfit_sanity={"enabled": False})
    return cfg, path


def save_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")


def run_method(method, bundle, cfg, device, out_dir, record):
    horizon = 10 if method == "dspark" else 12
    seed = 42
    out_dir.mkdir(parents=True, exist_ok=True)
    record.update(horizon=horizon, boundary_step=None if method == "dspark" else 6,
                  scope="natural_capture_v2_offline_diagnostic_microbatch_one",
                  config=cfg, checkpoint_reload_steps=[])
    module, geometry = build_module(bundle, cfg, device, seed=seed)
    validate_formal_config(cfg, geometry)
    record["geometry"] = {key: getattr(geometry, key) for key in
                          ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")}
    assert all(item.get("final_hidden") is not None for item in bundle["records"]), "teacher features missing"
    runtime.seed_all(seed, device)
    observer = RecipeObserver(module, method)
    def save_checkpoint(state):
        step = int(state["step"])
        if method == "dfly" and step in (5, 6):
            assert state["recipe_stage"] == (0 if step == 5 else 1)
            assert state["steps_total_planned"] == 12
            torch.save(state, out_dir / f"state_{step}.pt")
    try:
        with observer:
            result = run_training_loop(module, bundle["records"], cfg, bundle["contract"], device,
                                       total_steps=horizon, seed=seed, shuffle=False,
                                       log_path=out_dir / "continuous.jsonl", on_checkpoint=save_checkpoint, save_every=horizon,
                                       extra_save_steps={5, 6} if method == "dfly" else set())
            observer.validate({0} if method == "dspark" else {0, 1})
        assert result["summary"]["steps"] == horizon and not result["summary"]["nan_seen"]
        assert len(result["history"]) == horizon
        record["summary"] = result["summary"]
        record["attention"] = module.attention_runtime_report()
        record["history"] = result["history"]
        torch.save(result["final_state"], out_dir / "continuous_final.pt")
    finally:
        record["continuous_observation"] = observer.report()
    del module, observer, result
    gc.collect()
    if method == "dfly":
        for step in (5, 6):
            saved = torch.load(out_dir / f"state_{step}.pt", map_location="cpu", weights_only=False, mmap=True)
            module, _ = build_module(bundle, cfg, device, seed=seed)
            observer = RecipeObserver(module, method)
            with observer:
                resumed = run_training_loop(module, bundle["records"], cfg, bundle["contract"], device,
                                            total_steps=12, seed=seed, shuffle=False, resume_state=saved,
                                            log_path=out_dir / f"resume_{step}.jsonl")
                observer.validate({0, 1} if step == 5 else {1})
            expected = torch.load(out_dir / "continuous_final.pt", map_location="cpu", weights_only=False, mmap=True)
            # Compare all persisted state fields, including RNG and complete history.
            assert_state_equal(expected, resumed["final_state"])
            record["checkpoint_reload_steps"].append({"step": step, "exact_final_state": True,
                                                       "observation": observer.report()})
            del module, observer, saved, expected, resumed
            gc.collect()
    record["status"] = "pass"


def resume_one(bundle,cfg,device,reference_dir,step):
    """One recovery leg in a fresh process; mmap bounds host RAM, not geometry."""
    reference_dir=Path(reference_dir)
    saved=torch.load(reference_dir/f"state_{step}.pt",map_location="cpu",weights_only=False,mmap=True)
    assert saved["step"]==step and saved["steps_total_planned"]==12
    module,geometry=build_module(bundle,cfg,device,seed=42)
    validate_formal_config(cfg,geometry)
    with RecipeObserver(module,"dfly") as observer:
        resumed=run_training_loop(module,bundle["records"],cfg,bundle["contract"],device,
            total_steps=12,seed=42,shuffle=False,resume_state=saved)
        observer.validate({0,1} if step==5 else {1})
    del saved
    gc.collect()
    expected=torch.load(reference_dir/"continuous_final.pt",map_location="cpu",weights_only=False,mmap=True)
    assert_state_equal(expected,resumed["final_state"])
    return {"status":"pass","method":"dfly","horizon":12,"boundary_step":6,
            "checkpoint_reload_step":step,"exact_final_state":True,"observation":observer.report(),
            "summary":resumed["summary"],"history":resumed["history"],"config":cfg,
            "checkpoint_reading":"mmap CPU tensors in a fresh process; identical data/model/horizon"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, choices=(5,6), default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--method", choices=("all", "dspark", "dfly"), default="all")
    args = parser.parse_args(argv)
    report = {"schema": "veladraft-recipe-audit-v1", "requested_device": args.device,
              "bundle": str(args.bundle), "deterministic_algorithms": True, "results": []}
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        report.update(source_provenance(ROOT))
        device = runtime.ensure_runtime_device(args.device)
        torch.use_deterministic_algorithms(True)
        bundle = load_bundle(args.bundle)
        report["sample_count"] = len(bundle["records"])
        report["target"] = bundle["contract"]["target"]
        report["capture_manifest"] = bundle["manifest"]
        if args.resume_step is not None:
            if args.reference_dir is None:raise ValueError("--reference-dir required for recovery leg")
            cfg,_=diagnostic_config("dfly",args.config_dir,bundle,device)
            result=resume_one(bundle,cfg,device,args.reference_dir,args.resume_step)
            report.update(status="pass",results=[result])
            return 0
        for method in RECIPES if args.method == "all" else (args.method,):
            record = {"method": method, "status": "failed"}
            report["results"].append(record)
            try:
                cfg, path = diagnostic_config(method, args.config_dir, bundle, device)
                record["source_config"] = str(path)
                run_method(method, bundle, cfg, device, args.out / method, record)
            except Exception as error:
                record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            finally:
                save_json(args.out / f"{method}.json", record)
                save_json(args.out / "results.json", report)
        report["status"] = "pass" if all(r["status"] == "pass" for r in report["results"]) else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds"] = time.time() - started
        save_json(args.out / "results.json", report)
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
