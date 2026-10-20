#!/usr/bin/env python3
"""Seven-method synthetic-feature training smoke for an Ascend NPU.

This is the H2 execution body used by ``run_npu_smoke.py``. It builds the real
registered algorithm factory, trainer module, loss and AdamW optimizer for every
method, trains on deterministic synthetic target features, checks a 5+5 exact
resume, and exports/reloads the draft. No target checkpoint or image corpus is
needed; real teacher validation is a separate CLI run documented under H4.

The script is strict about the device: without ``--allow-cpu`` it refuses any
device other than ``npu``. A missing NPU writes ``NPU_UNVERIFIED`` and exits 2;
it never falls back to CPU.

Deterministic operators are enabled for the exact-resume gate. Without them,
trainable lookup gradients in DSpark/DFlash2 can differ at the bit level on NPU
despite identical losses and restored state.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_dflash import runtime
from vlm_dflash.algorithms import available_algorithms, get_algorithm
from vlm_dflash.dflash.features import collate_features
from vlm_dflash.dflash.mask import create_dflash_sdpa_mask
from vlm_dflash.dflash.train import run_training_loop
from vlm_dflash.export.hf import export_run, load_export
from vlm_dflash.npu_reporting import evaluate_method_records

VOCAB, HIDDEN, INTERMEDIATE, TARGET_LAYERS = 128, 32, 64, 2
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
BLOCK, ANCHORS, SEQ = 4, 8, 24
DTYPE = torch.float32


class _StopTraining(Exception):
    pass


def _config_for(method: str, steps: int) -> dict:
    cfg = {
        "drafter": {
            "num_hidden_layers": 2,
            "intermediate_size": INTERMEDIATE,
            "block_size": BLOCK,
            "num_anchors": ANCHORS,
            "mask_token_id": VOCAB - 1,
            "attention_backend": "sdpa_chunked",
            "attention_chunk_size": 7,
            "spatial_rope_enabled": False,
            "loss_decay_gamma": None if method == "jetspec" else 7.0,
            "capture_layer_ids": [0, 1],
        },
        "capture": {"store_final_hidden": True},
        "algorithm": {"name": method},
        "train": {
            "micro_batch_size": 2,
            "grad_accumulation_steps": 1,
            "num_epochs": max(1, steps),
            "lr": 1e-3,
            "log_every_steps": 0,
            "draft_dtype": "float32",
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
    }
    if method == "dflash2":
        cfg["algorithm"]["dflash2"] = {
            "selector_rank": 8,
            "selector_top_k": 4,
            "conv_group_size": 16,
        }
    elif method == "dspark":
        cfg["algorithm"]["dspark"] = {"markov_rank": 8, "enable_confidence_head": True}
    elif method == "domino":
        cfg["algorithm"]["domino"] = {"emb_dim": 16, "gru_hidden_dim": 32}
    elif method in ("dflare", "dfly"):
        cfg["algorithm"][method] = {"context_layer_ids": [0, 1]}
    return cfg


class _SyntheticFeatures(Dataset):
    def __init__(self, size: int = 8):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(1000 + int(index))
        input_ids = torch.randint(2, VOCAB - 1, (SEQ,), generator=generator)
        input_ids[0] = 2
        loss_mask = torch.ones(SEQ, dtype=torch.float32)
        loss_mask[:BLOCK] = 0
        hidden = torch.cat(
            [torch.randn(SEQ, HIDDEN, generator=generator) * 0.1 for _ in range(TARGET_LAYERS)],
            dim=-1,
        )
        final_hidden = torch.randn(SEQ, HIDDEN, generator=generator) * 0.1
        position3 = torch.arange(SEQ).view(1, 1, -1).repeat(3, 1, 1).squeeze(1)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "hidden": hidden,
            "position3": position3,
            "image_span": (0, 0),
            "sample_id": str(index),
            "final_hidden": final_hidden,
        }


def _contract(target: Path) -> dict:
    return {
        "target": {
            "model_path": str(target),
            "model_type": "qwen3",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "layers": {
            "num_target_layers": TARGET_LAYERS,
            "draft_layer_ids": [0, 1],
            "num_draft_layers": 2,
            "capture_layer_ids": [0, 1],
        },
        "rope": {"sections": [2, 3, 3], "interleaved": False},
        "text": {"hidden_size": HIDDEN},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
        "image": {"min_pixels": 100, "max_pixels": 200},
        "head": {"tie_word_embeddings": True},
    }


def _make_target(path: Path) -> Path:
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    path.mkdir(parents=True, exist_ok=True)
    Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    ).save_pretrained(path)
    return path


def _build_module(method: str, contract: dict, steps: int):
    cfg = _config_for(method, steps)
    algorithm = get_algorithm(method)
    algorithm.validate_config(cfg)
    draft_config = algorithm.build_draft_config(contract, cfg)
    torch.manual_seed(0)
    embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype=DTYPE)
    module = algorithm.build_trainer_module(draft_config, contract, cfg, embed.weight)
    return module, cfg, draft_config


def _clone_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_state(item) for item in value)
    return value


def _assert_state_equal(left, right, path: str = "state") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f"{path}: tensors differ")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"{path}: mapping keys differ")
        for key in left:
            _assert_state_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise AssertionError(f"{path}: sequence differs")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_state_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _manual_batch(contract: dict):
    dataset = _SyntheticFeatures()
    return collate_features([dataset[0], dataset[1]], contract["tokens"]["image_token_id"])


_SPECIFIC_GRADIENTS = {
    "dflare": ("layer_fusion_weights", "k_proj_target", "v_proj_target"),
    "dflash": (),
    "dflash2": ("attention_conv", "candidate_selector"),
    "dfly": ("layer_fusion_weights", "hidden_correction.down_proj"),
    "domino": ("prefix_gru", "embed_proj"),
    "dspark": ("markov_head", "confidence_head"),
    "jetspec": (),
}


def _specific_gradients(method: str, module, contract: dict, steps: int) -> tuple[bool, list[str]]:
    module.train()
    if method == "domino":
        module.set_training_progress(steps // 2, steps)
    batch = _manual_batch(contract)
    device = next(module.draft_model.parameters()).device
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    metrics = module(
        input_ids=batch["input_ids"],
        hidden_states=batch["hidden"],
        loss_mask=batch["loss_mask"],
        teacher_hidden=batch.get("final_hidden"),
    )
    if not torch.isfinite(metrics.loss):
        raise AssertionError("manual method-gradient forward produced a non-finite loss")
    metrics.loss.backward()
    trainable = [(name, parameter) for name, parameter in module.draft_model.named_parameters() if parameter.requires_grad]
    if not any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for _, parameter in trainable):
        raise AssertionError("no trainable draft parameter received a nonzero gradient")
    missing = []
    for substring in _SPECIFIC_GRADIENTS[method]:
        matched = [
            parameter
            for name, parameter in trainable
            if substring in name and parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        ]
        if not matched:
            missing.append(substring)
    if missing:
        raise AssertionError(f"{method}: no nonzero gradient for {missing}")
    return True, missing


def _write_export_run(
    run_dir: Path,
    contract: dict,
    cfg: dict,
    method: str,
    module,
    draft_config,
    device: torch.device,
    training_result: dict,
) -> Path:
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (run_dir / "data" / "manifest.json").write_text(
        json.dumps({"source": "unit", "num_train": 8, "num_val": 2, "rows_sha256": "d" * 64}),
        encoding="utf-8",
    )
    (run_dir / "train" / "train_summary.json").write_text(
        json.dumps(
            {
                **training_result["summary"],
                "feature_mode": "offline",
                "device": str(device),
                "device_type": device.type,
                "backend": device.type,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train" / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in training_result["history"]), encoding="utf-8"
    )
    algorithm = get_algorithm(method)
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": method,
        "identity": {"model_fingerprint": "f" * 64},
        "feature_mode": "offline",
        "step": training_result["summary"]["steps"],
        "model": _clone_state(module.draft_model.state_dict()),
        **algorithm.checkpoint_metadata(contract, cfg, draft_config),
        "attention_backend": "sdpa_chunked",
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return run_dir


def _export_round_trip(run_dir: Path, module, contract: dict) -> None:
    export_dir = export_run(run_dir)
    loaded, exported = load_export(export_dir)
    if exported["drafter"]["attention_backend"] != "sdpa":
        raise AssertionError("export did not record the portable SDPA serving backend")
    if exported["drafter"]["training_attention_backend"] != "sdpa_chunked":
        raise AssertionError("export did not preserve the training attention backend")
    device = next(module.parameters()).device
    loaded = loaded.to(device)
    anchors = torch.tensor([[2, 8]], dtype=torch.long, device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    mask = create_dflash_sdpa_mask(anchors, keep, SEQ // 2, BLOCK, device, DTYPE)
    target_hidden = torch.randn(1, SEQ // 2, TARGET_LAYERS * HIDDEN, device=device) * 0.1
    noise = torch.randn(1, 2 * BLOCK, HIDDEN, device=device) * 0.1
    position_ids = torch.arange(SEQ // 2 + 2 * BLOCK, device=device).unsqueeze(0)
    module.draft_model.eval()
    loaded.eval()
    with torch.no_grad():
        original = module.draft_model(
            position_ids=position_ids,
            attention_mask=mask,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
        reloaded = loaded(
            position_ids=position_ids,
            attention_mask=mask,
            noise_embedding=noise,
            target_hidden=target_hidden,
        )
    if original.shape != reloaded.shape or not torch.equal(original, reloaded):
        raise AssertionError("exported draft forward differs from the training module")


def _run_method(
    method: str,
    contract: dict,
    device: torch.device,
    *,
    steps: int,
    seed: int,
    artifact_dir: Path,
) -> dict:
    started = time.time()
    module, cfg, draft_config = _build_module(method, contract, steps)
    runtime.reset_peak_memory_stats(device)
    memory_samples = []

    def sample_memory(step):
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_samples.append({
            "step": int(step),
            "accelerator_allocated_bytes": runtime.memory_allocated(device),
            "host_peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
        })

    sample_memory(0)
    before = {name: parameter.detach().clone() for name, parameter in module.draft_model.named_parameters()}
    continuous = run_training_loop(
        module,
        _SyntheticFeatures(),
        cfg,
        contract,
        device,
        total_steps=steps,
        seed=seed,
        shuffle=False,
        save_every=max(1, steps // 10),
        on_checkpoint=lambda state: sample_memory(state["step"]),
    )
    sample_memory(steps)
    history = continuous["history"]
    if len(history) != steps:
        raise AssertionError(f"expected {steps} optimizer steps, got {len(history)}")
    if continuous["summary"]["nan_seen"]:
        raise AssertionError("training reported NaN in its loss history")
    if not all(torch.isfinite(torch.tensor(row["loss"])) for row in history):
        raise AssertionError("training history contains a non-finite loss")
    first_parameter = next(module.draft_model.parameters())
    if first_parameter.device.type != device.type:
        raise AssertionError(f"draft parameter is on {first_parameter.device}, expected {device}")

    changed = sum(
        1 for name, parameter in module.draft_model.named_parameters()
        if not torch.equal(parameter.detach().cpu(), before[name].cpu())
    )
    if changed == 0:
        raise AssertionError("no draft parameter changed after training")
    coverage = continuous.get("gradient_coverage") or {}
    if int(coverage.get("params_with_grad", 0)) <= 0:
        raise AssertionError("no parameter received a gradient")

    specific_ok, _ = _specific_gradients(method, module, contract, steps)

    # 5+5 (or half/half) exact resume against the uninterrupted run.
    half = max(1, steps // 2)
    interrupted_module, _, _ = _build_module(method, contract, steps)
    saved: dict = {}

    def checkpoint_hook(state):
        if int(state["step"]) == half:
            saved.update(state)
            raise _StopTraining

    try:
        run_training_loop(
            interrupted_module,
            _SyntheticFeatures(),
            cfg,
            contract,
            device,
            total_steps=steps,
            seed=seed,
            shuffle=False,
            save_every=half,
            on_checkpoint=checkpoint_hook,
        )
    except _StopTraining:
        pass
    if saved.get("step") != half:
        raise AssertionError(f"checkpoint hook did not capture step {half}")

    resumed_module, _, _ = _build_module(method, contract, steps)
    resumed = run_training_loop(
        resumed_module,
        _SyntheticFeatures(),
        cfg,
        contract,
        device,
        total_steps=steps,
        seed=seed,
        shuffle=False,
        resume_state=saved,
    )
    if resumed["summary"]["resumed_from_step"] != half:
        raise AssertionError("resume did not start from the saved step")
    for expected, actual in zip(continuous["history"][half:], resumed["history"][half:]):
        if int(actual["step"]) != int(expected["step"]):
            raise AssertionError("resumed step numbering diverged")
        if float(actual["loss"]) != float(expected["loss"]):
            raise AssertionError("resumed loss diverged from the uninterrupted run")
    for key in ("model", "master", "optimizer", "scheduler"):
        _assert_state_equal(continuous["final_state"][key], resumed["final_state"][key], key)

    run_dir = artifact_dir / method
    _write_export_run(run_dir, contract, cfg, method, module, draft_config, device, continuous)
    _export_round_trip(run_dir, module, contract)
    runtime.synchronize(device)
    peak = runtime.max_memory_allocated(device)
    return {
        "method": method,
        "steps": steps,
        "nan_seen": False,
        "exit_code": 0,
        "skipped": False,
        "forward": True,
        "backward": True,
        "update": bool(changed > 0),
        "resume": True,
        "export": True,
        "method_specific_grad": bool(specific_ok),
        "params_with_grad": int(coverage.get("params_with_grad", 0)),
        "changed_parameters": changed,
        "final_loss": float(history[-1]["loss"]),
        "elapsed_seconds": round(time.time() - started, 3),
        "peak_memory_bytes": None if peak is None else int(peak),
        "memory_samples": memory_samples,
        "device_type": device.type,
        "device_name": runtime.device_name(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--methods", default="all", help="comma-separated methods, or 'all'")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=str(ROOT / "runs" / "npu" / "seven_methods.json"))
    parser.add_argument("--artifact-dir", default=str(ROOT / "runs" / "npu" / "seven_methods_artifacts"))
    parser.add_argument("--allow-cpu", action="store_true", help="debug only: mark evidence CPU_REFERENCE")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
        help="require deterministic operators for the bitwise resume gate (default: enabled)",
    )
    args = parser.parse_args()
    torch.use_deterministic_algorithms(args.deterministic)

    registered = available_algorithms()
    methods = registered if args.methods == "all" else [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(registered))
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = {
        "format": "vlm-dflash-npu-seven-method-smoke-v1",
        "requested_device": args.device,
        "methods": methods,
        "steps_requested": args.steps,
        "seed": args.seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "runtime": runtime.environment_report(),
        },
        "records": [],
    }
    if unknown:
        payload.update({"status": "FAILED", "evidence_level": "NOT_VERIFIED", "error": f"unknown methods: {unknown}"})
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "FAILED", "error": payload["error"], "json": str(out_path)}))
        return 1

    try:
        device = runtime.ensure_runtime_device(args.device, context="seven_method_smoke.device")
    except Exception as error:
        payload.update(
            {
                "status": "NPU_UNVERIFIED",
                "evidence_level": "NPU_UNVERIFIED",
                "error": f"{type(error).__name__}: {error}",
                "note": "No usable NPU; no method ran and no CPU fallback was attempted.",
                "duration_seconds": round(time.time() - started, 3),
            }
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "NPU_UNVERIFIED", "error": payload["error"], "json": str(out_path)}))
        return 2
    if device.type != "npu" and not args.allow_cpu:
        payload.update(
            {
                "status": "FAILED",
                "evidence_level": "NOT_VERIFIED",
                "error": f"seven_method_smoke requires an NPU device, resolved {device}",
            }
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"status": "FAILED", "error": payload["error"], "json": str(out_path)}))
        return 1

    payload["resolved_device"] = str(device)
    payload["device_name"] = runtime.device_name(device)
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="vlm-dflash-npu-target-") as tmp:
        target = _make_target(Path(tmp) / "target")
        contract = _contract(target)
        for method in methods:
            try:
                record = _run_method(
                    method,
                    contract,
                    device,
                    steps=args.steps,
                    seed=args.seed,
                    artifact_dir=artifact_dir,
                )
            except Exception as error:
                records.append(
                    {
                        "method": method,
                        "steps": 0,
                        "nan_seen": None,
                        "exit_code": 1,
                        "skipped": False,
                        "forward": False,
                        "backward": False,
                        "update": False,
                        "resume": False,
                        "export": False,
                        "method_specific_grad": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc()[-4000:],
                    }
                )
            else:
                records.append(record)
                print(json.dumps({"method": method, "status": "passed", "steps": record["steps"]}))

    verdict = evaluate_method_records(records, methods, min_steps=max(10, args.steps))
    status = "NPU_SMOKE_PASSED" if verdict["status"] == "PASSED" else "FAILED"
    if device.type == "cpu":
        status = "CPU_REFERENCE" if verdict["status"] == "PASSED" else "FAILED"
    payload.update(
        {
            "status": status,
            "evidence_level": "CPU_REFERENCE" if status == "CPU_REFERENCE" else status,
            "verdict": verdict,
            "records": records,
            "duration_seconds": round(time.time() - started, 3),
            "artifact_dir": str(artifact_dir),
            "note": (
                "Synthetic target features; this validates the real training path on the device, "
                "not a real Qwen target forward or model quality."
            ),
        }
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "problems": verdict["problems"], "json": str(out_path)}))
    return 0 if verdict["status"] == "PASSED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
