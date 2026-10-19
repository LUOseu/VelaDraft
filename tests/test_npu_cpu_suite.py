"""End-to-end CPU evidence for every registered drafter (C05/C06/C08).

Each case builds the real algorithm factory, draft model, trainer module,
loss and optimizer on tiny synthetic target features. Nothing here proves an
Ascend kernel; it proves the shared training/resume/export paths are wired for
all seven methods without CUDA.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from vlm_dflash.algorithms import available_algorithms, get_algorithm
from vlm_dflash.dflash.features import collate_features
from vlm_dflash.dflash.mask import create_dflash_sdpa_mask
from vlm_dflash.dflash.train import run_training_loop
from vlm_dflash.export.hf import export_run, load_export
from vlm_dflash.npu_reporting import update_method_record

VOCAB, HIDDEN, INTERMEDIATE, TARGET_LAYERS = 128, 32, 64, 2
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
BLOCK, ANCHORS, SEQ = 4, 8, 24
DTYPE = torch.float32


_EVIDENCE: dict[str, dict] = {}


def _record(method: str, **fields) -> None:
    update_method_record(_EVIDENCE, method, **fields)


@pytest.fixture(scope="session", autouse=True)
def _dump_npu_evidence():
    """Persist machine-readable per-method evidence for ``run_cpu_suite.py``.

    The runner sets ``VLM_DFLASH_NPU_EVIDENCE_JSON`` to a temporary path. The
    file is written even when a case fails, so the aggregate can distinguish a
    failed assertion from a test that never ran.
    """
    yield
    target = os.environ.get("VLM_DFLASH_NPU_EVIDENCE_JSON")
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": "vlm-dflash-npu-method-evidence-v1",
                "records": [_EVIDENCE[method] for method in sorted(_EVIDENCE)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _config_for(method: str) -> dict:
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
            "num_epochs": 12,
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
    return cfg


@pytest.fixture(scope="module")
def method_env(tmp_path_factory):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    root = tmp_path_factory.mktemp("npu_cpu_suite")
    target = root / "target"
    target.mkdir()
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
    ).save_pretrained(target)
    contract = {
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
    return {"root": root, "contract": contract, "target": target}


class _SyntheticFeatures(Dataset):
    """Deterministic fixed features; no target model and no CUDA tensors."""

    def __init__(self, size: int = 8):
        self.size = int(size)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
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


def _build_module(method: str, env: dict):
    cfg = _config_for(method)
    algorithm = get_algorithm(method)
    algorithm.validate_config(cfg)
    draft_config = algorithm.build_draft_config(env["contract"], cfg)
    torch.manual_seed(0)
    embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype=DTYPE)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False, dtype=DTYPE)
    lm_head.weight = embed.weight
    module = algorithm.build_trainer_module(draft_config, env["contract"], cfg, embed.weight)
    return module, cfg, draft_config


def _manual_batch(env: dict):
    dataset = _SyntheticFeatures()
    return collate_features([dataset[0], dataset[1]], env["contract"]["tokens"]["image_token_id"])


@pytest.mark.parametrize("method", available_algorithms())
def test_all_seven_methods_run_ten_real_optimizer_updates(method: str, method_env):
    module, cfg, _ = _build_module(method, method_env)
    before = {name: parameter.detach().clone() for name, parameter in module.draft_model.named_parameters()}
    output = run_training_loop(
        module,
        _SyntheticFeatures(),
        cfg,
        method_env["contract"],
        torch.device("cpu"),
        total_steps=10,
        seed=5,
        shuffle=False,
    )
    history = output["history"]
    assert len(history) == 10
    assert output["summary"]["nan_seen"] is False
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    assert output["final_state"]["step"] == 10
    changed = sum(
        1 for name, parameter in module.draft_model.named_parameters() if not torch.equal(parameter, before[name])
    )
    assert changed > 0, method
    coverage = output["gradient_coverage"]
    assert coverage["params_with_grad"] > 0
    attention = output["summary"]["attention"]
    assert attention["requested_backend"] == "sdpa_chunked"
    assert attention["effective_backends"] == ["sdpa_chunked"]
    assert attention["memory_plan"]["chunked_total_gb"] < attention["memory_plan"]["dense_total_gb"]
    _record(
        method,
        steps=len(history),
        nan_seen=bool(output["summary"]["nan_seen"]),
        forward=True,
        backward=bool(coverage["params_with_grad"] > 0),
        update=bool(changed > 0),
    )


METHOD_GRADIENT_SUBSTRINGS = {
    "dflare": ("layer_fusion_weights", "k_proj_target", "v_proj_target"),
    "dflash": (),
    "dflash2": ("attention_conv", "candidate_selector"),
    "dfly": ("layer_fusion_weights", "hidden_correction.down_proj"),
    "domino": ("prefix_gru", "embed_proj"),
    "dspark": ("markov_head", "confidence_head"),
    "jetspec": (),
}


@pytest.mark.parametrize("method", available_algorithms())
def test_method_specific_modules_receive_nonzero_gradients(method: str, method_env):
    module, _, _ = _build_module(method, method_env)
    module.train()
    if method == "domino":
        # Mid-curriculum: at step 0 Domino's lambda_base is 1 and the correction
        # head is intentionally inactive.
        module.set_training_progress(5, 10)
    batch = _manual_batch(method_env)
    metrics = module(
        input_ids=batch["input_ids"],
        hidden_states=batch["hidden"],
        loss_mask=batch["loss_mask"],
        teacher_hidden=batch.get("final_hidden"),
    )
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    trainable = [(name, parameter) for name, parameter in module.draft_model.named_parameters() if parameter.requires_grad]
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for _, parameter in trainable)
    for substring in METHOD_GRADIENT_SUBSTRINGS[method]:
        matched = [
            parameter
            for name, parameter in trainable
            if substring in name and parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        ]
        assert matched, f"{method}: no nonzero gradient for {substring}"
    # The frozen target head/embedding never move.
    assert module.embed_tokens.weight.requires_grad is False
    assert module.lm_head.weight.requires_grad is False
    _record(method, forward=True, backward=True, method_specific_grad=True)


class _StopTraining(Exception):
    pass


def _assert_state_tree_equal(left, right, path: str = "state") -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_state_tree_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_state_tree_equal(a, b, f"{path}[{index}]")
        return
    if isinstance(left, float) and isinstance(right, float):
        assert left == pytest.approx(right, rel=0.0, abs=0.0), path
        return
    assert left == right, path


@pytest.mark.parametrize("method", available_algorithms())
def test_twelve_steps_equal_six_plus_exact_resume_plus_six(method: str, method_env):
    cfg = _config_for(method)
    continuous_module, _, _ = _build_module(method, method_env)
    continuous = run_training_loop(
        continuous_module,
        _SyntheticFeatures(),
        cfg,
        method_env["contract"],
        torch.device("cpu"),
        total_steps=12,
        seed=5,
        shuffle=False,
    )
    assert len(continuous["history"]) == 12

    interrupted_module, _, _ = _build_module(method, method_env)
    saved: dict = {}

    def checkpoint_hook(state):
        if int(state["step"]) == 6:
            saved.update(state)
            raise _StopTraining

    with pytest.raises(_StopTraining):
        run_training_loop(
            interrupted_module,
            _SyntheticFeatures(),
            cfg,
            method_env["contract"],
            torch.device("cpu"),
            total_steps=12,
            seed=5,
            shuffle=False,
            save_every=6,
            on_checkpoint=checkpoint_hook,
        )
    assert saved["step"] == 6
    assert len(saved["history"]) == 6

    resumed_module, _, _ = _build_module(method, method_env)
    resumed = run_training_loop(
        resumed_module,
        _SyntheticFeatures(),
        cfg,
        method_env["contract"],
        torch.device("cpu"),
        total_steps=12,
        seed=5,
        shuffle=False,
        resume_state=saved,
    )
    assert resumed["summary"]["resumed_from_step"] == 6
    assert len(resumed["history"]) == 12
    for expected, actual in zip(continuous["history"][6:], resumed["history"][6:]):
        assert actual["step"] == expected["step"]
        assert actual["loss"] == pytest.approx(expected["loss"], rel=0.0, abs=0.0)
        assert actual["lr"] == pytest.approx(expected["lr"], rel=0.0, abs=0.0)
        assert actual["grad_norm"] == pytest.approx(expected["grad_norm"], rel=0.0, abs=0.0)
    _assert_state_tree_equal(continuous["final_state"]["model"], resumed["final_state"]["model"], "model")
    _assert_state_tree_equal(continuous["final_state"]["master"], resumed["final_state"]["master"], "master")
    _assert_state_tree_equal(continuous["final_state"]["optimizer"], resumed["final_state"]["optimizer"], "optimizer")
    _assert_state_tree_equal(continuous["final_state"]["scheduler"], resumed["final_state"]["scheduler"], "scheduler")
    assert continuous["final_state"]["micro_step"] == resumed["final_state"]["micro_step"] == 12
    _record(method, steps=12, resume=True)


def _write_export_run(env: dict, method: str, module, cfg: dict, draft_config) -> Path:
    run_dir = env["root"] / f"export_{method}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    (run_dir / "contract.json").write_text(json.dumps(env["contract"]), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (run_dir / "data" / "manifest.json").write_text(
        json.dumps({"source": "unit", "num_train": 8, "num_val": 2, "rows_sha256": "d" * 64}),
        encoding="utf-8",
    )
    (run_dir / "train" / "train_summary.json").write_text(
        json.dumps(
            {
                "steps": 2,
                "final_loss": 1.0,
                "feature_mode": "offline",
                "device": "cpu",
                "device_type": "cpu",
                "backend": "cpu",
            }
        ),
        encoding="utf-8",
    )
    algorithm = get_algorithm(method)
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "algorithm": method,
        "identity": {"model_fingerprint": "f" * 64},
        "feature_mode": "offline",
        "step": 2,
        "model": {key: value.detach().clone() for key, value in module.draft_model.state_dict().items()},
        **algorithm.checkpoint_metadata(env["contract"], cfg, draft_config),
        "attention_backend": "sdpa_chunked",
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return run_dir


@pytest.mark.parametrize("method", available_algorithms())
def test_every_method_exports_and_reloads_its_draft(method: str, method_env):
    module, cfg, draft_config = _build_module(method, method_env)
    run_training_loop(
        module,
        _SyntheticFeatures(),
        cfg,
        method_env["contract"],
        torch.device("cpu"),
        total_steps=2,
        seed=3,
        shuffle=False,
    )
    run_dir = _write_export_run(method_env, method, module, cfg, draft_config)
    export_dir = export_run(run_dir)
    loaded, exported = load_export(export_dir)
    assert exported["method"] == method
    assert exported["drafter"]["attention_backend"] == "sdpa"
    assert exported["drafter"]["training_attention_backend"] == "sdpa_chunked"
    assert set(loaded.state_dict()) == set(module.draft_model.state_dict())

    anchors = torch.tensor([[2, 8]], dtype=torch.long)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    mask = create_dflash_sdpa_mask(
        anchors, keep, SEQ // 2, BLOCK, torch.device("cpu"), DTYPE
    )
    target_hidden = torch.randn(1, SEQ // 2, TARGET_LAYERS * HIDDEN) * 0.1
    noise = torch.randn(1, 2 * BLOCK, HIDDEN) * 0.1
    position_ids = torch.arange(SEQ // 2 + 2 * BLOCK).unsqueeze(0)
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
    assert original.shape == reloaded.shape == (1, 2 * BLOCK, HIDDEN)
    assert torch.equal(original, reloaded)
    _record(method, export=True)

