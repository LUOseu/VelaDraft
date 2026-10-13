"""Online feature source: CLI plumbing, live extraction, shared trainer wiring.

CPU-only, three layers of coverage:

1. config / CLI: mode resolution after the final merge, ``--features`` refusal
   before any stage, online runs regen but skip the offline capture;
2. source semantics with a stubbed per-sample extractor: record contract,
   re-extraction on every read, resume fast-forward without target calls, no
   feature store written, frozen-target refusal, digest/resume gates;
3. real-model integration on tiny random-init Qwen2.5-VL / Qwen3-VL: the
   actual ``build_sample_record`` path (processor contract -> vision tower ->
   MRoPE positions -> image span) feeding the real DFlash trainer loop, plus
   offline/online batch equality and the frozen-target guarantee.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.cli import build_parser, build_train_config, resolve_feature_mode
from vlm_dflash.dflash.features import FeatureDataset, collate_features, offline_feature_digest
from vlm_dflash.dflash.train import (
    _reconcile_train_log,
    preflight_resume,
    run_training_loop,
    train_dflash,
    validate_resume_state,
)
from vlm_dflash.provenance import IdentityMismatchError, identity_from_contract
from vlm_dflash.training_plan import PLAN_VERSION, training_plan_digest
from vlm_dflash.utils import read_jsonl, sha256_file, write_jsonl

import vlm_dflash.target.online as online_module


# --------------------------------------------------------------------------- #
# shared fixtures
# --------------------------------------------------------------------------- #

def _stub_contract(**overrides) -> dict:
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 196, "max_pixels": 196 * 1024, "merge_size": 2},
        "rope": {"sections": [2, 1, 1]},
        "layers": {"draft_layer_ids": [0, 1], "num_draft_layers": 2, "num_target_layers": 2},
        "tokens": {"image_token_id": 5},
        "mask": {"token_id": 255},
        "text": {"hidden_size": 8},
        "head": {"tie_word_embeddings": True},
    }
    contract.update(overrides)
    return contract


def _stub_cfg(*, store_final_hidden: bool = False, seq_length: int = 64) -> dict:
    return {
        "target": {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        "capture": {"seq_length": seq_length, "dtype": "bfloat16", "store_final_hidden": store_final_hidden},
        "train": {},
    }


def _write_online_run_dir(tmp_path: Path, *, rows: int = 4, answer_len: int = 12) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "regen").mkdir()
    (tmp_path / "images").mkdir(exist_ok=True)
    (run_dir / "data" / "meta.json").write_text(
        json.dumps({"media_dir": str(tmp_path / "images")}), encoding="utf-8"
    )
    payload = []
    for index in range(rows):
        # Real image bytes: the v2 content digest hashes what the loader would
        # actually read, so the files must exist.
        Image.new("RGB", (8, 8), (10 + index, 20, 30)).save(tmp_path / "images" / f"s{index}.png")
        payload.append(
            {
                "id": f"s{index}",
                "conversations": [
                    {"from": "human", "value": f"q{index}"},
                    {"from": "gpt", "value": "answer"},
                ],
                "images": [f"s{index}.png"],
                "regen": {
                    "prompt_ids": [10 + index] * 6,
                    "answer_ids": [100 + step for step in range(answer_len)],
                },
            }
        )
    write_jsonl(run_dir / "regen" / "target_data.jsonl", payload)
    return run_dir


class _StubTarget(torch.nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8, tied: bool = True):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight
        self.requires_grad_(False)
        self.eval()

    def get_input_embeddings(self):
        return self.embed


def _install_stub_extraction(monkeypatch, calls: list, *, store_final_hidden: bool = False,
                             context_hidden: int = 16, final_hidden_dim: int | None = None,
                             storage_dtype: torch.dtype | None = None):
    """Stubbed extractor: same record contract as the real capture forward.

    ``context_hidden`` is the fused routed-layer width; the teacher final hidden
    (used by distillation trainers) keeps the target's own hidden width, which
    the real capture never concatenates.
    """
    teacher_hidden_dim = context_hidden if final_hidden_dim is None else final_hidden_dim

    def fake_build(model, processor, row, media_dir, *, layer_ids, image_token_id, merge_size,
                   dtype, seq_length, store_final_hidden):
        calls.append(row.get("id"))
        prompt_len = len(row["regen"]["prompt_ids"])
        answer_len = min(len(row["regen"]["answer_ids"]), seq_length - prompt_len)
        length = prompt_len + answer_len
        # Tensors are produced under inference_mode on purpose: this is what the
        # real capture forward does, and the online source must copy them into
        # normal (backprop-able) tensors.
        hidden_dtype = storage_dtype or dtype
        with torch.inference_mode():
            record = {
                "input_ids": torch.arange(length, dtype=torch.int32),
                "hidden": torch.randn(length, context_hidden, dtype=torch.float32).to(hidden_dtype),
                "position3": torch.zeros(3, length, dtype=torch.int32),
                "image_span": (1, 4),
                "n_image_tokens": 4,
                "n_image_tokens_expected": 4,
                "loss_mask": torch.cat(
                    (
                        torch.zeros(prompt_len, dtype=torch.bool),
                        torch.ones(length - prompt_len, dtype=torch.bool),
                    )
                ),
                "sample_id": row.get("id"),
                "prompt_len": prompt_len,
                "answer_len": length - prompt_len,
                "truncated": False,
            }
            if store_final_hidden:
                record["final_hidden"] = torch.randn(length, teacher_hidden_dim, dtype=torch.float32).to(hidden_dtype)
        return record

    monkeypatch.setattr(online_module, "build_sample_record", fake_build)


def _stub_source(monkeypatch, run_dir, *, limit=None, contract=None, store_final_hidden=False,
                 tied=True, target=None):
    contract = contract or _stub_contract()
    target = target or _StubTarget(tied=tied)
    processor = object()
    monkeypatch.setattr(online_module, "load_target", lambda cfg, contract_: (target, processor))
    return online_module.OnlineFeatureSource(
        _stub_cfg(store_final_hidden=store_final_hidden), contract, run_dir, limit=limit
    )


# --------------------------------------------------------------------------- #
# 1. config / CLI plumbing
# --------------------------------------------------------------------------- #

def _train_args(tmp_path: Path, *extra: str):
    parser = build_parser()
    return parser.parse_args(
        [
            "train",
            "--target", "/models/Qwen2.5-VL-3B-Instruct",
            "--dataset", str(tmp_path / "data.jsonl"),
            *extra,
        ]
    )


def test_feature_mode_flag_and_defaults(tmp_path):
    args = _train_args(tmp_path, "--config", str(tmp_path / "cfg.yaml"), "--mode", "online")
    assert args.feature_mode == "online"
    # --feature-mode is the documented alias
    alias = _train_args(tmp_path, "--feature-mode", "offline")
    assert alias.feature_mode == "offline"
    # default: no flag, no config field -> offline
    assert _train_args(tmp_path).feature_mode is None

    cfg = {"train": {}}
    assert resolve_feature_mode(cfg, None) == "offline"
    assert resolve_feature_mode(cfg, "online") == "online"
    assert cfg["train"]["feature_mode"] == "online"


def test_build_train_config_online_suffixes_the_auto_run_name(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "target: {model: /models/tiny-vl}\n"
        "dataset: {source: jsonl, path: data.jsonl, slice: 4}\n",
        encoding="utf-8",
    )
    offline = build_train_config(_train_args(tmp_path, "--config", str(config)))
    assert offline["train"]["feature_mode"] == "offline"
    assert not offline["run"]["dir"].endswith("-online")

    online = build_train_config(_train_args(tmp_path, "--config", str(config), "--mode", "online"))
    assert online["train"]["feature_mode"] == "online"
    assert online["run"]["dir"].endswith("-online")
    assert online["run"]["name"] == online["run"]["dir"].split("/")[-1]


def test_feature_mode_merge_priority_and_validation(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "target: {model: /models/tiny-vl}\n"
        "dataset: {source: jsonl, path: data.jsonl, slice: 4}\n"
        "train: {feature_mode: offline}\n",
        encoding="utf-8",
    )
    # --set wins over YAML ...
    args = _train_args(tmp_path, "--config", str(config), "--set", "train.feature_mode=online")
    assert build_train_config(args)["train"]["feature_mode"] == "online"
    # ... and the explicit flag wins over --set (both are CLI; the flag is the
    # documented interface and must be the last word).
    args = _train_args(
        tmp_path, "--config", str(config), "--set", "train.feature_mode=online", "--mode", "offline"
    )
    assert build_train_config(args)["train"]["feature_mode"] == "offline"

    args = _train_args(tmp_path, "--config", str(config), "--set", "train.feature_mode=banana")
    with pytest.raises(SystemExit, match="feature_mode"):
        build_train_config(args)


class _FakeContract:
    @staticmethod
    def contract() -> dict:
        return _stub_contract()


def test_cmd_train_online_refuses_features_before_any_stage(monkeypatch, tmp_path):
    import vlm_dflash.cli as cli

    cfg = {
        "target": {"model": "/models/x"},
        "dataset": {"source": "jsonl", "path": "d.jsonl"},
        "train": {"feature_mode": "online"},
        "run": {"name": "x", "dir": str(tmp_path / "run")},
    }
    monkeypatch.setattr(cli, "build_train_config", lambda args: cfg)
    monkeypatch.setattr(
        cli, "stage_status", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stages must not run"))
    )
    args = _train_args(tmp_path, "--features", str(tmp_path / "existing_features"))
    with pytest.raises(SystemExit, match="cannot reuse a captured store"):
        cli.cmd_train(args)


def test_cmd_train_online_regens_and_skips_capture(monkeypatch, tmp_path):
    import vlm_dflash.cli as cli
    import vlm_dflash.contract as contract_module
    import vlm_dflash.data.prepare as prepare_module
    import vlm_dflash.dflash.train as train_module
    import vlm_dflash.target.capture as capture_module
    import vlm_dflash.target.regen as regen_module

    run_dir = tmp_path / "run"
    cfg = {
        "target": {"model": "/models/x"},
        "dataset": {"source": "jsonl", "path": "d.jsonl"},
        "train": {"feature_mode": "online"},
        "run": {"name": "x", "dir": str(run_dir)},
    }
    monkeypatch.setattr(cli, "build_train_config", lambda args: cfg)
    monkeypatch.setattr(contract_module, "ensure_contract", lambda cfg_, run_dir_: _stub_contract())

    calls: dict = {}
    monkeypatch.setattr(
        prepare_module, "build_dataset", lambda cfg_, out, limit=None: calls.setdefault("data", True)
    )
    monkeypatch.setattr(regen_module, "run_regen", lambda *a, **k: calls.setdefault("regen", True))
    monkeypatch.setattr(
        capture_module, "run_capture", lambda *a, **k: (_ for _ in ()).throw(AssertionError("capture must not run"))
    )

    def fake_train(cfg_, run_dir_, contract_, steps=None, limit=None, resume=False, features_dir=None,
                   feature_mode=None):
        calls["feature_mode"] = feature_mode
        calls["features_dir"] = features_dir
        return {"steps": 1, "final_loss": 1.0}

    monkeypatch.setattr(train_module, "train_dflash", fake_train)

    args = _train_args(tmp_path, "--mode", "online")
    assert cli.cmd_train(args) == 0
    assert calls["data"] is True
    assert calls["regen"] is True
    assert calls["feature_mode"] == "online"
    assert calls["features_dir"] is None


# --------------------------------------------------------------------------- #
# 2. resume gates and fast-forward semantics
# --------------------------------------------------------------------------- #

def _resume_state(**overrides) -> dict:
    state = {
        "format": "vlm-dflash-train-state-v1",
        "steps_total_planned": 6,
        "algorithm": "dflash",
        "feature_mode": "online",
        "feature_source_digest": "d" * 64,
        "training_plan_version": PLAN_VERSION,
        "training_plan_digest": "p" * 64,
        "identity": None,
    }
    state.update(overrides)
    return state


def _validate(state, *, algorithm_name="dflash", mode="online", digest="d" * 64, total_steps=6, contract=None,
              plan_version=PLAN_VERSION, plan_digest="p" * 64):
    validate_resume_state(
        state,
        algorithm_name=algorithm_name,
        mode=mode,
        feature_source_digest=digest,
        total_steps=total_steps,
        contract=contract or _stub_contract(),
        plan_version=plan_version,
        plan_digest=plan_digest,
    )


def test_resume_state_accepts_matching_state():
    _validate(_resume_state())


def test_resume_state_legacy_checkpoint_is_not_exactly_resumable():
    legacy = _resume_state()
    legacy.pop("feature_mode")
    legacy.pop("feature_source_digest")
    legacy.pop("training_plan_version")
    legacy.pop("training_plan_digest")
    # States that predate the plan gate recorded nothing to compare against:
    # exact resume cannot fabricate the missing fields, so they are refused in
    # both modes (an online attempt fails the mode gate first).
    with pytest.raises(ValueError, match="training-plan gate"):
        _validate(legacy, mode="offline")
    with pytest.raises(ValueError, match="mode mismatch"):
        _validate(legacy, mode="online")


def test_resume_state_requires_matching_training_plan():
    # Wrong plan version (or a missing / different digest) is refused: an exact
    # resume never compares digests across schemes or accepts a changed plan.
    with pytest.raises(ValueError, match="plan-version mismatch"):
        _validate(_resume_state(training_plan_version="vlm-dflash-training-plan-v0"))
    with pytest.raises(ValueError, match="training-plan gate"):
        _validate(_resume_state(training_plan_digest=None))
    with pytest.raises(ValueError, match="training-plan mismatch"):
        _validate(_resume_state(training_plan_digest="q" * 64))
    # The plan digest is part of the caller-provided plan, not re-derived from
    # the state: a matching version with a different current plan is refused.
    with pytest.raises(ValueError, match="training-plan mismatch"):
        _validate(_resume_state(), plan_digest="r" * 64)


def test_resume_state_rejects_mode_algorithm_and_source_mismatch():
    with pytest.raises(ValueError, match="mode mismatch"):
        _validate(_resume_state(feature_mode="offline"))
    with pytest.raises(ValueError, match="algorithm mismatch"):
        _validate(_resume_state(algorithm="dspark"))
    with pytest.raises(ValueError, match="feature-source mismatch"):
        _validate(_resume_state(), digest="e" * 64)
    with pytest.raises(ValueError, match="plan mismatch"):
        _validate(_resume_state(), total_steps=12)
    with pytest.raises(ValueError, match="no feature-source digest"):
        _validate(_resume_state(feature_source_digest=None))
    with pytest.raises(ValueError, match="format"):
        _validate(_resume_state(format="other"))


def test_resume_state_identity_gate():
    contract = _stub_contract()
    identity = identity_from_contract(contract)
    _validate(_resume_state(identity=identity), contract=contract)
    broken = dict(identity, model_fingerprint="0" * 64)
    with pytest.raises(IdentityMismatchError):
        _validate(_resume_state(identity=broken), contract=contract)


class _MainProcessOnly(torch.utils.data.Dataset):
    require_main_process = True

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("must not be read")


def test_training_loop_refuses_dataloader_workers_for_online_datasets():
    cfg = {"train": {"micro_batch_size": 1, "num_workers": 2, "draft_dtype": "float32"}}
    with pytest.raises(ValueError, match="num_workers=0"):
        run_training_loop(
            None,
            _MainProcessOnly(),
            cfg,
            {"tokens": {"image_token_id": 999}},
            torch.device("cpu"),
            total_steps=1,
            seed=0,
            shuffle=False,
        )


def _tiny_train_items(count: int = 12, seq: int = 48, seed: int = 0) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    items = []
    for _ in range(count):
        ids = torch.arange(seq) % 200 + 10
        mask = torch.ones(seq)
        mask[:8] = 0.0
        items.append(
            {
                "input_ids": ids,
                "loss_mask": mask,
                "hidden": torch.randn(seq, 128, generator=generator) * 0.2,
                "position3": torch.arange(seq).repeat(3, 1),
                "image_span": (0, 0),
                "sample_id": "tiny",
            }
        )
    return items


class _FastForwardDataset(torch.utils.data.Dataset):
    """Online-like dataset double: counts real (extraction) reads vs placeholders."""

    require_main_process = True

    def __init__(self, items: list[dict]):
        self.items = items
        self.real_reads = 0
        self.placeholder_reads = 0
        self._placeholders = 0

    def __len__(self):
        return len(self.items)

    def start_fast_forward(self, items: int) -> None:
        self._placeholders += max(0, int(items))

    def end_fast_forward(self) -> None:
        self._placeholders = 0

    def __getitem__(self, index: int) -> dict:
        if self._placeholders > 0:
            self._placeholders -= 1
            self.placeholder_reads += 1
            return {
                "input_ids": torch.zeros(2, dtype=torch.long),
                "loss_mask": torch.zeros(2),
                "hidden": torch.zeros(2, 128),
                "position3": torch.zeros(3, 2, dtype=torch.long),
                "image_span": (0, 0),
                "sample_id": "placeholder",
            }
        self.real_reads += 1
        return self.items[index]


def _tiny_dflash_module(seed: int = 0):
    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

    torch.manual_seed(seed)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
    )


def _run_fast_forward_loop(module, dataset, *, resume_state=None, sink=None):
    return run_training_loop(
        module,
        dataset,
        {
            "train": {
                "micro_batch_size": 2,
                "grad_accumulation_steps": 1,
                "num_epochs": 1,
                "lr": 1e-3,
                "log_every_steps": 0,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "min_lr_ratio": 0.1,
                "draft_dtype": "float32",
            }
        },
        {"tokens": {"image_token_id": 999999}},
        torch.device("cpu"),
        total_steps=6,
        seed=11,
        shuffle=True,
        log_path=None,
        resume_state=resume_state,
        on_checkpoint=sink,
        save_every=2,
    )


def test_online_resume_fast_forward_skips_extraction_and_keeps_losses_exact():
    """Resume replays the data order with placeholders: no re-extraction, same losses."""
    full_dataset = _FastForwardDataset(_tiny_train_items())
    uninterrupted = _run_fast_forward_loop(_tiny_dflash_module(), full_dataset)
    full_losses = [row["loss"] for row in uninterrupted["history"]]
    assert full_dataset.real_reads == 12  # every item extracted once
    assert full_dataset.placeholder_reads == 0

    saved: dict = {}
    interrupted_dataset = _FastForwardDataset(_tiny_train_items())
    interrupted = _run_fast_forward_loop(
        _tiny_dflash_module(), interrupted_dataset, sink=lambda state: saved.setdefault("state", state)
    )
    assert interrupted["history"] and int(saved["state"]["step"]) == 2

    resumed_dataset = _FastForwardDataset(_tiny_train_items())
    resumed = _run_fast_forward_loop(_tiny_dflash_module(), resumed_dataset, resume_state=saved["state"])
    resumed_losses = [row["loss"] for row in resumed["history"]]
    assert resumed["summary"]["resumed_from_step"] == 2
    assert resumed["summary"]["resume_fast_forward_batches"] == 2
    assert resumed["summary"]["resume_fast_forward_reduced_cost"] is True
    for first, second in zip(resumed_losses, full_losses):
        assert first == pytest.approx(second, abs=1e-9)
    # 2 fast-forward batches x 2 items answered with placeholders; the remaining
    # 4 batches were extracted for real.
    assert resumed_dataset.placeholder_reads == 4
    assert resumed_dataset.real_reads == 8


# --------------------------------------------------------------------------- #
# 3. online source semantics (stubbed extractor)
# --------------------------------------------------------------------------- #

def test_online_dataset_record_contract_and_reextraction(monkeypatch, tmp_path):
    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls)
    source = _stub_source(monkeypatch, run_dir)
    dataset = source.dataset()

    assert len(dataset) == 4
    item = dataset[0]
    assert set(item) == {"input_ids", "loss_mask", "hidden", "position3", "image_span", "sample_id"}
    assert item["input_ids"].dtype == torch.long
    assert item["loss_mask"].dtype == torch.float32
    assert item["position3"].dtype == torch.long
    assert item["image_span"] == (1, 4)
    # The answer-only supervision mask survives the source switch.
    assert float(item["loss_mask"][:6].sum()) == 0.0
    assert float(item["loss_mask"][6:].sum()) == item["loss_mask"][6:].numel()

    # Tensors produced under inference_mode must come back as normal tensors
    # that autograd can save (a plain detach() would raise here: "Inference
    # tensors cannot be saved for backward").
    weight = torch.ones_like(item["hidden"], requires_grad=True)
    (item["hidden"] * weight).sum().backward()
    assert weight.grad is not None
    assert torch.equal(weight.grad.float(), item["hidden"].float())

    # Every read re-runs the extractor (no hidden cache of teacher features).
    dataset[0]
    assert calls == ["s0", "s0"]
    dataset[1]
    assert calls == ["s0", "s0", "s1"]

    # Fast-forward credits answer cheap placeholders without touching the target.
    dataset.start_fast_forward(1)
    placeholder = dataset[0]
    assert placeholder["sample_id"].startswith("fast-forward:")
    assert len(calls) == 3
    dataset.end_fast_forward()
    dataset[0]
    assert len(calls) == 4

    summary = source.summary()
    assert summary["feature_mode"] == "online"
    assert summary["backend"] == "huggingface"
    assert summary["extractions"]["train"]["target_forwards"] == 4
    assert summary["target_forwards_total"] == 4
    assert summary["target_resident_bytes"] > 0
    assert len(summary["digest"]) == 64

    # No staged capture is written anywhere: online never produces a store.
    assert not (run_dir / "features").exists()


def test_online_source_refuses_unfrozen_or_empty_inputs(monkeypatch, tmp_path):
    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls)

    unfrozen = _StubTarget()
    unfrozen.embed.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="not frozen"):
        _stub_source(monkeypatch, run_dir, target=unfrozen)

    empty_dir = _write_online_run_dir(tmp_path / "empty", rows=0)
    with pytest.raises(ValueError, match="at least one regenerated row"):
        _stub_source(monkeypatch, empty_dir)


def test_online_digest_tracks_data_and_ignores_model_path(monkeypatch, tmp_path):
    calls: list = []
    _install_stub_extraction(monkeypatch, calls)
    run_dir = _write_online_run_dir(tmp_path)
    first = _stub_source(monkeypatch, run_dir).digest()

    moved_contract = _stub_contract()
    moved_contract["target"]["model_path"] = "/elsewhere/tiny-vl"
    assert _stub_source(monkeypatch, run_dir, contract=moved_contract).digest() == first

    changed_contract = _stub_contract()
    changed_contract["layers"]["draft_layer_ids"] = [2, 3]
    assert _stub_source(monkeypatch, run_dir, contract=changed_contract).digest() != first

    other_run = _write_online_run_dir(tmp_path / "other", rows=4, answer_len=16)
    assert _stub_source(monkeypatch, other_run).digest() != first
    assert _stub_source(monkeypatch, run_dir, limit=2).digest() != first


def test_online_digest_v2_covers_token_values_and_image_bytes(monkeypatch, tmp_path):
    """Equal-length token edits and same-size image swaps must change the digest."""
    _install_stub_extraction(monkeypatch, [])
    run_dir = _write_online_run_dir(tmp_path)
    first = _stub_source(monkeypatch, run_dir).digest()

    # Same length, same ids positionally: only the token *values* change.
    payload = read_jsonl(run_dir / "regen" / "target_data.jsonl")
    payload[1]["regen"]["answer_ids"][0] += 1
    write_jsonl(run_dir / "regen" / "target_data.jsonl", payload)
    second = _stub_source(monkeypatch, run_dir).digest()
    assert second != first

    # Same dimensions, different bytes: an image swap must not pass the gate.
    Image.new("RGB", (8, 8), (200, 10, 10)).save(tmp_path / "images" / "s2.png")
    assert _stub_source(monkeypatch, run_dir).digest() != second


def test_online_digest_v2_is_path_stable_and_verifies_the_data_manifest(monkeypatch, tmp_path):
    _install_stub_extraction(monkeypatch, [])
    run_dir = _write_online_run_dir(tmp_path)
    first = _stub_source(monkeypatch, run_dir).digest()

    # Same content copied elsewhere (run dir *and* media dir move): identical.
    copy_root = tmp_path / "copy"
    (copy_root / "run").parent.mkdir(parents=True)
    import shutil

    shutil.copytree(run_dir, copy_root / "run")
    shutil.copytree(tmp_path / "images", copy_root / "images")
    (copy_root / "run" / "data" / "meta.json").write_text(
        json.dumps({"media_dir": str(copy_root / "images")}), encoding="utf-8"
    )
    assert _stub_source(monkeypatch, copy_root / "run").digest() == first

    # A matching data/manifest.json is verified, not trusted blindly ...
    rows = read_jsonl(copy_root / "run" / "regen" / "target_data.jsonl")
    (copy_root / "run" / "data" / "manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": row["id"],
                        "image_sha256": sha256_file(copy_root / "images" / row["images"][0]),
                    }
                    for row in rows
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _stub_source(monkeypatch, copy_root / "run").digest() == first

    # ... and a stale manifest (recording the pre-swap bytes) is a loud error.
    Image.new("RGB", (8, 8), (0, 0, 255)).save(copy_root / "images" / "s0.png")
    with pytest.raises(RuntimeError, match="no longer match"):
        _stub_source(monkeypatch, copy_root / "run").digest()


def test_training_plan_digest_covers_semantic_fields_only():
    base = {
        "drafter": {
            "mask_token_id": 255,
            "block_size": 8,
            "num_anchors": 256,
            "num_hidden_layers": 3,
            "loss_decay_gamma": 7.0,
        },
        "train": {
            "seed": 42,
            "micro_batch_size": 2,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 6e-4,
            "warmup_ratio": 0.04,
            "min_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
    }
    digest = training_plan_digest(base, "dflash")
    assert training_plan_digest(json.loads(json.dumps(base)), "dflash") == digest
    assert training_plan_digest(base, "dflash", dataset_len=8) != training_plan_digest(
        base, "dflash", dataset_len=9
    )

    # Every effective training knob is covered, including shape-compatible ones.
    for section, key, value in (
        ("train", "seed", 43),
        ("train", "micro_batch_size", 1),
        ("train", "grad_accumulation_steps", 2),
        ("train", "num_epochs", 2),
        ("train", "lr", 1e-3),
        ("train", "warmup_ratio", 0.1),
        ("train", "min_lr_ratio", 0.2),
        ("train", "max_grad_norm", 0.5),
        ("train", "draft_dtype", "float32"),
        ("drafter", "block_size", 4),
        ("drafter", "num_anchors", 8),
        ("drafter", "num_hidden_layers", 2),
        ("drafter", "loss_decay_gamma", 4.0),
        ("drafter", "spatial_rope_enabled", False),
        ("drafter", "spatial_rope_mode", "add"),
        ("drafter", "sample_from_anchor", True),
        ("drafter", "attention_backend", "sdpa"),
    ):
        changed = json.loads(json.dumps(base))
        changed[section][key] = value
        assert training_plan_digest(changed, "dflash") != digest, (section, key)

    # Non-semantic fields (device, logging cadence, worker count, checkpoint
    # cadence) are deliberately excluded: they do not change the optimizer steps.
    for section, key, value in (
        ("train", "device", "cpu"),
        ("train", "log_every_steps", 1),
        ("train", "num_workers", 4),
        ("train", "save_every_steps", 3),
    ):
        changed = json.loads(json.dumps(base))
        changed[section][key] = value
        assert training_plan_digest(changed, "dflash") == digest, (section, key)


@pytest.mark.parametrize("algorithm", ["dflash", "dspark", "domino", "jetspec", "dflash2"])
def test_training_plan_digest_resolves_every_algorithm(algorithm):
    cfg = {"drafter": {"mask_token_id": 255, "block_size": 8}, "train": {}}
    digest = training_plan_digest(cfg, algorithm)
    assert len(digest) == 64
    if algorithm == "dflash":
        return
    defaults = _default_knob(algorithm)
    key = next(iter(defaults))
    # Algorithm-namespaced knobs resolve with defaults filled in: restating a
    # default is a no-op ...
    same = json.loads(json.dumps(cfg))
    same["algorithm"] = {algorithm: {key: defaults[key]}}
    assert training_plan_digest(same, algorithm) == digest
    # ... while overriding one (same shape) changes the digest instead of
    # slipping through as a missing key.
    changed = json.loads(json.dumps(cfg))
    changed["algorithm"] = {algorithm: {key: _changed_knob_value(algorithm, key)}}
    assert training_plan_digest(changed, algorithm) != digest


def _default_knob(algorithm: str) -> dict:
    from vlm_dflash.dspark.factory import DEFAULTS as DSPARK_DEFAULTS
    from vlm_dflash.domino.factory import DEFAULTS as DOMINO_DEFAULTS
    from vlm_dflash.jetspec.factory import DEFAULTS as JETSPEC_DEFAULTS
    from vlm_dflash.dflash2.factory import DEFAULTS as DFLASH2_DEFAULTS

    return {
        "dspark": DSPARK_DEFAULTS,
        "domino": DOMINO_DEFAULTS,
        "jetspec": JETSPEC_DEFAULTS,
        "dflash2": DFLASH2_DEFAULTS,
    }[algorithm]


def _changed_knob_value(algorithm: str, key: str):
    value = _default_knob(algorithm)[key]
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    return f"{value}-changed"


def test_online_final_hidden_flows_into_batches(monkeypatch, tmp_path):
    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls, store_final_hidden=True,
                             context_hidden=128, final_hidden_dim=64)
    source = _stub_source(monkeypatch, run_dir, store_final_hidden=True)
    dataset = source.dataset()
    batch = collate_features([dataset[0], dataset[1]], image_token_id=5)
    assert "final_hidden" in batch
    assert batch["hidden"].shape[-1] == 128
    assert batch["final_hidden"].shape[-1] == 64  # teacher width, not the fused width
    assert torch.isfinite(batch["final_hidden"]).all()


# --------------------------------------------------------------------------- #
# 4. real tiny-model integration
# --------------------------------------------------------------------------- #

class _FakeVLProcessor:
    """Processor double: real patchification + the processor's token contract.

    Not a random-tensor stand-in. It reproduces what the capture path consumes
    from an HF Qwen-VL processor (chat-template string, ``input_ids`` with the
    image placeholder span, ``attention_mask``, ``pixel_values``,
    ``image_grid_thw``, ``mm_token_type_ids``) while the vision/LM side is a
    real (tiny, random-init) Qwen model.
    """

    def __init__(self, *, image_token_id, vision_start_token_id, vision_end_token_id,
                 patch_size, temporal_patch_size, merge_size, question_tokens=5,
                 emit_image_tokens=None):
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.question_tokens = question_tokens
        self.text_base = 20
        self.emit_image_tokens = emit_image_tokens

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<|fake chat|>"

    def _pixel_values(self, image):
        raw = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        array = raw.to(torch.float32).view(image.height, image.width, 3)
        array = (array / 255.0 - 0.5) / 0.5
        patch, temporal = self.patch_size, self.temporal_patch_size
        grid_h, grid_w = image.height // patch, image.width // patch
        array = array.view(grid_h, patch, grid_w, patch, 3)
        array = array.unsqueeze(0).repeat(temporal, 1, 1, 1, 1, 1)  # (T, gh, gw, p, p, C)
        array = array.permute(1, 2, 0, 3, 4, 5).reshape(grid_h * grid_w, temporal * patch * patch * 3)
        return array, torch.tensor([[1, grid_h, grid_w]])

    def __call__(self, text, images, return_tensors="pt"):
        assert return_tensors == "pt"
        pixels, grid = self._pixel_values(images[0])
        image_tokens = int(grid.prod().item()) // (self.merge_size**2)
        if self.emit_image_tokens is not None:
            image_tokens = self.emit_image_tokens
        ids = (
            [self.vision_start_token_id]
            + [self.image_token_id] * image_tokens
            + [self.vision_end_token_id]
            + [self.text_base + (index % 7) for index in range(self.question_tokens)]
        )
        input_ids = torch.tensor([ids], dtype=torch.long)
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[:, 1 : 1 + image_tokens] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "mm_token_type_ids": mm_token_type_ids,
        }


def _tiny_qwen25_vl_model(seed: int = 0):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

    torch.manual_seed(seed)
    vision = Qwen2_5_VLVisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32,
        spatial_merge_size=2, patch_size=14, temporal_patch_size=2,
        fullatt_block_indexes=[0, 1], window_size=56,
    )
    config = Qwen2_5_VLConfig(
        vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1], "rope_theta": 10000.0},
        vision_config=vision, tie_word_embeddings=True,
        image_token_id=500, video_token_id=503, vision_start_token_id=501, vision_end_token_id=502,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
    processor = _FakeVLProcessor(
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        patch_size=14, temporal_patch_size=2, merge_size=2,
    )
    return model, processor


def _write_tiny_rows(run_dir: Path, processor, image_side: int = 56, rows: int = 2, answer_len: int = 12):
    from PIL import Image

    images_dir = run_dir / "data" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "meta.json").write_text(
        json.dumps({"media_dir": "images"}), encoding="utf-8"
    )
    payload = []
    for index in range(rows):
        image = Image.new("RGB", (image_side, image_side), (10 + index * 40, 20, 30 + index * 10))
        image.save(images_dir / f"s{index}.png")
        prompt_ids = processor(text=["x"], images=[image])["input_ids"][0].tolist()
        payload.append(
            {
                "id": f"s{index}",
                "conversations": [
                    {"from": "human", "value": f"describe image {index}"},
                    {"from": "gpt", "value": "answer"},
                ],
                "images": [f"s{index}.png"],
                "regen": {"prompt_ids": prompt_ids, "answer_ids": [100 + step for step in range(answer_len)]},
            }
        )
    write_jsonl(run_dir / "regen" / "target_data.jsonl", payload)
    return run_dir


def _tiny_contract(model_type: str = "qwen2_5_vl", *, tied: bool = True) -> dict:
    return _stub_contract(
        target={
            "model_path": "/models/tiny-vl",
            "model_type": model_type,
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        image={"min_pixels": 196, "max_pixels": 196 * 1024, "merge_size": 2},
        rope={"sections": [2, 1, 1]},
        text={"hidden_size": 32},
        tokens={"image_token_id": 500},
        head={"tie_word_embeddings": tied},
    )


def _sync_online_run_dir(tmp_path: Path, model, processor, *, rows: int = 2, answer_len: int = 12,
                         image_side: int = 56) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "regen").mkdir(parents=True)
    _write_tiny_rows(run_dir, processor, image_side=image_side, rows=rows, answer_len=answer_len)
    return run_dir


def _tiny_online_source(monkeypatch, tmp_path, *, model, processor, contract, store_final_hidden=False,
                        rows: int = 2, answer_len: int = 12, image_side: int = 56):
    run_dir = _sync_online_run_dir(tmp_path, model, processor, rows=rows, answer_len=answer_len,
                                   image_side=image_side)
    monkeypatch.setattr(online_module, "load_target", lambda cfg, contract_: (model, processor))
    source = online_module.OnlineFeatureSource(
        _stub_cfg(store_final_hidden=store_final_hidden), contract, run_dir
    )
    return source, run_dir


def test_tiny_qwen25_online_capture_matches_offline_store(monkeypatch, tmp_path):
    model, processor = _tiny_qwen25_vl_model()
    contract = _tiny_contract()
    source, run_dir = _tiny_online_source(
        monkeypatch, tmp_path, model=model, processor=processor, contract=contract
    )

    record = source.extract(0)
    assert record["hidden"].shape == (11 + 12, 2 * 32)  # 2 routed layers, target hidden 32
    assert record["image_span"] == (1, 4)
    assert record["n_image_tokens"] == record["n_image_tokens_expected"] == 4
    assert torch.equal(record["position3"][0], record["position3"][1]) is False  # real MRoPE
    assert record["loss_mask"][:11].sum() == 0 and record["loss_mask"][11:].all()

    # Determinism probe: two extractions of the same row must be bitwise equal
    # on CPU; otherwise the online/offline equality below would be meaningless.
    replay = source.extract(0)
    for key in ("input_ids", "loss_mask", "position3", "hidden"):
        assert torch.equal(record[key], replay[key]), key

    # Offline store written exactly like the staged capture, then both sources
    # must produce identical training batches through the shared collator.
    features_dir = run_dir / "features"
    features_dir.mkdir()
    torch.save({"samples": [record, source.extract(1)]}, features_dir / "shard_000.pt")
    write_jsonl(
        features_dir / "index.jsonl",
        [
            {"sample_id": "s0", "shard": "shard_000.pt", "shard_index": 0},
            {"sample_id": "s1", "shard": "shard_000.pt", "shard_index": 1},
        ],
    )
    (features_dir / "manifest.json").write_text("{}", encoding="utf-8")

    online_batch = collate_features([source.dataset()[0], source.dataset()[1]], image_token_id=500)
    offline_batch = collate_features(
        [FeatureDataset(features_dir)[0], FeatureDataset(features_dir)[1]], image_token_id=500
    )
    for key in ("input_ids", "loss_mask", "hidden", "position3", "image_mask"):
        assert torch.equal(online_batch[key], offline_batch[key]), key


def _dflash_module_from_online_source(source, contract, *, block_size=8, num_anchors=16):
    """Mirror factory.build_trainer_module: frozen head copies in target dtype."""
    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash.trainer_core import DFlashTrainerModule
    from vlm_dflash.target.sample_capture import extract_target_head

    embed_weight, lm_head_weight, tied = extract_target_head(source.model, contract)
    hidden, vocab = int(embed_weight.shape[1]), int(embed_weight.shape[0])
    torch.manual_seed(0)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=block_size,
        num_target_layers=len(contract["layers"]["draft_layer_ids"]),
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    # The trainer runs the draft in bf16 (``train.draft_dtype``); the frozen
    # head copies must share that dtype, exactly like the real factory where
    # the target itself runs in bf16.
    head_dtype = torch.bfloat16
    embed = torch.nn.Embedding(vocab, hidden, dtype=head_dtype)
    with torch.no_grad():
        embed.weight.copy_(embed_weight.to(head_dtype))
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, dtype=head_dtype)
    if lm_head_weight is None:
        lm_head.weight = embed.weight
    else:
        with torch.no_grad():
            lm_head.weight.copy_(lm_head_weight.to(head_dtype))
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
    )


def test_tiny_qwen25_online_run_trains_drafter_and_freezes_target(monkeypatch, tmp_path):
    model, processor = _tiny_qwen25_vl_model()
    contract = _tiny_contract()
    source, run_dir = _tiny_online_source(
        monkeypatch, tmp_path, model=model, processor=processor, contract=contract
    )
    module = _dflash_module_from_online_source(source, contract)

    target_before = [parameter.detach().clone() for parameter in model.parameters()]
    draft_before = {name: value.detach().clone() for name, value in module.draft_model.state_dict().items()}

    cfg = {
        "train": {
            "micro_batch_size": 1,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "draft_dtype": "bfloat16",
        }
    }
    output = run_training_loop(
        module,
        source.dataset(),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=2,
        seed=3,
        shuffle=False,
        log_path=None,
    )
    history = output["history"]
    assert len(history) == 2
    assert all(row["loss"] == row["loss"] for row in history)  # no NaN
    # The drafter actually received updates ...
    updated = any(
        not torch.equal(draft_before[name], value.detach())
        for name, value in module.draft_model.state_dict().items()
    )
    assert updated
    # ... and the teacher stayed frozen and grad-free.
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, (parameter.detach() for parameter in model.parameters()))
    )
    assert source.summary()["extractions"]["train"]["target_forwards"] == 2
    assert not (run_dir / "features").exists()


def test_online_capture_rejects_prompt_drift_and_token_count_mismatch(monkeypatch, tmp_path):
    model, processor = _tiny_qwen25_vl_model()
    contract = _tiny_contract()
    _source, run_dir = _tiny_online_source(
        monkeypatch, tmp_path, model=model, processor=processor, contract=contract
    )

    rows = [json.loads(line) for line in (run_dir / "regen" / "target_data.jsonl").read_text().splitlines()]
    rows[0]["regen"]["prompt_ids"] = rows[0]["regen"]["prompt_ids"][:-1] + [999]
    write_jsonl(run_dir / "regen" / "target_data.jsonl", rows)
    drifted_source = online_module.OnlineFeatureSource(_stub_cfg(), contract, run_dir)
    with pytest.raises(RuntimeError, match="Prompt token drift"):
        drifted_source.extract(0)

    # A processor that emits fewer image tokens than the grid implies must be
    # rejected instead of producing a misaligned span. The tiny real model's
    # placeholder check fires first in this arrangement; the capture's own span
    # check is exercised right below for the path where the model would tolerate
    # the mismatch.
    wrong_processor = _FakeVLProcessor(
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        patch_size=14, temporal_patch_size=2, merge_size=2, emit_image_tokens=2,
    )
    wrong_dir = _sync_online_run_dir(tmp_path / "wrong", model, wrong_processor)
    monkeypatch.setattr(online_module, "load_target", lambda cfg, contract_: (model, wrong_processor))
    wrong_source = online_module.OnlineFeatureSource(_stub_cfg(), contract, wrong_dir)
    with pytest.raises((ValueError, RuntimeError), match="do not match|token count mismatch"):
        wrong_source.extract(0)

    # Directly: the capture's image-token/grid check is the one that must reject
    # an inconsistent record (the model above failed on the same condition).
    from vlm_dflash.target import sample_capture as sample_capture_module

    def fake_forward(model_, inputs, full_ids, layer_ids, image_token_id, dtype, merge_size=2,
                     store_final_hidden=False):
        length = int(full_ids.numel())
        return {
            "input_ids": full_ids.to(torch.int32),
            "hidden": torch.zeros(length, 4, dtype=torch.float32),
            "position3": torch.zeros(3, length, dtype=torch.int32),
            "image_span": (1, 2),
            "n_image_tokens": 2,
            "n_image_tokens_expected": 4,
        }

    monkeypatch.setattr(sample_capture_module, "forward_and_capture", fake_forward)
    # Row 1 still has its original prompt ids (row 0 was mutated above).
    intact_row = json.loads((run_dir / "regen" / "target_data.jsonl").read_text().splitlines()[1])
    with pytest.raises(RuntimeError, match="Image token count mismatch"):
        sample_capture_module.build_sample_record(
            model, processor, intact_row, run_dir / "data" / "images",
            layer_ids=[0, 1], image_token_id=500, merge_size=2,
            dtype=torch.bfloat16, seq_length=64, store_final_hidden=False,
        )


def test_jetspec_dflash2_and_dspark_consume_online_batches(monkeypatch, tmp_path):
    """JetSpec/DFlash2 teacher_hidden and DSpark Markov heads over the online batch shape."""
    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls, store_final_hidden=True,
                             context_hidden=128, final_hidden_dim=64, storage_dtype=torch.float32)
    source = _stub_source(monkeypatch, run_dir, store_final_hidden=True)
    batch = collate_features([source.dataset()[0], source.dataset()[1]], image_token_id=5)
    assert "final_hidden" in batch

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash2.trainer import DFlash2TrainerModule
    from vlm_dflash.jetspec.trainer import JetSpecTrainerModule

    def qwen_config(**dflash_config):
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=512,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            block_size=8,
            num_target_layers=2,
            dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False, **dflash_config},
        )
        config._attn_implementation = "sdpa"
        return config

    def shared_head():
        embed = torch.nn.Embedding(256, 64)
        lm_head = torch.nn.Linear(64, 256, bias=False)
        lm_head.weight = embed.weight
        return embed, lm_head

    embed, lm_head = shared_head()
    jetspec = JetSpecTrainerModule(
        draft_model=Qwen3DFlashDraftModel(qwen_config()),
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=None,
        kd_temperature=1.0,
        kd_chunk_size=2048,
    )
    from vlm_dflash.dflash2.model import Qwen3DFlash2DraftModel

    embed, lm_head = shared_head()
    dflash2 = DFlash2TrainerModule(
        draft_model=Qwen3DFlash2DraftModel(
            qwen_config(
                dflash2={
                    "conv_kernel_size": 2,
                    "conv_group_size": 16,
                    "selector_rank": 8,
                    "selector_top_k": 4,
                }
            )
        ),
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        selector_loss_alpha=1.0,
        kd_chunk_size=512,
    )
    from vlm_dflash.dspark.model import Qwen3DSparkDraftModel
    from vlm_dflash.dspark.trainer import DSparkTrainerModule

    embed, lm_head = shared_head()
    dspark = DSparkTrainerModule(
        draft_model=Qwen3DSparkDraftModel(
            qwen_config(
                markov_rank=16,
                markov_head_type="vanilla",
                enable_confidence_head=True,
                confidence_head_with_markov=True,
            )
        ),
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        markov_rank=16,
        markov_head_type="vanilla",
    )
    for module in (jetspec, dflash2, dspark):
        metrics = module(
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden"],
            loss_mask=batch["loss_mask"],
            image_mask=batch["image_mask"],
            multimodal_position_ids=None,
            teacher_hidden=batch["final_hidden"],
        )
        assert torch.isfinite(metrics.loss)
        metrics.loss.backward()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.draft_model.parameters()
        )


def test_domino_progress_hook_runs_over_online_source(monkeypatch, tmp_path):
    """The shared loop drives Domino's curriculum hook for the online source too."""
    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.domino.trainer import DominoTrainerModule

    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls, context_hidden=128, storage_dtype=torch.float32)
    source = _stub_source(monkeypatch, run_dir)

    torch.manual_seed(0)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={
            "mask_token_id": 255,
            "target_layer_ids": [0, 1],
            "spatial_rope_enabled": False,
            "projector_type": "domino",
            "emb_dim": 16,
            "gru_hidden_dim": 32,
            "shift_label": True,
        },
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    with torch.no_grad():
        for parameter in draft.embed_proj.parameters():
            parameter.normal_(std=0.05)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    module = DominoTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
        sample_from_anchor=True,
        enable_correction_head=True,
    )
    output = run_training_loop(
        module,
        source.dataset(),
        {
            "train": {
                "micro_batch_size": 1,
                "grad_accumulation_steps": 1,
                "num_epochs": 1,
                "lr": 1e-3,
                "log_every_steps": 0,
                "max_grad_norm": 1.0,
                "warmup_ratio": 0.1,
                "min_lr_ratio": 0.1,
                "draft_dtype": "float32",
            }
        },
        _stub_contract(),
        torch.device("cpu"),
        total_steps=2,
        seed=5,
        shuffle=False,
        log_path=None,
    )
    history = output["history"]
    assert len(history) == 2
    assert all("lambda_base" in row for row in history)
    assert history[0]["lambda_base"] > history[-1]["lambda_base"]  # decaying curriculum
    assert source.summary()["extractions"]["train"]["target_forwards"] == 2


def test_tiny_qwen3vl_capture_interleaved_mrope_and_untied_head(monkeypatch, tmp_path):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    torch.manual_seed(0)
    text = Qwen3VLTextConfig(
        vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1]},
    )
    vision = Qwen3VLVisionConfig(
        depth=4, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32,
        deepstack_visual_indexes=(0, 2),
    )
    config = Qwen3VLConfig(
        text_config=text, vision_config=vision, image_token_id=500, video_token_id=503,
        vision_start_token_id=501, vision_end_token_id=502, tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    processor = _FakeVLProcessor(
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        patch_size=16, temporal_patch_size=2, merge_size=2,
    )
    contract = _tiny_contract(model_type="qwen3_vl", tied=False)
    source, run_dir = _tiny_online_source(
        monkeypatch, tmp_path, model=model, processor=processor, contract=contract, image_side=64
    )

    record = source.extract(0)
    assert record["image_span"] == (1, 4)
    assert record["hidden"].shape == (11 + 12, 2 * 32)

    # Untied head: the trainer must consume the target's own lm_head, and a
    # contract that disagrees with the checkpoint layout is a hard error.
    from vlm_dflash.target.sample_capture import extract_target_head

    embed_weight, lm_head_weight, tied = extract_target_head(model, contract)
    assert tied is False and lm_head_weight is not None

    # The mm_token_type_ids requirement is real for Qwen3-VL: dropping it from
    # the capture forward must fail rather than silently mis-position tokens.
    from vlm_dflash.target.sample_capture import forward_and_capture

    image = Image.new("RGB", (64, 64), (30, 40, 50))
    inputs = processor(text=["x"], images=[image])
    without_mm = {key: value for key, value in inputs.items() if key != "mm_token_type_ids"}
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        forward_and_capture(
            model, without_mm, record["input_ids"].long(), [0, 1], 500, torch.bfloat16, merge_size=2
        )

    mismatched = _tiny_contract(model_type="qwen3_vl", tied=True)
    with pytest.raises(RuntimeError, match="disagrees with the resolved contract"):
        extract_target_head(model, mismatched)


def test_online_head_extraction_reports_tied_and_untied(monkeypatch, tmp_path):
    run_dir = _write_online_run_dir(tmp_path)
    calls: list = []
    _install_stub_extraction(monkeypatch, calls)
    tied_source = _stub_source(monkeypatch, run_dir, tied=True)
    assert tied_source.tied_head is True and tied_source.lm_head_weight is None
    untied_contract = _stub_contract(head={"tie_word_embeddings": False})
    untied_source = _stub_source(monkeypatch, run_dir, tied=False, contract=untied_contract)
    assert untied_source.tied_head is False and untied_source.lm_head_weight is not None


# --------------------------------------------------------------------------- #
# 5. resume write-order, dtype boundary, offline/online alignment
# --------------------------------------------------------------------------- #

_TINY_DRAFT_KNOBS = {"mask_token_id": 255, "block_size": 8, "num_anchors": 16}


def _tiny_draft_config(*, vocab: int = 512, hidden: int = 32, block_size: int = 8, num_layers: int = 2):
    """A hand-built draft config standing in for ``factory.build_draft_config``."""
    torch.manual_seed(0)
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden // 4,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=block_size,
        num_target_layers=2,
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    return config


def _patch_tiny_draft_build(monkeypatch):
    """Let the real dflash algorithm build the tiny hand-made draft config."""
    import vlm_dflash.dflash.factory as dflash_factory

    monkeypatch.setattr(
        dflash_factory,
        "build_draft_config",
        lambda contract, cfg: _tiny_draft_config(block_size=int(cfg["drafter"]["block_size"])),
    )


def _patch_drafter_visual_check(monkeypatch):
    import vlm_dflash.dflash.visual_check as visual_check_module

    monkeypatch.setattr(
        visual_check_module,
        "check_drafter_visual_dependence",
        lambda *args, **kwargs: {"skipped": True, "reason": "unit test"},
    )


def _tiny_train_dflash_cfg(*, draft_dtype: str = "bfloat16") -> dict:
    return {
        "target": {"model": "/models/tiny-vl", "dtype": "float32", "attn_implementation": "sdpa"},
        "capture": {"seq_length": 64, "dtype": draft_dtype, "visual_path_check": False},
        "drafter": {
            **_TINY_DRAFT_KNOBS,
            "num_hidden_layers": 2,
            "attention_backend": "sdpa",
            "spatial_rope_enabled": False,
        },
        "train": {
            "micro_batch_size": 1,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "draft_dtype": draft_dtype,
            "device": "cpu",
            "overfit_sanity": {"enabled": False},
        },
    }


def _tiny_train_setup(monkeypatch, tmp_path, *, rows: int = 2, draft_dtype: str = "bfloat16"):
    """Real tiny Qwen2.5-VL target + real capture path, no model download."""
    model, processor = _tiny_qwen25_vl_model()
    contract = _tiny_contract()
    run_dir = _sync_online_run_dir(tmp_path, model, processor, rows=rows, answer_len=12)
    monkeypatch.setattr(online_module, "load_target", lambda cfg, contract_: (model, processor))
    cfg = _tiny_train_dflash_cfg(draft_dtype=draft_dtype)
    source = online_module.OnlineFeatureSource(cfg, contract, run_dir)
    return model, processor, contract, run_dir, cfg, source


def _tiny_online_train_run(monkeypatch, tmp_path, *, steps: int = 2):
    model, processor, contract, run_dir, cfg, _source = _tiny_train_setup(monkeypatch, tmp_path)
    _patch_tiny_draft_build(monkeypatch)
    _patch_drafter_visual_check(monkeypatch)
    summary = train_dflash(cfg, run_dir, contract, steps=steps, feature_mode="online")
    return cfg, contract, run_dir, summary


def test_train_dflash_resume_write_order_online(monkeypatch, tmp_path):
    cfg, contract, run_dir, summary = _tiny_online_train_run(monkeypatch, tmp_path)
    assert summary["steps"] == 2 and summary["feature_mode"] == "online"
    assert summary["dtypes"]["frozen_embed"] == "bfloat16"

    frozen_files = [
        run_dir / "train" / "ckpt_init.pt",
        run_dir / "train" / "train_log.jsonl",
        run_dir / "train" / "train_summary.json",
        run_dir / "train" / "overfit_sanity.json",
        run_dir / "provenance.json",
    ]
    before = {path: path.read_bytes() for path in frozen_files}

    # A changed plan (seed) is refused before any write: every original
    # artifact stays byte-identical.
    broken = json.loads(json.dumps(cfg))
    broken["train"]["seed"] = int(cfg["train"].get("seed", 42)) + 1
    with pytest.raises(ValueError, match="training-plan mismatch"):
        train_dflash(broken, run_dir, contract, steps=2, resume=True, feature_mode="online")
    for path, payload in before.items():
        assert path.read_bytes() == payload, path

    # A legal resume keeps the original ckpt_init byte-for-byte (the random-init
    # baseline is never regenerated) and continues from the recorded step.
    resumed = train_dflash(cfg, run_dir, contract, steps=2, resume=True, feature_mode="online")
    assert resumed["resumed_from_step"] >= 1
    assert (run_dir / "train" / "ckpt_init.pt").read_bytes() == before[run_dir / "train" / "ckpt_init.pt"]
    assert (run_dir / "train" / "overfit_sanity.json").read_bytes() == before[run_dir / "train" / "overfit_sanity.json"]


def test_cli_resume_preflight_refuses_before_writing(monkeypatch, tmp_path):
    cfg, contract, run_dir, _summary = _tiny_online_train_run(monkeypatch, tmp_path)
    from vlm_dflash import cli as cli_module

    # The real contract carries layers.block_size, which the YAML check reads.
    recorded_contract = json.loads(json.dumps(contract))
    recorded_contract["layers"]["block_size"] = int(cfg["drafter"]["block_size"])
    (run_dir / "contract.json").write_text(json.dumps(recorded_contract), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    frozen_files = [
        run_dir / "contract.json",
        run_dir / "config.json",
        run_dir / "train" / "ckpt_init.pt",
        run_dir / "train" / "ckpt_last.pt",
    ]
    before = {path: path.read_bytes() for path in frozen_files}

    # Stage digests are not the subject of this test; the upstream stages are
    # simulated as present and matching.
    monkeypatch.setattr(cli_module, "stage_status", lambda *args, **kwargs: "skip")
    args = SimpleNamespace(limit=None, steps=2, force=False)

    broken = json.loads(json.dumps(cfg))
    broken["train"]["seed"] = int(cfg["train"].get("seed", 42)) + 1
    with pytest.raises(SystemExit, match="training-plan mismatch"):
        cli_module._preflight_resume(broken, run_dir, args, "online", None)
    # A config pointed at another target must refuse before config.json is
    # rewritten (ensure_contract would only catch it afterwards).
    wrong_target = json.loads(json.dumps(cfg))
    wrong_target["target"]["model"] = "/models/other-vl"
    with pytest.raises(SystemExit, match="created for target"):
        cli_module._preflight_resume(wrong_target, run_dir, args, "online", None)
    for path, payload in before.items():
        assert path.read_bytes() == payload, path

    result = cli_module._preflight_resume(cfg, run_dir, args, "online", None)
    assert result["mode"] == "online" and result["total_steps"] == 2
    assert len(result["plan_digest"]) == 64 and result["step"] >= 1


def test_cmd_train_online_rejects_num_workers_before_any_stage(monkeypatch, tmp_path):
    import vlm_dflash.cli as cli_module

    cfg = {
        "target": {"model": "/models/x"},
        "dataset": {"source": "jsonl", "path": "d.jsonl"},
        "train": {"feature_mode": "online", "num_workers": 2},
        "run": {"name": "x", "dir": str(tmp_path / "run")},
    }
    monkeypatch.setattr(cli_module, "build_train_config", lambda args: cfg)
    monkeypatch.setattr(
        cli_module, "stage_status", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stages must not run"))
    )
    with pytest.raises(SystemExit, match="num_workers=0"):
        cli_module.cmd_train(_train_args(tmp_path))


def test_online_fp32_target_bf16_storage_trains_through_real_factory(monkeypatch, tmp_path):
    """The dtype boundary regression: an fp32 target must hand the trainer a
    bf16 frozen head (storage dtype), never the native fp32 weights."""
    model, processor, contract, run_dir, cfg, source = _tiny_train_setup(monkeypatch, tmp_path)

    assert next(model.parameters()).dtype == torch.float32
    assert source.embed_weight.dtype == torch.bfloat16
    assert source.lm_head_weight is None  # tied target: head shares the embedding

    from vlm_dflash.dflash.factory import build_trainer_module

    module = build_trainer_module(
        _tiny_draft_config(), contract, cfg, source.embed_weight, lm_head_weight=source.lm_head_weight
    )
    assert module.embed_tokens.weight.dtype == torch.bfloat16
    assert module.lm_head.weight.dtype == torch.bfloat16

    output = run_training_loop(
        module,
        source.dataset(),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=1,
        seed=0,
        shuffle=False,
        log_path=None,
    )
    history = output["history"]
    assert len(history) == 1 and history[0]["loss"] == history[0]["loss"]


def test_training_loop_refuses_frozen_head_dtype_mismatch():
    module = _tiny_dflash_module()  # frozen fp32 head/embed
    cfg = {
        "train": {
            "micro_batch_size": 1,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "draft_dtype": "bfloat16",
        }
    }
    with pytest.raises(RuntimeError, match="Frozen target head dtype mismatch"):
        run_training_loop(
            module,
            _FastForwardDataset(_tiny_train_items(2)),
            cfg,
            {"tokens": {"image_token_id": 999999}},
            torch.device("cpu"),
            total_steps=1,
            seed=0,
            shuffle=False,
            log_path=None,
        )


def test_offline_online_multistep_updates_align(monkeypatch, tmp_path):
    """Fixed teacher / data / init / seed / budget: staged and live features must
    produce the same optimizer trajectory (CPU: exact float equality)."""
    model, processor, contract, run_dir, _cfg, source = _tiny_train_setup(
        monkeypatch, tmp_path, draft_dtype="bfloat16"
    )
    records = [source.extract(index) for index in range(len(source.rows))]
    from vlm_dflash.target.sample_capture import extract_target_head

    embed_weight, lm_head_weight, _tied = extract_target_head(model, contract)
    # The trainer consumes the frozen head in the feature-storage dtype (bf16),
    # exactly like capture.save_target_head / the online source do.
    embed_weight = embed_weight.to(torch.bfloat16)
    if lm_head_weight is not None:
        lm_head_weight = lm_head_weight.to(torch.bfloat16)
    features_dir = run_dir / "features"
    _write_tiny_feature_store(features_dir, contract, records, embed_weight)

    cfg = {
        "drafter": {
            **_TINY_DRAFT_KNOBS,
            "num_hidden_layers": 2,
            "attention_backend": "sdpa",
            "spatial_rope_enabled": False,
        },
        "train": {
            "micro_batch_size": 1,
            "grad_accumulation_steps": 1,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            # Storage dtype is bf16 (the capture/online storage contract), so the
            # draft must compute in bf16: the frozen head is stored in bf16.
            "draft_dtype": "bfloat16",
        },
    }
    from vlm_dflash.dflash.factory import build_trainer_module

    def fresh_module():
        return build_trainer_module(
            _tiny_draft_config(), contract, cfg, embed_weight, lm_head_weight=lm_head_weight
        )

    torch.manual_seed(1234)
    offline_module = fresh_module()
    offline_output = run_training_loop(
        offline_module,
        FeatureDataset(features_dir),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=3,
        seed=7,
        shuffle=False,
        log_path=None,
    )
    torch.manual_seed(1234)
    online_module = fresh_module()
    online_output = run_training_loop(
        online_module,
        source.dataset(),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=3,
        seed=7,
        shuffle=False,
        log_path=None,
    )

    offline_losses = [row["loss"] for row in offline_output["history"]]
    online_losses = [row["loss"] for row in online_output["history"]]
    assert len(offline_losses) == len(online_losses) == 3
    assert offline_losses == online_losses
    for name, value in offline_module.draft_model.state_dict().items():
        assert torch.equal(value, online_module.draft_model.state_dict()[name]), name


def _write_tiny_feature_store(features_dir: Path, contract: dict, records: list[dict], embed_weight: torch.Tensor):
    features_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": records}, features_dir / "shard_000.pt")
    write_jsonl(
        features_dir / "index.jsonl",
        [
            {
                "sample_id": record["sample_id"],
                "shard": "shard_000.pt",
                "shard_index": index,
                "seq_len": int(record["input_ids"].numel()),
                "prompt_len": int(record["prompt_len"]),
                "answer_len": int(record["answer_len"]),
                "image_span": list(record["image_span"]),
                "n_image_tokens": int(record["n_image_tokens"]),
            }
            for index, record in enumerate(records)
        ],
    )
    seq_lengths = [int(record["input_ids"].numel()) for record in records]
    manifest = {
        "format": "vlm-dflash-features-v1",
        "identity": identity_from_contract(contract),
        "seq_length": max(seq_lengths),
        "storage_dtype": str(records[0]["hidden"].dtype).replace("torch.", ""),
        "num_samples": len(records),
        "num_truncated": 0,
        "draft_layer_ids": contract["layers"]["draft_layer_ids"],
        "stores_final_hidden": False,
        "tie_word_embeddings": True,
        "lm_head_saved": False,
        "seq_len_stats": {"min": min(seq_lengths), "max": max(seq_lengths), "mean": sum(seq_lengths) / len(seq_lengths)},
    }
    (features_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(
        {
            "format": "vlm-dflash-target-head-v1",
            "embed_tokens": embed_weight.to(records[0]["hidden"].dtype).cpu(),
            "tie_word_embeddings": True,
            "model_fingerprint": contract["target"]["model_fingerprint"],
        },
        features_dir / "target_head.pt",
    )


def test_offline_feature_digest_and_offline_resume_write_order(monkeypatch, tmp_path):
    model, processor, contract, run_dir, cfg, source = _tiny_train_setup(
        monkeypatch, tmp_path, draft_dtype="bfloat16"
    )
    records = [source.extract(index) for index in range(len(source.rows))]
    from vlm_dflash.target.sample_capture import extract_target_head

    embed_weight, _lm_head_weight, _tied = extract_target_head(model, contract)
    features_dir = run_dir / "features"
    _write_tiny_feature_store(features_dir, contract, records, embed_weight)

    # Content-addressed: a copy of the store keeps its identity, a changed
    # shard does not.
    copy_dir = tmp_path / "features_copy"
    shutil.copytree(features_dir, copy_dir)
    assert offline_feature_digest(copy_dir) == offline_feature_digest(features_dir)
    tampered = torch.load(copy_dir / "shard_000.pt", map_location="cpu", weights_only=False)
    tampered["samples"][0]["input_ids"] = tampered["samples"][0]["input_ids"] + 1
    torch.save(tampered, copy_dir / "shard_000.pt")
    assert offline_feature_digest(copy_dir) != offline_feature_digest(features_dir)

    _patch_tiny_draft_build(monkeypatch)
    summary = train_dflash(cfg, run_dir, contract, steps=2, feature_mode="offline")
    assert summary["steps"] == 2 and summary["feature_mode"] == "offline"

    frozen_files = [
        run_dir / "train" / "ckpt_init.pt",
        run_dir / "train" / "train_log.jsonl",
        run_dir / "train" / "train_summary.json",
        run_dir / "train" / "overfit_sanity.json",
        run_dir / "provenance.json",
    ]
    before = {path: path.read_bytes() for path in frozen_files}

    # A changed plan (batch geometry) is refused before any write.
    broken = json.loads(json.dumps(cfg))
    broken["train"]["micro_batch_size"] = int(cfg["train"]["micro_batch_size"]) + 1
    with pytest.raises(ValueError, match="training-plan mismatch"):
        train_dflash(broken, run_dir, contract, steps=2, resume=True, feature_mode="offline")
    for path, payload in before.items():
        assert path.read_bytes() == payload, path

    # The read-only preflight validates the same plan without touching the run.
    preflight = preflight_resume(cfg, run_dir, contract, steps=2, feature_mode="offline")
    assert preflight["total_steps"] == 2 and preflight["mode"] == "offline"

    # A legal resume keeps the original ckpt_init byte-for-byte.
    resumed = train_dflash(cfg, run_dir, contract, steps=2, resume=True, feature_mode="offline")
    assert resumed["resumed_from_step"] >= 1
    assert (run_dir / "train" / "ckpt_init.pt").read_bytes() == before[run_dir / "train" / "ckpt_init.pt"]


# --------------------------------------------------------------------------- #
# Second-review boundary fixes
# --------------------------------------------------------------------------- #

def _accum_items(count: int = 4, seq: int = 32, hidden: int = 128, seed: int = 0) -> list[dict]:
    """Anchor-valid batch items for the tiny trainer module (fused hidden)."""
    generator = torch.Generator().manual_seed(seed)
    items = []
    for _ in range(count):
        ids = torch.arange(seq) % 200 + 10
        mask = torch.ones(seq)
        mask[:8] = 0.0
        items.append(
            {
                "input_ids": ids,
                "loss_mask": mask,
                "hidden": torch.randn(seq, hidden, generator=generator) * 0.2,
                "position3": torch.arange(seq).repeat(3, 1),
                "image_span": (0, 0),
                "sample_id": "tiny",
            }
        )
    return items


def _accum_cfg(**overrides) -> dict:
    cfg = {
        "train": {
            "micro_batch_size": 1,
            "grad_accumulation_steps": 4,
            "num_epochs": 1,
            "lr": 1e-3,
            "log_every_steps": 0,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.1,
            "draft_dtype": "float32",
        }
    }
    for key, value in overrides.items():
        cfg["train"][key] = value
    return cfg


def test_pending_accumulation_across_epochs_is_not_a_stall():
    """One legal sample + accumulation=4: the first optimizer step spans four
    epochs; pending (gradient-bearing) micro-batches are progress, so the
    stall guard must not fire before the window fills."""
    module = _tiny_dflash_module()
    output = run_training_loop(
        module,
        _FastForwardDataset(_accum_items(1)),
        _accum_cfg(),
        {"tokens": {"image_token_id": 999999}},
        torch.device("cpu"),
        total_steps=1,
        seed=0,
        shuffle=False,
        log_path=None,
    )
    assert output["summary"]["steps"] == 1
    assert [row["step"] for row in output["history"]] == [1]


def test_no_valid_supervision_still_stalls():
    """The stall guard must still stop truly empty epochs (all batches raising
    NoValidAnchorsError), independent of the accumulation bookkeeping."""
    items = _accum_items(1)
    items[0]["loss_mask"] = torch.zeros_like(items[0]["loss_mask"])
    with pytest.raises(RuntimeError, match="no valid supervision"):
        run_training_loop(
            _tiny_dflash_module(),
            _FastForwardDataset(items),
            _accum_cfg(),
            {"tokens": {"image_token_id": 999999}},
            torch.device("cpu"),
            total_steps=1,
            seed=0,
            shuffle=False,
            log_path=None,
        )


def test_resume_replays_across_epoch_boundaries_exactly():
    """5 samples, micro-batch=2, accumulation=3: steps straddle epoch
    boundaries; a resume from a mid-run checkpoint must reproduce the
    uninterrupted remaining losses exactly (CPU float equality)."""
    cfg = _accum_cfg(micro_batch_size=2, grad_accumulation_steps=3)
    contract = {"tokens": {"image_token_id": 999999}}
    states: dict[int, dict] = {}

    def sink(state: dict) -> None:
        states[int(state["step"])] = state

    baseline = run_training_loop(
        _tiny_dflash_module(),
        _FastForwardDataset(_accum_items(5)),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=3,
        seed=0,
        shuffle=False,
        log_path=None,
        resume_state=None,
        on_checkpoint=sink,
        save_every=1,
    )
    baseline_losses = [row["loss"] for row in baseline["history"]]
    assert len(baseline_losses) == 3 and 2 in states

    resumed = run_training_loop(
        _tiny_dflash_module(),
        _FastForwardDataset(_accum_items(5)),
        cfg,
        contract,
        torch.device("cpu"),
        total_steps=3,
        seed=0,
        shuffle=False,
        log_path=None,
        resume_state=states[2],
        on_checkpoint=None,
        save_every=1,
    )
    resumed_losses = [row["loss"] for row in resumed["history"]]
    assert resumed["summary"]["resumed_from_step"] == 2
    assert [row["step"] for row in resumed["history"]] == [1, 2, 3]
    assert resumed_losses[:2] == baseline_losses[:2]
    assert resumed_losses[2] == baseline_losses[2]


def test_reconcile_train_log_moves_uncommitted_tail_and_half_lines(tmp_path):
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        '{"step": 1, "loss": 1.0}\n{"step": 2, "loss": 2.0}\n{"step": 3, "loss": 3.0}\n',
        encoding="utf-8",
    )
    _reconcile_train_log(log_path, 2)
    kept = [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()]
    recovery = tmp_path / "train_log.uncommitted.jsonl"
    moved = [json.loads(line)["step"] for line in recovery.read_text(encoding="utf-8").splitlines()]
    assert kept == [1, 2] and moved == [3]

    # A truncated final line (interrupted write) is uncommitted too.
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write('{"step": 3')
    _reconcile_train_log(log_path, 2)
    assert [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()] == [1, 2]
    assert recovery.read_text(encoding="utf-8").strip() == '{"step": 3'


def test_interrupted_run_resume_keeps_log_steps_unique(monkeypatch, tmp_path):
    """Reproduces the [1, 2, 3] -> [1, 2, 3, 3] report: an interrupted run
    whose log already contains a step the checkpoint does not cover. The
    resumed run must keep the log unique and ordered, with the replayed tail
    preserved as a recovery artifact."""
    import vlm_dflash.dflash.train as train_module

    model, processor, contract, run_dir, cfg, _source = _tiny_train_setup(monkeypatch, tmp_path)
    _patch_tiny_draft_build(monkeypatch)
    _patch_drafter_visual_check(monkeypatch)

    original = train_module.gather_train_state
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # step-1 and step-2 checkpoints, then the final state
            raise RuntimeError("injected interruption before the final resumable write")
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "gather_train_state", flaky)
    with pytest.raises(RuntimeError, match="injected interruption"):
        train_dflash(cfg, run_dir, contract, steps=3, feature_mode="online")
    monkeypatch.setattr(train_module, "gather_train_state", original)

    log_path = run_dir / "train" / "train_log.jsonl"
    steps_before = [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert steps_before == [1, 2, 3]
    ckpt_before = torch.load(run_dir / "train" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ckpt_before["step"] == 2

    resumed = train_dflash(cfg, run_dir, contract, steps=3, resume=True, feature_mode="online")
    assert resumed["steps"] == 3 and resumed["resumed_from_step"] == 2
    steps_after = [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert steps_after == [1, 2, 3]
    recovery = run_dir / "train" / "train_log.uncommitted.jsonl"
    assert [json.loads(line)["step"] for line in recovery.read_text(encoding="utf-8").splitlines()] == [3]

    # The completed run records its final resumable state: repeating the
    # resume is a no-op and leaves the log untouched.
    ckpt_done = torch.load(run_dir / "train" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ckpt_done["step"] == 3
    again = train_dflash(cfg, run_dir, contract, steps=3, resume=True, feature_mode="online")
    assert again["steps"] == 3 and again["resumed_from_step"] == 3
    assert [json.loads(line)["step"] for line in log_path.read_text(encoding="utf-8").splitlines()] == [1, 2, 3]


def test_visual_check_failure_closes_online_source(monkeypatch, tmp_path):
    """A failure during online startup (before the trainer exists) must still
    release the target: close() is called exactly once and drops its refs."""
    model, processor, contract, run_dir, cfg, _source = _tiny_train_setup(monkeypatch, tmp_path)
    _patch_tiny_draft_build(monkeypatch)
    cfg["capture"]["visual_path_check"] = True

    closed = []
    original_close = online_module.OnlineFeatureSource.close

    def counting_close(self):
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(online_module.OnlineFeatureSource, "close", counting_close)

    import vlm_dflash.target.visual_check as visual_check_module

    def boom(*args, **kwargs):
        raise RuntimeError("injected visual-check failure")

    monkeypatch.setattr(visual_check_module, "run_visual_path_check", boom)
    with pytest.raises(RuntimeError, match="injected visual-check failure"):
        train_dflash(cfg, run_dir, contract, steps=1, feature_mode="online")
    assert len(closed) == 1
    source = closed[0]
    assert source.model is None and source.processor is None
    assert source.embed_weight is None and source.lm_head_weight is None


def test_online_source_constructor_failure_closes_partial_state(monkeypatch, tmp_path):
    model, processor, contract, run_dir, cfg, _source = _tiny_train_setup(monkeypatch, tmp_path)
    closed = []
    original_close = online_module.OnlineFeatureSource.close

    def counting_close(self):
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(online_module.OnlineFeatureSource, "close", counting_close)
    original_extract = online_module.extract_target_head

    def boom(*args, **kwargs):
        raise RuntimeError("injected head-extraction failure")

    monkeypatch.setattr(online_module, "extract_target_head", boom)
    with pytest.raises(RuntimeError, match="injected head-extraction failure"):
        online_module.OnlineFeatureSource(cfg, contract, run_dir)
    assert len(closed) == 1
    assert closed[0].model is None and closed[0].processor is None

    # close() is idempotent on a healthy source too, and the released refs
    # include the frozen embedding/head copies handed to the trainer.
    monkeypatch.setattr(online_module, "extract_target_head", original_extract)
    source = online_module.OnlineFeatureSource(cfg, contract, run_dir)
    source.close()
    source.close()
    assert source.model is None and source.processor is None
    assert source.embed_weight is None and source.lm_head_weight is None


def test_cli_resume_preflight_external_features_dependency_chain(monkeypatch, tmp_path):
    """--features runs never build regen/capture; the resume preflight must
    check only the stages the run actually owns (data) plus the store digest
    and training plan."""
    import vlm_dflash.cli as cli_module
    from vlm_dflash.target.sample_capture import extract_target_head

    model, processor, contract, run_dir, cfg, source = _tiny_train_setup(monkeypatch, tmp_path)
    records = [source.extract(index) for index in range(len(source.rows))]
    embed_weight, _lm_head_weight, _tied = extract_target_head(model, contract)
    store = tmp_path / "shared_store"
    _write_tiny_feature_store(store, contract, records, embed_weight)

    _patch_tiny_draft_build(monkeypatch)
    _patch_drafter_visual_check(monkeypatch)
    summary = train_dflash(cfg, run_dir, contract, steps=2, feature_mode="offline", features_dir=store)
    assert summary["steps"] == 2
    # --features runs record no regen/capture *stage manifests* (cmd_train
    # skips those stages); the preflight must not demand them.
    assert not (run_dir / "regen" / "stage_manifest.json").exists()
    assert not (run_dir / "capture" / "stage_manifest.json").exists()

    recorded_contract = json.loads(json.dumps(contract))
    recorded_contract["layers"]["block_size"] = int(cfg["drafter"]["block_size"])
    (run_dir / "contract.json").write_text(json.dumps(recorded_contract), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    args = SimpleNamespace(limit=None, steps=2, force=False)

    consulted = []

    def fake_stage_status(run_dir_, stage, marker, digest, force):
        consulted.append(stage)
        return "skip"

    monkeypatch.setattr(cli_module, "stage_status", fake_stage_status)
    result = cli_module._preflight_resume(cfg, run_dir, args, "offline", store)
    assert result["mode"] == "offline" and result["total_steps"] == 2
    assert consulted == ["data"], f"preflight must not require regen/capture for --features: {consulted}"

    # Without --features the same run is internal-offline and *does* require
    # the regen stage it never built: the negative case still refuses.
    def regen_missing(run_dir_, stage, marker, digest, force):
        return "skip" if stage == "data" else "run"

    monkeypatch.setattr(cli_module, "stage_status", regen_missing)
    with pytest.raises(SystemExit, match="regen"):
        cli_module._preflight_resume(cfg, run_dir, SimpleNamespace(limit=None, steps=2, force=False), "offline", None)
