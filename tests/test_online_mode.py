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
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.cli import build_parser, build_train_config, resolve_feature_mode
from vlm_dflash.dflash.features import FeatureDataset, collate_features
from vlm_dflash.dflash.train import run_training_loop, validate_resume_state
from vlm_dflash.provenance import IdentityMismatchError, identity_from_contract
from vlm_dflash.utils import write_jsonl

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
        "identity": None,
    }
    state.update(overrides)
    return state


def _validate(state, *, algorithm_name="dflash", mode="online", digest="d" * 64, total_steps=6, contract=None):
    validate_resume_state(
        state,
        algorithm_name=algorithm_name,
        mode=mode,
        feature_source_digest=digest,
        total_steps=total_steps,
        contract=contract or _stub_contract(),
    )


def test_resume_state_accepts_matching_state():
    _validate(_resume_state())


def test_resume_state_legacy_offline_is_not_resumable_as_online():
    legacy = _resume_state()
    legacy.pop("feature_mode")
    legacy.pop("feature_source_digest")
    _validate(legacy, mode="offline")  # legacy checkpoint == offline
    with pytest.raises(ValueError, match="mode mismatch"):
        _validate(legacy, mode="online")


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
