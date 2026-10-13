"""Demo paths: synthetic sample generation, config building, checkpoint plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.demo import (
    align_config_with_drafter,
    build_demo_config,
    checkpoint_from_export,
    detect_drafter_algorithm,
    drafter_identity,
    drafter_recorded_layers,
    drafter_run_suffix,
    write_demo_source,
)
from vlm_dflash.export.hf import export_run


def _tiny_draft():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    torch.manual_seed(3)
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
    return config, Qwen3DFlashDraftModel(config)


def _tiny_export(tmp_path: Path) -> Path:
    config, draft = _tiny_draft()
    run_dir = tmp_path / "run"
    (run_dir / "train").mkdir(parents=True)
    contract = {
        "target": {
            "model_path": "/models/tiny-vl",
            "model_type": "qwen2_5_vl",
            "model_fingerprint": "f" * 64,
            "processor_fingerprint": "p" * 64,
        },
        "image": {"min_pixels": 100, "max_pixels": 200},
        "rope": {"sections": [16, 24, 24]},
        "layers": {"num_draft_layers": 2},
        "tokens": {"image_token_id": 1, "vision_start_token_id": 2},
    }
    (run_dir / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"drafter": {"spatial_rope_mode": "replace", "spatial_rope_enabled": False}}), encoding="utf-8"
    )
    checkpoint = {
        "format": "vlm-dflash-ckpt-v1",
        "draft_config": config.to_dict(),
        "attention_backend": "sdpa",
        "mask_token_id": 255,
        "block_size": 8,
        "num_anchors": 16,
        "loss_decay_gamma": 7.0,
        "target_layer_ids": [0, 1],
        "identity": {"model_fingerprint": "f" * 64, "image_min_pixels": 100, "image_max_pixels": 200},
        "step": 3,
        "model": {key: value.clone() for key, value in draft.state_dict().items()},
    }
    torch.save(checkpoint, run_dir / "train" / "ckpt_final.pt")
    return export_run(run_dir)


def test_write_demo_source_is_deterministic(tmp_path):
    pool_a, images_a = write_demo_source(tmp_path / "a", 4)
    pool_b, images_b = write_demo_source(tmp_path / "b", 4)
    rows_a = [json.loads(line) for line in pool_a.read_text(encoding="utf-8").splitlines()]
    rows_b = [json.loads(line) for line in pool_b.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows_a] == [row["id"] for row in rows_b]
    assert all({"id", "question", "answer", "image"} <= set(row) for row in rows_a)  # flat schema rows
    for name in ("demo_0000.png", "demo_0003.png"):
        assert (images_a / name).exists() and (images_b / name).exists()
        assert (images_a / name).read_bytes() == (images_b / name).read_bytes(), name


def test_write_demo_source_rejects_too_few_images(tmp_path):
    try:
        write_demo_source(tmp_path, 1)
    except ValueError as error:
        assert "at least 2" in str(error)
    else:  # pragma: no cover
        raise AssertionError("write_demo_source accepted a single image")


def test_build_demo_config_shape(tmp_path):
    cfg = build_demo_config(
        target="/models/Qwen2.5-VL-3B-Instruct",
        algorithm="dflash",
        run_dir=tmp_path / "runs" / "demo_dflash",
        pool_path=tmp_path / "pool.jsonl",
        images_dir=tmp_path / "images",
        num_train=4,
        num_val=4,
        num_prompts=4,
        max_new_tokens=64,
        steps=80,
        source_desc="unit",
    )
    assert cfg["dataset"]["source"] == "jsonl"
    assert cfg["eval"]["checkpoints"] == ["ckpt_init.pt"]
    assert cfg["eval"]["verification"] == "sequential"
    assert cfg["train"]["overfit_sanity"]["enabled"] is False
    # family-specific values must stay runtime-resolved, not baked in here
    assert "spatial_rope_sections" not in cfg["drafter"]
    assert cfg["drafter"]["mask_token_id"] == 151669  # default; contract validates it


def test_checkpoint_from_export_rebuilds_decoder(tmp_path):
    export_dir = _tiny_export(tmp_path)
    converted = tmp_path / "demo_run" / "train" / "ckpt_final.pt"
    method = checkpoint_from_export(export_dir, converted)
    assert method == "dflash"
    checkpoint = torch.load(converted, map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "vlm-dflash-ckpt-v1"
    assert checkpoint["algorithm"] == "dflash"
    assert checkpoint["block_size"] == 8
    assert checkpoint["identity"] == {
        "model_fingerprint": "f" * 64,
        "image_min_pixels": 100,
        "image_max_pixels": 200,
    }

    from vlm_dflash.algorithms import get_algorithm

    decoder, meta = get_algorithm("dflash").load_decoder(converted)
    assert meta["algorithm"] == "dflash"
    original_config, original_draft = _tiny_draft()
    assert set(decoder.state_dict()) == set(original_draft.state_dict())
    for key, value in original_draft.state_dict().items():
        assert torch.equal(value, decoder.state_dict()[key]), key


def test_checkpoint_from_export_rejects_plain_directory(tmp_path):
    plain = tmp_path / "not_an_export"
    plain.mkdir()
    (plain / "config.json").write_text("{}", encoding="utf-8")
    try:
        checkpoint_from_export(plain, tmp_path / "ckpt.pt")
    except ValueError as error:
        assert "not a vlm-dflash export" in str(error)
    else:  # pragma: no cover
        raise AssertionError("checkpoint_from_export accepted a non-export directory")


def test_drafter_identity_reads_export_run_and_pt(tmp_path):
    export_dir = _tiny_export(tmp_path)
    run_dir = tmp_path / "run"
    identity = drafter_identity(export_dir)
    assert identity["image_min_pixels"] == 100
    assert identity["image_max_pixels"] == 200

    from_pt = drafter_identity(run_dir / "train" / "ckpt_final.pt")
    assert from_pt == identity

    from_run = drafter_identity(run_dir)
    assert from_run == identity


def test_drafter_run_suffix_separates_contracts(tmp_path):
    export_dir = _tiny_export(tmp_path)
    assert drafter_run_suffix(export_dir) == export_dir.name
    assert drafter_run_suffix(export_dir / "train" / "ckpt_final.pt") == "ckpt_final"


def _demo_cfg(tmp_path, algorithm="dflash"):
    return build_demo_config(
        target="/models/Qwen2.5-VL-3B-Instruct",
        algorithm=algorithm,
        run_dir=tmp_path / "runs" / "demo",
        pool_path=tmp_path / "pool.jsonl",
        images_dir=tmp_path / "images",
        num_train=4,
        num_val=4,
        num_prompts=4,
        max_new_tokens=32,
        steps=0,
        source_desc="unit",
    )


def test_drafter_recorded_layers_reads_export_run_and_pt(tmp_path):
    export_dir = _tiny_export(tmp_path)
    run_dir = tmp_path / "run"
    assert drafter_recorded_layers(export_dir) == {
        "target_layer_ids": [0, 1],
        "capture_layer_ids": [0, 1],
    }
    # the raw run checkpoint records the consumed layers; the training store's
    # capture list is resolved by the export (legacy ckpts leave it None)
    from_pt = drafter_recorded_layers(run_dir / "train" / "ckpt_final.pt")
    assert from_pt == {"target_layer_ids": [0, 1], "capture_layer_ids": None}
    assert drafter_recorded_layers(run_dir) == from_pt


def test_align_config_with_drafter_adopts_recorded_layers(tmp_path):
    export_dir = _tiny_export(tmp_path)

    dflash_cfg = _demo_cfg(tmp_path)
    align_config_with_drafter(dflash_cfg, "dflash", export_dir)
    assert dflash_cfg["target"]["image_min_pixels"] == 100
    assert dflash_cfg["target"]["image_max_pixels"] == 200
    assert dflash_cfg["drafter"]["capture_layer_ids"] == [0, 1]
    # dflash-family routing is contract-derived; no algorithm context knob
    assert dflash_cfg["algorithm"] == "dflash"

    dfly_cfg = _demo_cfg(tmp_path, algorithm="dfly")
    align_config_with_drafter(dfly_cfg, "dfly", export_dir)
    assert dfly_cfg["drafter"]["capture_layer_ids"] == [0, 1]
    assert dfly_cfg["algorithm"]["name"] == "dfly"  # string form upgraded in place
    assert dfly_cfg["algorithm"]["dfly"]["context_layer_ids"] == [0, 1]


def test_align_config_with_drafter_leaves_capture_free_exports_alone(tmp_path):
    export = tmp_path / "legacy"
    export.mkdir()
    (export / "config.json").write_text(
        json.dumps(
            {
                "drafter": {"target_layer_ids": [0, 1]},
                "identity": {"image_min_pixels": 100, "image_max_pixels": 200},
            }
        ),
        encoding="utf-8",
    )
    (export / "model.safetensors").write_bytes(b"")
    cfg = _demo_cfg(tmp_path)
    align_config_with_drafter(cfg, "dflash", export)
    assert "capture_layer_ids" not in cfg["drafter"]
    assert cfg["target"]["image_min_pixels"] == 100


def test_detect_drafter_algorithm_reads_export_and_run_metadata(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "algorithm.json").write_text(json.dumps({"algorithm": "domino"}), encoding="utf-8")
    assert detect_drafter_algorithm(export) == "domino"

    config_only = tmp_path / "config_export"
    config_only.mkdir()
    (config_only / "config.json").write_text(json.dumps({"method": "dspark"}), encoding="utf-8")
    assert detect_drafter_algorithm(config_only) == "dspark"

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"algorithm": {"name": "jetspec"}}), encoding="utf-8")
    assert detect_drafter_algorithm(run) == "jetspec"

    assert detect_drafter_algorithm(tmp_path / "missing") is None
