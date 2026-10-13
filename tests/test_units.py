"""Pure-logic unit tests (no model download, CPU only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.contract import ContractError, build_target_layer_ids, validate_yaml_against_contract
from vlm_dflash.eval.accept import summarize_samples
from vlm_dflash.eval.equivalence import cut_at_first_stop, sequences_equal
from vlm_dflash.provenance import IdentityMismatchError, assert_identity


def test_build_target_layer_ids_matches_reference():
    assert build_target_layer_ids(28, 3) == [1, 13, 25]
    assert build_target_layer_ids(36, 3) == [1, 17, 33]
    with pytest.raises(ContractError):
        build_target_layer_ids(3, 1)


def test_equivalence_stop_aware():
    stops = {2}
    assert cut_at_first_stop([5, 6, 2, 9, 9], stops) == [5, 6, 2]
    assert sequences_equal([5, 6, 2, 9], [5, 6, 2], stops)
    assert not sequences_equal([5, 7, 2], [5, 6, 2], stops)
    # identical prefix without any stop token must match exactly
    assert sequences_equal([5, 6, 7], [5, 6, 7], stops)
    assert not sequences_equal([5, 6, 7, 8], [5, 6, 7], stops)


def test_summarize_samples_al_math():
    samples = [
        {
            "verify_steps": 2,
            "new_tokens": 6,
            "accepted_tokens": 4,
            "draft_tokens": 14,
            "acceptance_events": [
                {"generated_position": 1, "accepted_tokens": 3, "draft_tokens": 7},
                {"generated_position": 5, "accepted_tokens": 1, "draft_tokens": 7},
            ],
            "wall_seconds": 0.4,
        }
    ]
    summary = summarize_samples(samples, block_size=8)
    assert summary["mean_al"] == 3.0
    assert summary["acceptance_rate"] == pytest.approx(4 / 14)
    # Per-offset acceptance: offset 1 accepted by both rounds, offsets 2 and 3
    # accepted by the first round only (both rounds drafted all 7 offsets).
    assert summary["per_slot_acceptance"] == [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]


def test_per_slot_acceptance_uses_draft_opportunities():
    # The final round is clipped by the output budget: it drafted only 2 offsets,
    # so it must not be counted against offsets it never drafted.
    sample = {
        "verify_steps": 3,
        "new_tokens": 9,
        "accepted_tokens": 6,
        "draft_tokens": 16,
        "acceptance_events": [
            {"generated_position": 1, "accepted_tokens": 3, "draft_tokens": 7},
            {"generated_position": 5, "accepted_tokens": 1, "draft_tokens": 7},
            {"generated_position": 7, "accepted_tokens": 2, "draft_tokens": 2},
        ],
    }
    summary = summarize_samples([sample], block_size=8)
    per_slot = summary["per_slot_acceptance"]
    assert per_slot[0] == 1.0
    assert per_slot[1] == pytest.approx(2 / 3)
    assert per_slot[2] == pytest.approx(1 / 2)
    assert per_slot[3:] == [0.0, 0.0, 0.0, 0.0]


def test_identity_gate_raises():
    a = {"model_fingerprint": "x", "mask_token_id": 1}
    b = {"model_fingerprint": "x", "mask_token_id": 2}
    with pytest.raises(IdentityMismatchError):
        assert_identity(a, b, "test")
    assert_identity(a, dict(a), "test")


def test_identity_gate_ignores_target_model_path():
    """Same checkpoint in a new directory is the same target; content changes still fail."""
    a = {"target_model_path": "/models/a", "model_fingerprint": "x", "mask_token_id": 1}
    moved = {**a, "target_model_path": "/models/b"}
    assert_identity(a, moved, "test")
    # legacy records without the field compare fine too (union diff skips it)
    legacy = {"model_fingerprint": "x", "mask_token_id": 1}
    assert_identity(legacy, moved, "test")
    changed = {**moved, "model_fingerprint": "y"}
    with pytest.raises(IdentityMismatchError):
        assert_identity(a, changed, "test")


def test_validate_yaml_against_contract():
    contract = {
        "mask": {"token_id": 151669},
        "layers": {"block_size": 8, "num_draft_layers": 3},
        "rope": {"sections": [16, 24, 24]},
        "image": {"min_pixels": 200704, "max_pixels": 1003520},
    }
    cfg = {
        "drafter": {
            "mask_token_id": 151669,
            "block_size": 8,
            "num_hidden_layers": 3,
            "spatial_rope_sections": [16, 24, 24],
        },
        "target": {"image_min_pixels": 200704, "image_max_pixels": 1003520},
    }
    validate_yaml_against_contract(cfg, contract)
    bad = json.loads(json.dumps(cfg))
    bad["drafter"]["mask_token_id"] = 5
    with pytest.raises(ContractError):
        validate_yaml_against_contract(bad, contract)


def test_prepare_data_determinism(tmp_path):
    from PIL import Image

    from vlm_dflash.data.prepare import build_dataset
    from vlm_dflash.utils import read_json

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    rows = []
    for index in range(20):
        image_path = images_dir / f"{index:06d}.jpg"
        Image.new("RGB", (64, 64), color=(index * 7 % 255, 30, 60)).save(image_path)
        answer = "long " * (150 if index % 2 == 0 else 10)
        rows.append(
            {
                "id": f"{index:06d}",
                "image": image_path.name,
                "conversations": [
                    {"from": "human", "value": "<image>\nWhat is shown?"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps(rows))
    cfg = {
        "dataset": {
            "source": "llava_instruct",
            "annotation": str(annotation),
            "images_dir": str(images_dir),
            "slice": 5,
            "val_size": 2,
            "seed": 42,
            "min_answer_chars": 600,
        }
    }
    manifest_a = build_dataset(cfg, tmp_path / "out_a")
    manifest_b = build_dataset(cfg, tmp_path / "out_b")
    assert manifest_a["rows_sha256"] == manifest_b["rows_sha256"]
    assert manifest_a["num_train"] == 5 and manifest_a["num_val"] == 2
    # min_answer_chars filter keeps only the long answers (10 of 20 rows)
    assert all(row["answer_chars"] >= 600 for row in manifest_a["rows"])
    meta = read_json(tmp_path / "out_a" / "meta.json")
    assert meta["media_dir"] == "images"
