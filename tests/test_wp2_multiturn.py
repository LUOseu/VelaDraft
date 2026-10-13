"""WP2: schema-v2 multi-image / multi-turn data, regen bookkeeping and capture.

The v1 pipeline is exercised by the existing suites; these tests pin the v2
invariants: per-turn image placement, incremental ``full_ids`` bookkeeping,
multi-span capture records and the v1/v2 collate compatibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.data.schema import normalize_row
from vlm_dflash.data.sharegpt import row_image_turns, row_is_multi_turn
from vlm_dflash.dflash.features import FeatureDataset, collate_features
from vlm_dflash.target import regen as regen_module
from vlm_dflash.target.prompts import messages_from_row, render_conversation, render_prompt
from vlm_dflash.target.sample_capture import (
    build_sample_record,
    locate_image_span,
    locate_image_spans,
)
from vlm_dflash.utils import read_jsonl, write_jsonl

_END = 3  # end-of-message marker token id in the fake template


# --------------------------------------------------------------------------- #
# 1. schema v2
# --------------------------------------------------------------------------- #


def _save_image(path: Path, color=(10, 20, 30), side: int = 56) -> None:
    Image.new("RGB", (side, side), color).save(path)


def test_schema_v2_parses_multi_image_multi_turn(tmp_path):
    _save_image(tmp_path / "a.png")
    _save_image(tmp_path / "b.png", color=(200, 30, 40))
    row = {
        "id": "v2",
        "conversations": [
            {"from": "human", "value": "<image>\nfirst question"},
            {"from": "gpt", "value": "first answer"},
            {"from": "human", "value": "second question <image>"},
            {"from": "gpt", "value": "last answer"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 2],
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert example.is_conversation
    assert len(example.conversations) == 4
    assert example.image_paths == [tmp_path / "a.png", tmp_path / "b.png"]
    assert example.image_turns == [0, 2]
    assert example.answer == "last answer"
    assert example.image_path == tmp_path / "a.png"


def test_schema_v2_prepends_missing_placeholders(tmp_path):
    _save_image(tmp_path / "a.png")
    row = {
        "id": "v2",
        "conversations": [
            {"from": "human", "value": "no placeholder"},
            {"from": "gpt", "value": "answer"},
        ],
        "images": ["a.png"],
        "image_turns": [0],
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert example.conversations is not None
    assert example.conversations[0]["value"].startswith("<image>")


def test_schema_v2_rejects_inconsistent_rows(tmp_path):
    _save_image(tmp_path / "a.png")
    base = {
        "id": "bad",
        "conversations": [
            {"from": "human", "value": "<image>\nq"},
            {"from": "gpt", "value": "a"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ],
    }
    with pytest.raises(ValueError, match="image_turns"):
        normalize_row({**base, "images": ["a.png", "a.png"], "image_turns": [0]}, index=0, row_dir=tmp_path)
    with pytest.raises(ValueError, match="not a human turn"):
        normalize_row({**base, "images": ["a.png"], "image_turns": [1]}, index=0, row_dir=tmp_path)
    with pytest.raises(ValueError, match="placeholder"):
        normalize_row(
            {
                "id": "bad2",
                "conversations": [
                    {"from": "human", "value": "<image>\n<image>\nq"},
                    {"from": "gpt", "value": "a"},
                ],
                "images": ["a.png"],
                "image_turns": [0],
            },
            index=0,
            row_dir=tmp_path,
        )


def test_schema_v1_rows_stay_single_image(tmp_path):
    _save_image(tmp_path / "a.png")
    row = {
        "id": "v1",
        "conversations": [{"from": "human", "value": "<image>\nq"}, {"from": "gpt", "value": "a"}],
        "images": ["a.png"],
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert not example.is_conversation
    assert example.conversations is None and example.image_paths is None and example.image_turns is None
    with pytest.raises(ValueError, match="supports exactly one image"):
        normalize_row({**row, "images": ["a.png", "a.png"]}, index=0, row_dir=tmp_path)


def test_row_helpers_detect_v2_and_validate_turns(tmp_path):
    v1 = {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}], "images": ["x.png"]}
    assert not row_is_multi_turn(v1)
    assert row_image_turns(v1) == [0]
    v2 = {**v1, "images": ["x.png", "y.png"], "image_turns": [0, 0]}
    assert row_is_multi_turn(v2)
    assert row_image_turns(v2) == [0, 0]
    with pytest.raises(ValueError, match="image_turns"):
        row_image_turns({**v2, "image_turns": [0]})


# --------------------------------------------------------------------------- #
# 2. prepare: jsonl v2 round-trip
# --------------------------------------------------------------------------- #


def test_prepare_jsonl_writes_v2_rows_and_manifest(tmp_path):
    _save_image(tmp_path / "a.png")
    _save_image(tmp_path / "b.png", color=(66, 77, 88))
    rows = [
        {
            "id": "v2row",
            "conversations": [
                {"from": "human", "value": "<image>\nq0"},
                {"from": "gpt", "value": "a0"},
                {"from": "human", "value": "q1 <image>"},
                {"from": "gpt", "value": "a1"},
            ],
            "images": ["a.png", "b.png"],
            "image_turns": [0, 2],
        },
        {
            "id": "v1row",
            "conversations": [{"from": "human", "value": "<image>\nq"}, {"from": "gpt", "value": "a"}],
            "images": ["a.png"],
        },
    ]
    jsonl = tmp_path / "source.jsonl"
    jsonl.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cfg = {
        "dataset": {
            "source": "jsonl",
            "path": str(jsonl),
            "jsonl_images_dir": str(tmp_path),
            "slice": 2,
            "val_size": 0,
        }
    }
    manifest = build_dataset(cfg, tmp_path / "run" / "data", limit=2)
    out_rows = read_jsonl(tmp_path / "run" / "data" / "data.jsonl")
    by_id = {row["id"]: row for row in out_rows}
    v2 = by_id["v2row"]
    assert v2["schema"] == "vlm-dflash-data-v2"
    assert v2["image_turns"] == [0, 2]
    assert len(v2["images"]) == 2 and len(v2["conversations"]) == 4
    v1 = by_id["v1row"]
    assert "image_turns" not in v1 and len(v1["conversations"]) == 2
    manifest_v2 = next(entry for entry in manifest["rows"] if entry["id"] == "v2row")
    assert len(manifest_v2["image_sha256_list"]) == 2
    manifest_v1 = next(entry for entry in manifest["rows"] if entry["id"] == "v1row")
    assert "image_sha256_list" not in manifest_v1 and manifest_v1["image_sha256"]


# --------------------------------------------------------------------------- #
# 3. renderer
# --------------------------------------------------------------------------- #


class _RecordingProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        self.last_generation_prompt = add_generation_prompt
        return "rendered"


def test_renderer_places_images_at_turn_start():
    processor = _RecordingProcessor()
    row = {
        "id": "r",
        "conversations": [
            {"from": "human", "value": "<image>\nq0"},
            {"from": "gpt", "value": "dataset a0"},
            {"from": "human", "value": "before <image> after"},
            {"from": "gpt", "value": "a1"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 2],
    }
    messages, image_indices = messages_from_row(row)
    assert image_indices == [0, 1]
    assert messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "q0"}]
    assert messages[2]["content"] == [{"type": "image"}, {"type": "text", "text": "before  after"}]
    assert messages[1]["content"] == "dataset a0"
    truncated, used = messages_from_row(row, upto=3)
    assert used == [0, 1]
    assert [message["role"] for message in truncated] == ["user", "assistant", "user"]
    with pytest.raises(ValueError, match="no conversations"):
        messages_from_row({"id": "empty"})
    render_conversation(processor, messages)
    assert processor.last_generation_prompt is False
    render_conversation(processor, messages, add_generation_prompt=True)
    assert processor.last_generation_prompt is True
    # v1 renderer unchanged (single image, placeholder stripped, first text part after the image).
    processor2 = _RecordingProcessor()
    assert render_prompt(processor2, "<image>\nhello") == "rendered"
    assert processor2.last_messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "hello"}]


# --------------------------------------------------------------------------- #
# 4. concatenative fake processor + stub lm: regen bookkeeping
# --------------------------------------------------------------------------- #


class _Batch(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeChatProcessor:
    """Char-level tokenizer with a strictly concatenative chat template.

    One token per character; ``<IMG>`` markers expand to
    ``vision_start + image_token * (grid_h*grid_w/merge^2) + vision_end``.
    Because rendering is pure concatenation and tokenisation scans left to
    right, prefix-stability holds exactly like a real chat template would need
    it to, so regen's incremental ``full_ids`` checks are exercised for real.
    """

    image_token_id = 500
    vision_start_token_id = 501
    vision_end_token_id = 502
    patch = 14
    temporal = 2
    merge = 2

    def __init__(self):
        self.tokenizer = self

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(token)) if 0 < int(token) < 500 else "" for token in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        parts = []
        for message in messages:
            role = message["role"]
            parts.append("\x01" if role == "user" else "\x02")
            content = message["content"]
            if isinstance(content, list):
                for part in content:
                    parts.append("<IMG>" if part["type"] == "image" else part["text"])
            else:
                parts.append(content)
            parts.append("\x03")
        if add_generation_prompt:
            parts.append("\x02")
        return "".join(parts)

    def _image_tokens_and_pixels(self, image):
        import numpy as np

        array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        grid_h, grid_w = image.height // self.patch, image.width // self.patch
        patches = array.reshape(grid_h, self.patch, grid_w, self.patch, 3)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(grid_h * grid_w, self.patch * self.patch * 3)
        pixels = torch.tensor(patches).repeat(1, self.temporal)
        grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
        tokens = grid_h * grid_w // (self.merge * self.merge)
        return tokens, pixels, grid

    def __call__(self, text, images, return_tensors="pt", padding=False):
        assert return_tensors == "pt"
        image_list = list(images)
        token_lists, pixel_rows, grids, mm_types = [], [], [], []
        text = text[0] if isinstance(text, (list, tuple)) else text
        image_index = 0
        position = 0
        ids: list[int] = []
        while position < len(text):
            if text.startswith("<IMG>", position):
                assert image_index < len(image_list), "more <IMG> markers than images passed"
                tokens, pixels, grid = self._image_tokens_and_pixels(image_list[image_index])
                ids.append(self.vision_start_token_id)
                mm_types.append(0)
                ids.extend([self.image_token_id] * tokens)
                mm_types.extend([1] * tokens)
                ids.append(self.vision_end_token_id)
                mm_types.append(0)
                pixel_rows.append(pixels)
                grids.append(grid)
                image_index += 1
                position += 5
                continue
            ids.append(ord(text[position]))
            mm_types.append(0)
            position += 1
        assert image_index == len(image_list), "fewer <IMG> markers than images passed"
        if pixel_rows:
            pixels = torch.cat(pixel_rows, dim=0)
            grid = torch.cat(grids, dim=0)
        else:
            pixels = torch.zeros((0, self.patch * self.patch * 3 * self.temporal))
            grid = torch.zeros((0, 3), dtype=torch.long)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _Batch(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=torch.tensor([mm_types], dtype=torch.long),
        )


class _StubLM:
    def __init__(self, replies: list[list[int]]):
        # The regen batch path resolves the family adapter (image-argument
        # shaping); the stub stands in for a Qwen-VL target.
        self.config = SimpleNamespace(model_type="qwen2_5_vl")
        self.generation_config = SimpleNamespace(eos_token_id=_END)
        self.replies = [list(reply) for reply in replies]
        self.calls = 0

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, do_sample=False, **kwargs):
        assert do_sample is False
        reply = self.replies[self.calls]
        self.calls += 1
        suffix = torch.tensor([reply], dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


def _write_media(tmp_path: Path, names: list[str]) -> None:
    for index, name in enumerate(names):
        _save_image(tmp_path / name, color=(20 + index * 30, 40, 60 + index * 10))


def test_regen_multi_turn_bookkeeping(tmp_path):
    _write_media(tmp_path, ["a.png", "b.png"])
    processor = _FakeChatProcessor()
    row = {
        "id": "mt",
        "conversations": [
            {"from": "human", "value": "<image>\nq0"},
            {"from": "gpt", "value": "dataset a0"},
            {"from": "human", "value": "q1 <image>"},
            {"from": "gpt", "value": "dataset a1"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 2],
    }
    model = _StubLM([[ord("h"), ord("i"), _END], [ord("y"), ord("o")]])
    record, drop, per_turn = regen_module._regen_multi_turn_row(
        model,
        processor,
        row,
        tmp_path,
        stop_ids={_END},
        primary_stop=_END,
        device=torch.device("cpu"),
        max_new_tokens=16,
    )
    assert drop is None and record is not None
    regen = record["regen"]
    assert regen["n_turns"] == 2
    assert [len(turn["answer_ids"]) for turn in regen["turns"]] == [3, 3]
    # Turn 2 hit max_new_tokens without a stop; the primary stop was appended.
    assert regen["turns"][1]["forced_stop"] is True and regen["turns"][1]["answer_ids"] == [ord("y"), ord("o"), _END]
    assert regen["forced_stop"] == 1
    full_ids = regen["full_ids"]
    for turn, span in zip(regen["turns"], regen["answer_spans"]):
        assert full_ids[span[0] : span[1]] == turn["answer_ids"]
    assert regen["answer_spans"][0][1] <= regen["answer_spans"][1][0]
    # The stored sequence is exactly one full-conversation render (prefix-wise).
    overrides = {1: "hi", 3: "yo"}
    messages, _ = messages_from_row(row, assistant_overrides=overrides)
    text = render_conversation(processor, messages)
    fresh = processor(text=[text], images=[Image.open(tmp_path / name) for name in row["images"]])
    assert fresh["input_ids"][0].tolist()[: len(full_ids)] == full_ids
    # Per-turn prompts condition on the generated (not dataset) history.
    assert processor.decode(regen["turns"][1]["prompt_ids"]).endswith("hi\x03\x01q1\x03\x02")


def test_regen_multi_turn_skips_empty_answer(tmp_path):
    _write_media(tmp_path, ["a.png"])
    processor = _FakeChatProcessor()
    row = {
        "id": "empty",
        "conversations": [
            {"from": "human", "value": "<image>\nq"},
            {"from": "gpt", "value": "ignored"},
        ],
        "images": ["a.png"],
        "image_turns": [0],
    }
    record, drop, _ = regen_module._regen_multi_turn_row(
        _StubLM([[]]),
        processor,
        row,
        tmp_path,
        stop_ids={_END},
        primary_stop=_END,
        device=torch.device("cpu"),
        max_new_tokens=16,
    )
    assert record is None
    assert drop["reason"] == "empty_answer"


def test_run_regen_dispatches_v1_and_v2_rows(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "data" / "meta.json").write_text(json.dumps({"media_dir": "images"}), encoding="utf-8")
    images_dir = run_dir / "data" / "images"
    images_dir.mkdir()
    _write_media(images_dir, ["a.png", "b.png"])
    rows = [
        {
            "id": "v1",
            "conversations": [{"from": "human", "value": "<image>\nq"}, {"from": "gpt", "value": "old"}],
            "images": ["a.png"],
        },
        {
            "id": "v2",
            "conversations": [
                {"from": "human", "value": "<image>\nq0"},
                {"from": "gpt", "value": "old0"},
                {"from": "human", "value": "q1 <image>"},
                {"from": "gpt", "value": "old1"},
            ],
            "images": ["a.png", "b.png"],
            "image_turns": [0, 2],
        },
    ]
    write_jsonl(run_dir / "data" / "data.jsonl", rows)
    processor = _FakeChatProcessor()
    model = _StubLM([[ord("a"), _END], [ord("x"), _END], [ord("y"), ord("z"), _END]])
    monkeypatch.setattr(regen_module, "load_target", lambda cfg, contract: (model, processor))
    contract = {
        "target": {"model_path": "/models/tiny-vl", "model_type": "qwen2_5_vl", "model_fingerprint": "f" * 64, "processor_fingerprint": "p" * 64},
        "image": {"min_pixels": 196, "max_pixels": 196 * 1024, "merge_size": 2},
        "rope": {"sections": [2, 1, 1]},
        "layers": {"draft_layer_ids": [0, 1], "num_draft_layers": 2, "num_target_layers": 2},
        "tokens": {"image_token_id": 500},
        "mask": {"token_id": 255},
        "text": {"hidden_size": 8},
        "head": {"tie_word_embeddings": True},
    }
    stats = regen_module.run_regen(
        {"regen": {"max_new_tokens": 16, "min_response_tokens": 1, "batch_size": 2}}, run_dir, contract
    )
    out_rows = read_jsonl(run_dir / "regen" / "target_data.jsonl")
    assert [row["id"] for row in out_rows] == ["v1", "v2"]
    assert set(out_rows[0]["regen"]) == {"prompt_ids", "answer_ids", "stop_token_id"}
    assert out_rows[1]["regen"]["n_turns"] == 2
    assert stats["num_kept"] == 2
    assert stats["multi_turn"]["num_kept"] == 1
    assert stats["multi_turn"]["num_turns"] == {"min": 2, "max": 2}
    assert len(stats["multi_turn"]["per_turn"]) == 2


# --------------------------------------------------------------------------- #
# 5. capture: multi-span record with a tiny real Qwen2.5-VL model
# --------------------------------------------------------------------------- #


def _tiny_qwen25_vl_model():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

    torch.manual_seed(0)
    vision = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        spatial_merge_size=2,
        patch_size=14,
        temporal_patch_size=2,
        fullatt_block_indexes=[0, 1],
        window_size=56,
    )
    config = Qwen2_5_VLConfig(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1], "rope_theta": 10000.0},
        vision_config=vision,
        tie_word_embeddings=True,
        image_token_id=500,
        video_token_id=503,
        vision_start_token_id=501,
        vision_end_token_id=502,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
    return model, _FakeChatProcessor()


def _stored_multi_turn_row(tmp_path, processor, answers: list[str]) -> tuple[dict, Path]:
    _write_media(tmp_path, ["a.png", "b.png"])
    row = {
        "id": "capture-mt",
        "conversations": [
            {"from": "human", "value": "<image>\nq0"},
            {"from": "gpt", "value": "dataset0"},
            {"from": "human", "value": "q1 <image>"},
            {"from": "gpt", "value": "dataset1"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 2],
    }
    images = [Image.open(tmp_path / name).convert("RGB") for name in row["images"]]
    assistant_indices = [1, 3]
    overrides, turns, full_ids, spans = {}, [], [], []
    for msg_index, answer in zip(assistant_indices, answers):
        messages, image_indices = messages_from_row(row, assistant_overrides=overrides, upto=msg_index)
        text = render_conversation(processor, messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[images[position] for position in image_indices])
        prompt_ids = inputs["input_ids"][0].tolist()
        answer_ids = [ord(character) for character in answer] + [_END]
        full_ids.extend(prompt_ids[len(full_ids) :])
        full_ids.extend(answer_ids)
        spans.append([len(full_ids) - len(answer_ids), len(full_ids)])
        overrides[msg_index] = answer
        turns.append(
            {
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "answer_text": answer,
                "stop_token_id": _END,
                "forced_stop": False,
            }
        )
    row["schema"] = "vlm-dflash-data-v2"
    row["regen"] = {"turns": turns, "full_ids": full_ids, "answer_spans": spans, "n_turns": 2, "forced_stop": 0}
    return row, tmp_path


def test_capture_multi_image_multi_turn_record(tmp_path):
    model, processor = _tiny_qwen25_vl_model()
    row, media_dir = _stored_multi_turn_row(tmp_path, processor, ["one", "two"])
    record = build_sample_record(
        model,
        processor,
        row,
        media_dir,
        layer_ids=[0, 1],
        image_token_id=500,
        merge_size=2,
        dtype=torch.float32,
        seq_length=256,
        store_final_hidden=True,
    )
    assert record["n_turns"] == 2
    assert record["n_images"] == 2
    assert len(record["image_spans"]) == 2
    counts = [end - start + 1 for start, end in record["image_spans"]]
    assert counts == [4, 4]
    assert record["n_image_tokens"] == 8 == record["n_image_tokens_expected"]
    assert record["image_token_counts_expected"] == [4, 4]
    # loss mask is the union of the two answer segments, nothing else.
    mask = record["loss_mask"]
    expected = torch.zeros_like(mask)
    for start, end in record["answer_spans"]:
        expected[start:end] = True
    assert torch.equal(mask.float(), expected.float())
    assert record["answer_len"] == 4 + 4  # "one" + stop, "two" + stop
    assert record["truncated"] is False
    assert record["turn_spans"][0][0] == 0
    assert record["turn_spans"][-1][1] == int(record["input_ids"].numel())
    assert record["position3"].shape == (3, int(record["input_ids"].numel()))
    assert record["hidden"].shape[0] == int(record["input_ids"].numel())
    # Multi-image MRoPE positions: monotone along the sequence per axis, and
    # the second image occupies its own spatial block (not collapsed into the
    # first one) - the invariant the drafter's spatial-RoPE path consumes.
    positions = record["position3"]
    # Axes 0/1 (h/w) stream monotonically; the temporal axis legitimately
    # cycles inside an image block on this fixture.
    for axis in (0, 1):
        assert torch.all(positions[axis][1:] >= positions[axis][:-1]), f"axis {axis} not monotone"
    (start_a, end_a), (start_b, end_b) = record["image_spans"]
    assert int(positions[0, start_b:].min()) > int(positions[0, start_a : end_a + 1].max())


def test_capture_multi_turn_truncates_last_answer(tmp_path):
    model, processor = _tiny_qwen25_vl_model()
    row, media_dir = _stored_multi_turn_row(tmp_path, processor, ["one", "twotwotwo"])
    prompt_len_last = len(row["regen"]["turns"][-1]["prompt_ids"])
    record = build_sample_record(
        model,
        processor,
        row,
        media_dir,
        layer_ids=[0, 1],
        image_token_id=500,
        merge_size=2,
        dtype=torch.float32,
        seq_length=prompt_len_last + 8,
        store_final_hidden=False,
    )
    assert record["truncated"] is True
    assert int(record["input_ids"].numel()) == prompt_len_last + 8
    # The first answer is untouched; the last one is cut at the seq budget.
    assert record["answer_spans"][0][1] == len(row["regen"]["turns"][0]["prompt_ids"]) + 4
    assert record["answer_spans"][-1][1] == prompt_len_last + 8
    assert record["loss_mask"].sum().item() == 4 + 8


# --------------------------------------------------------------------------- #
# 6. span location + store collate compatibility
# --------------------------------------------------------------------------- #


def test_locate_image_spans_finds_runs():
    ids = torch.tensor([0, 500, 500, 7, 500, 500, 500, 1])
    assert locate_image_spans(ids, 500) == [(1, 2), (4, 6)]
    with pytest.raises(ValueError, match="No image tokens"):
        locate_image_spans(torch.tensor([1, 2, 3]), 500)
    with pytest.raises(ValueError, match="not contiguous"):
        # v1 helper still refuses multi-span sequences.
        locate_image_span(ids, 500)


def _write_store(features_dir: Path, records: list[dict], spans: bool) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": records}, features_dir / "shard_000.pt")
    index_rows = []
    for index, record in enumerate(records):
        entry = {
            "sample_id": record["sample_id"],
            "shard": "shard_000.pt",
            "shard_index": index,
            "seq_len": int(record["input_ids"].numel()),
            "prompt_len": int(record["prompt_len"]),
            "answer_len": int(record["answer_len"]),
            "image_span": list(record["image_span"]),
            "n_image_tokens": int(record["n_image_tokens"]),
        }
        if spans and record.get("image_spans"):
            entry["image_spans"] = [list(span) for span in record["image_spans"]]
            entry["n_turns"] = int(record.get("n_turns", 1))
        index_rows.append(entry)
    write_jsonl(features_dir / "index.jsonl", index_rows)
    manifest = {
        "format": "vlm-dflash-features-v1",
        "num_samples": len(records),
        "seq_length": max(int(record["input_ids"].numel()) for record in records),
    }
    if spans:
        manifest["schema_version"] = 2
    (features_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _record(seq: int, spans: list[tuple[int, int]] | None, turns: int = 1) -> dict:
    input_ids = torch.arange(seq) % 100 + 5
    for start, end in spans or []:
        input_ids[start : end + 1] = 500
    loss_mask = torch.zeros(seq, dtype=torch.bool)
    loss_mask[seq - 4 :] = True
    record = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden": torch.randn(seq, 8),
        "position3": torch.zeros(3, seq, dtype=torch.long),
        "image_span": spans[0] if spans else (0, 0),
        "n_image_tokens": int(sum(end - start + 1 for start, end in spans)) if spans else 0,
        "prompt_len": seq - 4,
        "answer_len": 4,
        "sample_id": "s",
    }
    if spans is not None:
        record["image_spans"] = [list(span) for span in spans]
        record["n_turns"] = turns
        record["turn_spans"] = [[0, seq]]
    return record


def test_feature_store_v1_and_v2_items_collate(tmp_path):
    v1_dir = tmp_path / "v1"
    _write_store(v1_dir, [_record(12, None)], spans=False)
    v2_dir = tmp_path / "v2"
    _write_store(v2_dir, [_record(16, [(1, 4), (9, 12)], turns=2)], spans=True)
    v1_item = FeatureDataset(v1_dir)[0]
    assert "image_spans" not in v1_item and v1_item["image_span"] == (0, 0)
    v2_item = FeatureDataset(v2_dir)[0]
    assert v2_item["image_spans"] == [(1, 4), (9, 12)]
    assert v2_item["n_turns"] == 2 and v2_item["turn_spans"] == [(0, 16)]
    batch = collate_features([v1_item, v2_item], image_token_id=500)
    assert batch["image_mask"][0].sum().item() == 0
    assert batch["image_mask"][1].sum().item() == 8
    assert batch["loss_mask"][1, 16 - 4] == 1.0
