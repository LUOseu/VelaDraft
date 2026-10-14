"""R4: ordered text/image layout and explicit rejection of unsupported media.

Regression tests for the review finding that ``_text_from_content`` flattened
typed message parts into one string (losing the text/image order inside a turn)
while ``messages_from_row`` moved every image to the turn start, and that rows
with ``video_url`` parts were accepted with the video silently dropped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.data.schema import normalize_row
from vlm_dflash.target.prompts import messages_from_row, render_conversation
from vlm_dflash.utils import read_jsonl


def _save(path: Path, name: str, color=(10, 20, 30)) -> None:
    Image.new("RGB", (32, 32), color=color).save(path / name)


class _ConcatProcessor:
    """Concatenative chat template (no tokenizer) for order assertions."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        parts = []
        for message in messages:
            parts.append("\x01" if message["role"] == "user" else "\x02")
            content = message["content"]
            if isinstance(content, list):
                for part in content:
                    if part["type"] == "image":
                        parts.append("<IMG>")
                    else:
                        parts.append(part["text"])
            else:
                parts.append(content)
            parts.append("\x03")
        if add_generation_prompt:
            parts.append("\x02")
        return "".join(parts)


def _interleaved_row() -> dict:
    return {
        "id": "interleaved",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A "},
                    {"type": "image"},
                    {"type": "text", "text": " B "},
                    {"type": "image"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 0],
        "schema": "vlm-dflash-data-v2",
    }


def _prepared(row: dict, tmp_path: Path) -> dict:
    """Run the row through the schema the way prepare does before regen/capture."""
    example = normalize_row(row, index=0, row_dir=tmp_path)
    prepared = {
        "id": example.id,
        "conversations": example.conversations,
        "images": row["images"],
    }
    if row.get("image_turns") is not None:
        prepared["image_turns"] = row["image_turns"]
    return prepared


def test_interleaved_parts_preserve_order(tmp_path):
    _save(tmp_path, "a.png")
    _save(tmp_path, "b.png", color=(200, 30, 40))
    example = normalize_row(_interleaved_row(), index=0, row_dir=tmp_path)
    segments = example.conversations[0]["segments"]
    assert segments == [
        {"type": "text", "text": "A "},
        {"type": "image", "index": 0},
        {"type": "text", "text": " B "},
        {"type": "image", "index": 1},
    ]
    messages, image_indices = messages_from_row(_prepared(_interleaved_row(), tmp_path))
    assert image_indices == [0, 1]
    assert messages[0]["content"] == [
        {"type": "text", "text": "A "},
        {"type": "image"},
        {"type": "text", "text": " B "},
        {"type": "image"},
    ]
    rendered = render_conversation(_ConcatProcessor(), messages)
    assert "A <IMG> B <IMG>" in rendered
    assert rendered.index("A ") < rendered.index("<IMG>") < rendered.index(" B ") < rendered.rindex("<IMG>")


def test_string_placeholders_become_ordered_segments(tmp_path):
    _save(tmp_path, "a.png")
    _save(tmp_path, "b.png", color=(200, 30, 40))
    row = {
        "id": "string-interleaved",
        "conversations": [
            {"from": "human", "value": "A <image> B <image>"},
            {"from": "gpt", "value": "answer"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 0],
        "schema": "vlm-dflash-data-v2",
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert example.conversations[0]["segments"] == [
        {"type": "text", "text": "A "},
        {"type": "image", "index": 0},
        {"type": "text", "text": " B "},
        {"type": "image", "index": 1},
    ]
    messages, image_indices = messages_from_row(_prepared(row, tmp_path))
    assert image_indices == [0, 1]
    rendered = render_conversation(_ConcatProcessor(), messages)
    assert "A <IMG> B <IMG>" in rendered


@pytest.mark.parametrize(
    "part, needle",
    [
        ({"type": "video_url", "video_url": {"url": "file://v.mp4"}}, "video"),
        ({"type": "audio", "audio": {"url": "file://a.wav"}}, "audio"),
        ({"type": "tool_call", "name": "x"}, "Unsupported content part"),
    ],
)
def test_unsupported_message_parts_are_rejected(tmp_path, part, needle):
    _save(tmp_path, "a.png")
    for row in (
        {
            "id": "v2",
            "messages": [
                {"role": "user", "content": [part, {"type": "text", "text": "q"}]},
                {"role": "assistant", "content": "a"},
            ],
            "images": ["a.png"],
            "image_turns": [0],
            "schema": "vlm-dflash-data-v2",
        },
        {
            "id": "v1",
            "messages": [
                {"role": "user", "content": [part, {"type": "text", "text": "q"}]},
                {"role": "assistant", "content": "a"},
            ],
            "images": ["a.png"],
        },
    ):
        with pytest.raises(ValueError, match=needle):
            normalize_row(row, index=0, row_dir=tmp_path)


def test_assistant_image_marker_is_rejected(tmp_path):
    _save(tmp_path, "a.png")
    row = {
        "id": "assistant-image",
        "conversations": [
            {"from": "human", "value": "<image>\nq"},
            {"from": "gpt", "value": "answer <image>"},
        ],
        "images": ["a.png"],
        "image_turns": [0],
        "schema": "vlm-dflash-data-v2",
    }
    with pytest.raises(ValueError, match="assistant"):
        normalize_row(row, index=0, row_dir=tmp_path)


def test_v1_mid_text_placeholder_is_rejected(tmp_path):
    _save(tmp_path, "a.png")
    row = {
        "id": "v1-mid",
        "conversations": [
            {"from": "human", "value": "A <image> B"},
            {"from": "gpt", "value": "answer"},
        ],
        "images": ["a.png"],
    }
    with pytest.raises(ValueError, match="leading|start"):
        normalize_row(row, index=0, row_dir=tmp_path)
    with pytest.raises(ValueError, match="leading|start"):
        normalize_row({**row, "id": "v1-late", "conversations": [
            {"from": "human", "value": "q <image>"},
            {"from": "gpt", "value": "answer"},
        ]}, index=0, row_dir=tmp_path)


def test_v1_leading_placeholder_bytes_unchanged(tmp_path):
    _save(tmp_path, "a.png")
    row = {
        "id": "v1-ok",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is this?"},
            {"from": "gpt", "value": "answer"},
        ],
        "images": ["a.png"],
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert example.question == "<image>\nWhat is this?"


def test_placeholder_count_mismatch_is_rejected(tmp_path):
    _save(tmp_path, "a.png")
    row = {
        "id": "too-many",
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\nq"},
            {"from": "gpt", "value": "answer"},
        ],
        "images": ["a.png"],
        "image_turns": [0],
        "schema": "vlm-dflash-data-v2",
    }
    with pytest.raises(ValueError, match="image"):
        normalize_row(row, index=0, row_dir=tmp_path)


def test_late_turn_image_is_not_readable_by_earlier_turn(tmp_path):
    _save(tmp_path, "a.png")
    _save(tmp_path, "b.png", color=(1, 2, 3))
    row = {
        "id": "late-image",
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q0"}]},
            {"role": "assistant", "content": "dataset a0"},
            {"role": "user", "content": [{"type": "text", "text": "q1"}, {"type": "image"}]},
            {"role": "assistant", "content": "dataset a1"},
        ],
        "images": ["a.png", "b.png"],
        "image_turns": [0, 2],
        "schema": "vlm-dflash-data-v2",
    }
    # Everything up to and including the first assistant turn: only image 0.
    prepared = _prepared(row, tmp_path)
    messages, image_indices = messages_from_row(prepared, upto=2)
    assert image_indices == [0]
    assert sum(part["type"] == "image" for part in messages[0]["content"]) == 1
    # The full conversation uses image 1 only after its own user turn.
    messages, image_indices = messages_from_row(prepared)
    assert image_indices == [0, 1]
    second_user = [message for message in messages if message["role"] == "user"][1]
    assert [part["type"] for part in second_user["content"]] == ["text", "image"]


def test_segments_survive_prepare_jsonl(tmp_path):
    _save(tmp_path, "a.png")
    _save(tmp_path, "b.png", color=(200, 30, 40))
    rows = [_interleaved_row()]
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cfg = {
        "dataset": {
            "source": "jsonl",
            "path": str(rows_path),
            "images_dir": str(tmp_path),
            "slice": 1,
            "val_size": 0,
            "seed": 42,
        }
    }
    build_dataset(cfg, tmp_path / "out")
    prepared = read_jsonl(tmp_path / "out" / "data.jsonl")[0]
    segments = prepared["conversations"][0]["segments"]
    assert [(seg["type"], seg.get("index")) for seg in segments] == [
        ("text", None),
        ("image", 0),
        ("text", None),
        ("image", 1),
    ]
    # Regen/capture/eval read the prepared row through the same renderer.
    messages, image_indices = messages_from_row(prepared)
    assert image_indices == [0, 1]
    assert "A <IMG> B <IMG>" in render_conversation(_ConcatProcessor(), messages)


def test_image_url_parts_are_positional_markers(tmp_path):
    _save(tmp_path, "a.png")
    row = {
        "id": "image-url",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://ignored.png"}},
                    {"type": "text", "text": "q"},
                ],
            },
            {"role": "assistant", "content": "a"},
        ],
        "images": ["a.png"],
        "image_turns": [0],
        "schema": "vlm-dflash-data-v2",
    }
    example = normalize_row(row, index=0, row_dir=tmp_path)
    assert example.conversations[0]["segments"][0] == {"type": "image", "index": 0}


def test_llava_source_rejects_mid_text_placeholder(tmp_path):
    """The v1 llava_instruct reader must reject non-convertible layouts too."""
    _save(tmp_path, "a.png")
    rows = [
        {
            "id": "bad",
            "image": "a.png",
            "conversations": [
                {"from": "human", "value": "A <image> B"},
                {"from": "gpt", "value": "answer"},
            ],
        }
    ]
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    cfg = {
        "dataset": {
            "source": "llava_instruct",
            "annotation": str(annotation),
            "images_dir": str(tmp_path),
            "slice": 1,
            "val_size": 0,
            "seed": 42,
        }
    }
    with pytest.raises(ValueError, match="leading|start"):
        build_dataset(cfg, tmp_path / "out")
