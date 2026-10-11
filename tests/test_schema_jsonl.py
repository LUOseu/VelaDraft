"""JSONL dataset schema: row normalisation across all supported shapes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.data.schema import iter_jsonl, normalize_row


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), (120, 30, 200)).save(path)


def test_normalize_supported_row_shapes(tmp_path):
    image = tmp_path / "a.jpg"
    _write_image(image)

    messages_row = {
        "id": "m1",
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color?"}]},
            {"role": "assistant", "content": "Purple."},
        ],
        "images": ["a.jpg"],
    }
    example = normalize_row(messages_row, index=0, row_dir=tmp_path)
    assert example.id == "m1"
    assert example.answer == "Purple."
    assert example.image_path == image
    assert "<image>" in example.question

    sharegpt_row = {
        "id": "s1",
        "conversations": [{"from": "human", "value": "<image>\nDescribe."}, {"from": "gpt", "value": "A thing."}],
        "images": ["a.jpg"],
    }
    example = normalize_row(sharegpt_row, index=1, row_dir=tmp_path)
    assert example.answer == "A thing."

    flat_row = {"question": "Flat?", "answer": "Yes.", "image": str(image)}
    example = normalize_row(flat_row, index=2, row_dir=tmp_path)
    assert example.answer == "Yes."
    assert example.image_path == image


def test_iter_jsonl_missing_image_mentions_row(tmp_path):
    _write_image(tmp_path / "b.jpg")
    path = tmp_path / "data.jsonl"
    valid = {
        "id": "x",
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        "images": ["b.jpg"],
    }
    broken = {"id": "y", "question": "Q2", "answer": "A2", "image": "missing.jpg"}
    path.write_text("\n".join(json.dumps(row) for row in (valid, broken)), encoding="utf-8")
    with pytest.raises(FileNotFoundError) as error:
        iter_jsonl(path)
    assert "row 1" in str(error.value)

    path.write_text(json.dumps(valid), encoding="utf-8")
    examples = iter_jsonl(path)
    assert len(examples) == 1
    assert examples[0].question.startswith("<image>")
    assert examples[0].image_path == tmp_path / "b.jpg"


def test_normalize_rejects_multi_image(tmp_path):
    _write_image(tmp_path / "a.jpg")
    row = {
        "question": "Q",
        "answer": "A",
        "images": ["a.jpg", "a.jpg"],
    }
    with pytest.raises(ValueError):
        normalize_row(row, index=0, row_dir=tmp_path)
