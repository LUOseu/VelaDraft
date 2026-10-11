"""Unified multimodal example schema and source readers.

Every dataset source normalises into :class:`MultimodalExample`
(one image + one question/answer pair), which is the single shape the prepare
stage consumes. Supported external entry points:

* JSONL rows in any of three shapes:
  - ``{"messages": [{"role", "content"}...], "images": [...]}`` (OpenAI style,
    ``content`` may be a string or a list of ``{"type": "text"/"image"}`` parts)
  - ``{"conversations": [{"from", "value"}...], "images": [...]}`` (ShareGPT/LLaVA)
  - ``{"question"/"prompt", "answer"/"response", "image"/"image_path"}``
* HuggingFace ``datasets`` rows via question/answer/image columns or a
  messages column (see ``dataset.hf`` config keys).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import read_jsonl

_QUESTION_KEYS = ("question", "prompt", "q", "instruction")
_ANSWER_KEYS = ("answer", "response", "a", "output")
_IMAGE_KEYS = ("image", "image_path", "img", "image_file")
_ID_KEYS = ("id", "sample_id", "uid")


@dataclass
class MultimodalExample:
    id: str
    question: str
    answer: str
    image_path: Path
    src_index: int = 0


def _text_from_content(content: Any) -> str:
    """Flatten a message ``content`` (string or typed parts) into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text") and part.get("text"):
                parts.append(str(part["text"]))
        return "\n".join(parts)
    return str(content or "")


def _first_turn_pair(messages: list[dict]) -> tuple[str, str] | None:
    """First human->assistant pair (single-turn MVP); supports both key styles."""
    question: str | None = None
    for message in messages:
        role = str(message.get("role", message.get("from", ""))).lower()
        text = message.get("content", message.get("value", message.get("text", "")))
        text = _text_from_content(text)
        if role in ("user", "human") and question is None:
            question = text
        elif role in ("assistant", "gpt", "model") and question is not None:
            return question, text
    return None


def _resolve_image(raw: Any, *, row_dir: Path, images_dir: Path | None) -> Path:
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError(f"v0.1 supports exactly one image per sample, got {len(raw)}.")
        raw = raw[0]
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Cannot interpret image field: {raw!r}")
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (images_dir, row_dir):
        if base is not None:
            candidate = Path(base) / path
            if candidate.exists():
                return candidate
    return (images_dir or row_dir) / path


def normalize_row(row: dict, *, index: int, row_dir: Path, images_dir: Path | None = None) -> MultimodalExample:
    """Normalise one raw dataset row into :class:`MultimodalExample`."""
    sample_id = str(next((row[k] for k in _ID_KEYS if row.get(k) not in (None, "")), f"{index:07d}"))

    messages = row.get("messages") or row.get("conversations")
    if messages:
        pair = _first_turn_pair(list(messages))
        if pair is None:
            raise ValueError(f"Row {sample_id!r} has no complete human->assistant turn.")
        question, answer = pair
        raw_images = row.get("images") or row.get("image") or []
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        image = _resolve_image(raw_images, row_dir=row_dir, images_dir=images_dir)
    else:
        question = next((str(row[k]) for k in _QUESTION_KEYS if row.get(k) is not None), None)
        answer = next((str(row[k]) for k in _ANSWER_KEYS if row.get(k) is not None), None)
        raw_image = next((row[k] for k in _IMAGE_KEYS if row.get(k) is not None), None)
        if question is None or answer is None or raw_image is None:
            raise ValueError(
                f"Row {index} does not match any supported schema (messages / conversations / question+answer+image)."
            )
        image = _resolve_image(raw_image, row_dir=row_dir, images_dir=images_dir)

    if "<image>" not in question:
        question = "<image>\n" + question
    if not image.exists():
        raise FileNotFoundError(f"Row {sample_id!r} image not found: {image}")
    return MultimodalExample(id=sample_id, question=question, answer=answer, image_path=image, src_index=index)


def iter_jsonl(path: str | Path, images_dir: str | Path | None = None) -> list[MultimodalExample]:
    """Read a JSONL file of multimodal samples in any supported row shape."""
    path = Path(path)
    base = Path(images_dir).resolve() if images_dir is not None else None
    examples = []
    for index, row in enumerate(read_jsonl(path)):
        try:
            examples.append(normalize_row(row, index=index, row_dir=path.parent, images_dir=base))
        except (ValueError, FileNotFoundError) as error:
            raise type(error)(f"{path} row {index}: {error}") from error
    return examples
