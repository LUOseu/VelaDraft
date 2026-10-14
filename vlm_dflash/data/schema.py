"""Unified multimodal example schema and source readers.

Every dataset source normalises into :class:`MultimodalExample`, the shape the
prepare stage consumes. Two row generations are supported:

* v1 (default): one image + the first question/answer pair. Unchanged.
* v2 (schema ``vlm-dflash-data-v2``, or any row carrying ``image_turns``):
  multi-image and/or multi-turn conversations. The images list stays flat and
  ``image_turns[i]`` records the conversation index each image belongs to
  (default ``0`` for every image when omitted). Rows without ``image_turns``
  and without the v2 marker keep the v1 single-image restriction.

Supported external entry points:

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
_V2_SCHEMA_MARKERS = {"vlm-dflash-data-v2", "v2"}
_ROLE_ALIASES = {"user": "human", "human": "human", "assistant": "gpt", "gpt": "gpt", "model": "gpt"}


@dataclass
class MultimodalExample:
    id: str
    question: str
    answer: str
    image_path: Path
    src_index: int = 0
    #: Full conversation (v2 rows); ``None`` for v1 rows (first pair only).
    conversations: list[dict] | None = None
    #: All resolved image paths, in ``images`` order (v2 rows).
    image_paths: list[Path] | None = None
    #: Conversation index each image belongs to, same order as ``image_paths``.
    image_turns: list[int] | None = None

    @property
    def is_conversation(self) -> bool:
        """Whether this row was parsed with schema-v2 (multi-image/multi-turn) semantics."""
        return self.conversations is not None


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


def _resolve_one_image(raw: Any, *, row_dir: Path, images_dir: Path | None) -> Path:
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


def _resolve_image(raw: Any, *, row_dir: Path, images_dir: Path | None) -> Path:
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError(f"v0.1 supports exactly one image per sample, got {len(raw)}.")
        raw = raw[0]
    return _resolve_one_image(raw, row_dir=row_dir, images_dir=images_dir)


def _conversation_turns(messages: list[dict]) -> list[dict]:
    """Normalise a message list into ShareGPT-style ``{"from", "value"}`` turns."""
    turns = []
    for message in messages:
        role = str(message.get("role", message.get("from", ""))).lower()
        if role not in _ROLE_ALIASES:
            raise ValueError(f"Unsupported conversation role {role!r}.")
        text = message.get("content", message.get("value", message.get("text", "")))
        turns.append({"from": _ROLE_ALIASES[role], "value": _text_from_content(text)})
    if not turns:
        raise ValueError("Conversation is empty.")
    for index, turn in enumerate(turns):
        if turn["from"] != ("human" if index % 2 == 0 else "gpt"):
            raise ValueError(
                f"Conversation must alternate human/gpt starting with human, got {turn['from']!r} at turn {index}."
            )
    return turns


def _normalize_conversation_row(
    row: dict,
    messages: list[dict],
    *,
    sample_id: str,
    index: int,
    row_dir: Path,
    images_dir: Path | None,
) -> MultimodalExample:
    """Parse a schema-v2 row: full conversation, flat images + ``image_turns``."""
    conversations = _conversation_turns(list(messages))
    raw_images = row.get("images", row.get("image")) or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]
    if not isinstance(raw_images, (list, tuple)) or not raw_images:
        raise ValueError(f"Row {sample_id!r} has no images; schema v2 rows need at least one image.")
    raw_turns = row.get("image_turns")
    if raw_turns is None:
        raw_turns = [0] * len(raw_images)
    if not isinstance(raw_turns, (list, tuple)):
        raise ValueError(f"Cannot interpret image_turns field: {raw_turns!r}")
    if len(raw_turns) != len(raw_images):
        raise ValueError(
            f"Row {sample_id!r} has {len(raw_images)} image(s) but {len(raw_turns)} image_turns entr(ies)."
        )
    image_turns = []
    for turn in raw_turns:
        if isinstance(turn, bool) or not isinstance(turn, int) or not 0 <= turn < len(conversations):
            raise ValueError(f"Row {sample_id!r} image_turns entry {turn!r} is not a valid conversation index.")
        if conversations[turn]["from"] != "human":
            raise ValueError(f"Row {sample_id!r} attaches an image to turn {turn}, which is not a human turn.")
        image_turns.append(int(turn))
    attached: dict[int, int] = {}
    for turn in image_turns:
        attached[turn] = attached.get(turn, 0) + 1
    for turn_index, turn in enumerate(conversations):
        if turn["from"] != "human":
            continue
        count = attached.get(turn_index, 0)
        present = turn["value"].count("<image>")
        if present > count:
            raise ValueError(
                f"Row {sample_id!r} turn {turn_index} has {present} <image> placeholder(s) but {count} image(s) attached."
            )
        if present < count:
            turn["value"] = "<image>\n" * (count - present) + turn["value"]
    image_paths = [_resolve_one_image(raw, row_dir=row_dir, images_dir=images_dir) for raw in raw_images]
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Row {sample_id!r} image not found: {path}")
    human_values = [turn["value"] for turn in conversations if turn["from"] == "human"]
    gpt_values = [turn["value"] for turn in conversations if turn["from"] == "gpt"]
    if not human_values or not gpt_values:
        raise ValueError(f"Row {sample_id!r} has no complete human->assistant turn.")
    return MultimodalExample(
        id=sample_id,
        question=human_values[0],
        answer=gpt_values[-1],
        image_path=image_paths[0],
        src_index=index,
        conversations=conversations,
        image_paths=image_paths,
        image_turns=image_turns,
    )


def normalize_row(row: dict, *, index: int, row_dir: Path, images_dir: Path | None = None) -> MultimodalExample:
    """Normalise one raw dataset row into :class:`MultimodalExample`."""
    sample_id = str(next((row[k] for k in _ID_KEYS if row.get(k) not in (None, "")), f"{index:07d}"))

    messages = row.get("messages") or row.get("conversations")
    if messages:
        declared_v2 = str(row.get("schema") or "").strip().lower() in _V2_SCHEMA_MARKERS
        if declared_v2 or row.get("image_turns") is not None:
            return _normalize_conversation_row(
                row, list(messages), sample_id=sample_id, index=index, row_dir=row_dir, images_dir=images_dir
            )
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
