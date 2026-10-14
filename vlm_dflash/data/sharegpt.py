"""ShareGPT-jsonl dataset IO (the format the reference ecosystem uses).

``data.jsonl`` rows carry ``{"conversations": [{"from", "value"}, ...],
"images": [relative_path]}``; ``meta.json`` records ``media_dir`` (resolved
relative to the JSONL's directory when not absolute).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..utils import read_json, read_jsonl


def load_meta(data_dir: str | os.PathLike) -> dict:
    return read_json(Path(data_dir) / "meta.json")


def load_sharegpt(path: str | os.PathLike, meta_path: str | os.PathLike | None = None) -> tuple[list[dict], Path]:
    """Return (rows, media_dir) for a ShareGPT jsonl plus its meta.json.

    ``meta_path`` defaults to ``meta.json`` next to the jsonl; stages that write
    derived jsonl files (e.g. ``regen/target_data.jsonl``) pass the data-stage
    meta explicitly.
    """
    path = Path(path)
    rows = read_jsonl(path)
    meta_file = Path(meta_path) if meta_path is not None else path.parent / "meta.json"
    meta = read_json(meta_file)
    media_dir = Path(meta["media_dir"])
    if not media_dir.is_absolute():
        media_dir = (meta_file.parent / media_dir).resolve()
    return rows, media_dir


def row_question(row: dict) -> str:
    return row["conversations"][0]["value"]


def row_answer(row: dict) -> str:
    return row["conversations"][1]["value"]


def row_conversations(row: dict) -> list[dict]:
    """Full conversation turns, or the first pair for rows that only carry one."""
    conversations = row.get("conversations") or []
    if not conversations:
        raise ValueError(f"Row {row.get('id')!r} has no conversations.")
    return conversations


def row_last_answer(row: dict) -> str:
    """Text of the last assistant turn (the supervised answer of a v2 row)."""
    for turn in reversed(row_conversations(row)):
        if str(turn.get("from")) in ("gpt", "assistant", "model"):
            return str(turn.get("value", ""))
    raise ValueError(f"Row {row.get('id')!r} has no assistant turn.")


def row_image_names(row: dict) -> list[str]:
    images = row.get("images") or []
    if isinstance(images, str):
        images = [images]
    return [str(image) for image in images]


def row_image_turns(row: dict) -> list[int]:
    """Conversation index per image; missing ``image_turns`` means all on turn 0."""
    images = row_image_names(row)
    raw_turns = row.get("image_turns")
    if raw_turns is None:
        return [0] * len(images)
    if len(raw_turns) != len(images):
        raise ValueError(
            f"Row {row.get('id')!r} has {len(images)} image(s) but {len(raw_turns)} image_turns entr(ies)."
        )
    return [int(turn) for turn in raw_turns]


def row_image_paths(row: dict, media_dir: Path) -> list[Path]:
    """Resolved paths of every image of the row, in ``images`` order."""
    names = row_image_names(row)
    if not names:
        raise ValueError(f"Row {row.get('id')!r} has no images.")
    return [media_dir / name for name in names]


def row_is_multi_turn(row: dict) -> bool:
    """Whether the row carries schema-v2 content (multi-image and/or multi-turn)."""
    conversations = row.get("conversations") or []
    return len(row_image_names(row)) > 1 or len(conversations) > 2 or row.get("image_turns") is not None


def row_image_path(row: dict, media_dir: Path) -> Path:
    images = row_image_names(row)
    if len(images) != 1:
        raise ValueError(
            f"Row {row.get('id')!r} has {len(images)} image(s); the single-image helper needs exactly one. "
            "Use row_image_paths for schema-v2 rows."
        )
    return media_dir / images[0]
