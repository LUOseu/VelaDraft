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


def row_image_path(row: dict, media_dir: Path) -> Path:
    images = row.get("images") or []
    if len(images) != 1:
        raise ValueError(f"v0.1 supports exactly one image per sample, got {len(images)}.")
    return media_dir / images[0]
