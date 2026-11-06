"""Frozen fixture manifest handling for the serving gate.

The manifest is deliberately data-only: partners can replace the image files
and prompts with their own 32 samples as long as the same category counts and
hashes are recorded.  No sampling or prompt rewriting happens at run time.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .contracts import ContractError, sha256_file, sha256_text, write_json_atomic

FIXTURE_SCHEMA = "veladraft-serving-fixtures-v1"
NATURAL_CATEGORIES = ("ocr", "chart_doc", "vqa", "caption")
NATURAL_PER_CATEGORY = 8
DIAGNOSTIC_COUNT = 8
BOUNDARY_COUNT = 8


class FixtureError(ValueError):
    pass


@dataclass
class Fixture:
    request_id: str
    category: str
    prompt: str
    max_tokens: int = 128
    image_path: Optional[Path] = None
    image_sha256: Optional[str] = None
    expect: str = "success"
    notes: str = ""
    boundary_variant: Optional[str] = None

    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)


def _read_image_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FixtureError(f"cannot read image {path}: {exc}") from exc


def image_data_uri(path: str | Path, mime: Optional[str] = None) -> str:
    path = Path(path)
    data = _read_image_bytes(path)
    if mime is None:
        suffix = path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _load_entries(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    entries = manifest.get(key)
    if not isinstance(entries, list):
        raise FixtureError(f"manifest.{key} must be a list")
    return entries


def load_manifest(path: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIXTURE_SCHEMA:
        raise FixtureError(f"unexpected fixture schema {manifest.get('schema_version')!r}")
    if manifest.get("frozen") is not True:
        raise FixtureError("fixture manifest must set frozen=true")
    root = path.parent
    requests = _load_entries(manifest, "requests")
    boundary = _load_entries(manifest, "boundary")
    diagnostic = _load_entries(manifest, "diagnostic")
    if len(boundary) != BOUNDARY_COUNT:
        raise FixtureError(f"expected {BOUNDARY_COUNT} boundary requests, got {len(boundary)}")
    if len(diagnostic) != DIAGNOSTIC_COUNT:
        raise FixtureError(f"expected {DIAGNOSTIC_COUNT} diagnostic requests, got {len(diagnostic)}")
    counts = {category: 0 for category in NATURAL_CATEGORIES}
    seen = set()
    for entry in requests:
        category = entry.get("category")
        if category not in counts:
            raise FixtureError(f"unknown natural category {category!r}")
        counts[category] += 1
        if entry["request_id"] in seen:
            raise FixtureError(f"duplicate request_id {entry['request_id']!r}")
        seen.add(entry["request_id"])
        _check_entry(entry, root, verify_hashes)
    for category, count in counts.items():
        if count != NATURAL_PER_CATEGORY:
            raise FixtureError(f"category {category!r} has {count} requests; expected {NATURAL_PER_CATEGORY}")
    for entry in boundary + diagnostic:
        _check_entry(entry, root, verify_hashes, require_image=False)
        if entry["request_id"] in seen:
            raise FixtureError(f"duplicate request_id {entry['request_id']!r}")
        seen.add(entry["request_id"])
    manifest["_root"] = str(root)
    return manifest


def _check_entry(entry: dict[str, Any], root: Path, verify_hashes: bool, require_image: bool = True) -> None:
    for key in ("request_id", "prompt"):
        if not isinstance(entry.get(key), str):
            raise FixtureError(f"fixture entry needs a string {key!r}: {entry!r}")
        if key == "request_id" and not entry[key].strip():
            raise FixtureError(f"fixture entry needs a non-empty request_id: {entry!r}")
        if key == "prompt" and not entry[key].strip() and entry.get("variant") != "empty_text":
            raise FixtureError(f"fixture entry needs a non-empty prompt: {entry!r}")
    image = entry.get("image")
    if image:
        image_path = (root / image).resolve()
        if not image_path.is_file():
            raise FixtureError(f"missing image {image_path}")
        if verify_hashes:
            expected = entry.get("image_sha256")
            if not expected:
                raise FixtureError(f"{entry['request_id']}: image_sha256 is required")
            actual = sha256_file(image_path)
            if actual != expected:
                raise FixtureError(
                    f"{entry['request_id']}: image sha256 mismatch ({actual} != {expected})"
                )
    elif require_image and entry.get("expect", "success") == "success":
        raise FixtureError(f"{entry['request_id']}: image is required for successful fixtures")
    if verify_hashes:
        expected_prompt = entry.get("prompt_sha256")
        if not expected_prompt:
            raise FixtureError(f"{entry['request_id']}: prompt_sha256 is required")
        actual_prompt = sha256_text(entry["prompt"])
        if actual_prompt != expected_prompt:
            raise FixtureError(
                f"{entry['request_id']}: prompt sha256 mismatch ({actual_prompt} != {expected_prompt})"
            )


def _fixture(entry: dict[str, Any], category: str, root: Path) -> Fixture:
    image = entry.get("image")
    return Fixture(
        request_id=entry["request_id"],
        category=category,
        prompt=entry["prompt"],
        max_tokens=int(entry.get("max_tokens", 128)),
        image_path=(root / image).resolve() if image else None,
        image_sha256=entry.get("image_sha256"),
        expect=entry.get("expect", "success"),
        notes=entry.get("notes", ""),
        boundary_variant=entry.get("variant"),
    )


def natural_fixtures(manifest: dict[str, Any]) -> list[Fixture]:
    root = Path(manifest["_root"])
    return [_fixture(entry, entry["category"], root) for entry in manifest["requests"]]


def diagnostic_fixtures(manifest: dict[str, Any]) -> list[Fixture]:
    root = Path(manifest["_root"])
    return [_fixture(entry, "diagnostic", root) for entry in manifest["diagnostic"]]


def boundary_fixtures(manifest: dict[str, Any]) -> list[Fixture]:
    root = Path(manifest["_root"])
    return [_fixture(entry, "boundary", root) for entry in manifest["boundary"]]


def image_url_content(path: str | Path, mime: Optional[str] = None) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_data_uri(path, mime)}}


def build_messages(fixture: Fixture, *, include_images: bool = True) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages for a fixture.

    Natural and diagnostic requests are one user turn.  Boundary variants
    exercise two-turn, empty-text, no-image, invalid-media, multi-image and
    malformed-structure paths.
    """

    variant = fixture.boundary_variant
    if variant == "two_turn":
        if fixture.image_path is None:
            raise FixtureError("two_turn boundary needs an image")
        return [
            {
                "role": "user",
                "content": [
                    image_url_content(fixture.image_path),
                    {"type": "text", "text": fixture.prompt},
                ],
            },
            {"role": "assistant", "content": "I can see the image."},
            {"role": "user", "content": "Now answer in one short sentence."},
        ]
    if variant == "no_image":
        return [{"role": "user", "content": fixture.prompt}]
    if variant == "empty_text":
        if fixture.image_path is None:
            raise FixtureError("empty_text boundary needs an image")
        return [
            {
                "role": "user",
                "content": [
                    image_url_content(fixture.image_path),
                    {"type": "text", "text": ""},
                ],
            }
        ]
    if variant == "invalid_media":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,not-base64!!"}},
                    {"type": "text", "text": fixture.prompt},
                ],
            }
        ]
    if variant == "two_images":
        if fixture.image_path is None:
            raise FixtureError("two_images boundary needs an image")
        content = [
            image_url_content(fixture.image_path),
            image_url_content(fixture.image_path),
            {"type": "text", "text": fixture.prompt},
        ]
        return [{"role": "user", "content": content}]
    if variant == "bad_structure":
        return [{"role": "user", "content": {"not": "a valid content list"}}]
    if variant == "long_prompt":
        return [
            {
                "role": "user",
                "content": [
                    image_url_content(fixture.image_path) if fixture.image_path else {"type": "text", "text": ""},
                    {"type": "text", "text": fixture.prompt},
                ],
            }
        ]
    if variant == "different_resolution":
        if fixture.image_path is None:
            raise FixtureError("different_resolution boundary needs an image")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - PIL is a serving dependency
            raise FixtureError("Pillow is required for the different_resolution fixture") from exc
        image = Image.open(fixture.image_path).convert("RGB")
        image.thumbnail((320, 320))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": fixture.prompt},
                ],
            }
        ]
    if variant:
        raise FixtureError(f"unknown boundary variant {variant!r}")
    if fixture.image_path is None:
        return [{"role": "user", "content": fixture.prompt}]
    content = []
    if include_images:
        content.append(image_url_content(fixture.image_path))
    content.append({"type": "text", "text": fixture.prompt})
    return [{"role": "user", "content": content}]


def request_body(
    fixture: Fixture,
    *,
    model: str,
    include_images: bool = True,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    return_token_ids: bool = True,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_messages(fixture, include_images=include_images),
        "temperature": 0.0,
        "max_tokens": int(max_tokens if max_tokens is not None else fixture.max_tokens),
        "stream": bool(stream),
        "return_token_ids": bool(return_token_ids),
        "request_id": request_id or fixture.request_id,
    }


def build_manifest_document(
    *,
    requests: list[dict[str, Any]],
    boundary: list[dict[str, Any]],
    diagnostic: list[dict[str, Any]],
    source_notes: dict[str, Any],
    pixel_bounds: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": FIXTURE_SCHEMA,
        "created": "2026-10-17",
        "frozen": True,
        "source_notes": source_notes,
        "pixel_bounds": pixel_bounds,
        "counts": {
            "ocr": NATURAL_PER_CATEGORY,
            "chart_doc": NATURAL_PER_CATEGORY,
            "vqa": NATURAL_PER_CATEGORY,
            "caption": NATURAL_PER_CATEGORY,
            "diagnostic": DIAGNOSTIC_COUNT,
            "boundary": BOUNDARY_COUNT,
        },
        "requests": requests,
        "boundary": boundary,
        "diagnostic": diagnostic,
    }


def write_manifest(path: str | Path, document: dict[str, Any]) -> None:
    write_json_atomic(path, document)
