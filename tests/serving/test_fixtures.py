from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_dflash.serving.contracts import sha256_file, sha256_text
from vlm_dflash.serving.fixtures import (
    FixtureError,
    build_manifest_document,
    load_manifest,
    natural_fixtures,
    boundary_fixtures,
)


def _image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (120, 30, 200)).save(path, format="JPEG")
    return path


def _entry(root: Path, request_id: str, category: str, prompt: str, *, variant=None, expect="success", image=True):
    entry = {
        "request_id": request_id,
        "category": category,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "max_tokens": 16,
        "expect": expect,
    }
    if variant:
        entry["variant"] = variant
    if image:
        path = _image(root / "images" / f"{request_id}.jpg")
        entry["image"] = str(path.relative_to(root))
        entry["image_sha256"] = sha256_file(path)
    return entry


def _manifest(tmp_path: Path):
    natural = []
    for category in ("ocr", "chart_doc", "vqa", "caption"):
        for index in range(8):
            natural.append(_entry(tmp_path, f"{category}-{index}", category, f"Question for {category} {index}"))
    boundary = [
        _entry(tmp_path, "b-two-turn", "boundary", "Describe the image.", variant="two_turn"),
        _entry(tmp_path, "b-resolution", "boundary", "Describe the image.", variant="different_resolution"),
        _entry(tmp_path, "b-empty", "boundary", "Describe the image.", variant="empty_text"),
        _entry(tmp_path, "b-no-image", "boundary", "Say hello.", variant="no_image", image=False),
        _entry(tmp_path, "b-invalid", "boundary", "Describe the image.", variant="invalid_media", expect="error", image=False),
        _entry(tmp_path, "b-long", "boundary", "x " * 9000, variant="long_prompt"),
        _entry(tmp_path, "b-bad", "boundary", "Describe the image.", variant="bad_structure"),
        _entry(tmp_path, "b-two-images", "boundary", "Describe the image.", variant="two_images"),
    ]
    diagnostic = []
    for index in range(8):
        diagnostic.append(
            _entry(
                tmp_path,
                f"diag-{index}",
                "diagnostic",
                f"Ignore the image. Continue exactly: alpha-{index} beta-{index} gamma-{index}. alpha-{index} beta-{index}",
            )
        )
    doc = build_manifest_document(
        requests=natural,
        boundary=boundary,
        diagnostic=diagnostic,
        source_notes={"test": True},
        pixel_bounds={"min_pixels": 200704, "max_pixels": 1003520},
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_manifest_counts_and_load(tmp_path):
    path = _manifest(tmp_path)
    manifest = load_manifest(path)
    assert len(natural_fixtures(manifest)) == 32
    assert len(boundary_fixtures(manifest)) == 8
    assert {fixture.category for fixture in natural_fixtures(manifest)} == {"ocr", "chart_doc", "vqa", "caption"}


def test_tampered_image_hash_is_rejected(tmp_path):
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["requests"][0]["image_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureError, match="image sha256 mismatch"):
        load_manifest(path)


def test_missing_category_count_is_rejected(tmp_path):
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["requests"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FixtureError, match="expected 8"):
        load_manifest(path)
