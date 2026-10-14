"""R1: collision-safe media materialisation and manifest traceability.

Regression tests for the review finding that ``build_dataset`` named output
images ``{sample_id}_{source basename}``: two different source files with the
same basename (from different directories) silently mapped to one output file
and the second copy was skipped, so both manifest references decoded the first
image's pixels. Also pins the read-only audit script used to check data that
was prepared before the fix.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.data.prepare import build_dataset
from vlm_dflash.utils import read_json, read_jsonl


def _save(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_cfg(path: Path, images_dir: Path, *, slice_size: int, val_size: int = 0) -> dict:
    return {
        "dataset": {
            "source": "jsonl",
            "path": str(path),
            "images_dir": str(images_dir),
            "slice": slice_size,
            "val_size": val_size,
            "seed": 42,
        }
    }


def _multi_image_row(row_id: str, images: list[str], image_turns: list[int]) -> dict:
    conversations = []
    for turn in range(0, max(image_turns) + 1, 2):
        attached = image_turns.count(turn)
        prefix = "<image>\n" * attached
        conversations.append({"from": "human", "value": f"{prefix}question {turn}"})
        conversations.append({"from": "gpt", "value": f"answer {turn}"})
    return {
        "id": row_id,
        "conversations": conversations,
        "images": images,
        "image_turns": image_turns,
        "schema": "vlm-dflash-data-v2",
    }


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _run_audit(data_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_prepared_media.py"
    return subprocess.run(
        [sys.executable, str(script), "--data-dir", str(data_dir), *extra],
        capture_output=True,
        text=True,
    )


def test_same_basename_different_content_is_not_merged(tmp_path):
    """a/same.png (red) + b/same.png (blue) in one row must stay distinct."""
    _save(tmp_path / "a" / "same.png", (255, 0, 0))
    _save(tmp_path / "b" / "same.png", (0, 0, 255))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [_multi_image_row("collision", ["a/same.png", "b/same.png"], [0, 2])],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=1)

    manifest = build_dataset(cfg, tmp_path / "out")
    row = read_jsonl(tmp_path / "out" / "data.jsonl")[0]
    names = row["images"]
    assert len(set(names)) == 2, f"two distinct source images collapsed onto {names}"

    out_media = tmp_path / "out" / "images"
    pixels = [Image.open(out_media / name).convert("RGB").getpixel((0, 0)) for name in names]
    assert pixels[0] == (255, 0, 0) and pixels[1] == (0, 0, 255), "image content was swapped/lost"

    entry = manifest["rows"][0]
    assert entry["images"] == names
    assert entry["image_sha256_list"] == [_sha256(out_media / name) for name in names]
    # The manifest traces each materialised file back to its original reference.
    assert entry["image_sources"] == [
        str((tmp_path / "a" / "same.png").resolve()),
        str((tmp_path / "b" / "same.png").resolve()),
    ]


def test_intentional_duplicate_reference_reuses_one_verified_file(tmp_path):
    """One image referenced twice is legal and dedupes to a single output file."""
    _save(tmp_path / "a" / "same.png", (10, 20, 30))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [_multi_image_row("dup", ["a/same.png", "a/same.png"], [0, 2])],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=1)

    manifest = build_dataset(cfg, tmp_path / "out")
    row = read_jsonl(tmp_path / "out" / "data.jsonl")[0]
    assert row["images"][0] == row["images"][1]
    assert len(list((tmp_path / "out" / "images").iterdir())) == 1
    entry = manifest["rows"][0]
    assert entry["image_sha256_list"] == [_sha256(tmp_path / "out" / "images" / row["images"][0])] * 2
    assert entry["image_sources"] == [
        str((tmp_path / "a" / "same.png").resolve()),
        str((tmp_path / "a" / "same.png").resolve()),
    ]


def test_rerun_prepare_is_idempotent_and_refuses_tampered_output(tmp_path):
    _save(tmp_path / "a" / "img.png", (1, 2, 3))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [
            {
                "id": "v1row",
                "image": "a/img.png",
                "conversations": [
                    {"from": "human", "value": "<image>\nq"},
                    {"from": "gpt", "value": "a"},
                ],
            }
        ],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=1)
    first = build_dataset(cfg, tmp_path / "out")
    names_first = sorted(p.name for p in (tmp_path / "out" / "images").iterdir())
    second = build_dataset(cfg, tmp_path / "out")
    names_second = sorted(p.name for p in (tmp_path / "out" / "images").iterdir())
    assert names_first == names_second
    assert first["rows"] == second["rows"]

    # A tampered output file must fail loud instead of being silently reused.
    tampered = tmp_path / "out" / "images" / names_first[0]
    tampered.write_bytes(b"not the image the name claims to be")
    with pytest.raises(RuntimeError, match="does not match"):
        build_dataset(cfg, tmp_path / "out")


def test_duplicate_sample_ids_do_not_collide(tmp_path):
    """Duplicate ids with different pixels get distinct, deterministic files."""
    _save(tmp_path / "a" / "one.png", (200, 0, 0))
    _save(tmp_path / "b" / "two.png", (0, 200, 0))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [_multi_image_row("dup-id", ["a/one.png"], [0]), _multi_image_row("dup-id", ["b/two.png"], [0])],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=2)

    manifest = build_dataset(cfg, tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "data.jsonl")
    assert [row["id"] for row in rows] == ["dup-id", "dup-id"]
    first_name, second_name = rows[0]["images"][0], rows[1]["images"][0]
    assert first_name != second_name
    out_media = tmp_path / "out" / "images"
    assert Image.open(out_media / first_name).convert("RGB").getpixel((0, 0)) == (200, 0, 0)
    assert Image.open(out_media / second_name).convert("RGB").getpixel((0, 0)) == (0, 200, 0)
    assert [entry["image_sha256_list"][0] for entry in manifest["rows"]] == [
        _sha256(out_media / first_name),
        _sha256(out_media / second_name),
    ]
    # Duplicate ids are distinguishable by src_index in the manifest.
    assert [entry["src_index"] for entry in manifest["rows"]] == [0, 1]


def test_v1_manifest_traces_its_media_reference(tmp_path):
    _save(tmp_path / "a" / "img.png", (9, 9, 9))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [
            {
                "id": "v1",
                "image": "a/img.png",
                "conversations": [
                    {"from": "human", "value": "<image>\nq"},
                    {"from": "gpt", "value": "a"},
                ],
            }
        ],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=1)
    manifest = build_dataset(cfg, tmp_path / "out")
    entry = manifest["rows"][0]
    assert entry["image_source"] == str((tmp_path / "a" / "img.png").resolve())
    assert entry["image"] == read_jsonl(tmp_path / "out" / "data.jsonl")[0]["images"][0]
    assert entry["image_sha256"] == _sha256(tmp_path / "out" / "images" / entry["image"])


def test_audit_script_flags_corrupt_and_colliding_outputs(tmp_path):
    _save(tmp_path / "a" / "same.png", (255, 0, 0))
    rows_path = _write_rows(
        tmp_path / "rows.jsonl",
        [_multi_image_row("row-a", ["a/same.png"], [0])],
    )
    cfg = _jsonl_cfg(rows_path, tmp_path, slice_size=1)
    build_dataset(cfg, tmp_path / "out")

    clean = _run_audit(tmp_path / "out")
    assert clean.returncode == 0, clean.stderr + clean.stdout

    # Tamper: name still carries the old content hash, bytes changed.
    target = next((tmp_path / "out" / "images").iterdir())
    target.write_bytes(b"tampered")
    dirty = _run_audit(tmp_path / "out")
    assert dirty.returncode == 2
    report = json.loads(dirty.stdout)
    assert report["issues"], "audit must report the tampered file"
    assert any("mismatch" in issue["kind"] or "hash" in issue["kind"] for issue in report["issues"])


def test_audit_script_flags_legacy_manifest_hash_conflict(tmp_path):
    """Legacy (pre-fix) manifests can still be audited: identical file name
    referenced with two different recorded hashes is a detected collision."""
    media = tmp_path / "out" / "images"
    media.mkdir(parents=True)
    (media / "x_same.png").write_bytes(b"whatever")
    (tmp_path / "out" / "data.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "rows": [
            {"id": "a", "split": "train", "image": "x_same.png", "image_sha256": "0" * 64},
            {"id": "b", "split": "train", "image": "x_same.png", "image_sha256": "1" * 64},
        ]
    }
    (tmp_path / "out" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = _run_audit(tmp_path / "out")
    assert result.returncode == 2
    report = json.loads(result.stdout)
    kinds = {issue["kind"] for issue in report["issues"]}
    assert "reference_hash_conflict" in kinds or "manifest_hash_mismatch" in kinds
