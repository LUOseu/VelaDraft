"""dataset.split_field: a curated pool's own train/val/test membership."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vlm_dflash.data.prepare import _predefined_split_labels, build_dataset
from vlm_dflash.utils import read_jsonl, write_jsonl


def _pool(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """``rows`` = (id, split, image name); images are colored by name."""
    colors = {name: index for index, name in enumerate(dict.fromkeys(image for _, _, image in rows))}
    (tmp_path / "media").mkdir(parents=True, exist_ok=True)
    records = []
    for row_id, split, image in rows:
        image_path = tmp_path / "media" / f"{image}.png"
        if not image_path.exists():
            Image.new("RGB", (32, 32), color=(colors[image] * 9 % 256, 7, 9)).save(image_path)
        record = {"id": row_id, "image": str(image_path), "question": f"q {row_id}", "answer": "a"}
        if split is not None:
            record["split"] = split
        records.append(record)
    write_jsonl(tmp_path / "pool.jsonl", records)
    return tmp_path / "pool.jsonl"


def _cfg(path: Path, **dataset) -> dict:
    return {"dataset": {"source": "jsonl", "path": str(path), "split_field": "split", "seed": 1, **dataset}}


def _ids(path: Path) -> list[str]:
    return [row["id"] for row in read_jsonl(path)]


def test_predefined_splits_keep_membership_and_file_order(tmp_path):
    rows = [("t2", "train", "a"), ("d1", "dev", "b"), ("t1", "train", "c"), ("x1", "test", "d"),
            ("t3", "train", "a"), ("v2", "validation", "e")]
    manifest = build_dataset(_cfg(_pool(tmp_path, rows)), tmp_path / "out")
    assert _ids(tmp_path / "out" / "data.jsonl") == ["t2", "t1", "t3"]
    assert _ids(tmp_path / "out" / "val.jsonl") == ["d1", "v2"]
    assert _ids(tmp_path / "out" / "test.jsonl") == ["x1"]
    assert (manifest["num_train"], manifest["num_val"], manifest["num_test"]) == (3, 2, 1)
    assert manifest["image_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_slice_and_limit_cap_the_predefined_train_split(tmp_path):
    rows = [(f"t{i}", "train", f"img{i}") for i in range(5)] + [("d0", "dev", "dev")]
    pool = _pool(tmp_path, rows)
    assert build_dataset(_cfg(pool, slice=2), tmp_path / "a")["num_train"] == 2
    assert build_dataset(_cfg(pool), tmp_path / "b", limit=3)["num_train"] == 3
    assert _ids(tmp_path / "b" / "data.jsonl") == ["t0", "t1", "t2"]


def test_image_shared_across_splits_is_rejected(tmp_path):
    rows = [("t1", "train", "same"), ("d1", "dev", "same")]
    with pytest.raises(ValueError, match="span more than one split"):
        build_dataset(_cfg(_pool(tmp_path, rows)), tmp_path / "out")


@pytest.mark.parametrize("label", ["holdout", None])
def test_unknown_or_missing_split_label_is_rejected(tmp_path, label):
    rows = [("t1", "train", "a"), ("z1", label, "b")]
    with pytest.raises(ValueError, match="expected one of"):
        build_dataset(_cfg(_pool(tmp_path, rows)), tmp_path / "out")


def test_labels_helper_is_inactive_without_split_field():
    assert _predefined_split_labels({"source": "jsonl"}, [{"src_index": 0}]) is None


def test_labels_tolerate_whitespace_and_case(tmp_path):
    rows = [("t1", " Train ", "a"), ("d1", "DEV\n", "b")]
    manifest = build_dataset(_cfg(_pool(tmp_path, rows)), tmp_path / "out")
    assert (manifest["num_train"], manifest["num_val"]) == (1, 1)
    assert manifest["split_mode"] == "predefined by 'split', file order kept"


def test_media_check_covers_only_kept_rows(tmp_path, capsys):
    rows = [("t0", "train", "a"), ("t1", "train", "shared"), ("d1", "dev", "shared")]
    pool = _pool(tmp_path, rows)
    # t1 shares an image with d1, but slice=1 drops it before the check.
    manifest = build_dataset(_cfg(pool, slice=1), tmp_path / "ok")
    assert manifest["num_train"] == 1
    assert "train split capped to 1 of 2 rows" in capsys.readouterr().out
    with pytest.raises(ValueError, match="span more than one split"):
        build_dataset(_cfg(pool), tmp_path / "bad")


def test_missing_val_rows_are_reported(tmp_path, capsys):
    build_dataset(_cfg(_pool(tmp_path, [("t1", "train", "a"), ("x1", "test", "b")])), tmp_path / "out")
    assert "has no val rows" in capsys.readouterr().out


def test_non_jsonl_source_fails_before_loading_the_pool(monkeypatch, tmp_path):
    from vlm_dflash.data import prepare

    def must_not_load(cfg):
        raise AssertionError("the pool was loaded before the split_field check")

    monkeypatch.setitem(prepare._SOURCES, "coco_caption", must_not_load)
    with pytest.raises(ValueError, match="only supported for dataset.source=jsonl"):
        build_dataset({"dataset": {"source": "coco_caption", "split_field": "split", "slice": 1}}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_random_split_still_fills_test_then_val_then_train(tmp_path):
    rows = [(f"r{i}", None, f"img{i}") for i in range(6)]
    cfg = {"dataset": {"source": "jsonl", "path": str(_pool(tmp_path, rows)), "seed": 3, "slice": 3,
                       "val_size": 1, "test_size": 2}}
    manifest = build_dataset(cfg, tmp_path / "out")
    assert (manifest["num_train"], manifest["num_val"], manifest["num_test"]) == (3, 1, 2)
    assert manifest["split_mode"] == "image-grouped split, seed=3"
    ids = _ids(tmp_path / "out" / "data.jsonl") + _ids(tmp_path / "out" / "val.jsonl") + _ids(tmp_path / "out" / "test.jsonl")
    assert len(set(ids)) == 6
