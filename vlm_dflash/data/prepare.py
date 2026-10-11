"""Build the training/validation ShareGPT dataset from a public source.

v0.1 sources:

* ``llava_instruct`` — ``liuhaotian/LLaVA-Instruct-150K`` style JSON with
  ``{"id", "image", "conversations"}`` rows and a COCO image directory. The
  first human/assistant turn is kept (single-turn MVP) and answers shorter than
  ``min_answer_chars`` are filtered out to form the long-answer training set.
* ``coco_caption`` — COCO ``captions_train2017.json`` + images; short caption
  answers, used for pipeline smoke runs only.
* ``jsonl`` — any JSONL of multimodal rows (messages / ShareGPT / flat
  question-answer-image shapes; see :mod:`vlm_dflash.data.schema`), with
  optional ``dataset.jsonl_images_dir`` for relative image paths.
* ``hf`` — a HuggingFace ``datasets`` source exposing question/answer/image
  columns or a messages column (see ``dataset.hf`` config keys).

Outputs (under ``<run>/data``): ``data.jsonl``, ``val.jsonl``, copied images,
``meta.json`` (media_dir) and ``manifest.json`` with per-row image hashes and a
``rows_sha256`` reproducibility digest.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..utils import (
    ensure_dir,
    log,
    now_iso,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)


def _iter_llava_instruct(cfg: dict) -> list[dict]:
    dataset = cfg["dataset"]
    annotation = Path(dataset["annotation"])
    images_dir = Path(dataset["images_dir"])
    raw = read_json(annotation)
    min_chars = int(dataset.get("min_answer_chars", 0))
    max_chars = dataset.get("max_answer_chars")
    pool = []
    for src_index, row in enumerate(raw):
        conversations = row.get("conversations") or []
        if len(conversations) < 2:
            continue
        question, answer = conversations[0].get("value", ""), conversations[1].get("value", "")
        if "<image>" not in question:
            continue
        image_name = row.get("image")
        if not isinstance(image_name, str) or not image_name:
            continue
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        if len(answer) < min_chars or (max_chars is not None and len(answer) > int(max_chars)):
            continue
        pool.append(
            {
                "src_index": src_index,
                "sample_id": str(row.get("id", f"{src_index:07d}")),
                "question": question,
                "answer": answer,
                "image_path": image_path,
            }
        )
    return pool


def _iter_coco_caption(cfg: dict) -> list[dict]:
    dataset = cfg["dataset"]
    annotation = Path(dataset["annotation"])  # captions_train2017.json
    images_dir = Path(dataset["images_dir"])
    raw = read_json(annotation)
    images = {int(img["id"]): img["file_name"] for img in raw["images"]}
    question = dataset.get("smoke_question", "Please describe this image in detail.")
    pool = []
    for annotation_entry in sorted(raw["annotations"], key=lambda a: (int(a["image_id"]), int(a["id"]))):
        image_name = images.get(int(annotation_entry["image_id"]))
        if image_name is None:
            continue
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        pool.append(
            {
                "src_index": int(annotation_entry["id"]),
                "sample_id": f"coco_{int(annotation_entry['image_id']):012d}_{int(annotation_entry['id'])}",
                "question": question,
                "answer": str(annotation_entry["caption"]).strip(),
                "image_path": image_path,
            }
        )
    return pool


def _iter_jsonl(cfg: dict) -> list[dict]:
    from .schema import iter_jsonl

    dataset = cfg["dataset"]
    images_dir = dataset.get("jsonl_images_dir") or dataset.get("images_dir")
    examples = iter_jsonl(dataset["path"], images_dir=images_dir)
    return [
        {
            "src_index": example.src_index,
            "sample_id": example.id,
            "question": example.question,
            "answer": example.answer,
            "image_path": example.image_path,
        }
        for example in examples
    ]


def _iter_hf(cfg: dict) -> list[dict]:
    import datasets  # imported lazily; only the ``hf`` source needs it

    dataset = cfg["dataset"]
    hf = dataset["hf"]
    ds = datasets.load_dataset(hf["path"], hf.get("config"), split=hf.get("split", "train"))
    image_col = hf.get("image_col", "image")
    min_chars = int(dataset.get("min_answer_chars", 0))
    image_root = ensure_dir(Path(dataset.get("hf_image_cache", "hf_images")))
    pool = []

    if hf.get("messages_col"):
        from .schema import normalize_row

        messages_col = hf["messages_col"]
        for src_index, row in enumerate(ds):
            image = row[image_col]
            if not hasattr(image, "save"):
                continue
            image_path = image_root / f"hf_{src_index:07d}.jpg"
            image.convert("RGB").save(image_path)
            raw = dict(row)
            raw["images"] = [str(image_path)]
            raw.pop(image_col, None)
            example = normalize_row(raw, index=src_index, row_dir=image_root)
            if len(example.answer) < min_chars:
                continue
            pool.append(
                {
                    "src_index": src_index,
                    "sample_id": example.id,
                    "question": example.question,
                    "answer": example.answer,
                    "image_path": example.image_path,
                }
            )
        return pool

    question_col = hf.get("question_col", "question")
    answer_col = hf.get("answer_col", "answer")
    for src_index, row in enumerate(ds):
        question, answer = str(row[question_col]), str(row[answer_col])
        if "<image>" not in question:
            question = "<image>\n" + question
        if len(answer) < min_chars:
            continue
        image = row[image_col]
        image_path = image_root / f"hf_{src_index:07d}.jpg"
        if hasattr(image, "save"):
            image.convert("RGB").save(image_path)
            pool.append(
                {
                    "src_index": src_index,
                    "sample_id": f"hf_{src_index:07d}",
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                }
            )
    return pool


_SOURCES = {
    "llava_instruct": _iter_llava_instruct,
    "coco_caption": _iter_coco_caption,
    "jsonl": _iter_jsonl,
    "hf": _iter_hf,
}


def build_dataset(cfg: dict, data_dir: str | Path, limit: int | None = None) -> dict:
    """Materialise ``data.jsonl``/``val.jsonl``/images/manifest under ``data_dir``."""
    import random

    dataset_cfg = cfg["dataset"]
    source = dataset_cfg["source"]
    if source not in _SOURCES:
        raise ValueError(f"Unknown dataset source {source!r}; expected one of {sorted(_SOURCES)}.")
    data_dir = ensure_dir(data_dir)
    images_out = ensure_dir(data_dir / "images")

    pool = _SOURCES[source](cfg)
    log(f"prepare_data: source={source} pool={len(pool)}")
    seed = int(dataset_cfg.get("seed", 42))
    train_size = int(limit if limit is not None else dataset_cfg["slice"])
    val_size = int(dataset_cfg.get("val_size", 64))
    total = min(train_size + val_size, len(pool))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(pool)), total))
    train_idx = chosen[: min(train_size, total)]
    val_idx = chosen[min(train_size, total) :]

    manifest_rows = []

    def materialise(indices: list[int], jsonl_name: str) -> list[dict]:
        rows_out = []
        for index in indices:
            sample = pool[index]
            src_image = sample["image_path"]
            dst_name = f"{sample['sample_id']}_{src_image.name}"
            dst = images_out / dst_name
            if not dst.exists():
                shutil.copyfile(src_image, dst)
            row = {
                "id": sample["sample_id"],
                "conversations": [
                    {"from": "human", "value": sample["question"]},
                    {"from": "gpt", "value": sample["answer"]},
                ],
                "images": [dst_name],
            }
            rows_out.append(row)
            manifest_rows.append(
                {
                    "id": sample["sample_id"],
                    "split": "train" if jsonl_name == "data.jsonl" else "val",
                    "src_index": sample["src_index"],
                    "image": dst_name,
                    "image_sha256": sha256_file(dst),
                    "question_chars": len(sample["question"]),
                    "answer_chars": len(sample["answer"]),
                }
            )
        return rows_out

    train_rows = materialise(train_idx, "data.jsonl")
    val_rows = materialise(val_idx, "val.jsonl")
    write_jsonl(data_dir / "data.jsonl", train_rows)
    write_jsonl(data_dir / "val.jsonl", val_rows)
    write_json(data_dir / "meta.json", {"media_dir": "images", "source": source})

    manifest = {
        "created": now_iso(),
        "source": source,
        "source_config": {k: v for k, v in dataset_cfg.items() if k not in {"hf"}},
        "seed": seed,
        "slice": train_size,
        "val_size": val_size,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "rows_sha256": sha256_json(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(data_dir / "manifest.json", manifest)
    log(f"prepare_data: train={len(train_rows)} val={len(val_rows)} digest={manifest['rows_sha256'][:12]}")
    return manifest
