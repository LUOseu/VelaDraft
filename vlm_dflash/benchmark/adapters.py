"""Benchmark dataset adapters: public workload -> pinned sharegpt_jsonl export.

The adapters are deterministic: fixed split, fixed row order (sorted by the
dataset's own id), fixed subset size, fixed seed and fixed image bytes. Each
export writes the images plus a ``manifest.json`` recording the dataset id,
split, per-shard hashes, selected ids and per-row question/image hashes, so a
later run can be checked against the pinned protocol instead of trusting it.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..utils import ensure_dir, log, now_iso, sha256_file, sha256_json, write_json, write_jsonl

TEXTVQA_DATASET = "lmms-lab/textvqa"
TEXTVQA_SPLIT = "validation"
DOCVQA_DATASET = "lmms-lab/DocVQA"
DOCVQA_CONFIG = "DocVQA"
DOCVQA_SPLIT = "validation"
CHARTQA_DATASET = "lmms-lab/ChartQA"
CHARTQA_SPLIT = "test"

_PINNED_KEYS = (
    "id",
    "question_id",
    "source_index",
    "doc_id",
    "ucsf_document_id",
    "ucsf_document_page_no",
    "image_id",
    "image_path",
    "question_sha256",
    "image_sha256",
)


def _image_suffix(image_bytes: bytes) -> str:
    """File suffix for embedded image bytes (PNG / JPEG magic; fail loud)."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    raise ValueError("Unrecognised embedded image bytes: expected PNG or JPEG magic.")


def _write_image(path: Path, image_bytes: bytes) -> None:
    if not path.exists() or path.read_bytes() != image_bytes:
        path.write_bytes(image_bytes)


def select_indices(num_rows: int, num_prompts: int, seed: int) -> list[int]:
    """Deterministic subset: seeded sample over sorted rows, returned in row order."""
    if num_prompts > num_rows:
        raise ValueError(f"asked for {num_prompts} prompts but the split has {num_rows} rows.")
    chosen = sorted(random.Random(seed).sample(range(num_rows), num_prompts))
    return chosen


def export_textvqa(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    num_prompts: int = 128,
    seed: int = 42,
) -> dict:
    """Export the TextVQA validation split into the pinned benchmark format.

    ``source_dir`` is a HuggingFace ``datasets``-style download of
    ``lmms-lab/textvqa`` (``data/validation-*.parquet`` with embedded image
    bytes). Returns the written manifest.
    """
    import pyarrow.parquet as pq

    source_dir = Path(source_dir)
    shards = sorted((source_dir / "data").glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No validation-*.parquet under {source_dir}/data; point --source at an "
            f"lmms-lab/textvqa download."
        )
    rows = []
    shard_hashes = {}
    for shard in shards:
        shard_rows = pq.read_table(shard).to_pylist()
        rows.extend(shard_rows)
        shard_hashes[shard.name] = sha256_file(shard)
    rows.sort(key=lambda row: (int(row["question_id"]), str(row["image_id"])))
    indices = select_indices(len(rows), num_prompts, seed)

    out_dir = ensure_dir(Path(out_dir))
    images_dir = ensure_dir(out_dir / "images")
    prompt_rows, manifest_rows = [], []
    for index in indices:
        row = rows[index]
        question_id = int(row["question_id"])
        sample_id = f"textvqa-{question_id:06d}"
        image_name = f"{sample_id}.jpg"
        image_path = images_dir / image_name
        image_bytes = row["image"]["bytes"]
        _write_image(image_path, image_bytes)
        question = str(row["question"]).strip()
        prompt_rows.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": question},
                    # The benchmark feeds only the prompt (question + image) to
                    # the model; answers are intentionally not exported.
                    {"from": "gpt", "value": ""},
                ],
                "images": [image_name],
            }
        )
        manifest_rows.append(
            {
                "id": sample_id,
                "question_id": question_id,
                "image_id": str(row["image_id"]),
                "image": image_name,
                "image_sha256": sha256_file(image_path),
                "question_sha256": sha256_json(question),
            }
        )

    write_jsonl(out_dir / "textvqa_v1.jsonl", prompt_rows)
    write_json(out_dir / "meta.json", {"media_dir": "images", "source": TEXTVQA_DATASET})
    manifest = {
        "created": now_iso(),
        "dataset": TEXTVQA_DATASET,
        "split": TEXTVQA_SPLIT,
        "source_dir": str(source_dir.resolve()),
        "num_rows_in_split": len(rows),
        "num_prompts": num_prompts,
        "seed": seed,
        "subset": f"seeded sample of {num_prompts} (seed={seed}) after sorting by question_id",
        "shards": shard_hashes,
        "rows_sha256": sha256_json(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    log(
        f"benchmark adapter: textvqa -> {out_dir} ({num_prompts} prompts, "
        f"rows_sha256={manifest['rows_sha256'][:12]})"
    )
    return manifest


def export_docvqa(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    num_prompts: int = 128,
    seed: int = 42,
) -> dict:
    """Export the DocVQA validation split into the pinned benchmark format.

    ``source_dir`` is a HuggingFace ``datasets``-style download of
    ``lmms-lab/DocVQA`` (``DocVQA/validation-*.parquet`` with embedded page
    image bytes). Rows are sorted by ``(questionId, docId)`` before the seeded
    subset is drawn; upstream question / UCSF page ids stay in the manifest.
    Answers are intentionally not exported (the benchmark feeds prompts only).
    """
    import pyarrow.parquet as pq

    source_dir = Path(source_dir)
    shards = sorted((source_dir / DOCVQA_CONFIG).glob("validation-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No {DOCVQA_CONFIG}/validation-*.parquet under {source_dir}; point --source at "
            f"an lmms-lab/DocVQA download."
        )
    rows = []
    shard_hashes = {}
    for shard in shards:
        rows.extend(pq.read_table(shard).to_pylist())
        shard_hashes[f"{DOCVQA_CONFIG}/{shard.name}"] = sha256_file(shard)
    rows.sort(key=lambda row: (int(row["questionId"]), int(row["docId"])))
    indices = select_indices(len(rows), num_prompts, seed)

    out_dir = ensure_dir(Path(out_dir))
    images_dir = ensure_dir(out_dir / "images")
    prompt_rows, manifest_rows = [], []
    for index in indices:
        row = rows[index]
        question_id = int(row["questionId"])
        sample_id = f"docvqa-{question_id:06d}"
        image_bytes = row["image"]["bytes"]
        image_name = f"{sample_id}{_image_suffix(image_bytes)}"
        image_path = images_dir / image_name
        _write_image(image_path, image_bytes)
        question = str(row["question"]).strip()
        prompt_rows.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": ""},
                ],
                "images": [image_name],
            }
        )
        manifest_rows.append(
            {
                "id": sample_id,
                "question_id": question_id,
                "doc_id": int(row["docId"]),
                "ucsf_document_id": str(row["ucsf_document_id"]),
                "ucsf_document_page_no": str(row["ucsf_document_page_no"]),
                "image": image_name,
                "image_path": str(row["image"]["path"]),
                "image_sha256": sha256_file(image_path),
                "question_sha256": sha256_json(question),
            }
        )

    write_jsonl(out_dir / "docvqa_v1.jsonl", prompt_rows)
    write_json(out_dir / "meta.json", {"media_dir": "images", "source": DOCVQA_DATASET})
    manifest = {
        "created": now_iso(),
        "dataset": DOCVQA_DATASET,
        "config": DOCVQA_CONFIG,
        "split": DOCVQA_SPLIT,
        "source_dir": str(source_dir.resolve()),
        "num_rows_in_split": len(rows),
        "num_prompts": num_prompts,
        "seed": seed,
        "subset": (
            f"seeded sample of {num_prompts} (seed={seed}) after sorting by (questionId, docId)"
        ),
        "shards": shard_hashes,
        "rows_sha256": sha256_json(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    log(
        f"benchmark adapter: docvqa -> {out_dir} ({num_prompts} prompts, "
        f"rows_sha256={manifest['rows_sha256'][:12]})"
    )
    return manifest


def export_chartqa(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    num_prompts: int = 128,
    seed: int = 42,
) -> dict:
    """Export the ChartQA test split into the pinned benchmark format.

    ``source_dir`` is a HuggingFace ``datasets``-style download of
    ``lmms-lab/ChartQA`` (``data/test-*.parquet`` with embedded chart image
    bytes). Rows are sorted by ``(image path, question)`` before the seeded
    subset is drawn; the upstream ``type`` column (human vs augmented) stays
    in the manifest. Answers are intentionally not exported.
    """
    import pyarrow.parquet as pq

    source_dir = Path(source_dir)
    shards = sorted((source_dir / "data").glob("test-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No data/test-*.parquet under {source_dir}; point --source at an "
            f"lmms-lab/ChartQA download."
        )
    rows = []
    shard_hashes = {}
    for shard in shards:
        rows.extend(pq.read_table(shard).to_pylist())
        shard_hashes[f"data/{shard.name}"] = sha256_file(shard)
    rows.sort(key=lambda row: (str(row["image"]["path"]), str(row["question"])))
    indices = select_indices(len(rows), num_prompts, seed)

    out_dir = ensure_dir(Path(out_dir))
    images_dir = ensure_dir(out_dir / "images")
    prompt_rows, manifest_rows = [], []
    for index in indices:
        row = rows[index]
        sample_id = f"chartqa-{index:06d}"
        image_bytes = row["image"]["bytes"]
        image_name = f"{sample_id}{_image_suffix(image_bytes)}"
        image_path = images_dir / image_name
        _write_image(image_path, image_bytes)
        question = str(row["question"]).strip()
        prompt_rows.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": ""},
                ],
                "images": [image_name],
            }
        )
        manifest_rows.append(
            {
                "id": sample_id,
                "source_index": index,
                "type": str(row.get("type", "")),
                "image": image_name,
                "image_path": str(row["image"]["path"]),
                "image_sha256": sha256_file(image_path),
                "question_sha256": sha256_json(question),
            }
        )

    write_jsonl(out_dir / "chartqa_v1.jsonl", prompt_rows)
    write_json(out_dir / "meta.json", {"media_dir": "images", "source": CHARTQA_DATASET})
    manifest = {
        "created": now_iso(),
        "dataset": CHARTQA_DATASET,
        "split": CHARTQA_SPLIT,
        "source_dir": str(source_dir.resolve()),
        "num_rows_in_split": len(rows),
        "num_prompts": num_prompts,
        "seed": seed,
        "subset": (
            f"seeded sample of {num_prompts} (seed={seed}) after sorting by (image path, question)"
        ),
        "shards": shard_hashes,
        "rows_sha256": sha256_json(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    log(
        f"benchmark adapter: chartqa -> {out_dir} ({num_prompts} prompts, "
        f"rows_sha256={manifest['rows_sha256'][:12]})"
    )
    return manifest


def pinned_prompt_manifest(manifest: dict) -> list[dict]:
    """The repo-tracked prompt set definition (ids + hashes, no image bytes)."""
    return [
        {key: row[key] for key in _PINNED_KEYS if key in row}
        for row in manifest["rows"]
    ]
