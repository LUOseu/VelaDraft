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

import os
import re
import shutil
from collections import Counter
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


#: Accepted ``dataset.split_field`` values (and their aliases).
_SPLIT_ALIASES = {"train": "train", "val": "val", "dev": "val", "validation": "val", "test": "test"}


def _predefined_split_labels(dataset_cfg: dict, pool: list[dict]) -> list[str] | None:
    """Per-pool-row split (``train``/``val``/``test``) from ``dataset.split_field``, or ``None``.

    Lets a curated pool freeze its own membership (e.g. stratified dev/test
    frozen before the training corpus) instead of the seeded random split.
    """
    field = dataset_cfg.get("split_field")
    if not field:
        return None
    labels = []
    for sample in pool:
        name = _SPLIT_ALIASES.get(str(sample.get("split")).strip().lower())
        if name is None:
            raise ValueError(
                f"dataset.split_field={field!r}: {dataset_cfg['path']} row {sample['src_index']} has split "
                f"{sample.get('split')!r}; expected one of {sorted(_SPLIT_ALIASES)}."
            )
        labels.append(name)
    return labels


def _iter_llava_instruct(cfg: dict) -> list[dict]:
    from .schema import check_v1_question_layout

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
        # LLaVA-150k historically puts the placeholder last ("...\n<image>";
        # about half of the file) and the v1 renderer strips it and renders the
        # image first either way - the layout every recorded 2K llava run and
        # capture store was built with. Accept that trailing form; genuinely
        # interleaved rows (text after the placeholder) stay rejected.
        check_v1_question_layout(
            question, context=f"llava_instruct row {row.get('id', src_index)!r}", allow_trailing=True
        )
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
    examples = iter_jsonl(dataset["path"], images_dir=images_dir, split_field=dataset.get("split_field"))
    pool = []
    for example in examples:
        entry = {
            "src_index": example.src_index,
            "sample_id": example.id,
            "question": example.question,
            "answer": example.answer,
            "image_path": example.image_path,
        }
        if dataset.get("split_field"):
            entry["split"] = example.split
        if example.is_conversation:
            entry["conversations"] = example.conversations
            entry["image_paths"] = example.image_paths
            entry["image_turns"] = example.image_turns
        pool.append(entry)
    return pool


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
            entry = {
                "src_index": src_index,
                "sample_id": example.id,
                "question": example.question,
                "answer": example.answer,
                "image_path": example.image_path,
            }
            if example.is_conversation:
                entry["conversations"] = example.conversations
                entry["image_paths"] = example.image_paths
                entry["image_turns"] = example.image_turns
            pool.append(entry)
        return pool

    question_col = hf.get("question_col", "question")
    answer_col = hf.get("answer_col", "answer")
    from .schema import check_v1_question_layout

    for src_index, row in enumerate(ds):
        question, answer = str(row[question_col]), str(row[answer_col])
        if "<image>" not in question:
            question = "<image>\n" + question
        else:
            check_v1_question_layout(question, context=f"hf row {src_index}")
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


def _safe_sample_component(sample_id: str) -> str:
    """A filesystem-safe stem for the sample id (path separators etc. removed)."""
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(sample_id)).strip("._")
    return cleaned or "sample"


LINK_MODES = ("copy", "hardlink", "symlink")


def _materialise_media(
    src_image: Path, *, sample_id: str, images_out: Path, link_mode: str = "copy"
) -> tuple[str, str]:
    """Materialise one media file under a collision-free, content-verified name.

    The output name embeds the sample id and the first 16 hex digits of the
    file's sha256, so two different source files can never share an output name
    (the old ``{id}_{basename}`` scheme silently merged ``a/same.png`` with
    ``b/same.png``). Reusing an existing file is only allowed after verifying
    its bytes hash to the same content; an intentional duplicate reference
    (same bytes) therefore shares one file, while a tampered/foreign file at
    that name is a loud error.

    ``link_mode`` controls how the bytes reach the data directory:

    * ``copy`` (default): byte copy via :func:`shutil.copyfile`;
    * ``hardlink``: ``os.link`` to the source inode (zero extra disk on the
      same filesystem), falling back to a copy when the link syscall is not
      permitted (cross-device, no-hardlink filesystems);
    * ``symlink``: a relative-free symlink to the resolved source path.

    Hardlinked/symlinked outputs keep the same content hash as the source, so
    the manifest traceability and the read-only media audit are unchanged.

    Returns ``(dst_name, content_sha256)``.
    """
    if link_mode not in LINK_MODES:
        raise ValueError(f"image_link_mode must be one of {LINK_MODES}, got {link_mode!r}.")
    content_sha256 = sha256_file(src_image)
    stem = _safe_sample_component(sample_id)
    dst_name = f"{stem}_{content_sha256[:16]}{src_image.suffix.lower()}"
    dst = images_out / dst_name
    if dst.exists() or dst.is_symlink():
        if sha256_file(dst) != content_sha256:
            raise RuntimeError(
                f"prepare_data: existing output image {dst_name!r} does not match source "
                f"{str(src_image)!r} (content hash mismatch). Refusing to reuse it; remove the "
                "stale file or rebuild into a fresh data directory."
            )
    elif link_mode == "hardlink":
        try:
            os.link(src_image, dst)
        except OSError:
            shutil.copyfile(src_image, dst)
    elif link_mode == "symlink":
        os.symlink(src_image.resolve(), dst)
    else:
        shutil.copyfile(src_image, dst)
    return dst_name, content_sha256


def _random_group_split(
    groups: dict[int, list[int]], seed: int, train_size: int, val_size: int, test_size: int
) -> tuple[list[int], list[int], list[int]]:
    """Seeded image-grouped split: whole groups fill test, then val, then train."""
    import random

    rng = random.Random(seed)
    group_order = sorted(groups)
    rng.shuffle(group_order)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for root in group_order:
        members = groups[root]
        if len(test_idx) < test_size:
            test_idx.extend(members)
        elif len(val_idx) < val_size:
            val_idx.extend(members)
        else:
            train_idx.extend(members)
            if len(train_idx) >= train_size:
                break
    # Shuffle each split once more so file order is a seeded random order (the
    # evaluator's periodic subset is a fixed prefix of val.jsonl and must not
    # inherit the source annotation's block structure).
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _predefined_split(
    groups: dict[int, list[int]], labels: list[str], cap: int | None
) -> tuple[list[int], list[int], list[int]]:
    """Rows' own splits (file order); ``cap`` limits train; the kept rows must be media-disjoint."""
    train_idx = [index for index, name in enumerate(labels) if name == "train"]
    if cap is not None and len(train_idx) > int(cap):
        log(f"prepare_data: train split capped to {int(cap)} of {len(train_idx)} rows by dataset.slice/--limit")
        train_idx = train_idx[: int(cap)]
    val_idx = [index for index, name in enumerate(labels) if name == "val"]
    test_idx = [index for index, name in enumerate(labels) if name == "test"]
    if not val_idx:
        log("prepare_data: warning: the predefined split has no val rows; eval reads val.jsonl.")
    kept = set(train_idx) | set(val_idx) | set(test_idx)
    spanning = [
        root for root, members in groups.items() if len({labels[m] for m in members if m in kept}) > 1
    ]
    if spanning:
        raise ValueError(
            f"dataset.split_field: {len(spanning)} image-content group(s) span more than one split "
            f"(first pool rows: {[groups[root][0] for root in spanning[:5]]}); splits must be media-disjoint."
        )
    return train_idx, val_idx, test_idx


def build_dataset(cfg: dict, data_dir: str | Path, limit: int | None = None) -> dict:
    """Materialise ``data.jsonl``/``val.jsonl``/images/manifest under ``data_dir``."""
    dataset_cfg = cfg["dataset"]
    source = dataset_cfg["source"]
    if source not in _SOURCES:
        raise ValueError(f"Unknown dataset source {source!r}; expected one of {sorted(_SOURCES)}.")
    if dataset_cfg.get("split_field") and source != "jsonl":
        raise ValueError("dataset.split_field is only supported for dataset.source=jsonl.")
    data_dir = ensure_dir(data_dir)
    images_out = ensure_dir(data_dir / "images")

    pool = _SOURCES[source](cfg)
    log(f"prepare_data: source={source} pool={len(pool)}")
    seed = int(dataset_cfg.get("seed", 42))
    labels = _predefined_split_labels(dataset_cfg, pool)
    if labels is None:
        train_size = int(limit if limit is not None else dataset_cfg["slice"])
        val_size = int(dataset_cfg.get("val_size", 64))
        test_size = int(dataset_cfg.get("test_size", 0))
    link_mode = str(dataset_cfg.get("image_link_mode", "copy")).lower()
    if link_mode not in LINK_MODES:
        raise ValueError(f"dataset.image_link_mode must be one of {LINK_MODES}, got {link_mode!r}.")

    # ---- image-content grouping -------------------------------------------
    # Rows that reference the same image bytes (across different questions or
    # layouts) form one group; whole groups are assigned to exactly one split
    # (or, with ``dataset.split_field``, verified to sit in one), so
    # train/val/test are disjoint by image *content*, not by row index.
    media_signatures: list[tuple[str, ...]] = []
    hash_cache: dict[Path, str] = {}

    def content_hash(path: Path) -> str:
        resolved = Path(path)
        cached = hash_cache.get(resolved)
        if cached is None:
            cached = sha256_file(resolved)
            hash_cache[resolved] = cached
        return cached

    for sample in pool:
        paths = sample.get("image_paths")
        if paths is None:
            paths = [sample["image_path"]]
        media_signatures.append(tuple(content_hash(path) for path in paths))

    parent = list(range(len(pool)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        # Keep the smallest pool index as the root so group order is a pure
        # function of the pool (deterministic before the shuffle).
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    hash_owner: dict[str, int] = {}
    for index, signatures in enumerate(media_signatures):
        for content_hash in signatures:
            owner = hash_owner.setdefault(content_hash, index)
            if owner != index:
                union(owner, index)

    groups: dict[int, list[int]] = {}
    for index in range(len(pool)):
        groups.setdefault(find(index), []).append(index)

    if labels is None:
        train_idx, val_idx, test_idx = _random_group_split(groups, seed, train_size, val_size, test_size)
        split_mode = f"image-grouped split, seed={seed}"
    else:
        cap = limit if limit is not None else dataset_cfg.get("slice")
        train_idx, val_idx, test_idx = _predefined_split(groups, labels, cap)
        train_size, val_size, test_size = len(train_idx), len(val_idx), len(test_idx)
        split_mode = f"predefined by '{dataset_cfg['split_field']}', file order kept"
    log(
        f"prepare_data: groups={len(groups)} train={len(train_idx)} val={len(val_idx)} "
        f"test={len(test_idx)} ({split_mode})"
    )

    manifest_rows = []

    split_of = {"data.jsonl": "train", "val.jsonl": "val", "test.jsonl": "test"}

    def materialise(indices: list[int], jsonl_name: str) -> list[dict]:
        rows_out = []
        for index in indices:
            sample = pool[index]
            if sample.get("conversations") is not None:
                dst_names = []
                content_hashes = []
                for src_image in sample["image_paths"]:
                    dst_name, content_hash = _materialise_media(
                        src_image,
                        sample_id=sample["sample_id"],
                        images_out=images_out,
                        link_mode=link_mode,
                    )
                    dst_names.append(dst_name)
                    content_hashes.append(content_hash)
                rows_out.append(
                    {
                        "id": sample["sample_id"],
                        "conversations": sample["conversations"],
                        "images": dst_names,
                        "image_turns": list(sample["image_turns"]),
                        "schema": "vlm-dflash-data-v2",
                    }
                )
                manifest_rows.append(
                    {
                        "id": sample["sample_id"],
                        "split": split_of[jsonl_name],
                        "src_index": sample["src_index"],
                        "image": dst_names[0],
                        "images": dst_names,
                        "image_sha256_list": content_hashes,
                        "image_sources": [str(path) for path in sample["image_paths"]],
                        "question_chars": sum(
                            len(turn["value"]) for turn in sample["conversations"] if turn["from"] == "human"
                        ),
                        "answer_chars": len(sample["answer"]),
                    }
                )
                continue
            src_image = sample["image_path"]
            dst_name, content_hash = _materialise_media(
                src_image, sample_id=sample["sample_id"], images_out=images_out, link_mode=link_mode
            )
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
                    "split": split_of[jsonl_name],
                    "src_index": sample["src_index"],
                    "image": dst_name,
                    "image_sha256": content_hash,
                    "image_source": str(src_image),
                    "question_chars": len(sample["question"]),
                    "answer_chars": len(sample["answer"]),
                }
            )
        return rows_out

    id_counts = Counter(pool[index]["sample_id"] for index in train_idx + val_idx + test_idx)
    duplicates = sorted(sample_id for sample_id, count in id_counts.items() if count > 1)
    if duplicates:
        log(
            "prepare_data: warning: duplicate sample id(s) in the selected rows "
            f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}; rows stay distinct in the "
            "manifest (by src_index) and identical media bytes dedupe to one verified file."
        )
    train_rows = materialise(train_idx, "data.jsonl")
    val_rows = materialise(val_idx, "val.jsonl")
    if val_size > 0 and not val_rows:
        log(
            "prepare_data: warning: the val split is empty - the pool has "
            f"{len(pool)} row(s) and the training slice/limit ({train_size}) already "
            "consumes all of them. Eval reads val.jsonl, so add rows or lower the "
            "training slice (train + val_size rows are needed)."
        )
    write_jsonl(data_dir / "data.jsonl", train_rows)
    write_jsonl(data_dir / "val.jsonl", val_rows)
    if test_size:
        test_rows = materialise(test_idx, "test.jsonl")
        write_jsonl(data_dir / "test.jsonl", test_rows)
    else:
        test_rows = []
    write_json(data_dir / "meta.json", {"media_dir": "images", "source": source})

    def image_hashes(indices: list[int]) -> set[str]:
        return {content for index in indices for content in media_signatures[index]}

    split_images = {
        "train": image_hashes(train_idx),
        "val": image_hashes(val_idx),
        "test": image_hashes(test_idx),
    }
    manifest = {
        "created": now_iso(),
        "source": source,
        "source_config": {k: v for k, v in dataset_cfg.items() if k not in {"hf"}},
        "seed": seed,
        "split_mode": split_mode,
        "slice": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "image_link_mode": link_mode,
        "num_pool": len(pool),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "num_groups": len(groups),
        "num_images": {name: len(hashes) for name, hashes in split_images.items()},
        "image_overlap": {
            "train_val": len(split_images["train"] & split_images["val"]),
            "train_test": len(split_images["train"] & split_images["test"]),
            "val_test": len(split_images["val"] & split_images["test"]),
        },
        "rows_sha256": sha256_json(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(data_dir / "manifest.json", manifest)
    log(
        f"prepare_data: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"images={{'train': {len(split_images['train'])}, 'val': {len(split_images['val'])}, "
        f"'test': {len(split_images['test'])}}} overlap={{train/val: "
        f"{manifest['image_overlap']['train_val']}, train/test: "
        f"{manifest['image_overlap']['train_test']}, val/test: "
        f"{manifest['image_overlap']['val_test']}}} digest={manifest['rows_sha256'][:12]}"
    )
    return manifest
