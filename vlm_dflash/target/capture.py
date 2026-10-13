"""Offline capture of fused target hidden states for staged DFlash training.

For every regenerated sample this runs one teacher-forced target forward (via
the shared single-sample helper in :mod:`vlm_dflash.target.sample_capture`),
concatenates the contract's routed layers into the draft context feature,
records the MRoPE 3-axis positions, validates the image-token span, and writes
fp/bf16 shards plus an index. Training never needs the target model in memory
afterwards; ``feature_mode=online`` skips this stage and extracts the same
records live instead (see :mod:`vlm_dflash.target.online`).
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..data.sharegpt import load_sharegpt
from ..provenance import identity_from_contract, record_stage
from ..utils import ensure_dir, log, write_json, write_jsonl
from .load import load_target
from .sample_capture import (
    build_sample_record,
    compute_multimodal_positions,
    extend_mm_token_type_ids,
    extract_target_head,
    forward_and_capture,
    locate_image_span,
    storage_dtype,
)

__all__ = [
    "build_sample_record",
    "compute_multimodal_positions",
    "extend_mm_token_type_ids",
    "forward_and_capture",
    "locate_image_span",
    "run_capture",
    "save_target_head",
    "storage_dtype",
]


def save_target_head(model, features_dir: str | Path, dtype: torch.dtype, contract: dict) -> bool:
    """Persist the frozen embed/head pair the draft trainer must consume.

    Inference samples draft logits through ``target.lm_head``, so training has to
    use the same head: tied targets share the embedding matrix, untied targets
    (e.g. Qwen3-VL-8B) store their own output projection under ``"lm_head"``.

    Returns:
        True when the target head is untied and its weight was stored.
    """
    embedding_weight, lm_head_weight, tied = extract_target_head(model, contract)
    head_payload = {
        "format": "vlm-dflash-target-head-v1",
        "embed_tokens": embedding_weight.to(dtype).cpu(),
        "tie_word_embeddings": bool(tied),
        "model_fingerprint": contract["target"]["model_fingerprint"],
    }
    if not tied:
        head_payload["lm_head"] = lm_head_weight.to(dtype).cpu()
    torch.save(head_payload, Path(features_dir) / "target_head.pt")
    return not tied


def run_capture(cfg: dict, run_dir: str | Path, contract: dict, limit: int | None = None) -> dict:
    run_dir = Path(run_dir)
    rows, media_dir = load_sharegpt(
        run_dir / "regen" / "target_data.jsonl", meta_path=run_dir / "data" / "meta.json"
    )
    if limit is not None:
        rows = rows[:limit]
    cap_cfg = cfg.get("capture", {})
    seq_length = int(cap_cfg.get("seq_length", 2048))
    shard_size = int(cap_cfg.get("shard_size", 32))
    dtype = storage_dtype(cap_cfg.get("dtype", "bfloat16"))
    layer_ids = [int(layer) for layer in contract["layers"]["draft_layer_ids"]]
    image_token_id = int(contract["tokens"]["image_token_id"])
    merge_size = int(contract["image"]["merge_size"])
    store_final_hidden = bool(cap_cfg.get("store_final_hidden", False))

    model, processor = load_target(cfg, contract)
    features_dir = ensure_dir(run_dir / "features")
    index_rows, shard_buffer = [], []
    shard_number = 0
    truncated = 0

    def flush_shard() -> None:
        nonlocal shard_buffer, shard_number
        if not shard_buffer:
            return
        path = features_dir / f"shard_{shard_number:03d}.pt"
        torch.save({"samples": shard_buffer}, path)
        for offset, record in enumerate(shard_buffer):
            index_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "shard": path.name,
                    "shard_index": offset,
                    "seq_len": int(record["input_ids"].numel()),
                    "prompt_len": record["prompt_len"],
                    "answer_len": record["answer_len"],
                    "image_span": list(record["image_span"]),
                    "n_image_tokens": record["n_image_tokens"],
                }
            )
        shard_buffer = []
        shard_number += 1

    log(f"capture: {len(rows)} samples, seq_length={seq_length}, dtype={dtype}")
    for row_number, row in enumerate(rows):
        record = build_sample_record(
            model,
            processor,
            row,
            media_dir,
            layer_ids=layer_ids,
            image_token_id=image_token_id,
            merge_size=merge_size,
            dtype=dtype,
            seq_length=seq_length,
            store_final_hidden=store_final_hidden,
        )
        if record["truncated"]:
            truncated += 1
        shard_buffer.append(record)
        if len(shard_buffer) >= shard_size:
            flush_shard()
        if (row_number + 1) % 32 == 0:
            log(f"capture: {row_number + 1}/{len(rows)}")
    flush_shard()
    write_jsonl(features_dir / "index.jsonl", index_rows)

    lm_head_saved = save_target_head(model, features_dir, dtype, contract)

    manifest = {
        "format": "vlm-dflash-features-v1",
        "identity": identity_from_contract(contract),
        "seq_length": seq_length,
        "storage_dtype": str(dtype).replace("torch.", ""),
        "num_samples": len(index_rows),
        "num_truncated": truncated,
        "draft_layer_ids": layer_ids,
        "stores_final_hidden": store_final_hidden,
        "tie_word_embeddings": not lm_head_saved,
        "lm_head_saved": lm_head_saved,
        "seq_len_stats": {
            "min": min((r["seq_len"] for r in index_rows), default=None),
            "max": max((r["seq_len"] for r in index_rows), default=None),
            "mean": (sum(r["seq_len"] for r in index_rows) / len(index_rows)) if index_rows else None,
        },
    }
    write_json(features_dir / "manifest.json", manifest)

    visual_check = None
    if cap_cfg.get("visual_path_check", True) and len(rows) >= 2:
        from .visual_check import run_visual_path_check

        visual_check = run_visual_path_check(
            cfg,
            run_dir,
            contract,
            model,
            processor,
            rows,
            media_dir,
            dtype,
            limit=min(int(cap_cfg.get("visual_check_samples", 2)), len(rows)),
        )

    record_stage(
        run_dir,
        "capture",
        {
            "identity": identity_from_contract(contract),
            "manifest": {k: v for k, v in manifest.items() if k != "identity"},
            "visual_path_check": visual_check,
            "code": "vlm_dflash.target.capture.run_capture",
        },
    )
    return manifest
