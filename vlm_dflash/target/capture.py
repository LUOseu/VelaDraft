"""Offline capture of fused target hidden states for staged DFlash training.

For every regenerated sample this runs one teacher-forced target forward with
``output_hidden_states=True``, concatenates the contract's routed layers into
the draft context feature, records the MRoPE 3-axis positions, validates the
image-token span, and writes fp/bf16 shards plus an index. Training never needs
the target model in memory afterwards.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import load_sharegpt, row_answer, row_image_path, row_question
from ..provenance import identity_from_contract, load_provenance, record_stage
from ..utils import ensure_dir, log, write_json, write_jsonl
from .load import load_target
from .prompts import encode_prompt


def storage_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"Unsupported capture storage dtype {name!r}.")
    return mapping[name]


def compute_multimodal_positions(model, input_ids, attention_mask, image_grid_thw, mm_token_type_ids=None):
    """Call the target's own MRoPE position builder, passing only supported kwargs."""
    builder = getattr(getattr(model, "model", None), "compute_3d_position_ids", None)
    if not callable(builder):
        raise RuntimeError("Target does not expose compute_3d_position_ids; v0.1 requires Qwen2.5-VL.")
    signature = inspect.signature(builder)
    candidate = {
        "input_ids": input_ids,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": None,
        "inputs_embeds": None,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "second_per_grid_ts": None,
        "mm_token_type_ids": mm_token_type_ids,
    }
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    kwargs = {
        key: value for key, value in candidate.items() if accepts_var_kw or key in signature.parameters
    }
    positions = builder(**kwargs)
    if positions is None:
        raise RuntimeError("compute_3d_position_ids returned None; image/token-type inputs missing.")
    if positions.dim() == 4:
        positions = positions[..., 0, :]
    if positions.dim() == 3:
        positions = positions[:, 0, :]
    return positions


def extend_mm_token_type_ids(mm_token_type_ids: torch.Tensor, full_length: int) -> torch.Tensor:
    """Extend processor ``mm_token_type_ids`` [1, prompt] with text (0) for the answer."""
    prompt_length = mm_token_type_ids.shape[1]
    if full_length <= prompt_length:
        return mm_token_type_ids[:, :full_length]
    padding = torch.zeros(
        (mm_token_type_ids.shape[0], full_length - prompt_length),
        dtype=mm_token_type_ids.dtype,
        device=mm_token_type_ids.device,
    )
    return torch.cat([mm_token_type_ids, padding], dim=1)


def locate_image_span(input_ids: torch.Tensor, image_token_id: int) -> tuple[int, int, int]:
    """Return ``(start, end_inclusive, count)`` of the contiguous image token span."""
    matches = (input_ids == image_token_id).nonzero().flatten()
    if matches.numel() == 0:
        raise ValueError("No image tokens found in the sequence.")
    start, end = int(matches[0]), int(matches[-1])
    if int(matches.numel()) != end - start + 1:
        raise ValueError("Image tokens are not contiguous; v0.1 requires a single leading image.")
    return start, end, int(matches.numel())


@torch.inference_mode()
def forward_and_capture(
    model,
    inputs: dict,
    full_ids: torch.Tensor,
    layer_ids: list[int],
    image_token_id: int,
    dtype: torch.dtype,
    merge_size: int = 2,
) -> dict:
    """One capture forward; returns the sample record consumed by training."""
    device = next(model.parameters()).device
    input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
        "logits_to_keep": 1,
    }
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in inputs:
            forward_kwargs[key] = inputs[key].to(device)
    output = model(**forward_kwargs)
    hidden_states = output.hidden_states
    fused = torch.cat([hidden_states[layer_id + 1][0] for layer_id in layer_ids], dim=-1)
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = extend_mm_token_type_ids(mm_token_type_ids.to(device), input_ids.shape[1])
    positions = compute_multimodal_positions(
        model, input_ids, attention_mask, inputs.get("image_grid_thw"), mm_token_type_ids=mm_token_type_ids
    )
    span_start, span_end, span_count = locate_image_span(full_ids.cpu(), image_token_id)
    grid = inputs.get("image_grid_thw")
    # image_grid_thw counts 14px patches; every merge_size^2 patches become one token.
    expected = int(grid.prod().item()) // (merge_size * merge_size) if grid is not None else None
    return {
        "input_ids": full_ids.to(torch.int32),
        "hidden": fused.to(dtype).cpu(),
        "position3": positions.to(torch.int32).cpu(),
        "image_span": (span_start, span_end),
        "n_image_tokens": span_count,
        "n_image_tokens_expected": expected,
    }


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
        image = Image.open(row_image_path(row, media_dir)).convert("RGB")
        inputs = encode_prompt(processor, row_question(row), image)
        stored_prompt = row["regen"]["prompt_ids"]
        fresh_prompt = inputs["input_ids"][0].tolist()
        if fresh_prompt != stored_prompt:
            raise RuntimeError(
                f"Prompt token drift between regen and capture for sample {row.get('id')}: "
                "the chat template or processor changed between stages."
            )
        answer_ids = row["regen"]["answer_ids"]
        prompt_len = len(stored_prompt)
        if prompt_len + 8 > seq_length:
            raise RuntimeError(f"Prompt of sample {row.get('id')} ({prompt_len} tokens) exceeds seq_length.")
        answer_len = min(len(answer_ids), seq_length - prompt_len)
        if answer_len < len(answer_ids):
            truncated += 1
        full_ids = torch.tensor(stored_prompt + answer_ids[:answer_len], dtype=torch.long)
        record = forward_and_capture(model, inputs, full_ids, layer_ids, image_token_id, dtype, merge_size=merge_size)
        if record["n_image_tokens_expected"] != record["n_image_tokens"]:
            raise RuntimeError(
                f"Image token count mismatch for sample {row.get('id')}: "
                f"found {record['n_image_tokens']} tokens in input_ids but grid_thw expects "
                f"{record['n_image_tokens_expected']}."
            )
        record.update(
            {
                "sample_id": row.get("id"),
                "prompt_len": prompt_len,
                "answer_len": answer_len,
                "loss_mask": torch.cat(
                    (
                        torch.zeros(prompt_len, dtype=torch.bool),
                        torch.ones(answer_len, dtype=torch.bool),
                    )
                ),
            }
        )
        shard_buffer.append(record)
        if len(shard_buffer) >= shard_size:
            flush_shard()
        if (row_number + 1) % 32 == 0:
            log(f"capture: {row_number + 1}/{len(rows)}")
    flush_shard()
    write_jsonl(features_dir / "index.jsonl", index_rows)

    embedding_weight = model.get_input_embeddings().weight.detach()
    lm_head_weight = getattr(model, "lm_head", None)
    tied = lm_head_weight is not None and lm_head_weight.weight.data_ptr() == embedding_weight.data_ptr()
    torch.save(
        {
            "format": "vlm-dflash-target-head-v1",
            "embed_tokens": embedding_weight.to(dtype).cpu(),
            "tie_word_embeddings": bool(tied),
            "model_fingerprint": contract["target"]["model_fingerprint"],
        },
        features_dir / "target_head.pt",
    )

    manifest = {
        "format": "vlm-dflash-features-v1",
        "identity": identity_from_contract(contract),
        "seq_length": seq_length,
        "storage_dtype": str(dtype).replace("torch.", ""),
        "num_samples": len(index_rows),
        "num_truncated": truncated,
        "draft_layer_ids": layer_ids,
        "tie_word_embeddings": bool(tied),
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
