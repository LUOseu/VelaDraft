"""Single-sample teacher-forced capture, shared by staged capture and online training.

One code path produces the per-sample training record for both feature
sources: the offline ``capture_features`` stage writes these records into
shards, while the online source (``feature_mode=online``) feeds the same
records straight into the shared trainer without ever writing a feature
store. Keeping the two paths byte-identical on the record fields is what
makes an online run train against exactly the same teacher signal as the
staged pipeline.

The record contract (consumed by ``dflash.features.collate_features``):
``input_ids``, ``hidden`` (routed target layers concatenated on the last
axis), ``position3`` (MRoPE from the target adapter), ``image_span``,
``sample_id``, ``loss_mask``, and optionally ``final_hidden`` (the
post-final-norm tensor the target ``lm_head`` consumes).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import row_image_path, row_image_paths, row_is_multi_turn, row_question
from .prompts import encode_conversation, encode_prompt, messages_from_row


def storage_dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"Unsupported capture storage dtype {name!r}.")
    return mapping[name]


def compute_multimodal_positions(model, input_ids, attention_mask, image_grid_thw, mm_token_type_ids=None):
    """Delegate to the target family adapter's MRoPE position builder."""
    from ..adapters import adapter_for_model

    return adapter_for_model(model).compute_multimodal_positions(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )


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


def locate_image_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """All contiguous image-token runs, ``(start, end_inclusive)`` in appearance order.

    One run per group of consecutive ``<image>`` parts (images attached to the
    same turn render back to back); multi-turn rows produce several runs.
    """
    matches = (input_ids == image_token_id).nonzero().flatten().tolist()
    if not matches:
        raise ValueError("No image tokens found in the sequence.")
    spans: list[tuple[int, int]] = []
    start = previous = matches[0]
    for position in matches[1:]:
        if position == previous + 1:
            previous = position
            continue
        spans.append((start, previous))
        start = previous = position
    spans.append((start, previous))
    return spans


def extract_target_head(model, contract: dict):
    """Resolve the frozen embed/head pair the draft trainer must consume.

    Inference samples draft logits through ``target.lm_head``, so training has to
    use the same head: tied targets share the embedding matrix, untied targets
    (e.g. Qwen3-VL-8B) keep their own output projection.

    Returns:
        ``(embed_weight, lm_head_weight_or_None, tied)``. ``lm_head_weight`` is
        ``None`` for tied targets (the module shares the embedding row).
    """
    embedding_weight = model.get_input_embeddings().weight.detach()
    lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    tied = lm_head_weight is not None and lm_head_weight.data_ptr() == embedding_weight.data_ptr()
    expected_tied = bool(contract.get("head", {}).get("tie_word_embeddings", tied))
    if tied != expected_tied:
        raise RuntimeError(
            "Loaded target head disagrees with the resolved contract: "
            f"runtime tie_word_embeddings={tied}, contract={expected_tied}."
        )
    if not tied and lm_head_weight is None:
        raise RuntimeError(
            "The target exposes no lm_head weight; the draft's training head cannot be reconstructed "
            "and training would silently diverge from the eval path (which samples through target.lm_head)."
        )
    return embedding_weight, (None if tied else lm_head_weight.detach()), bool(tied)


@torch.inference_mode()
def forward_and_capture(
    model,
    inputs: dict,
    full_ids: torch.Tensor,
    layer_ids: list[int],
    image_token_id: int,
    dtype: torch.dtype,
    merge_size: int = 2,
    store_final_hidden: bool = False,
    allow_multi_span: bool = False,
) -> dict:
    """One capture forward; returns the sample record consumed by training.

    ``store_final_hidden`` additionally records the target's final hidden state
    (the post-final-norm tensor the target lm_head consumes), which distillation
    objectives (JetSpec, DFlash2) project to full-vocab teacher logits at
    training time. Old feature stores do not have this field; those trainers
    fail loud on them instead of silently training against the wrong teacher.

    ``allow_multi_span`` admits several contiguous image-token runs (multi-image
    / multi-turn rows); the record then also carries ``image_spans``,
    ``n_images`` and the per-image expected token counts.
    """
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
    # The target's own multimodal forward *must* see the same token-type map
    # the processor emitted (extended over the regen answer with text=0):
    # Qwen3-VL raises without it, and Qwen2.5-VL would otherwise fall back to
    # plain 1-D positions here while eval/prefill use MRoPE positions.
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = extend_mm_token_type_ids(mm_token_type_ids.to(device), input_ids.shape[1])
        forward_kwargs["mm_token_type_ids"] = mm_token_type_ids
    output = model(**forward_kwargs)
    hidden_states = output.hidden_states
    fused = torch.cat([hidden_states[layer_id + 1][0] for layer_id in layer_ids], dim=-1)
    # hidden_states[-1] is the post-final-norm tensor; ``lm_head`` applied to it
    # reproduces the target's own output logits (verified by the capture-time
    # teacher-logits check in the smoke pipeline).
    final_hidden = hidden_states[-1][0].to(dtype).cpu() if store_final_hidden else None
    positions = compute_multimodal_positions(
        model, input_ids, attention_mask, inputs.get("image_grid_thw"), mm_token_type_ids=mm_token_type_ids
    )
    spans = locate_image_spans(full_ids.cpu(), image_token_id)
    if not allow_multi_span and len(spans) != 1:
        raise ValueError("Image tokens are not contiguous; v0.1 requires a single leading image.")
    grid = inputs.get("image_grid_thw")
    # image_grid_thw counts patches; every merge_size^2 patches become one token.
    expected_per_image = (
        [int(row.prod().item()) // (merge_size * merge_size) for row in grid] if grid is not None else None
    )
    span_count = sum(end - start + 1 for start, end in spans)
    expected = sum(expected_per_image) if expected_per_image is not None else None
    record = {
        "input_ids": full_ids.to(torch.int32),
        "hidden": fused.to(dtype).cpu(),
        "position3": positions.to(torch.int32).cpu(),
        "image_span": spans[0],
        "n_image_tokens": span_count,
        "n_image_tokens_expected": expected,
    }
    if allow_multi_span:
        record["image_spans"] = [tuple(span) for span in spans]
        record["n_images"] = len(grid) if grid is not None else None
        record["image_token_counts_expected"] = expected_per_image
    if final_hidden is not None:
        record["final_hidden"] = final_hidden
    return record


def _build_single_turn_record(
    model,
    processor,
    row: dict,
    media_dir: Path,
    *,
    layer_ids: list[int],
    image_token_id: int,
    merge_size: int,
    dtype: torch.dtype,
    seq_length: int,
    store_final_hidden: bool,
) -> dict:
    """One regenerated row -> the training record (offline capture and online share this).

    Replays the prompt through the same processor/chat template the regen stage
    used, asserts the prompt token ids did not drift, truncates the answer to
    ``seq_length``, then runs one teacher-forced target forward. Raises on any
    mismatch instead of producing a silently misaligned record.
    """
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
    full_ids = torch.tensor(stored_prompt + answer_ids[:answer_len], dtype=torch.long)
    record = forward_and_capture(
        model,
        inputs,
        full_ids,
        layer_ids,
        image_token_id,
        dtype,
        merge_size=merge_size,
        store_final_hidden=store_final_hidden,
    )
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
            "truncated": answer_len < len(answer_ids),
            "loss_mask": torch.cat(
                (
                    torch.zeros(prompt_len, dtype=torch.bool),
                    torch.ones(answer_len, dtype=torch.bool),
                )
            ),
        }
    )
    return record


def _build_multi_turn_record(
    model,
    processor,
    row: dict,
    media_dir: Path,
    *,
    layer_ids: list[int],
    image_token_id: int,
    merge_size: int,
    dtype: torch.dtype,
    seq_length: int,
    store_final_hidden: bool,
) -> dict:
    """Schema-v2 row -> training record: replay the full-conversation render once.

    The stored ``regen.full_ids`` is asserted to be the exact token prefix of a
    fresh processor render (chat-template drift fails loud, like the v1 prompt
    replay), every answer span is validated against its stored answer ids, and
    ``loss_mask`` becomes the union of the (clipped) answer spans.
    """
    conversations = row.get("conversations") or []
    regen = row.get("regen") or {}
    turns = regen.get("turns")
    if not turns:
        raise RuntimeError(f"Multi-turn row {row.get('id')!r} has no regen['turns']; re-run the regen stage.")
    assistant_indices = [
        index
        for index, turn in enumerate(conversations)
        if str(turn.get("from", "")).lower() in ("gpt", "assistant", "model")
    ]
    if len(assistant_indices) != len(turns):
        raise RuntimeError(
            f"Row {row.get('id')!r} has {len(assistant_indices)} assistant turn(s) but {len(turns)} regenerated turn(s)."
        )
    overrides = {msg_index: turn["answer_text"] for msg_index, turn in zip(assistant_indices, turns)}
    image_paths = row_image_paths(row, media_dir)
    messages, image_indices = messages_from_row(row, assistant_overrides=overrides)
    images = [Image.open(image_paths[position]).convert("RGB") for position in image_indices]
    inputs = encode_conversation(processor, messages, images, add_generation_prompt=False)
    stored_full_ids = [int(token) for token in regen["full_ids"]]
    fresh_prompt = inputs["input_ids"][0].tolist()
    if fresh_prompt[: len(stored_full_ids)] != stored_full_ids:
        raise RuntimeError(
            f"Prompt token drift between regen and capture for sample {row.get('id')}: "
            "the chat template or processor changed between stages."
        )
    answer_spans = [tuple(int(value) for value in span) for span in regen["answer_spans"]]
    if len(answer_spans) != len(turns):
        raise RuntimeError(f"Row {row.get('id')!r} has {len(turns)} turn(s) but {len(answer_spans)} answer span(s).")
    for turn, span in zip(turns, answer_spans):
        if stored_full_ids[span[0] : span[1]] != [int(token) for token in turn["answer_ids"]]:
            raise RuntimeError(
                f"Answer span mismatch for sample {row.get('id')}: stored answer ids do not match full_ids."
            )
    prompt_len_last = len(turns[-1]["prompt_ids"])
    if prompt_len_last + 8 > seq_length:
        raise RuntimeError(f"Final prompt of sample {row.get('id')} ({prompt_len_last} tokens) exceeds seq_length.")
    keep = min(len(stored_full_ids), seq_length)
    full_ids = torch.tensor(stored_full_ids[:keep], dtype=torch.long)
    record = forward_and_capture(
        model,
        inputs,
        full_ids,
        layer_ids,
        image_token_id,
        dtype,
        merge_size=merge_size,
        store_final_hidden=store_final_hidden,
        allow_multi_span=True,
    )
    loss_mask = torch.zeros(keep, dtype=torch.bool)
    clipped_spans = []
    for start, end in answer_spans:
        start, end = min(start, keep), min(end, keep)
        if end > start:
            loss_mask[start:end] = True
        clipped_spans.append((start, end))
    turn_spans, start = [], 0
    for turn in turns:
        end = len(turn["prompt_ids"]) + len(turn["answer_ids"])
        turn_spans.append((min(start, keep), min(end, keep)))
        start = end
    if start != len(stored_full_ids):
        raise RuntimeError(
            f"Turn span bookkeeping for sample {row.get('id')} does not cover its full token sequence."
        )
    record.update(
        {
            "sample_id": row.get("id"),
            "prompt_len": keep - int(loss_mask.sum()),
            "answer_len": int(loss_mask.sum()),
            "truncated": keep < len(stored_full_ids),
            "loss_mask": loss_mask,
            "answer_spans": [list(span) for span in clipped_spans],
            "turn_spans": [list(span) for span in turn_spans],
            "n_turns": len(turns),
        }
    )
    return record


def build_sample_record(
    model,
    processor,
    row: dict,
    media_dir: Path,
    *,
    layer_ids: list[int],
    image_token_id: int,
    merge_size: int,
    dtype: torch.dtype,
    seq_length: int,
    store_final_hidden: bool,
) -> dict:
    """One regenerated row -> the training record (offline capture and online share this).

    Dispatches on the row generation: schema-v2 rows (multi-image / multi-turn)
    replay the whole conversation render and build a multi-segment loss mask;
    v1 rows keep the original single-image, single-answer path bit-for-bit.
    """
    if row_is_multi_turn(row):
        return _build_multi_turn_record(
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
    return _build_single_turn_record(
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
