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
    """All plain contiguous image-token runs, ``(start, end_inclusive)``.

    Family-aware capture goes through the adapter's ``image_token_spans``
    (tile-encoded families bridge their structure tokens); this helper keeps
    the strict runs-of-image-tokens view.
    """
    from ..adapters.base import locate_image_token_spans

    return locate_image_token_spans(input_ids, image_token_id)


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
def forward_and_capture_batch(
    model,
    prepared_list: list[dict],
    layer_ids: list[int],
    image_token_id: int,
    dtype: torch.dtype,
    merge_size: int | None = None,
    store_final_hidden: bool = False,
    tokenizer=None,
) -> list[dict]:
    """Batched capture forward: one padded model call for several prepared rows.

    ``prepared_list`` comes from :func:`prepare_sample_record` (render +
    validation already done). Rows are padded to the batch max; every output
    is sliced back to the row's own length before a record is built, so pad
    positions never reach a record. Mathematically each row runs the same
    teacher-forced forward as :func:`forward_and_capture` does alone.
    """
    device = next(model.parameters()).device
    lengths = [int(item["full_ids"].numel()) for item in prepared_list]
    max_length = max(lengths)
    # Some families ship a config pad id from a different vocabulary (e.g.
    # SmolVLM2's pad_token_id 128002 vs. a 49280-row embedding table). Pad
    # positions are masked out, so any in-range id is a valid filler.
    pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    vocab_size = int(model.get_input_embeddings().weight.shape[0])
    if pad_token_id is None or not 0 <= int(pad_token_id) < vocab_size:
        pad_token_id = 0
    input_ids = torch.full((len(prepared_list), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(prepared_list), max_length), dtype=torch.long)
    for row_index, item in enumerate(prepared_list):
        length = lengths[row_index]
        input_ids[row_index, :length] = item["full_ids"]
        attention_mask[row_index, :length] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    forward_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
        "use_cache": False,
        "logits_to_keep": 1,
    }
    from ..adapters import adapter_for_model

    adapter = adapter_for_model(model)
    forward_kwargs.update(adapter.capture_forward_inputs(prepared_list, device))
    # The target's own multimodal forward *must* see the same token-type map
    # the processor emitted (extended over the regen answer with text=0):
    # Qwen3-VL raises without it, and Qwen2.5-VL would otherwise fall back to
    # plain 1-D positions here while eval/prefill use MRoPE positions. Families
    # that do not consume the map (tile-encoded processors) skip this entirely.
    extended_mm: list[torch.Tensor | None] = [None] * len(prepared_list)
    if "mm_token_type_ids" in adapter.info.forward_input_keys:
        mm_rows = [item["inputs"].get("mm_token_type_ids") for item in prepared_list]
        mm_present = [value is not None for value in mm_rows]
        if any(mm_present) and not all(mm_present):
            raise ValueError("Batch capture mixes rows with and without mm_token_type_ids; split the batch.")
        if all(mm_present):
            stacked = torch.zeros((len(prepared_list), max_length), dtype=mm_rows[0].dtype, device=device)
            for row_index, value in enumerate(mm_rows):
                extended = extend_mm_token_type_ids(value.to(device), lengths[row_index])
                extended_mm[row_index] = extended
                stacked[row_index, : lengths[row_index]] = extended[0]
            forward_kwargs["mm_token_type_ids"] = stacked
    output = model(**forward_kwargs)
    hidden_states = output.hidden_states
    # A device-mapped target leaves each layer's output on its own GPU.
    fuse_device = hidden_states[layer_ids[0] + 1].device
    records = []
    for row_index, item in enumerate(prepared_list):
        length = lengths[row_index]
        fused = torch.cat(
            [hidden_states[layer_id + 1][row_index, :length].to(fuse_device) for layer_id in layer_ids], dim=-1
        )
        # hidden_states[-1] is the post-final-norm tensor; ``lm_head`` applied to it
        # reproduces the target's own output logits (verified by the capture-time
        # teacher-logits check in the smoke pipeline).
        final_hidden = (
            hidden_states[-1][row_index, :length].to(dtype).cpu() if store_final_hidden else None
        )
        row_inputs = input_ids[row_index : row_index + 1, :length]
        row_mask = attention_mask[row_index : row_index + 1, :length]
        grid = item["inputs"].get("image_grid_thw")
        positions = compute_multimodal_positions(
            model,
            row_inputs,
            row_mask,
            grid.to(device) if grid is not None else None,
            mm_token_type_ids=extended_mm[row_index],
        )
        row_ids = item["full_ids"].cpu()
        spans = adapter.image_token_spans(row_ids, image_token_id, tokenizer=tokenizer)
        if not item.get("allow_multi_span") and len(spans) != 1:
            raise ValueError("Image tokens are not contiguous; v0.1 requires a single leading image.")
        # Grid families: image_grid_thw counts patches and every merge_size^2
        # patches become one token; tile families report tiles x fixed length.
        # The count is always of ``image_token_id`` itself: a bridged family
        # span may also contain its structure/whitespace markers.
        expected_per_image = adapter.expected_image_token_counts(item["inputs"], merge_size=merge_size)
        span_count = int(sum((row_ids[start : end + 1] == image_token_id).sum() for start, end in spans))
        expected = sum(expected_per_image) if expected_per_image is not None else None
        record = {
            "input_ids": item["full_ids"].to(torch.int32),
            "hidden": fused.to(dtype).cpu(),
            "position3": positions.to(torch.int32).cpu(),
            "image_span": spans[0],
            "n_image_tokens": span_count,
            "n_image_tokens_expected": expected,
        }
        if item.get("allow_multi_span"):
            record["image_spans"] = [tuple(span) for span in spans]
            record["n_images"] = len(grid) if grid is not None else len(spans)
            record["image_token_counts_expected"] = expected_per_image
        if final_hidden is not None:
            record["final_hidden"] = final_hidden
        records.append(record)
    return records


@torch.inference_mode()
def forward_and_capture(
    model,
    inputs: dict,
    full_ids: torch.Tensor,
    layer_ids: list[int],
    image_token_id: int,
    dtype: torch.dtype,
    merge_size: int | None = None,
    store_final_hidden: bool = False,
    allow_multi_span: bool = False,
    tokenizer=None,
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
    prepared = {
        "inputs": inputs,
        "full_ids": full_ids,
        "allow_multi_span": bool(allow_multi_span),
    }
    return forward_and_capture_batch(
        model,
        [prepared],
        layer_ids,
        image_token_id,
        dtype,
        merge_size=merge_size,
        store_final_hidden=store_final_hidden,
        tokenizer=tokenizer,
    )[0]


def _prepare_single_turn(
    processor,
    row: dict,
    media_dir: Path,
    *,
    seq_length: int,
) -> dict:
    """v1 row -> rendered inputs + token ids + record metadata (no forward).

    Replays the prompt through the same processor/chat template the regen stage
    used, asserts the prompt token ids did not drift and truncates the answer
    to ``seq_length``. Raises on any mismatch instead of producing a silently
    misaligned record; the forward half is ``finalize_prepared_records``.
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
    return {
        "inputs": inputs,
        "full_ids": full_ids,
        "sample_id": row.get("id"),
        "allow_multi_span": False,
        "check_image_token_count": True,
        "meta": {
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
        },
    }


def _prepare_multi_turn(
    processor,
    row: dict,
    media_dir: Path,
    *,
    seq_length: int,
) -> dict:
    """Schema-v2 row -> rendered inputs + full ids + multi-segment metadata (no forward).

    The stored ``regen.full_ids`` is asserted to be the exact token prefix of a
    fresh processor render (chat-template drift fails loud, like the v1 prompt
    replay), every answer span is validated against its stored answer ids, and
    ``loss_mask`` becomes the union of the (clipped) answer spans.

    The row's ``regen.mode`` selects the history/supervision protocol:
    ``last_turn_only`` keeps the dataset history and supervises only the last
    regenerated answer; ``all_turns`` (and legacy records without a mode --
    they predate the field and are all-turn artifacts) applies every stored
    regenerated answer as history overrides and supervises all of them. A
    stored record whose mode and ``assistant_indices`` disagree is refused so a
    mode change can never silently reuse old supervision.
    """
    conversations = row.get("conversations") or []
    regen = row.get("regen") or {}
    turns = regen.get("turns")
    if not turns:
        raise RuntimeError(f"Multi-turn row {row.get('id')!r} has no regen['turns']; re-run the regen stage.")
    mode = str(regen.get("mode") or "all_turns").lower()
    if mode not in ("last_turn_only", "all_turns"):
        raise RuntimeError(
            f"Row {row.get('id')!r} has unknown regen.mode {mode!r}; re-run the regen stage."
        )
    assistant_indices = [
        index
        for index, turn in enumerate(conversations)
        if str(turn.get("from", "")).lower() in ("gpt", "assistant", "model")
    ]
    regenerated_indices = regen.get("assistant_indices")
    regenerated_indices = list(assistant_indices) if regenerated_indices is None else [int(i) for i in regenerated_indices]
    if mode == "last_turn_only":
        if regenerated_indices != [assistant_indices[-1]]:
            raise RuntimeError(
                f"Row {row.get('id')!r} records regen.mode=last_turn_only but assistant_indices="
                f"{regenerated_indices}; only the last assistant turn ({assistant_indices[-1]}) may be "
                "regenerated/supervised in this mode."
            )
    elif regenerated_indices != list(assistant_indices):
        raise RuntimeError(
            f"Row {row.get('id')!r} records regen.mode=all_turns but assistant_indices={regenerated_indices} "
            f"does not cover every assistant turn {assistant_indices}."
        )
    if len(regenerated_indices) != len(turns):
        raise RuntimeError(
            f"Row {row.get('id')!r} has {len(regenerated_indices)} regenerated turn(s) but {len(turns)} "
            "stored regen turn(s)."
        )
    overrides = {msg_index: turn["answer_text"] for msg_index, turn in zip(regenerated_indices, turns)}
    image_paths = row_image_paths(row, media_dir)
    messages, image_indices = messages_from_row(row, assistant_overrides=overrides)
    all_images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = encode_conversation(processor, messages, [all_images[position] for position in image_indices], add_generation_prompt=False)
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
    loss_mask = torch.zeros(keep, dtype=torch.bool)
    clipped_spans = []
    for start, end in answer_spans:
        start, end = min(start, keep), min(end, keep)
        if end > start:
            loss_mask[start:end] = True
        clipped_spans.append((start, end))
    if mode == "last_turn_only":
        # Per-conversation-turn spans from prefix renders: a rendered prefix
        # must be a token prefix of full_ids (a second drift guard on the new
        # protocol), and the last assistant turn spans through the end of the
        # sequence. Some templates render a conversation's trailing assistant
        # turn unlike a historical one (Qwen3.5/3.6 add an empty thinking
        # block), so a prefix ending in an assistant turn is not a token prefix
        # there. When full_ids holds one turn-start marker per turn, a
        # historical assistant turn ends where the next turn starts instead,
        # and every user-turn prefix render must end on a marker too.
        target_index = regenerated_indices[-1]
        turn_spans, previous_end = [], 0
        turn_starts = _turn_start_positions(processor, stored_full_ids, len(conversations))
        for turn_index in range(len(conversations)):
            if turn_index == target_index:
                end = len(stored_full_ids)
            elif turn_starts is not None and turn_index in assistant_indices:
                end = turn_starts[turn_index + 1]
            else:
                prefix_messages, prefix_image_indices = messages_from_row(
                    row, assistant_overrides=overrides, upto=turn_index + 1
                )
                prefix_inputs = encode_conversation(
                    processor,
                    prefix_messages,
                    [all_images[position] for position in prefix_image_indices],
                    add_generation_prompt=False,
                )
                prefix_ids = prefix_inputs["input_ids"][0].tolist()
                if stored_full_ids[: len(prefix_ids)] != prefix_ids:
                    raise RuntimeError(
                        f"Prefix render drift for sample {row.get('id')} at turn {turn_index}: the chat "
                        "template does not reproduce the committed prefix; refusing to store misaligned ids."
                    )
                end = len(prefix_ids)
                if turn_starts is not None and end != turn_starts[turn_index + 1]:
                    raise RuntimeError(
                        f"Turn-start markers of sample {row.get('id')} disagree with the prefix render at turn "
                        f"{turn_index} (marker at {turn_starts[turn_index + 1]}, prefix ends at {end})."
                    )
            turn_spans.append((min(previous_end, keep), min(end, keep)))
            previous_end = end
        if previous_end != len(stored_full_ids):
            raise RuntimeError(
                f"Turn span bookkeeping for sample {row.get('id')} does not cover its full token sequence."
            )
    else:
        turn_spans, start = [], 0
        for turn in turns:
            end = len(turn["prompt_ids"]) + len(turn["answer_ids"])
            turn_spans.append((min(start, keep), min(end, keep)))
            start = end
        if start != len(stored_full_ids):
            raise RuntimeError(
                f"Turn span bookkeeping for sample {row.get('id')} does not cover its full token sequence."
            )
    return {
        "inputs": inputs,
        "full_ids": full_ids,
        "sample_id": row.get("id"),
        "allow_multi_span": True,
        "check_image_token_count": False,
        "meta": {
            "sample_id": row.get("id"),
            "prompt_len": keep - int(loss_mask.sum()),
            "answer_len": int(loss_mask.sum()),
            "truncated": keep < len(stored_full_ids),
            "loss_mask": loss_mask,
            "answer_spans": [list(span) for span in clipped_spans],
            "turn_spans": [list(span) for span in turn_spans],
            # Regenerated/supervised answers (``last_turn_only``: exactly one).
            "n_turns": len(turns),
            # Conversation turns (human+assistant), for the multi-turn manifest flags.
            "n_conversation_turns": len(conversations),
            "multi_turn_mode": mode,
        },
    }


def _turn_start_positions(processor, full_ids: list[int], num_turns: int) -> list[int] | None:
    """Token positions where each conversation turn starts (``<|im_start|>`` markers).

    Returns ``None`` when the tokenizer has no such marker or the sequence does
    not hold exactly one marker per turn, so callers keep failing loudly.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    marker = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if marker is None or marker < 0 or marker == getattr(tokenizer, "unk_token_id", None):
        return None
    starts = [index for index, token in enumerate(full_ids) if token == marker]
    return starts if len(starts) == num_turns else None


def prepare_sample_record(
    processor,
    row: dict,
    media_dir: Path,
    *,
    seq_length: int,
) -> dict:
    """Render + validate one row; the target forward happens in finalize.

    The prepared dict (processor inputs, full token ids, record metadata) is
    what ``forward_and_capture_batch`` consumes, so one forward can serve a
    whole batch of rows without changing any per-row record field.
    """
    if row_is_multi_turn(row):
        return _prepare_multi_turn(processor, row, media_dir, seq_length=seq_length)
    return _prepare_single_turn(processor, row, media_dir, seq_length=seq_length)


def finalize_prepared_records(
    model,
    prepared_list: list[dict],
    *,
    layer_ids: list[int],
    image_token_id: int,
    merge_size: int | None,
    dtype: torch.dtype,
    store_final_hidden: bool,
    tokenizer=None,
) -> list[dict]:
    """Run the (batched) capture forward for prepared rows and attach metadata."""
    if len(prepared_list) == 1:
        # Single-row capture keeps the original call shape (and monkeypatch
        # seam) used by the staged offline path.
        prepared = prepared_list[0]
        extra = {"allow_multi_span": True} if prepared.get("allow_multi_span") else {}
        records = [
            forward_and_capture(
                model,
                prepared["inputs"],
                prepared["full_ids"],
                layer_ids,
                image_token_id,
                dtype,
                merge_size=merge_size,
                store_final_hidden=store_final_hidden,
                tokenizer=tokenizer,
                **extra,
            )
        ]
    else:
        records = forward_and_capture_batch(
            model,
            prepared_list,
            layer_ids,
            image_token_id,
            dtype,
            merge_size=merge_size,
            store_final_hidden=store_final_hidden,
            tokenizer=tokenizer,
        )
    for prepared, record in zip(prepared_list, records):
        if prepared.get("check_image_token_count") and (
            record["n_image_tokens_expected"] != record["n_image_tokens"]
        ):
            raise RuntimeError(
                f"Image token count mismatch for sample {prepared.get('sample_id')}: "
                f"found {record['n_image_tokens']} tokens in input_ids but grid_thw expects "
                f"{record['n_image_tokens_expected']}."
            )
        record.update(prepared["meta"])
    return records


def build_sample_record(
    model,
    processor,
    row: dict,
    media_dir: Path,
    *,
    layer_ids: list[int],
    image_token_id: int,
    merge_size: int | None,
    dtype: torch.dtype,
    seq_length: int,
    store_final_hidden: bool,
) -> dict:
    """One regenerated row -> the training record (offline capture and online share this).

    Dispatches on the row generation: schema-v2 rows (multi-image / multi-turn)
    replay the whole conversation render and build a multi-segment loss mask;
    v1 rows keep the original single-image, single-answer path bit-for-bit.
    ``prepare_sample_record`` + ``finalize_prepared_records`` are the same two
    halves split apart for batched online capture.
    """
    prepared = prepare_sample_record(processor, row, media_dir, seq_length=seq_length)
    return finalize_prepared_records(
        model,
        [prepared],
        layer_ids=layer_ids,
        image_token_id=image_token_id,
        merge_size=merge_size,
        dtype=dtype,
        store_final_hidden=store_final_hidden,
        tokenizer=getattr(processor, "tokenizer", None),
    )[0]
