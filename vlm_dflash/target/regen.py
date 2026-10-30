"""Target-native regeneration of assistant answers (greedy, batch).

The regenerated answers (token ids included) become the supervision targets for
the drafter. Storing the exact token ids here is what makes the later capture
step bit-exact: capture re-renders the prompt and asserts it matches.

Schema-v2 rows (multi-image / multi-turn) support two supervision modes
(``regen.multi_turn_mode``):

* ``last_turn_only`` (default): the dataset's own earlier assistant turns are
  kept verbatim as context and only the last assistant turn is regenerated and
  supervised -- eval renders the same original history, so train and eval
  follow one protocol.
* ``all_turns`` (opt-in, experimental): every assistant turn is regenerated
  progressively, conditioned on the already-generated earlier turns (never on
  the dataset's own assistant text), and every regenerated answer is
  supervised. Eval still renders the dataset history, so all-turn artifacts
  are only comparable within their own protocol.

The per-turn token ids are concatenated into ``regen.full_ids`` and checked
against one full-conversation render: any chat-template drift fails loud,
exactly like the v1 prompt-drift check. Legacy regen records without a
recorded ``mode`` stay identifiable as all-turn artifacts and keep rendering
through the legacy path.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import (
    load_sharegpt,
    row_image_path,
    row_image_paths,
    row_image_turns,
    row_is_multi_turn,
    row_question,
)
from ..provenance import identity_from_contract, record_stage
from ..utils import ensure_dir, log, sha256_text, write_json, write_jsonl
from .load import load_target
from .prompts import (
    encode_prompt,
    messages_from_row,
    multimodal_token_ids,
    render_conversation,
    render_prompt,
    same_text_render,
)

_HUMAN_ROLES = ("human", "user")
_GPT_ROLES = ("gpt", "assistant", "model")

#: Supported multi-turn supervision modes (see the module docstring).
MULTI_TURN_MODES = ("last_turn_only", "all_turns")
DEFAULT_MULTI_TURN_MODE = "last_turn_only"


def _length_summary(lengths: list[int], cap: int | None) -> dict:
    """Sorted length summary (min/p10/median/p90/max) + cap-hit counts."""
    if not lengths:
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    ordered = sorted(lengths)

    def quantile(q: float) -> int:
        position = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[position]

    summary = {
        "count": len(ordered),
        "min": ordered[0],
        "p10": quantile(0.1),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }
    if cap is not None:
        summary["cap"] = int(cap)
        summary["at_cap"] = sum(1 for value in ordered if value >= cap)
        summary["at_cap_fraction"] = summary["at_cap"] / len(ordered)
    return summary


def _cut_at_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[: index + 1]
    return ids


def _regen_single_turn_rows(
    model,
    processor,
    rows: list[dict],
    media_dir: Path,
    *,
    stop_ids: set[int],
    device,
    max_new_tokens: int,
    min_response_tokens: int,
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    """v1 batch path: one image, one greedy answer per row. Unchanged semantics."""
    out_rows, skipped = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images, texts = [], []
        for row in batch:
            image = Image.open(row_image_path(row, media_dir)).convert("RGB")
            images.append(image)
            texts.append(render_prompt(processor, row_question(row)))
        from ..adapters import adapter_for_model

        inputs = processor(
            text=texts,
            images=adapter_for_model(model).batch_processor_images(images),
            padding=True,
            return_tensors="pt",
        ).to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
            )
        for offset, row in enumerate(batch):
            attention = inputs["attention_mask"][offset]
            unpadded_prompt_ids = inputs["input_ids"][offset][attention.bool()].tolist()
            answer_ids = _cut_at_stop(generated[offset, prompt_len:].tolist(), stop_ids)
            if len(answer_ids) < min_response_tokens:
                skipped.append({"id": row.get("id"), "reason": "too_short", "answer_tokens": len(answer_ids)})
                continue
            answer_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
            out_rows.append(
                {
                    "id": row.get("id"),
                    "conversations": [
                        {"from": "human", "value": row_question(row)},
                        {"from": "gpt", "value": answer_text},
                    ],
                    "images": row["images"],
                    "regen": {
                        "prompt_ids": unpadded_prompt_ids,
                        "answer_ids": answer_ids,
                        "stop_token_id": answer_ids[-1] if answer_ids and answer_ids[-1] in stop_ids else None,
                    },
                }
            )
        log(f"regen: processed {min(start + batch_size, len(rows))}/{len(rows)} kept={len(out_rows)}")
    return out_rows, skipped


def _assistant_turn_indices(conversations: list[dict], sample_id) -> list[int]:
    indices = []
    for index, turn in enumerate(conversations):
        role = str(turn.get("from", "")).lower()
        if role in _GPT_ROLES:
            indices.append(index)
        elif role not in _HUMAN_ROLES:
            raise ValueError(f"Row {sample_id!r} has unsupported role {role!r} at turn {index}.")
    if not indices:
        raise ValueError(f"Row {sample_id!r} has no assistant turn.")
    return indices


def _verify_full_render(processor, row: dict, images: list, full_ids: list[int], overrides: dict[int, str]) -> None:
    """Assert ``full_ids`` is one full-conversation chat-template render.

    A token prefix mismatch whose decoded text is identical is a tokenizer
    non-roundtrip split of a generated answer (see ``same_text_render``) and is
    accepted; any other mismatch is chat-template drift and fails loud.
    Non-empty trailing text after ``full_ids`` is refused so a stored sequence
    can never silently drop part of the conversation.
    """
    full_messages, full_image_indices = messages_from_row(row, assistant_overrides=overrides)
    full_text = render_conversation(processor, full_messages, add_generation_prompt=False)
    full_inputs = processor(
        text=[full_text], images=[images[position] for position in full_image_indices], return_tensors="pt"
    )
    rendered = full_inputs["input_ids"][0].tolist()
    if rendered[: len(full_ids)] != full_ids:
        if same_text_render(
            processor.tokenizer, full_ids, rendered, protected_ids=multimodal_token_ids(processor, full_inputs)
        ):
            return
        raise RuntimeError(
            f"Multi-turn full-render drift for sample {row.get('id')}: the incremental token ids do not "
            "match one apply_chat_template render of the whole conversation; refusing to store them."
        )
    tail = rendered[len(full_ids) :]
    if tail and processor.tokenizer.decode(tail, skip_special_tokens=False).strip():
        raise RuntimeError(
            f"Multi-turn render of sample {row.get('id')} produced unexpected trailing tokens {tail!r}."
        )


def _regen_multi_turn_row(
    model,
    processor,
    row: dict,
    media_dir: Path,
    *,
    stop_ids: set[int],
    primary_stop: int,
    device,
    max_new_tokens: int,
    mode: str = DEFAULT_MULTI_TURN_MODE,
) -> tuple[dict | None, dict | None, list[int]]:
    """Greedy regen for one schema-v2 row under the selected supervision mode.

    Returns ``(record, skipped_or_None, per_turn_answer_tokens)``. A row with
    an empty generated answer is skipped (``skipped`` non-null). Answers that
    hit ``max_new_tokens`` without a stop token get the primary stop appended so
    the stored sequence is exactly a valid chat-template render (recorded as
    ``forced_stop`` in the per-turn stats). ``per_turn_answer_tokens`` lists the
    regenerated answers only (one entry under ``last_turn_only``).
    """
    if mode not in MULTI_TURN_MODES:
        raise ValueError(f"Unknown regen.multi_turn_mode {mode!r}; expected one of {list(MULTI_TURN_MODES)}.")
    conversations = row.get("conversations") or []
    assistant_indices = _assistant_turn_indices(conversations, row.get("id"))
    image_paths = row_image_paths(row, media_dir)
    images = [Image.open(path).convert("RGB") for path in image_paths]

    if mode == "last_turn_only":
        target_index = assistant_indices[-1]
        # Original history (dataset assistant texts, never regenerated) plus the
        # final user turn; the generation prompt closes it.
        messages, image_indices = messages_from_row(row, upto=target_index)
        text = render_conversation(processor, messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[images[position] for position in image_indices], return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0].tolist()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
            )
        answer_ids = _cut_at_stop(generated[0, len(prompt_ids) :].tolist(), stop_ids)
        if not answer_ids:
            return None, {"id": row.get("id"), "reason": "empty_answer", "turn": target_index}, []
        forced = answer_ids[-1] not in stop_ids
        if forced:
            answer_ids = answer_ids + [primary_stop]
        answer_text = processor.tokenizer.decode(answer_ids[:-1], skip_special_tokens=False)
        full_ids = prompt_ids + answer_ids
        regen = {
            "mode": "last_turn_only",
            "assistant_indices": [target_index],
            "turns": [
                {
                    "prompt_ids": prompt_ids,
                    "answer_ids": answer_ids,
                    "answer_text": answer_text,
                    "stop_token_id": int(answer_ids[-1]),
                    "forced_stop": bool(forced),
                }
            ],
            "full_ids": full_ids,
            "answer_spans": [[len(prompt_ids), len(full_ids)]],
            "n_turns": 1,
            "forced_stop": int(forced),
        }
        record = {
            "id": row.get("id"),
            "conversations": conversations,
            "images": row["images"],
            "image_turns": row_image_turns(row),
            "schema": "vlm-dflash-data-v2",
            "regen": regen,
        }
        _verify_full_render(processor, row, images, full_ids, {target_index: answer_text})
        return record, None, [len(answer_ids)]

    turns, answer_spans, overrides = [], [], {}
    full_ids: list[int] = []
    forced_stop = 0
    for msg_index in assistant_indices:
        # Everything before the assistant turn being generated (earlier turns
        # use the already-generated text, never the dataset's own answers).
        messages, image_indices = messages_from_row(row, assistant_overrides=overrides, upto=msg_index)
        used_images = [images[position] for position in image_indices]
        text = render_conversation(processor, messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=used_images, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"][0].tolist()
        if prompt_ids[: len(full_ids)] != full_ids:
            raise RuntimeError(
                f"Multi-turn prompt drift between regen steps for sample {row.get('id')}: the chat "
                "template does not reproduce the committed prefix; refusing to store misaligned ids."
            )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
            )
        answer_ids = _cut_at_stop(generated[0, len(prompt_ids) :].tolist(), stop_ids)
        if not answer_ids:
            return None, {"id": row.get("id"), "reason": "empty_answer", "turn": msg_index}, []
        forced = answer_ids[-1] not in stop_ids
        if forced:
            answer_ids = answer_ids + [primary_stop]
            forced_stop += 1
        full_ids.extend(prompt_ids[len(full_ids) :])
        full_ids.extend(answer_ids)
        answer_spans.append([len(full_ids) - len(answer_ids), len(full_ids)])
        answer_text = processor.tokenizer.decode(answer_ids[:-1], skip_special_tokens=False)
        overrides[msg_index] = answer_text
        turns.append(
            {
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "answer_text": answer_text,
                "stop_token_id": int(answer_ids[-1]),
                "forced_stop": bool(forced),
            }
        )
    _verify_full_render(processor, row, images, full_ids, overrides)
    record = {
        "id": row.get("id"),
        "conversations": conversations,
        "images": row["images"],
        "image_turns": row_image_turns(row),
        "schema": "vlm-dflash-data-v2",
        "regen": {
            "mode": "all_turns",
            "assistant_indices": list(assistant_indices),
            "turns": turns,
            "full_ids": full_ids,
            "answer_spans": answer_spans,
            "n_turns": len(turns),
            "forced_stop": forced_stop,
        },
    }
    return record, None, [len(turn["answer_ids"]) for turn in turns]


def _template_sha(processor, row: dict) -> str:
    if row_is_multi_turn(row):
        messages, _ = messages_from_row(row)
        return sha256_text(render_conversation(processor, messages, add_generation_prompt=False))
    return sha256_text(render_prompt(processor, row_question(row)))


def run_regen(cfg: dict, run_dir: str | Path, contract: dict, limit: int | None = None) -> dict:
    run_dir = Path(run_dir)
    regen_cfg = cfg.get("regen", {})
    max_new_tokens = int(regen_cfg.get("max_new_tokens", 160))
    min_response_tokens = int(regen_cfg.get("min_response_tokens", 128))
    batch_size = int(regen_cfg.get("batch_size", 8))
    multi_turn_mode = str(regen_cfg.get("multi_turn_mode", DEFAULT_MULTI_TURN_MODE)).strip().lower()
    if multi_turn_mode not in MULTI_TURN_MODES:
        raise ValueError(
            f"regen.multi_turn_mode must be one of {list(MULTI_TURN_MODES)}, got {multi_turn_mode!r}."
        )
    rows, media_dir = load_sharegpt(run_dir / "data" / "data.jsonl")
    if limit is not None:
        rows = rows[:limit]

    model, processor = load_target(cfg, contract)
    device = next(model.parameters()).device
    eos = getattr(model.generation_config, "eos_token_id", None)
    stop_id_list = list(eos) if isinstance(eos, (list, tuple)) else ([eos] if eos is not None else [])
    stop_ids = set(stop_id_list)
    primary_stop = stop_id_list[0] if stop_id_list else None

    out_rows, skipped = [], []
    v2_count = 0
    v2_skipped: list[dict] = []
    v2_totals: list[int] = []
    v2_turns_count: list[int] = []
    v2_forced_stop = 0
    per_turn_tokens: list[list[int]] = []
    buffer: list[dict] = []

    def flush_single_turn() -> None:
        nonlocal buffer
        if not buffer:
            return
        kept, dropped = _regen_single_turn_rows(
            model,
            processor,
            buffer,
            media_dir,
            stop_ids=stop_ids,
            device=device,
            max_new_tokens=max_new_tokens,
            min_response_tokens=min_response_tokens,
            batch_size=batch_size,
        )
        out_rows.extend(kept)
        skipped.extend(dropped)
        buffer = []

    log(
        f"regen: {len(rows)} samples, batch={batch_size}, max_new={max_new_tokens}, "
        f"min_answer={min_response_tokens}, multi_turn_mode={multi_turn_mode}"
    )
    for row in rows:
        if not row_is_multi_turn(row):
            buffer.append(row)
            continue
        flush_single_turn()
        v2_count += 1
        if primary_stop is None:
            raise RuntimeError("Multi-turn regen needs a primary eos token, but the target has none.")
        record, drop, turn_tokens = _regen_multi_turn_row(
            model,
            processor,
            row,
            media_dir,
            stop_ids=stop_ids,
            primary_stop=primary_stop,
            device=device,
            max_new_tokens=max_new_tokens,
            mode=multi_turn_mode,
        )
        if drop is not None:
            v2_skipped.append(drop)
            skipped.append(drop)
            continue
        total = sum(turn_tokens)
        if total < min_response_tokens:
            dropped = {"id": row.get("id"), "reason": "too_short", "answer_tokens": total}
            v2_skipped.append(dropped)
            skipped.append(dropped)
            continue
        out_rows.append(record)
        v2_totals.append(total)
        v2_turns_count.append(int(record["regen"]["n_turns"]))
        v2_forced_stop += int(record["regen"]["forced_stop"])
        per_turn_tokens.append(turn_tokens)
        log(f"regen: kept multi-turn sample {row.get('id')} ({record['regen']['n_turns']} turns, {total} answer tokens)")
    flush_single_turn()

    regen_dir = ensure_dir(run_dir / "regen")
    write_jsonl(regen_dir / "target_data.jsonl", out_rows)
    write_jsonl(regen_dir / "skipped.jsonl", skipped)
    v1_rows = [row for row in out_rows if "turns" not in row["regen"]]
    stats = {
        "num_input": len(rows),
        "num_kept": len(out_rows),
        "num_skipped": len(skipped),
        "skip_reasons": {"too_short": sum(1 for entry in skipped if entry["reason"] == "too_short")},
        "max_new_tokens": max_new_tokens,
        "min_response_tokens": min_response_tokens,
        "batch_size": batch_size,
        "greedy": True,
        # Effective token-selection protocol, pinned explicitly at every
        # generate() call site (the target's generation_config.json otherwise
        # leaks repetition_penalty=1.05 / do_sample=true into these calls).
        "generation_protocol": {
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "temperature": None,
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": sorted(stop_ids),
            "pad_token_id": getattr(getattr(model, "generation_config", None), "pad_token_id", None),
        },
        # Supervision protocol for schema-v2 rows (data records carry the same
        # value per row; legacy records without a mode are all-turn artifacts).
        "multi_turn_mode": multi_turn_mode,
        "answer_token_histogram": _length_summary(
            [len(r["regen"]["answer_ids"]) for r in v1_rows], cap=max_new_tokens
        ),
        "prompt_token_histogram": _length_summary(
            [len(r["regen"]["prompt_ids"]) for r in v1_rows], cap=None
        ),
        "prompt_template_sha256": _template_sha(processor, rows[0]) if rows else None,
    }
    if v2_count:
        max_turn_slots = max((len(entry) for entry in per_turn_tokens), default=0)
        stats["skip_reasons"].setdefault("empty_answer", 0)
        stats["skip_reasons"]["empty_answer"] = sum(1 for entry in skipped if entry["reason"] == "empty_answer")
        stats["multi_turn"] = {
            "mode": multi_turn_mode,
            "num_input": v2_count,
            "num_kept": len(v2_totals),
            "num_skipped": len(v2_skipped),
            # Regenerated (and, under last_turn_only, supervised) answers per row.
            "num_turns": {
                "min": min(v2_turns_count, default=None),
                "max": max(v2_turns_count, default=None),
            },
            "answer_tokens_total": {
                "min": min(v2_totals, default=None),
                "max": max(v2_totals, default=None),
            },
            "forced_stop_answers": v2_forced_stop,
            "per_turn": [
                {
                    "turn": turn,
                    "min_answer_tokens": min((entry[turn] for entry in per_turn_tokens if len(entry) > turn), default=None),
                    "max_answer_tokens": max((entry[turn] for entry in per_turn_tokens if len(entry) > turn), default=None),
                }
                for turn in range(max_turn_slots)
            ],
        }
    write_json(regen_dir / "regen_stats.json", stats)
    record_stage(
        run_dir,
        "regen",
        {
            "identity": identity_from_contract(contract),
            "stats": stats,
            "code": "vlm_dflash.target.regen.run_regen",
        },
    )
    if not out_rows:
        raise RuntimeError("Regen kept zero samples; check max_new_tokens/min_response_tokens.")
    return stats
