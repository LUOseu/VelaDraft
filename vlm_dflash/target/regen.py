"""Target-native regeneration of assistant answers (greedy, batch).

The regenerated answers (token ids included) become the supervision targets for
the drafter. Storing the exact token ids here is what makes the later capture
step bit-exact: capture re-renders the prompt and asserts it matches.

Schema-v2 rows (multi-image / multi-turn) generate each assistant turn
progressively, conditioned on the already-generated earlier turns (never on
the dataset's own assistant text). The per-turn token ids are concatenated
into ``regen.full_ids`` and checked against one full-conversation render:
any chat-template drift fails loud, exactly like the v1 prompt-drift check.
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
from .prompts import encode_prompt, messages_from_row, render_conversation, render_prompt

_HUMAN_ROLES = ("human", "user")
_GPT_ROLES = ("gpt", "assistant", "model")


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
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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
) -> tuple[dict | None, dict | None, list[int]]:
    """Progressive greedy regen for one schema-v2 row.

    Returns ``(record, skipped_or_None, per_turn_answer_tokens)``. A row with
    an empty generated answer is skipped (``skipped`` non-null). Answers that
    hit ``max_new_tokens`` without a stop token get the primary stop appended so
    the stored sequence is exactly a valid chat-template render (recorded as
    ``forced_stop`` in the per-turn stats).
    """
    conversations = row.get("conversations") or []
    assistant_indices = _assistant_turn_indices(conversations, row.get("id"))
    image_paths = row_image_paths(row, media_dir)
    images = [Image.open(path).convert("RGB") for path in image_paths]
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
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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
    full_messages, full_image_indices = messages_from_row(row, assistant_overrides=overrides)
    full_text = render_conversation(processor, full_messages, add_generation_prompt=False)
    full_inputs = processor(text=[full_text], images=[images[position] for position in full_image_indices], return_tensors="pt")
    rendered = full_inputs["input_ids"][0].tolist()
    if rendered[: len(full_ids)] != full_ids:
        raise RuntimeError(
            f"Multi-turn full-render drift for sample {row.get('id')}: the incremental token ids do not "
            "match one apply_chat_template render of the whole conversation; refusing to store them."
        )
    tail = rendered[len(full_ids) :]
    if tail and processor.tokenizer.decode(tail, skip_special_tokens=False).strip():
        raise RuntimeError(
            f"Multi-turn render of sample {row.get('id')} produced unexpected trailing tokens {tail!r}."
        )
    record = {
        "id": row.get("id"),
        "conversations": conversations,
        "images": row["images"],
        "image_turns": row_image_turns(row),
        "schema": "vlm-dflash-data-v2",
        "regen": {
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
    rows, media_dir = load_sharegpt(run_dir / "data" / "data.jsonl")
    if limit is not None:
        rows = rows[:limit]
    regen_cfg = cfg.get("regen", {})
    max_new_tokens = int(regen_cfg.get("max_new_tokens", 160))
    min_response_tokens = int(regen_cfg.get("min_response_tokens", 128))
    batch_size = int(regen_cfg.get("batch_size", 8))

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

    log(f"regen: {len(rows)} samples, batch={batch_size}, max_new={max_new_tokens}, min_answer={min_response_tokens}")
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
        "answer_token_histogram": {
            "min": min((len(r["regen"]["answer_ids"]) for r in v1_rows), default=None),
            "max": max((len(r["regen"]["answer_ids"]) for r in v1_rows), default=None),
        },
        "prompt_template_sha256": _template_sha(processor, rows[0]) if rows else None,
    }
    if v2_count:
        max_turn_slots = max((len(entry) for entry in per_turn_tokens), default=0)
        stats["skip_reasons"].setdefault("empty_answer", 0)
        stats["skip_reasons"]["empty_answer"] = sum(1 for entry in skipped if entry["reason"] == "empty_answer")
        stats["multi_turn"] = {
            "num_input": v2_count,
            "num_kept": len(v2_totals),
            "num_skipped": len(v2_skipped),
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
