"""Target-native regeneration of assistant answers (greedy, batch).

The regenerated answers (token ids included) become the supervision targets for
the drafter. Storing the exact token ids here is what makes the later capture
step bit-exact: capture re-renders the prompt and asserts it matches.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import load_sharegpt, row_answer, row_image_path, row_question
from ..provenance import identity_from_contract, record_stage
from ..utils import ensure_dir, log, sha256_text, write_json, write_jsonl
from .load import load_target
from .prompts import encode_prompt, render_prompt


def _cut_at_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[: index + 1]
    return ids


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
    stop_ids = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is not None:
        stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    out_rows, skipped = [], []
    log(f"regen: {len(rows)} samples, batch={batch_size}, max_new={max_new_tokens}, min_answer={min_response_tokens}")
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

    regen_dir = ensure_dir(run_dir / "regen")
    write_jsonl(regen_dir / "target_data.jsonl", out_rows)
    write_jsonl(regen_dir / "skipped.jsonl", skipped)
    stats = {
        "num_input": len(rows),
        "num_kept": len(out_rows),
        "num_skipped": len(skipped),
        "skip_reasons": {"too_short": len(skipped)},
        "max_new_tokens": max_new_tokens,
        "min_response_tokens": min_response_tokens,
        "batch_size": batch_size,
        "greedy": True,
        "answer_token_histogram": {
            "min": min((len(r["regen"]["answer_ids"]) for r in out_rows), default=None),
            "max": max((len(r["regen"]["answer_ids"]) for r in out_rows), default=None),
        },
        "prompt_template_sha256": (
            sha256_text(render_prompt(processor, row_question(rows[0]))) if rows else None
        ),
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
