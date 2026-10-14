"""Hard checks that the visual path is real (span, replay determinism, swap)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import row_image_path, row_image_paths, row_is_multi_turn, row_question
from ..utils import ensure_dir, log, write_json
from .prompts import encode_conversation, encode_prompt, messages_from_row
from .sample_capture import forward_and_capture


def _span_and_downstream_deltas(hidden_a: torch.Tensor, hidden_b: torch.Tensor, span: tuple[int, int]) -> dict:
    start, end = span
    span_a, span_b = hidden_a[start : end + 1].float(), hidden_b[start : end + 1].float()
    downstream_a, downstream_b = hidden_a[end + 1 :].float(), hidden_b[end + 1 :].float()

    def metrics(a, b):
        if a.numel() == 0 or b.numel() == 0:
            return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0, "min_cosine": 1.0, "changed_positions": 0}
        diff = (a - b).abs()
        per_position = diff.amax(dim=-1)
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return {
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "min_cosine": float(cos.min()),
            "changed_positions": int((per_position > 1e-2).sum()),
        }

    return {"span": metrics(span_a, span_b), "downstream": metrics(downstream_a, downstream_b)}


@torch.inference_mode()
def run_visual_path_check(
    cfg: dict,
    run_dir: str | Path,
    contract: dict,
    model,
    processor,
    rows: list[dict],
    media_dir: Path,
    dtype: torch.dtype,
    limit: int = 2,
    check_path: str | Path | None = None,
    pair_path: str | Path | None = None,
) -> dict:
    """Span/replay/swap checks; writes ``capture/visual_path_check.json`` (+ swap pair).

    ``check_path`` / ``pair_path`` default to the staged-capture locations; the
    online path passes ``run_dir/online/...`` so online diagnostics never read
    or overwrite an old staged capture's swap pair.
    """
    run_dir = Path(run_dir)
    check_path = Path(check_path) if check_path is not None else run_dir / "capture" / "visual_path_check.json"
    pair_path = Path(pair_path) if pair_path is not None else run_dir / "features" / "swap_pair.pt"
    ensure_dir(check_path.parent)
    ensure_dir(pair_path.parent)
    from ..contract import contract_capture_layer_ids

    layer_ids = contract_capture_layer_ids(contract)
    image_token_id = int(contract["tokens"]["image_token_id"])
    merge_size = contract["image"].get("merge_size")
    cfg_check = cfg.get("capture", {}).get("visual_check", {})
    min_downstream_changed = int(cfg_check.get("min_downstream_changed", 16))
    min_span_cosine_drop = float(cfg_check.get("min_span_cosine_drop", 1e-3))

    def capture_sample(row: dict, images) -> dict:
        """Capture one row with the provided image(s); v2 rows take a list of PIL images."""
        if row_is_multi_turn(row):
            regen = row["regen"]
            turns = regen["turns"]
            assistant_indices = [
                index
                for index, turn in enumerate(row.get("conversations") or [])
                if str(turn.get("from", "")).lower() in ("gpt", "assistant", "model")
            ]
            # Legacy records (no mode/assistant_indices) regenerated every turn;
            # last_turn_only records regenerate exactly the last one.
            regenerated_indices = regen.get("assistant_indices") or assistant_indices
            overrides = {
                msg_index: turn["answer_text"] for msg_index, turn in zip(regenerated_indices, turns)
            }
            messages, image_indices = messages_from_row(row, assistant_overrides=overrides)
            # ``images`` arrives in row order; the processor consumes them in
            # the conversation's appearance order (R4: interleaved rows may
            # place images in any order inside their turns).
            inputs = encode_conversation(processor, messages, [images[position] for position in image_indices])
            full_ids = torch.tensor([int(token) for token in row["regen"]["full_ids"]], dtype=torch.long)
            return forward_and_capture(
                model,
                inputs,
                full_ids,
                layer_ids,
                image_token_id,
                dtype,
                merge_size=merge_size,
                allow_multi_span=True,
                tokenizer=getattr(processor, "tokenizer", None),
            )
        inputs = encode_prompt(processor, row_question(row), images)
        prompt_ids = row["regen"]["prompt_ids"]
        full_ids = torch.tensor(prompt_ids + row["regen"]["answer_ids"], dtype=torch.long)
        return forward_and_capture(
            model,
            inputs,
            full_ids,
            layer_ids,
            image_token_id,
            dtype,
            merge_size=merge_size,
            tokenizer=getattr(processor, "tokenizer", None),
        )

    base_row = rows[0]
    swap_row = rows[1] if len(rows) > 1 else rows[0]
    multi_turn = row_is_multi_turn(base_row)
    if multi_turn:
        base_paths = row_image_paths(base_row, media_dir)
        swap_paths = row_image_paths(swap_row, media_dir)
        image_a = [Image.open(path).convert("RGB") for path in base_paths]
        image_b = [
            Image.open(swap_paths[index % len(swap_paths)]).convert("RGB").resize(image_a[index].size, Image.BICUBIC)
            for index in range(len(image_a))
        ]
    else:
        image_a = Image.open(row_image_path(base_row, media_dir)).convert("RGB")
        swap_image = Image.open(row_image_paths(swap_row, media_dir)[0]).convert("RGB")
        image_b = swap_image.resize(image_a.size, Image.BICUBIC)

    sample_a = capture_sample(base_row, image_a)
    replay = capture_sample(base_row, image_a)
    swap = capture_sample(base_row, image_b)

    bitwise_equal = bool(torch.equal(sample_a["hidden"], replay["hidden"]))
    replay_max_diff = float((sample_a["hidden"].float() - replay["hidden"].float()).abs().max())
    token_count_match = sample_a["n_image_tokens"] == swap["n_image_tokens"]
    span = tuple(sample_a["image_span"])

    if token_count_match:
        deltas = _span_and_downstream_deltas(sample_a["hidden"], swap["hidden"], span)
    else:
        deltas = None

    span_changed = (
        deltas is not None
        and not torch.allclose(
            sample_a["hidden"][span[0] : span[1] + 1].float(), swap["hidden"][span[0] : span[1] + 1].float()
        )
        and deltas["span"]["min_cosine"] < 1.0 - min_span_cosine_drop
    )
    downstream_changed = deltas is not None and deltas["downstream"]["changed_positions"] >= min_downstream_changed
    passed = bitwise_equal and token_count_match and span_changed and downstream_changed
    result = {
        "sample_id": base_row.get("id"),
        "swap_sample_id": swap_row.get("id"),
        "expected_image_tokens": sample_a["n_image_tokens_expected"],
        "found_image_tokens": sample_a["n_image_tokens"],
        "span_ok": sample_a["n_image_tokens_expected"] == sample_a["n_image_tokens"],
        "replay_bitwise_equal": bitwise_equal,
        "replay_max_abs_diff": replay_max_diff,
        "swap_token_count_match": token_count_match,
        "swap_deltas": deltas,
        "thresholds": {
            "min_downstream_changed": min_downstream_changed,
            "min_span_cosine_drop": min_span_cosine_drop,
        },
        "passed": passed,
    }
    write_json(check_path, result)
    if token_count_match:
        if multi_turn:
            keep = int(sample_a["input_ids"].numel())
            loss_mask = torch.zeros(keep, dtype=torch.bool)
            for start, end in base_row["regen"]["answer_spans"]:
                loss_mask[min(int(start), keep) : min(int(end), keep)] = True
        else:
            loss_mask = torch.cat(
                (
                    torch.zeros(len(base_row["regen"]["prompt_ids"]), dtype=torch.bool),
                    torch.ones(len(base_row["regen"]["answer_ids"]), dtype=torch.bool),
                )
            )
        pair_payload = {
            "format": "vlm-dflash-swap-pair-v1",
            "input_ids": sample_a["input_ids"],
            "loss_mask": loss_mask,
            "position3": sample_a["position3"],
            "image_span": list(span),
            "hidden_a": sample_a["hidden"],
            "hidden_b": swap["hidden"],
        }
        if multi_turn:
            pair_payload["image_spans"] = [list(value) for value in sample_a["image_spans"]]
        torch.save(
            pair_payload,
            pair_path,
        )
    log(f"visual_path_check: passed={passed} replay_bitwise={bitwise_equal} span_changed={span_changed} "
        f"downstream_changed={downstream_changed}")
    return result
