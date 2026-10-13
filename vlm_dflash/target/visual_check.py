"""Hard checks that the visual path is real (span, replay determinism, swap)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.sharegpt import row_answer, row_image_path, row_question
from ..utils import ensure_dir, log, write_json
from .prompts import encode_prompt
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
    merge_size = int(contract["image"]["merge_size"])
    cfg_check = cfg.get("capture", {}).get("visual_check", {})
    min_downstream_changed = int(cfg_check.get("min_downstream_changed", 16))
    min_span_cosine_drop = float(cfg_check.get("min_span_cosine_drop", 1e-3))

    def capture_sample(row: dict, image) -> dict:
        inputs = encode_prompt(processor, row_question(row), image)
        prompt_ids = row["regen"]["prompt_ids"]
        full_ids = torch.tensor(prompt_ids + row["regen"]["answer_ids"], dtype=torch.long)
        return forward_and_capture(model, inputs, full_ids, layer_ids, image_token_id, dtype, merge_size=merge_size)

    base_row = rows[0]
    swap_row = rows[1] if len(rows) > 1 else rows[0]
    image_a = Image.open(row_image_path(base_row, media_dir)).convert("RGB")
    image_b = Image.open(row_image_path(swap_row, media_dir)).convert("RGB").resize(image_a.size, Image.BICUBIC)

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
        torch.save(
            {
                "format": "vlm-dflash-swap-pair-v1",
                "input_ids": sample_a["input_ids"],
                "loss_mask": torch.cat(
                    (
                        torch.zeros(len(base_row["regen"]["prompt_ids"]), dtype=torch.bool),
                        torch.ones(len(base_row["regen"]["answer_ids"]), dtype=torch.bool),
                    )
                ),
                "position3": sample_a["position3"],
                "image_span": list(span),
                "hidden_a": sample_a["hidden"],
                "hidden_b": swap["hidden"],
            },
            pair_path,
        )
    log(f"visual_path_check: passed={passed} replay_bitwise={bitwise_equal} span_changed={span_changed} "
        f"downstream_changed={downstream_changed}")
    return result
