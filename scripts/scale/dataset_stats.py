#!/usr/bin/env python
"""Compile the R3/R5 dataset statistics artifact (data_stats.json).

Covers the round's data contract: input/kept/dropped counts, independent image
counts and train/dev/test image intersection, effective supervision tokens
(unique, not repeated epochs), answer-length distribution, cap-hit fraction,
sequence-length truncation, image pixel-band classification and content
hashes. Reads only artifacts already under the run dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_dflash.utils import read_json, read_jsonl, sha256_file, write_json


def _percentile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    position = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[position]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import yaml

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = read_json(run_dir / "data" / "manifest.json")
    regen_stats = read_json(run_dir / "regen" / "regen_stats.json")
    freeze = read_json(run_dir / "regen" / "freeze_record.json")
    frozen = read_jsonl(run_dir / "regen" / "target_data.jsonl")

    answer_lengths = sorted(len(row["regen"]["answer_ids"]) for row in frozen)
    prompt_lengths = sorted(len(row["regen"].get("prompt_ids", [])) for row in frozen)
    max_new = int(regen_stats["max_new_tokens"])
    answer_tokens = sum(answer_lengths)
    prompt_tokens = sum(prompt_lengths)
    over_seq = sum(
        1
        for row in frozen
        if len(row["regen"]["answer_ids"]) + len(row["regen"].get("prompt_ids", [])) > int(cfg["capture"]["seq_length"])
    )

    # image pixel band classification over the frozen rows (raw pixel heuristic;
    # the processor smart-resizes into [min_pixels, max_pixels]).
    min_px = int(cfg["target"]["image_min_pixels"])
    max_px = int(cfg["target"]["image_max_pixels"])
    media_dir = run_dir / "data" / "images"
    from PIL import Image

    below = above = in_band = 0
    for row in frozen:
        image_name = row["images"][0] if isinstance(row["images"], list) else row["images"]
        with Image.open(media_dir / image_name) as image:
            pixels = image.size[0] * image.size[1]
        if pixels < min_px:
            below += 1
        elif pixels > max_px:
            above += 1
        else:
            in_band += 1

    def file_sha(path: Path) -> str | None:
        return sha256_file(path) if path.exists() else None

    stats = {
        "created_freeze": freeze["created"],
        "dataset": {
            "config": {k: cfg["dataset"].get(k) for k in ("source", "slice", "val_size", "test_size", "seed", "min_answer_chars", "image_link_mode")},
            "num_pool": manifest.get("num_pool"),
            "num_train_selected": manifest["num_train"],
            "num_val": manifest["num_val"],
            "num_test": manifest["num_test"],
            "num_groups": manifest.get("num_groups"),
            "num_images": manifest.get("num_images"),
            "image_overlap": manifest.get("image_overlap"),
            "rows_sha256": manifest.get("rows_sha256"),
            "val_sha256": file_sha(run_dir / "data" / "val.jsonl"),
            "test_sha256": file_sha(run_dir / "data" / "test.jsonl"),
            "data_sha256": file_sha(run_dir / "data" / "data.jsonl"),
        },
        "regen": {
            "num_input": regen_stats["num_input"],
            "num_kept": regen_stats["num_kept"],
            "num_skipped": regen_stats["num_skipped"],
            "skip_reasons": regen_stats["skip_reasons"],
            "generation_protocol": regen_stats.get("generation_protocol"),
            "batch_size": regen_stats["batch_size"],
            "multi_turn_mode": regen_stats.get("multi_turn_mode"),
            "answer_token_histogram": regen_stats["answer_token_histogram"],
            "prompt_token_histogram": regen_stats.get("prompt_token_histogram"),
        },
        "freeze": freeze,
        "frozen_train": {
            "num_rows": len(frozen),
            "answer_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "answer_len": {
                "min": answer_lengths[0],
                "p10": _percentile(answer_lengths, 0.1),
                "median": _percentile(answer_lengths, 0.5),
                "p90": _percentile(answer_lengths, 0.9),
                "max": answer_lengths[-1],
                "mean": answer_tokens / len(answer_lengths),
                "at_cap": sum(1 for value in answer_lengths if value >= max_new),
            },
            "prompt_len": {
                "min": prompt_lengths[0],
                "median": _percentile(prompt_lengths, 0.5),
                "p90": _percentile(prompt_lengths, 0.9),
                "max": prompt_lengths[-1],
                "mean": prompt_tokens / len(prompt_lengths),
            },
            "rows_over_seq_length": over_seq,
            "rows_over_seq_fraction": over_seq / max(1, len(frozen)),
            "distinct_images": len({tuple(row["images"]) if isinstance(row["images"], list) else row["images"] for row in frozen}),
        },
        "image_bands": {
            "min_pixels": min_px,
            "max_pixels": max_px,
            "below_min_upscaled": below,
            "above_max_downscaled": above,
            "in_band": in_band,
            "note": "raw original pixels vs configured band; the processor resizes every image into the band",
        },
        "supervision_accounting": {
            "unique_supervised_answer_tokens": answer_tokens,
            "epochs_planned": 3,
            "note": "repeated epochs re-present the same tokens; they are NOT new independent supervision",
        },
    }
    out_path = run_dir / "data_stats.json"
    write_json(out_path, stats)
    print(json.dumps({k: stats[k] for k in ("dataset", "frozen_train", "image_bands")}, indent=2)[:4000])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
