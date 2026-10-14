#!/usr/bin/env python3
"""Generate the synthetic schema-v2 dataset used by the WP2 multi-turn smoke (E2).

Writes ``source.jsonl`` plus an ``images/`` directory of simple coloured
shapes: every row is a 2-image / 2-turn conversation (image_turns ``[0, 2]``).
The dataset assistant texts describe the drawn shapes faithfully; the regen
stage still replaces them with the frozen target's own greedy answers on the
training split, while the held-out split keeps them as the eval history (so
multi-turn eval prompts carry a realistic first answer, not a placeholder).

Usage: ``python scripts/make_wp2_smoke_data.py --out data/e2_smoke [--train 16 --val 4]``
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

_COLORS = [
    (196, 60, 60),
    (60, 140, 196),
    (70, 170, 90),
    (200, 160, 40),
    (150, 90, 190),
    (60, 180, 180),
]
_SHAPES = ("circle", "square", "triangle", "diamond")

_QUESTION_FIRST = "Describe the main shape in the first image in one short sentence."
_QUESTION_SECOND = "Describe the main shape in the second image and mention its colour in one complete sentence."

_COLOR_NAMES = {
    (196, 60, 60): "red",
    (60, 140, 196): "blue",
    (70, 170, 90): "green",
    (200, 160, 40): "yellow",
    (150, 90, 190): "purple",
    (60, 180, 180): "teal",
}


def _draw(shape: str, color: tuple[int, int, int], side: int) -> Image.Image:
    image = Image.new("RGB", (side, side), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    margin = side // 6
    box = (margin, margin, side - margin, side - margin)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(side // 2, margin), (side - margin, side - margin), (margin, side - margin)], fill=color)
    else:  # diamond
        draw.polygon([(side // 2, margin), (side - margin, side // 2), (side // 2, side - margin), (margin, side // 2)], fill=color)
    return image


def build(out_dir: Path, train: int, val: int, seed: int, side: int) -> dict:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for index in range(train + val):
        split = "train" if index < train else "val"
        sample_id = f"e2_{split}_{index:04d}"
        shape_a, shape_b = rng.sample(_SHAPES, 2)
        color_a, color_b = rng.sample(_COLORS, 2)
        names = [f"{sample_id}_a.png", f"{sample_id}_b.png"]
        _draw(shape_a, color_a, side).save(images_dir / names[0])
        _draw(shape_b, color_b, side).save(images_dir / names[1])
        answer_a = f"The first image shows a {_COLOR_NAMES[color_a]} {shape_a} on a light background."
        answer_b = f"The second image shows a {_COLOR_NAMES[color_b]} {shape_b} on a light background."
        rows.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{_QUESTION_FIRST}"},
                    {"from": "gpt", "value": answer_a},
                    {"from": "human", "value": f"<image>\n{_QUESTION_SECOND}"},
                    {"from": "gpt", "value": answer_b},
                ],
                "images": names,
                "image_turns": [0, 2],
                "schema": "vlm-dflash-data-v2",
            }
        )
    with (out_dir / "source.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"num_rows": len(rows), "num_train": train, "num_val": val, "out_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side", type=int, default=320)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.train, args.val, args.seed, args.side), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
