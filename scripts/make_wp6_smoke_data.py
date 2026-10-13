#!/usr/bin/env python3
"""Generate the synthetic single-image dataset used by the WP6 family smoke.

Writes ``source.jsonl`` plus an ``images/`` directory of simple coloured shapes:
every row is a single-image / single-turn conversation (image_turns ``[0]``).
The described shapes/colours are faithful, but the regen stage still replaces
the training-split answers with the frozen target's own greedy answers; the
held-out split keeps the written answers as eval history.

Usage: ``python scripts/make_wp6_smoke_data.py --out data/e6_smoke [--train 16 --val 4]``
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
_COLOR_NAMES = {
    (196, 60, 60): "red",
    (60, 140, 196): "blue",
    (70, 170, 90): "green",
    (200, 160, 40): "yellow",
    (150, 90, 190): "purple",
    (60, 180, 180): "teal",
}
_QUESTION = "What shape is in the image? Answer with its colour and shape in one short sentence."


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
        sample_id = f"e6_{split}_{index:04d}"
        shape = rng.choice(_SHAPES)
        color = rng.choice(_COLORS)
        name = f"{sample_id}.png"
        _draw(shape, color, side).save(images_dir / name)
        answer = f"The image shows a {_COLOR_NAMES[color]} {shape} on a light background."
        rows.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{_QUESTION}"},
                    {"from": "gpt", "value": answer},
                ],
                "images": [name],
                "image_turns": [0],
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
