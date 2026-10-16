#!/usr/bin/env python
"""Materialise a contiguous slice of a prepared data stage as its own run-input.

Regenerating 24k rows in one process is a 2-4h single point of failure, so the
regen is run as two (or more) independent processes in disjoint run dirs. This
helper builds the input side of one such piece:

* ``<out>/data/data.jsonl`` -- rows ``[offset, offset+limit)`` in the original
  order (so concatenating the pieces' regen outputs in piece order reproduces a
  single-pass row order);
* ``<out>/data/images/*``    -- hardlinks to the shared media dir (0 extra bytes);
* ``<out>/data/meta.json``   -- copied verbatim;
* ``<out>/data/manifest.json`` -- the same manifest with ``rows`` filtered to
  this piece.

Nothing else is written; no regen/capture stage runs here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="prepared stage dir with data.jsonl + images/")
    parser.add_argument("--out", required=True, help="run dir to create")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = read_jsonl(data_dir / "data.jsonl")
    piece = rows[args.offset : None if args.limit is None else args.offset + args.limit]
    out_data = Path(args.out) / "data"
    (out_data / "images").mkdir(parents=True, exist_ok=True)

    with (out_data / "data.jsonl").open("w") as fh:
        for row in piece:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    linked = missing = 0
    for row in piece:
        names = row.get("images") or []
        if isinstance(names, str):
            names = [names]
        for name in names:
            src = data_dir / "images" / name
            dst = out_data / "images" / name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copyfile(src, dst)
                linked += 1
            except FileNotFoundError:
                missing += 1

    shutil.copyfile(data_dir / "meta.json", out_data / "meta.json")

    manifest = json.loads((data_dir / "manifest.json").read_text())
    ids = {row["id"] for row in piece}
    manifest["rows"] = [row for row in manifest.get("rows", []) if row.get("id") in ids]
    manifest["slice_of"] = {"source_data_dir": str(data_dir), "offset": args.offset, "limit": len(piece)}
    (out_data / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(
        f"split_input_rows: rows[{args.offset}:{args.offset + len(piece)}] of {len(rows)} -> {out_data} "
        f"({linked} images linked, {missing} missing, manifest rows={len(manifest['rows'])})"
    )


if __name__ == "__main__":
    main()
