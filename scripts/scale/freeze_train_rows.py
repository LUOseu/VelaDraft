#!/usr/bin/env python
"""Freeze the exact training corpus for the R3/R5 round.

Reads ``<run>/regen/target_data.jsonl`` (all regen-kept rows), keeps
``N = min(target, kept)`` rows chosen by a deterministic seeded shuffle,
archives the full kept list as ``target_data_all.jsonl`` and overwrites
``target_data.jsonl`` with exactly the frozen rows (single source of truth
both training runs hardlink). Writes ``freeze_record.json`` with counts,
hashes and the dropped-row ids so the member list is reproducible.

Refuses to freeze below ``--floor`` (the round cannot claim 15K-20K effective
samples with fewer rows; expand the input pool and regen first).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vlm_dflash.utils import ensure_dir, now_iso, read_jsonl, sha256_file, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--floor", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    regen_dir = Path(args.run_dir) / "regen"
    current = regen_dir / "target_data.jsonl"
    archived = regen_dir / "target_data_all.jsonl"
    if archived.exists():
        raise SystemExit(f"{archived} already exists; freeze already ran. Refusing to re-freeze.")
    rows = read_jsonl(current)
    kept = len(rows)
    if kept < args.floor:
        raise SystemExit(
            f"kept={kept} < floor={args.floor}: do not freeze this corpus; expand the input pool."
        )
    n = min(args.target, kept)

    rng = random.Random(args.seed)
    order = list(range(kept))
    rng.shuffle(order)
    chosen = set(order[:n])
    frozen = [rows[i] for i in sorted(order[:n])]
    dropped = [rows[i] for i in range(kept) if i not in chosen]

    write_jsonl(archived, rows)
    write_jsonl(current, frozen)

    answer_tokens = sum(len(row["regen"]["answer_ids"]) for row in frozen)
    prompt_tokens = sum(len(row["regen"].get("prompt_ids", [])) for row in frozen)
    record = {
        "created": now_iso(),
        "num_kept_total": kept,
        "num_frozen": len(frozen),
        "num_dropped_excess": len(dropped),
        "seed": args.seed,
        "selection": "random.Random(seed).shuffle(range(kept)); take first N; frozen rows kept in original file order",
        "frozen_ids_sha256": hashlib.sha256(
            "\n".join(str(row.get("id")) for row in frozen).encode("utf-8")
        ).hexdigest(),
        "dropped_ids_sha256": hashlib.sha256(
            "\n".join(str(row.get("id")) for row in dropped).encode("utf-8")
        ).hexdigest(),
        "target_data_all_sha256": sha256_file(archived),
        "frozen_file_sha256": sha256_file(current),
        "answer_tokens_frozen": answer_tokens,
        "prompt_tokens_frozen": prompt_tokens,
        "note": "target_data.jsonl now contains exactly the frozen rows; both R3 and R5 train on it",
    }
    write_json(regen_dir / "freeze_record.json", record)
    print(
        f"freeze: kept={kept} frozen={len(frozen)} dropped={len(dropped)} "
        f"answer_tokens={answer_tokens} prompt_tokens={prompt_tokens} "
        f"frozen_sha={record['frozen_file_sha256'][:12]}"
    )


if __name__ == "__main__":
    main()
