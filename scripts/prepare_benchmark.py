#!/usr/bin/env python3
"""Export a public benchmark workload into the pinned sharegpt_jsonl format.

    python scripts/prepare_benchmark.py --workload textvqa \
        --source /path/to/lmms-lab__textvqa --out benchmarks/data/textvqa

Also writes the repo-tracked prompt set definition (ids + question/image
hashes, no image bytes) to ``benchmarks/prompts/<workload>_v1.jsonl`` so the
pinned protocol is reviewable in git. Images and prompt jsonl stay under
``benchmarks/data/`` (gitignored). Deterministic: fixed split, fixed row order,
fixed subset size, fixed seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.benchmark.adapters import export_textvqa, pinned_prompt_manifest  # noqa: E402
from vlm_dflash.utils import log, write_jsonl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTERS = {"textvqa": export_textvqa}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, choices=sorted(_ADAPTERS))
    parser.add_argument("--source", required=True, help="local download of the upstream dataset")
    parser.add_argument("--out", default=None, help="export directory (default: benchmarks/data/<workload>)")
    parser.add_argument("--prompts-out", default=None, help="prompt manifest path (default: benchmarks/prompts/<workload>_v1.jsonl)")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "data" / args.workload
    prompts_out = Path(args.prompts_out) if args.prompts_out else REPO_ROOT / "benchmarks" / "prompts" / f"{args.workload}_v1.jsonl"
    manifest = _ADAPTERS[args.workload](
        args.source, out_dir, num_prompts=args.num_prompts, seed=args.seed
    )
    write_jsonl(prompts_out, pinned_prompt_manifest(manifest))
    log(f"prepare_benchmark: prompt manifest -> {prompts_out}")
    log(f"prepare_benchmark: rows_sha256={manifest['rows_sha256']}")


if __name__ == "__main__":
    main()
