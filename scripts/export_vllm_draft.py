#!/usr/bin/env python
"""Convert a vlm-dflash DFlash export into a vLLM (speculators) draft directory.

Usage (server or workstation):

    python scripts/export_vllm_draft.py \\
        --export /tmp/wp5/zoo_export/qwen25vl-3b-dflash \\
        --out /tmp/wp5/vllm_draft --target /path/to/Qwen2.5-VL-3B-Instruct

The conversion copies the weights unchanged and writes the
``vllm-project/speculators`` config vLLM's native ``dflash`` speculative
method consumes; see ``vlm_dflash/export/vllm_bridge.py`` for the layout and
the layer-id convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.export.vllm_bridge import export_vllm_draft  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="vlm-dflash export directory")
    parser.add_argument("--out", required=True, help="output directory for the vLLM draft")
    parser.add_argument("--target", default=None, help="target model path (default: recorded in the export)")
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=None,
        help="draft tokens per step (default: block_size - 1)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --force to overwrite")
    summary = export_vllm_draft(
        args.export,
        out_dir,
        target_path=args.target,
        speculative_tokens=args.speculative_tokens,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
