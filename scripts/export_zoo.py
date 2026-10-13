#!/usr/bin/env python
"""Export the v0.3 model-zoo checkpoints with canonical names.

Every export directory is self-describing: ``model.safetensors``,
``config.json``, ``draft_config.json``, ``algorithm.json``,
``target_contract.json``, ``training_manifest.json`` (with the trainable
parameter count), ``model_card.md`` and ``checksums.json``. Each export is
reloaded with ``load_export`` before the script reports success, so a broken
checkpoint cannot be published by accident.

Usage (server, from the repository root):

    python scripts/export_zoo.py --out assets/zoo
    python scripts/export_zoo.py --out assets/zoo --only qwen25vl-3b-dflash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.export.hf import export_run, load_export  # noqa: E402
from vlm_dflash.utils import log  # noqa: E402

# Canonical model-zoo names -> run directory holding ckpt_final.pt.
ZOO_ITEMS = {
    "qwen25vl-3b-dflash": "runs/qwen25vl3b_2k",
    "qwen25vl-3b-dspark": "runs/qwen25vl3b_dspark_2k",
    "qwen25vl-3b-domino": "runs/qwen25vl3b_domino_2k",
    "qwen25vl-3b-jetspec": "runs/qwen25vl3b_jetspec_2k",
    "qwen25vl-3b-dflash2-exp": "runs/qwen25vl3b_dflash2_2k",
    "qwen25vl-3b-dflare": "runs/qwen25vl3b_dflare_2k",
    "qwen25vl-3b-dfly": "runs/qwen25vl3b_dfly_2k",
    "qwen3vl-8b-dflash": "runs/qwen3vl8b_2k",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory for the zoo exports")
    parser.add_argument("--only", default=None, help="comma-separated subset of zoo names")
    parser.add_argument("--repo-root", default=".", help="repository root (default: cwd)")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_root = Path(args.out).resolve()
    names = list(ZOO_ITEMS) if not args.only else [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in ZOO_ITEMS]
    if unknown:
        raise SystemExit(f"unknown zoo item(s) {unknown}; available: {sorted(ZOO_ITEMS)}")

    failures: list[str] = []
    for name in names:
        run_dir = root / ZOO_ITEMS[name]
        if not (run_dir / "train" / "ckpt_final.pt").exists():
            failures.append(f"{name}: no checkpoint at {run_dir}/train/ckpt_final.pt")
            continue
        export_dir = export_run(run_dir, out_dir=out_root / name)
        model, _export_config = load_export(export_dir)
        parameters = sum(p.numel() for p in model.parameters())
        log(f"zoo: {name} -> {export_dir} (reload ok, {parameters / 1e6:.1f}M params)")

    if failures:
        for failure in failures:
            log(f"zoo: FAILED {failure}")
        return 1
    log(f"zoo: exported {len(names)} checkpoint(s) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
