#!/usr/bin/env python3
"""Convert resumable train-state checkpoints into eval/export-loadable drafts.

``write_train_state`` saves ``ckpt_last.pt`` (format ``vlm-dflash-train-state-v1``:
weights + fp32 masters + optimizer + RNG) for exact resume. Eval/export load
``vlm-dflash-ckpt-v1`` drafts, which additionally carry the provenance block
(``draft_config``, ``algorithm``, ``identity``, ``feature_mode``, plan digest,
...). That block is written into ``ckpt_init.pt`` at training start, so a
mid-run snapshot converts by grafting its ``model``/``step`` onto the init's
header. This is how stability-curve checkpoints become evaluable without
re-running training.

    python scripts/convert_train_state_ckpt.py --run-dir runs/<run> \
        --state ckpt_600.pt --out ckpt_eval_600.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def convert_train_state(template: dict, state: dict, state_name: str = "") -> dict:
    """Graft ``state``'s model/step onto ``template`` (an init checkpoint)."""
    if template.get("format") != "vlm-dflash-ckpt-v1":
        raise ValueError(f"template must be format 'vlm-dflash-ckpt-v1', got {template.get('format')!r}")
    if state.get("format") != "vlm-dflash-train-state-v1":
        raise ValueError(f"state must be format 'vlm-dflash-train-state-v1', got {state.get('format')!r}")
    if "model" not in state:
        raise ValueError("train-state payload has no 'model' weights")
    out = {key: value for key, value in template.items() if key != "model"}
    out["model"] = state["model"]
    out["step"] = int(state.get("step", 0))
    out["converted_from"] = {
        "train_state": state_name,
        "step": int(state.get("step", 0)),
        "note": "weights from train-state; header (draft_config/provenance) from train/ckpt_init.pt",
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--state", action="append", required=True,
                        help="train-state file inside <run-dir>/train (repeatable)")
    parser.add_argument("--out", default=None,
                        help="output name (single --state only); default ckpt_eval_<tag>.pt")
    parser.add_argument("--init", default="ckpt_init.pt", help="template checkpoint name")
    args = parser.parse_args()
    train_dir = Path(args.run_dir).resolve() / "train"
    template = torch.load(train_dir / args.init, map_location="cpu", weights_only=False)
    if args.out and len(args.state) > 1:
        raise SystemExit("--out is only valid with a single --state")
    for state_name in args.state:
        state = torch.load(train_dir / state_name, map_location="cpu", weights_only=False)
        converted = convert_train_state(template, state, state_name=state_name)
        tag = Path(state_name).stem.removeprefix("ckpt_")
        out_name = args.out or f"ckpt_eval_{tag}.pt"
        torch.save(converted, train_dir / out_name)
        print(f"wrote {train_dir / out_name} (step {converted['step']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
