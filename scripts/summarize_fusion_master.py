#!/usr/bin/env python3
"""Summarize the fp32 optimizer-master fusion logits of a DFlare/DFly checkpoint.

The live draft parameters are bf16, where a single AdamW step is below the
quantisation grid; the fp32 masters in ``train/ckpt_last.pt`` (written every
``save_every`` steps) carry the true trajectory. This script writes a small
JSON next to the checkpoint: prior logits, trained logits, softmax
distributions, per-layer entropy and the L1 drift from the prior.

Usage:
    python scripts/summarize_fusion_master.py runs/<run> [--checkpoint ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def fusion_prior(num_draft: int, num_context: int, alpha: float = 2.0) -> torch.Tensor:
    """The DFlare/DFly distance prior (`-alpha * |t - t*_d|`), fp32."""
    distance = torch.arange(num_context, dtype=torch.float32)
    prior = torch.empty(num_draft, num_context, dtype=torch.float32)
    for draft_layer in range(num_draft):
        preferred = round(draft_layer / max(num_draft - 1, 1) * (num_context - 1))
        prior[draft_layer] = -alpha * (distance - preferred).abs()
    return prior


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="run directory (contains train/ckpt_last.pt)")
    parser.add_argument("--checkpoint", default="train/ckpt_last.pt")
    parser.add_argument("--out", default="train/fusion_master_summary.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    master = state["master"]["layer_fusion_weights"].float()
    num_draft, num_context = master.shape
    prior = fusion_prior(num_draft, num_context)
    probabilities = F.softmax(master, dim=-1)
    prior_probabilities = F.softmax(prior, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    summary = {
        "format": "vlm-dflash-fusion-master-v1",
        "source": str(ckpt_path),
        "step": int(state.get("step", -1)),
        "note": (
            "fp32 optimizer master fusion logits; the live bf16 parameter is grid-quantised, "
            "so in-log fusion metrics move coarsely"
        ),
        "num_draft_layers": num_draft,
        "num_context_layers": num_context,
        "fusion_master_logits": master.tolist(),
        "prior_logits": prior.tolist(),
        "fusion_probabilities": probabilities.tolist(),
        "prior_probabilities": prior_probabilities.tolist(),
        "prob_l1_drift_from_prior": float((probabilities - prior_probabilities).abs().mean()),
        "entropy_per_layer": entropy.tolist(),
        "entropy_mean": float(entropy.mean()),
    }
    correction = state.get("master", {}).get("hidden_correction.down_proj.weight")
    if correction is not None:
        summary["hidden_correction_down_proj_abs_mean"] = float(correction.float().abs().mean())

    out_path = run_dir / args.out
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if not key.endswith(("logits", "probabilities"))}, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
