#!/usr/bin/env python3
"""Replay a finished run's offline feature store to re-audit gradient coverage.

The training-time audit (``train/gradient_coverage.json``) runs at
``train.coverage_audit_fraction`` (default 0 -> the first optimizer step).
That is the right moment to catch "head created but never updated"; but a
zero-initialised residual branch (DFly's hidden correction) legitimately
reports zero gradient on its gate/up path at step 0 and only lights up once
the zero-initialised down projection has moved off zero. This probe loads a
saved checkpoint, replays real feature batches through the trainer module,
and re-audits after each optimizer step so that "the branch turns on" is
verifiable on the canonical run instead of only in a unit test.

Uses the run's own config/contract/feature store; no target model needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.algorithms.registry import algorithm_name
from vlm_dflash.config import load_config
from vlm_dflash.contract import ensure_contract
from vlm_dflash.dflash.features import FeatureDataset, capture_layer_layout, collate_features
from vlm_dflash.utils import gradient_coverage_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="train/ckpt_final.pt")
    parser.add_argument("--steps", type=int, default=2, help="optimizer steps to replay")
    parser.add_argument("--batch", type=int, default=1, help="feature samples per micro-batch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--lr", type=float, default=None, help="defaults to train.lr from config")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="defaults to train.max_grad_norm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    train_cfg = cfg.get("train", {})
    algorithm = get_algorithm(algorithm_name(cfg.get("algorithm")))
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 6e-4))
    max_grad_norm = float(
        args.max_grad_norm if args.max_grad_norm is not None else train_cfg.get("max_grad_norm", 1.0)
    )

    features_dir = run_dir / "features"
    target_head = torch.load(features_dir / "target_head.pt", map_location="cpu", weights_only=False)
    if target_head["model_fingerprint"] != contract["target"]["model_fingerprint"]:
        raise RuntimeError("features/target_head.pt was produced from a different target fingerprint.")
    tied_head = bool(target_head.get("tie_word_embeddings", True))
    lm_head_weight = target_head.get("lm_head")
    if tied_head and lm_head_weight is not None:
        raise RuntimeError("tied target but features/target_head.pt stores a separate lm_head.")
    if not tied_head and lm_head_weight is None:
        raise RuntimeError("untied target but features/target_head.pt has no lm_head weight.")

    checkpoint = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "vlm-dflash-ckpt-v1":
        raise ValueError(f"Unrecognized checkpoint format: {checkpoint.get('format')!r}")
    draft_config = algorithm.build_draft_config(contract, cfg)
    module = algorithm.build_trainer_module(
        draft_config, contract, cfg, target_head["embed_tokens"], lm_head_weight=lm_head_weight
    )
    module.draft_model.load_state_dict(checkpoint["model"])

    spatial = bool(
        getattr(
            module,
            "uses_multimodal_position_ids",
            getattr(module.draft_model, "spatial_rope_enabled", False),
        )
    )
    image_token_id = int(contract["tokens"]["image_token_id"])
    dataset = FeatureDataset(features_dir)
    # The store may concatenate a superset of the layers the method consumes
    # (e.g. a 5-layer capture sliced to [1, 17, 33]); training configures the
    # module with the recorded layout before feeding raw store batches, and so
    # must the replay probe.
    store_capture, store_hidden_size = capture_layer_layout(
        dataset.store.manifest, contract_hidden_size=int(contract["text"]["hidden_size"])
    )
    module.configure_feature_layout(store_capture, store_hidden_size)

    module.to(device)
    module.lm_head.to(device)
    module.embed_tokens.to(device)
    module.draft_model.to(dtype)
    master_parameters = {
        name: parameter.detach().float().clone().requires_grad_(True)
        for name, parameter in module.draft_model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        list(master_parameters.values()),
        lr=lr,
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    module.train()
    print(
        f"replay: {len(dataset)} samples, steps={args.steps}, batch={args.batch}, "
        f"device={device}, dtype={args.dtype}, lr={lr:g}",
        flush=True,
    )

    report: list[dict] = []
    for step in range(args.steps):
        start = (step * args.batch) % len(dataset)
        rows = [dataset[(start + offset) % len(dataset)] for offset in range(args.batch)]
        batch = collate_features(rows, image_token_id)
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        metrics = module(
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden"],
            loss_mask=batch["loss_mask"],
            image_mask=batch["image_mask"],
            multimodal_position_ids=batch["position3"] if spatial else None,
            teacher_hidden=batch.get("final_hidden"),
        )
        metrics.loss.backward()
        coverage = gradient_coverage_report(module)
        coverage["step"] = step
        coverage["loss"] = float(metrics.loss.detach())
        report.append(coverage)
        print(
            f"step {step}: loss={coverage['loss']:.4f} coverage="
            f"{coverage['params_with_nonzero_grad']}/{coverage['trainable_tensors']} tensors with nonzero grad",
            flush=True,
        )
        for group in coverage["groups"]:
            if group["trainable_params"]:
                print(
                    f"  {group['group']}: nonzero={group['params_with_nonzero_grad']}/"
                    f"{group['trainable_tensors']} tensors grad_norm={group['grad_norm']:.4g}",
                    flush=True,
                )
        with torch.no_grad():
            for name, parameter in module.draft_model.named_parameters():
                master = master_parameters[name]
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
                parameter.grad = None
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(master_parameters.values()), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in module.draft_model.named_parameters():
                parameter.copy_(master_parameters[name])

    print(json.dumps({"run_dir": str(run_dir), "checkpoint": args.checkpoint, "steps": report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
