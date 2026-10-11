"""Staged offline DFlash draft training (target never loaded here).

Reads captured feature shards, trains the vendored DFlash trainer core with
AdamW + warmup/cosine, saves a random-init checkpoint (the acceptance
baseline), a final checkpoint, the training log, an overfit sanity run on a
tiny subset, and the drafter visual-dependence check.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ..provenance import assert_identity, identity_from_contract, record_stage
from ..utils import ensure_dir, log, set_seed, write_json
from .factory import build_draft_config, build_trainer_module
from .features import FeatureDataset, collate_features


def _make_optimizer_and_scheduler(parameters, train_cfg: dict, total_steps: int):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_cfg.get("lr", 6e-4)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.04))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training_loop(
    module,
    dataset,
    cfg: dict,
    contract: dict,
    device: torch.device,
    total_steps: int | None,
    seed: int,
    shuffle: bool,
    log_path: Path | None = None,
) -> dict:
    train_cfg = cfg.get("train", {})
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    accumulation = max(1, int(train_cfg.get("grad_accumulation_steps", 1)))
    log_every = int(train_cfg.get("log_every_steps", 10))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    epochs = int(train_cfg.get("num_epochs", 6))
    image_token_id = int(contract["tokens"]["image_token_id"])
    spatial = bool(module.draft_model.spatial_rope_enabled)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_features(batch, image_token_id),
        drop_last=False,
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulation))
    if total_steps is None:
        total_steps = steps_per_epoch * epochs

    module.to(device)
    # lm_head / embed_tokens are deliberately non-registered references
    # (mirroring the reference trainer), so .to(device) does not move them.
    module.lm_head.to(device)
    module.embed_tokens.to(device)
    # Train the draft in bf16 like the reference run (the draft's rope buffers
    # must share the compute dtype to keep attention inputs consistent); keep
    # fp32 master copies for the optimizer updates.
    module.draft_model.to(torch.bfloat16)
    master_parameters = {
        name: parameter.detach().float().clone().requires_grad_(True)
        for name, parameter in module.draft_model.named_parameters()
    }
    optimizer, scheduler = _make_optimizer_and_scheduler(list(master_parameters.values()), train_cfg, total_steps)
    module.train()
    history, nan_seen, skipped_batches = [], False, 0
    step = 0
    micro_step = 0
    stall_epochs = 0
    started = time.time()
    from .trainer_core import NoValidAnchorsError

    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        progressed = False
        for batch in loader:
            batch = {key: value for key, value in batch.items() if key != "lengths"}
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            try:
                metrics = module(
                    input_ids=batch["input_ids"],
                    hidden_states=batch["hidden"],
                    loss_mask=batch["loss_mask"],
                    image_mask=batch["image_mask"],
                    multimodal_position_ids=batch["position3"] if spatial else None,
                )
            except NoValidAnchorsError:
                skipped_batches += 1
                continue
            loss = metrics.loss
            if not torch.isfinite(loss):
                nan_seen = True
                raise RuntimeError(f"Non-finite training loss at step {step}: {loss.item()}")
            (loss / accumulation).backward()
            micro_step += 1
            if micro_step % accumulation == 0:
                with torch.no_grad():
                    for name, parameter in module.draft_model.named_parameters():
                        master = master_parameters[name]
                        master.grad = None if parameter.grad is None else parameter.grad.detach().float()
                        parameter.grad = None
                grad_norm = torch.nn.utils.clip_grad_norm_(list(master_parameters.values()), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, parameter in module.draft_model.named_parameters():
                        parameter.copy_(master_parameters[name])
                step += 1
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "accuracy": float(metrics.accuracy),
                    "accept_len": float(metrics.accept_len),
                    "valid_tokens": int(metrics.valid_tokens),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm),
                }
                if metrics.pos_correct is not None and metrics.pos_count is not None:
                    row["slot0_accuracy"] = float(metrics.pos_correct[0] / metrics.pos_count[0].clamp_min(1))
                history.append(row)
                progressed = True
                if log_path is not None:
                    with open(log_path, "a", encoding="utf-8") as fh:
                        import json as _json

                        fh.write(_json.dumps(row) + "\n")
                if log_every and step % log_every == 0:
                    log(f"step {step}/{total_steps} loss={row['loss']:.4f} acc={row['accuracy']:.3f} "
                        f"al={row['accept_len']:.2f}")
                if step >= total_steps:
                    break
        if not progressed:
            stall_epochs += 1
            if stall_epochs >= 3:
                raise RuntimeError(
                    "Training made no progress for 3 full epochs (all batches raised NoValidAnchorsError); "
                    "check that loss_mask covers at least block_size+1 tokens per sample."
                )
        else:
            stall_epochs = 0

    losses = [row["loss"] for row in history]
    window = max(1, len(losses) // 10)
    summary = {
        "steps": len(losses),
        "first10_mean_loss": sum(losses[:window]) / window if losses else None,
        "last10_mean_loss": sum(losses[-window:]) / window if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and (sum(losses[-window:]) / window) < (sum(losses[:window]) / window)),
        "nan_seen": nan_seen,
        "skipped_no_anchor_batches": skipped_batches,
        "wall_seconds": time.time() - started,
        "peak_gpu_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        ),
    }
    return {"summary": summary, "history": history}


def run_overfit_sanity(cfg: dict, contract: dict, features_dir: Path, run_dir: Path, device, module, init_state) -> dict:
    overfit_cfg = (cfg.get("train", {}).get("overfit_sanity", {}) or {})
    result_path = run_dir / "train" / "overfit_sanity.json"
    if not overfit_cfg.get("enabled", True):
        result = {"enabled": False}
        write_json(result_path, result)
        return result
    num_samples = int(overfit_cfg.get("num_samples", 16))
    steps = int(overfit_cfg.get("steps", 200))
    dataset = FeatureDataset(features_dir)
    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    run_cfg = dict(cfg)
    run_cfg["train"] = {
        **cfg.get("train", {}),
        "micro_batch_size": min(num_samples, len(subset)),
        "grad_accumulation_steps": 1,
        "num_epochs": 1,
    }
    module.draft_model.load_state_dict(init_state)
    output = run_training_loop(
        module,
        subset,
        run_cfg,
        contract,
        device,
        total_steps=steps,
        seed=int(cfg.get("train", {}).get("seed", 42)),
        shuffle=False,
    )
    history = output["history"]
    initial = history[0]["loss"] if history else None
    final = max(1e-9, sum(row["loss"] for row in history[-max(1, len(history) // 10):]) / max(1, len(history) // 10))
    drop = 1.0 - final / initial if initial else 0.0
    slot0 = history[-1].get("slot0_accuracy") if history else None
    ce_drop_threshold = float(overfit_cfg.get("ce_drop", 0.6))
    slot0_threshold = float(overfit_cfg.get("slot0_acc", 0.9))
    result = {
        "enabled": True,
        "num_samples": len(subset),
        "steps": len(history),
        "initial_loss": initial,
        "final_loss": final,
        "ce_drop_fraction": drop,
        "slot0_accuracy_final": slot0,
        "thresholds": {"ce_drop": ce_drop_threshold, "slot0_acc": slot0_threshold},
        "passed": bool(drop >= ce_drop_threshold and (slot0 is not None and slot0 >= slot0_threshold)),
    }
    write_json(result_path, result)
    log(f"overfit sanity: drop={drop:.3f} slot0={slot0} passed={result['passed']}")
    return result


def train_dflash(cfg: dict, run_dir: str | Path, contract: dict, steps: int | None = None, limit: int | None = None) -> dict:
    run_dir = Path(run_dir)
    features_dir = run_dir / "features"
    dataset = FeatureDataset(features_dir)
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    manifest = FeatureDataset(features_dir).store.manifest
    assert_identity(identity_from_contract(contract), manifest["identity"], "train vs features")

    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda:0"))
    set_seed(seed)

    draft_config = build_draft_config(contract, cfg)
    target_head = torch.load(features_dir / "target_head.pt", map_location="cpu", weights_only=False)
    if target_head["model_fingerprint"] != contract["target"]["model_fingerprint"]:
        raise RuntimeError("features/target_head.pt was produced from a different target fingerprint.")
    module = build_trainer_module(draft_config, contract, cfg, target_head["embed_tokens"])
    init_state = {key: value.clone() for key, value in module.draft_model.state_dict().items()}

    train_dir = ensure_dir(run_dir / "train")
    checkpoint_common = {
        "format": "vlm-dflash-ckpt-v1",
        "draft_config": draft_config.to_dict(),
        "attention_backend": str(cfg["drafter"].get("attention_backend", "flex_attention")),
        "mask_token_id": int(cfg["drafter"]["mask_token_id"]),
        "block_size": int(cfg["drafter"]["block_size"]),
        "num_anchors": int(cfg["drafter"].get("num_anchors", 256)),
        "loss_decay_gamma": float(cfg["drafter"].get("loss_decay_gamma", 7.0)),
        "target_layer_ids": [int(x) for x in contract["layers"]["draft_layer_ids"]],
        "identity": identity_from_contract(contract),
    }
    torch.save({**checkpoint_common, "step": 0, "model": init_state}, train_dir / "ckpt_init.pt")

    overfit = run_overfit_sanity(cfg, contract, features_dir, run_dir, device, module, init_state)
    module.draft_model.load_state_dict(init_state)

    log_path = train_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    output = run_training_loop(module, dataset, cfg, contract, device, total_steps=steps, seed=seed, shuffle=True, log_path=log_path)
    torch.save({**checkpoint_common, "step": output["summary"]["steps"], "model": module.draft_model.state_dict()},
               train_dir / "ckpt_final.pt")

    from .visual_check import check_drafter_visual_dependence

    visual = check_drafter_visual_dependence(cfg, contract, run_dir, device, module)
    summary = {**output["summary"], "overfit_sanity": overfit, "visual_dependence": visual}
    write_json(train_dir / "train_summary.json", summary)
    record_stage(run_dir, "train", {"identity": identity_from_contract(contract), "summary": summary,
                                    "code": "vlm_dflash.dflash.train.train_dflash"})
    return summary
