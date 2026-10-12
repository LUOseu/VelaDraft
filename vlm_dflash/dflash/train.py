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

from ..algorithms import get_algorithm
from ..provenance import assert_identity, identity_from_contract, record_stage
from ..utils import ensure_dir, log, set_seed, write_json
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


def compute_total_steps(dataset_len: int, train_cfg: dict, steps: int | None = None) -> int:
    """Planned optimizer steps (mirrors the loop's own loader arithmetic)."""
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    accumulation = max(1, int(train_cfg.get("grad_accumulation_steps", 1)))
    epochs = int(train_cfg.get("num_epochs", 6))
    steps_per_epoch = max(1, math.ceil(math.ceil(dataset_len / micro_batch) / accumulation))
    return steps if steps is not None else steps_per_epoch * epochs


def _clone_tree(value):
    """Recursively detach/clone tensors so a saved state never aliases live buffers."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_tree(item) for item in value)
    return value


def gather_train_state(module, master_parameters, optimizer, scheduler, *, step, micro_step, skipped_batches, history, steps_total_planned) -> dict:
    """Self-contained resume payload: weights, fp32 masters, optimizer, schedule.

    All tensors are cloned (``.cpu()`` alone does not copy on CPU devices), so
    the returned payload stays valid while training continues past the save.
    """
    return {
        "format": "vlm-dflash-train-state-v1",
        "step": int(step),
        "micro_step": int(micro_step),
        "skipped_no_anchor_batches": int(skipped_batches),
        "steps_total_planned": int(steps_total_planned),
        "model": _clone_tree(module.draft_model.state_dict()),
        "master": _clone_tree(master_parameters),
        "optimizer": _clone_tree(optimizer.state_dict()),
        "scheduler": _clone_tree(scheduler.state_dict()),
        "history": _clone_tree(list(history)),
        # RNG states: anchor sampling consumes the torch RNG, so an exact
        # resume must restore it (otherwise resumed steps diverge from the
        # uninterrupted run even with identical weights).
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


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
    resume_state: dict | None = None,
    on_checkpoint=None,
    save_every: int | None = None,
) -> dict:
    train_cfg = cfg.get("train", {})
    micro_batch = int(train_cfg.get("micro_batch_size", 8))
    accumulation = max(1, int(train_cfg.get("grad_accumulation_steps", 1)))
    log_every = int(train_cfg.get("log_every_steps", 10))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    epochs = int(train_cfg.get("num_epochs", 6))
    image_token_id = int(contract["tokens"]["image_token_id"])
    spatial = bool(getattr(module, "uses_multimodal_position_ids", getattr(module.draft_model, "spatial_rope_enabled", False)))

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
    # fp32 master copies for the optimizer updates. ``train.draft_dtype`` can
    # be set to float32 for debugging on CPU.
    draft_dtype_name = str(train_cfg.get("draft_dtype", "bfloat16"))
    if draft_dtype_name not in ("bfloat16", "float32"):
        raise ValueError(f"train.draft_dtype must be bfloat16 or float32, got {draft_dtype_name!r}.")
    module.draft_model.to(torch.bfloat16 if draft_dtype_name == "bfloat16" else torch.float32)
    master_parameters = {
        name: parameter.detach().float().clone().requires_grad_(True)
        for name, parameter in module.draft_model.named_parameters()
    }
    optimizer, scheduler = _make_optimizer_and_scheduler(list(master_parameters.values()), train_cfg, total_steps)
    module.train()
    history, nan_seen, skipped_batches = [], False, 0
    step = 0
    micro_step = 0
    consumed = 0
    if resume_state is not None:
        if resume_state.get("format") != "vlm-dflash-train-state-v1":
            raise ValueError(f"Unrecognized resume state format: {resume_state.get('format')!r}")
        module.draft_model.load_state_dict(resume_state["model"])
        for name, parameter in master_parameters.items():
            parameter.data.copy_(resume_state["master"][name].float())
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        step = int(resume_state["step"])
        micro_step = int(resume_state.get("micro_step", 0))
        skipped_batches = int(resume_state.get("skipped_no_anchor_batches", 0))
        consumed = micro_step + skipped_batches
        history = list(resume_state.get("history", []))
        if resume_state.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])
        log(
            f"train: resumed from step {step} "
            f"(micro_step={micro_step}, skipped={skipped_batches}, history_rows={len(history)})"
        )
    stall_epochs = 0
    gradient_coverage: dict | None = None
    started = time.time()
    from .trainer_core import NoValidAnchorsError

    skippable_errors = tuple(getattr(module, "skippable_errors", ()) or ()) or (NoValidAnchorsError,)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    # Fast-forward a resumed run to the same micro-batch position the original
    # uninterrupted run had reached (the shuffle order is reproducible from the
    # seed, so skipping consumed items restores the exact data order).
    while consumed > 0:
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
        else:
            consumed -= 1
    while step < total_steps:
        progressed = False
        for batch in iterator:
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
            except skippable_errors:
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
                if step == 1:
                    # Gradient coverage audit (GOAL §13): a new head that never
                    # receives a gradient still trains the rest of the model and
                    # looks plausible, so check coverage once per run.
                    from ..utils import gradient_coverage_report

                    coverage = gradient_coverage_report(module)
                    coverage["source"] = "step1_micro_batch"
                    gradient_coverage = coverage
                    log(
                        "gradient coverage: "
                        f"{coverage['params_with_grad']}/{coverage['trainable_params']} trainable tensors "
                        f"with grad (fraction={coverage['fraction_with_grad']:.3f})"
                    )
                    for group in coverage["groups"]:
                        if group["trainable_params"] or group["group"] == "frozen":
                            log(
                                f"  {group['group']}: trainable={group['trainable_params']} "
                                f"with_grad={group['params_with_grad']} grad_norm={group['grad_norm']:.4g}"
                            )
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "accuracy": float(metrics.accuracy),
                    "accept_len": float(metrics.accept_len),
                    "valid_tokens": int(metrics.valid_tokens),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm),
                }
                if getattr(metrics, "extra_metrics", None):
                    row.update({key: float(value) for key, value in metrics.extra_metrics.items()})
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
                if on_checkpoint is not None and save_every and step % save_every == 0 and step < total_steps:
                    on_checkpoint(
                        gather_train_state(
                            module,
                            master_parameters,
                            optimizer,
                            scheduler,
                            step=step,
                            micro_step=micro_step,
                            skipped_batches=skipped_batches,
                            history=history,
                            steps_total_planned=total_steps,
                        )
                    )
                if step >= total_steps:
                    break
        iterator = iter(loader)
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
        "resumed_from_step": int(resume_state["step"]) if resume_state is not None else 0,
        "wall_seconds": time.time() - started,
        "peak_gpu_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        ),
    }
    return {"summary": summary, "history": history, "gradient_coverage": gradient_coverage}


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


def train_dflash(cfg: dict, run_dir: str | Path, contract: dict, steps: int | None = None, limit: int | None = None, resume: bool = False) -> dict:
    run_dir = Path(run_dir)
    features_dir = run_dir / "features"
    dataset = FeatureDataset(features_dir)
    if limit is not None:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    manifest = FeatureDataset(features_dir).store.manifest
    assert_identity(identity_from_contract(contract), manifest["identity"], "train vs features")

    algorithm = get_algorithm(cfg.get("algorithm"))
    algorithm.validate_config(cfg)

    train_cfg = cfg.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    device = torch.device(train_cfg.get("device", "cuda:0"))
    set_seed(seed)

    draft_config = algorithm.build_draft_config(contract, cfg)
    target_head = torch.load(features_dir / "target_head.pt", map_location="cpu", weights_only=False)
    if target_head["model_fingerprint"] != contract["target"]["model_fingerprint"]:
        raise RuntimeError("features/target_head.pt was produced from a different target fingerprint.")
    tied_head = bool(target_head.get("tie_word_embeddings", True))
    lm_head_weight = target_head.get("lm_head")
    if not tied_head and lm_head_weight is None:
        raise RuntimeError(
            "features/target_head.pt has no untied lm_head weight (captured before the head fix). "
            "Re-run capture_features so training uses the same output head as inference."
        )
    if tied_head and lm_head_weight is not None:
        raise RuntimeError("features/target_head.pt records a tied target but also stores a separate lm_head.")
    module = algorithm.build_trainer_module(
        draft_config, contract, cfg, target_head["embed_tokens"], lm_head_weight=lm_head_weight
    )
    init_state = {key: value.clone() for key, value in module.draft_model.state_dict().items()}

    train_dir = ensure_dir(run_dir / "train")
    checkpoint_common = {
        "format": "vlm-dflash-ckpt-v1",
        # Which drafter algorithm trained these weights: eval/export hard-fail
        # when a checkpoint is scored with a different algorithm than it was
        # trained with (same idea as the head/identity tags below).
        "algorithm": algorithm.name,
        **algorithm.checkpoint_metadata(contract, cfg, draft_config),
        # Which frozen output head the draft logits were trained against. Eval
        # refuses checkpoints whose head does not match the target's tie state.
        "head": "tied_embedding" if tied_head else "target_lm_head",
        "identity": identity_from_contract(contract),
    }
    torch.save({**checkpoint_common, "step": 0, "model": init_state}, train_dir / "ckpt_init.pt")

    total_steps = compute_total_steps(len(dataset), train_cfg, steps)
    resume_state = None
    if resume:
        resume_path = train_dir / "ckpt_last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but no resume checkpoint at {resume_path}.")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_state.get("format") != "vlm-dflash-train-state-v1":
            raise ValueError(f"Unrecognized resume state format: {resume_state.get('format')!r}")
        planned = int(resume_state.get("steps_total_planned", -1))
        if planned != total_steps:
            raise ValueError(
                f"Resume plan mismatch: checkpoint planned {planned} total steps, this config computes "
                f"{total_steps}. Re-run with the same --steps / epochs / dataset slice."
            )
        if resume_state.get("identity") is not None:
            assert_identity(identity_from_contract(contract), resume_state["identity"], "resume vs contract")
        overfit = {"enabled": True, "resumed": True}
    else:
        overfit = run_overfit_sanity(cfg, contract, features_dir, run_dir, device, module, init_state)
        module.draft_model.load_state_dict(init_state)

    log_path = train_dir / "train_log.jsonl"
    if log_path.exists() and resume_state is None:
        log_path.unlink()
    save_every = int(train_cfg.get("save_every_steps", max(1, total_steps // 4)))

    def write_train_state(state: dict) -> None:
        tmp_path = train_dir / "ckpt_last.pt.tmp"
        torch.save({**checkpoint_common, **state}, tmp_path)
        tmp_path.replace(train_dir / "ckpt_last.pt")
        log(f"train: wrote resumable checkpoint (step {state['step']}) -> {train_dir / 'ckpt_last.pt'}")

    output = run_training_loop(
        module,
        dataset,
        cfg,
        contract,
        device,
        total_steps=total_steps,
        seed=seed,
        shuffle=True,
        log_path=log_path,
        resume_state=resume_state,
        on_checkpoint=write_train_state,
        save_every=save_every,
    )
    torch.save({**checkpoint_common, "step": output["summary"]["steps"], "model": module.draft_model.state_dict()},
               train_dir / "ckpt_final.pt")

    from .visual_check import check_drafter_visual_dependence

    visual = check_drafter_visual_dependence(cfg, contract, run_dir, device, module)
    if output.get("gradient_coverage"):
        write_json(train_dir / "gradient_coverage.json", output["gradient_coverage"])
    # Algorithm-specific calibration diagnostics (e.g. DSpark's confidence
    # head): only modules that implement the hook produce an artifact.
    calibration = None
    if hasattr(module, "calibration_report"):
        from torch.utils.data import DataLoader

        calibration_dataset = FeatureDataset(features_dir)
        calibration_image_token_id = int(contract["tokens"]["image_token_id"])
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=int(train_cfg.get("micro_batch_size", 8)),
            shuffle=False,
            collate_fn=lambda batch: collate_features(batch, calibration_image_token_id),
        )
        calibration = module.calibration_report(calibration_loader, device)
        if calibration is not None:
            calibration["source"] = "train_features_head_batches"
            write_json(train_dir / "confidence_calibration.json", calibration)
            log(f"train: confidence calibration -> {train_dir / 'confidence_calibration.json'}")
    summary = {
        **output["summary"],
        "overfit_sanity": overfit,
        "visual_dependence": visual,
        "confidence_calibration": calibration,
    }
    write_json(train_dir / "train_summary.json", summary)
    record_stage(run_dir, "train", {"identity": identity_from_contract(contract), "summary": summary,
                                    "code": "vlm_dflash.dflash.train.train_dflash"})
    return summary
