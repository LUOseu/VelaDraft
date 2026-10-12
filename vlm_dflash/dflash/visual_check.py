"""Train-stage check that the drafter conditions on the captured visual context.

Consumes the swap pair saved by the capture visual check: the same token ids
with target features from image A vs image B must produce different draft block
logits and different block losses; perturbing the image-span feature rows must
also move them. Anchors are resampled under a fixed seed before every pass, so
the only difference between passes is the context features. Logits/losses are
compared in fp32 (the model itself stays in its trained bf16 dtype).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..utils import log, write_json

_ANCHOR_SEED = 1234
_MIN_DELTA = 1e-5


def draft_block_logits(module, input_ids, loss_mask, hidden, position3):
    """Draft block logits for one batch; mirrors the trainer forward's prologue."""
    torch.manual_seed(_ANCHOR_SEED)
    anchors, keep_mask, noise_embedding, full_positions, attention_mask, _ = module._prepare_block_inputs(
        input_ids, loss_mask, causal=getattr(module, "causal_in_block", False)
    )
    multimodal_positions = None
    if module.draft_model.spatial_rope_enabled and position3 is not None:
        gather_indices = anchors.clamp(min=0).unsqueeze(0).expand(3, -1, -1)
        anchor_positions = torch.gather(position3, 2, gather_indices)
        offsets = torch.arange(module.block_size, device=position3.device)
        draft_positions = (anchor_positions.unsqueeze(-1) + offsets.view(1, 1, 1, -1)).flatten(2)
        multimodal_positions = torch.cat((position3, draft_positions), dim=-1)
    output_hidden = module.draft_model(
        position_ids=full_positions,
        multimodal_position_ids=multimodal_positions,
        noise_embedding=noise_embedding,
        target_hidden=hidden,
        attention_mask=attention_mask,
        image_mask=None,
    )
    return module.lm_head(output_hidden), anchors, keep_mask


def fp32_block_ce(module, logits, input_ids, loss_mask, anchors, keep_mask) -> float:
    """Decay-weighted block CE recomputed in fp32 from bf16 logits (check only)."""
    bsz, seq_len = input_ids.shape
    block_size = module.block_size
    _, target_ids, block_mask = module._build_block_targets(
        input_ids, loss_mask, anchors, keep_mask, seq_len
    )
    pred_logits = logits.view(bsz, anchors.shape[1], block_size, -1)[:, :, 1:, :].reshape(
        bsz, -1, logits.shape[-1]
    )
    pred_targets = target_ids[:, :, 1:].reshape(bsz, -1)
    pred_mask = block_mask[:, :, 1:].reshape(bsz, -1)
    depth = torch.arange(block_size - 1, device=logits.device, dtype=torch.float32)
    gamma = module.loss_decay_gamma
    per_block = torch.exp(-depth / gamma) if gamma else torch.ones_like(depth)
    weights = pred_mask.float() * per_block.repeat(anchors.shape[1])
    nll = F.cross_entropy(
        pred_logits.reshape(-1, logits.shape[-1]).float(),
        pred_targets.reshape(-1),
        reduction="none",
    )
    return float((nll * weights.reshape(-1)).sum() / weights.sum().clamp_min(1e-6))


def check_drafter_visual_dependence(cfg: dict, contract: dict, run_dir: str | Path, device, module) -> dict:
    run_dir = Path(run_dir)
    pair_path = run_dir / "features" / "swap_pair.pt"
    out_path = run_dir / "train" / "visual_path_check.json"
    if not pair_path.exists():
        result = {"skipped": True, "reason": "features/swap_pair.pt missing (capture visual check did not run)"}
        write_json(out_path, result)
        return result

    pair = torch.load(pair_path, map_location="cpu", weights_only=False)
    input_ids = pair["input_ids"].long().unsqueeze(0).to(device)
    loss_mask = pair["loss_mask"].unsqueeze(0).to(device)
    position3 = pair["position3"].unsqueeze(1).to(device)
    span_start, span_end = pair["image_span"]
    hidden_a = pair["hidden_a"].unsqueeze(0).to(device)
    hidden_b = pair["hidden_b"].unsqueeze(0).to(device)

    module.eval()
    with torch.inference_mode():
        logits_a, anchors, keep_mask = draft_block_logits(module, input_ids, loss_mask, hidden_a, position3)
        logits_b, _, _ = draft_block_logits(module, input_ids, loss_mask, hidden_b, position3)
        torch.manual_seed(_ANCHOR_SEED)
        span = hidden_a[:, span_start : span_end + 1].float()
        # Scale-relative perturbation (half the span's std): large enough that the
        # induced block-CE change stays comfortably above the 1e-5 floor even on a
        # well-trained drafter whose loss is locally flat in the context features.
        sigma = max(0.5 * float(span.std()), 1e-3)
        noise = torch.randn_like(span) * sigma
        perturbed = hidden_a.clone()
        perturbed[:, span_start : span_end + 1] = (span + noise).to(perturbed.dtype)
        logits_p, _, _ = draft_block_logits(module, input_ids, loss_mask, perturbed, position3)

        def ce(logits):
            return fp32_block_ce(module, logits, input_ids, loss_mask, anchors, keep_mask)

        loss_a, loss_b, loss_p = ce(logits_a), ce(logits_b), ce(logits_p)
        swap_logits_diff = (logits_a.float() - logits_b.float()).abs()
        perturb_logits_diff = (logits_a.float() - logits_p.float()).abs()

    result = {
        "swap": {
            "logits_max_abs_diff": float(swap_logits_diff.max()),
            "logits_mean_abs_diff": float(swap_logits_diff.mean()),
            "block_ce_a": loss_a,
            "block_ce_b": loss_b,
            "block_ce_delta": abs(loss_a - loss_b),
        },
        "perturb_span": {
            "logits_max_abs_diff": float(perturb_logits_diff.max()),
            "logits_mean_abs_diff": float(perturb_logits_diff.mean()),
            "block_ce_a": loss_a,
            "block_ce_perturbed": loss_p,
            "block_ce_delta": abs(loss_a - loss_p),
        },
        "draft_logits_differ": bool(swap_logits_diff.max() > 0),
        "loss_changes_on_swap": bool(abs(loss_a - loss_b) > _MIN_DELTA),
        "loss_changes_on_perturb": bool(abs(loss_a - loss_p) > _MIN_DELTA),
    }
    result["passed"] = bool(
        result["draft_logits_differ"] and result["loss_changes_on_swap"] and result["loss_changes_on_perturb"]
    )
    write_json(out_path, result)
    log(
        "drafter visual check: "
        f"logits_diff={result['swap']['logits_max_abs_diff']:.4g} "
        f"ce_delta_swap={result['swap']['block_ce_delta']:.4g} "
        f"ce_delta_perturb={result['perturb_span']['block_ce_delta']:.4g} passed={result['passed']}"
    )
    return result
