"""Train-stage check that the drafter conditions on the captured visual context.

Consumes the swap pair saved by the capture visual check: the same token ids
with target features from image A vs image B must produce different draft block
logits and different block losses; perturbing the image-span feature rows must
also move them. Anchors are resampled under a fixed seed before every pass, so
the only difference between passes is the context features. Logits/losses are
compared in fp32 (the model itself stays in its trained bf16 dtype).

Both passes run the full drafter path the training forward uses -- including the
algorithm's pre-head hook (``module._correct_draft_hidden``, e.g. DFly's
TreeFlash correction with teacher-forced previous tokens) -- and report the
pre-hook ("backbone") and post-hook ("full") logits separately. This is a
teacher-forced diagnostic over fixed anchors, not a real rollout: the previous
tokens are gathered from ``input_ids``, never sampled.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..utils import log, write_json

_ANCHOR_SEED = 1234
_MIN_DELTA = 1e-5


def draft_block_logits(module, input_ids, loss_mask, hidden, position3):
    """Pre- and post-hook draft block logits for one batch.

    Mirrors the trainer forward's prologue (feature slice, fixed-seed anchors,
    block inputs, MRoPE block positions), then runs the same teacher-forced
    previous-token / pre-head hook the training forward applies. Returns
    ``(backbone_logits, full_logits, anchors, keep_mask)``: ``backbone_logits``
    are the raw backbone output through the frozen LM head, ``full_logits`` the
    served distribution (DFly's correction included; identical to the backbone
    for every hook that is the identity, and at correction zero-init).
    """
    # Superset stores hold more layers than the model consumes; apply the same
    # explicit-id slice the training forward applies before the draft forward.
    hidden = module._select_context_layers(hidden)
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
    backbone_logits = module.lm_head(output_hidden)
    bsz, seq_len = input_ids.shape
    n = anchors.shape[1]
    bs = int(module.block_size)
    label_indices, _target_ids, _block_mask = module._build_block_targets(
        input_ids,
        loss_mask,
        anchors,
        keep_mask,
        seq_len,
        label_start=int(getattr(module, "input_shift", 0)),
    )
    prev_token_ids = module._teacher_forced_prev_tokens(input_ids, label_indices, n)
    corrected_hidden = module._correct_draft_hidden(
        output_hidden,
        prev_token_ids=prev_token_ids,
        label_indices=label_indices,
        bsz=bsz,
        n=n,
        bs=bs,
    )
    return backbone_logits, module.lm_head(corrected_hidden), anchors, keep_mask


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


def check_drafter_visual_dependence(
    cfg: dict, contract: dict, run_dir: str | Path, device, module, pair_path: str | Path | None = None
) -> dict:
    """Pair-logits/loss sensitivity check against a recorded swap pair.

    ``pair_path`` defaults to ``features/swap_pair.pt`` (staged capture); the
    online path passes ``online/swap_pair.pt`` so the check uses the live
    source's own pair instead of a leftover from an earlier staged run.
    """
    run_dir = Path(run_dir)
    pair_path = Path(pair_path) if pair_path is not None else run_dir / "features" / "swap_pair.pt"
    out_path = run_dir / "train" / "visual_path_check.json"
    if not pair_path.exists():
        try:
            shown = pair_path.relative_to(run_dir)
        except ValueError:
            shown = pair_path
        result = {"skipped": True, "reason": f"{shown} missing (the visual capture check did not run)"}
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
        backbone_a, logits_a, anchors, keep_mask = draft_block_logits(
            module, input_ids, loss_mask, hidden_a, position3
        )
        backbone_b, logits_b, _, _ = draft_block_logits(module, input_ids, loss_mask, hidden_b, position3)
        torch.manual_seed(_ANCHOR_SEED)
        span = hidden_a[:, span_start : span_end + 1].float()
        # Scale-relative perturbation (half the span's std): large enough that the
        # induced block-CE change stays comfortably above the 1e-5 floor even on a
        # well-trained drafter whose loss is locally flat in the context features.
        sigma = max(0.5 * float(span.std()), 1e-3)
        noise = torch.randn_like(span) * sigma
        perturbed = hidden_a.clone()
        perturbed[:, span_start : span_end + 1] = (span + noise).to(perturbed.dtype)
        backbone_p, logits_p, _, _ = draft_block_logits(module, input_ids, loss_mask, perturbed, position3)

        def ce(logits):
            return fp32_block_ce(module, logits, input_ids, loss_mask, anchors, keep_mask)

        loss_a, loss_b, loss_p = ce(logits_a), ce(logits_b), ce(logits_p)
        backbone_loss_a, backbone_loss_b, backbone_loss_p = (
            ce(backbone_a),
            ce(backbone_b),
            ce(backbone_p),
        )
        swap_logits_diff = (logits_a.float() - logits_b.float()).abs()
        perturb_logits_diff = (logits_a.float() - logits_p.float()).abs()
        backbone_swap_diff = (backbone_a.float() - backbone_b.float()).abs()
        backbone_perturb_diff = (backbone_a.float() - backbone_p.float()).abs()
        full_vs_backbone = (logits_a.float() - backbone_a.float()).abs()

    correction = getattr(module.draft_model, "hidden_correction", None)
    correction_delta = getattr(module, "_last_correction_delta_abs_mean", None)
    result = {
        # Teacher-forced pre-head hook: enabled for DFly, identity otherwise.
        "correction": {
            "enabled": bool(correction is not None),
            "applied_in_diagnostic": True,
            "delta_abs_mean": None if correction_delta is None else float(correction_delta),
        },
        # Served distribution: backbone through the correction hook (the path
        # inference runs; the gate below is decided on these numbers).
        "full": {
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
        },
        # Pre-hook backbone only: kept so a correction that hides a context
        # insensitivity (or adds one) stays visible in the same report.
        "backbone": {
            "swap": {
                "logits_max_abs_diff": float(backbone_swap_diff.max()),
                "logits_mean_abs_diff": float(backbone_swap_diff.mean()),
                "block_ce_a": backbone_loss_a,
                "block_ce_b": backbone_loss_b,
                "block_ce_delta": abs(backbone_loss_a - backbone_loss_b),
            },
            "perturb_span": {
                "logits_max_abs_diff": float(backbone_perturb_diff.max()),
                "logits_mean_abs_diff": float(backbone_perturb_diff.mean()),
                "block_ce_a": backbone_loss_a,
                "block_ce_perturbed": backbone_loss_p,
                "block_ce_delta": abs(backbone_loss_a - backbone_loss_p),
            },
        },
        "full_vs_backbone": {
            "logits_max_abs_diff": float(full_vs_backbone.max()),
            "logits_mean_abs_diff": float(full_vs_backbone.mean()),
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
        f"logits_diff={result['full']['swap']['logits_max_abs_diff']:.4g} "
        f"ce_delta_swap={result['full']['swap']['block_ce_delta']:.4g} "
        f"ce_delta_perturb={result['full']['perturb_span']['block_ce_delta']:.4g} "
        f"correction_delta={result['correction']['delta_abs_mean']} passed={result['passed']}"
    )
    return result
