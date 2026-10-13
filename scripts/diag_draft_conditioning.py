#!/usr/bin/env python3
"""Dev diagnostic: bisect the trained-vs-eval draft conditioning gap.

For one regenerated sample this compares, at the same anchor positions:

* TRAIN-STYLE: the exact DFlash block the training loss supervises
  (full-sequence fused hidden + DFlash block mask + block positions = anchor+offset).
* EVAL-STYLE: the first speculative round of ``spec_generate``
  (prompt-only hidden context, arange positions, MRoPE future offsets from the
  last prompt position, fresh DynamicCache, ``attention_mask=None``).

and reports: MRoPE/1D position equality between the two paths, per-slot
predictions vs the ground-truth continuation, and the logit difference between
the two conditioning paths for the same anchor.

Not part of the v0.1 pipeline; used to locate the Qwen3-VL acceptance bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.adapters import adapter_for_model
from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.config import apply_overrides, load_config, resolve_run_dir
from vlm_dflash.contract import ensure_contract
from vlm_dflash.data.sharegpt import load_sharegpt, row_answer, row_image_path, row_question
from vlm_dflash.dflash.factory import build_trainer_module
from vlm_dflash.target.capture import forward_and_capture
from vlm_dflash.target.load import load_target
from vlm_dflash.target.prompts import encode_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default="ckpt_final.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--anchor-offsets", default="0,32,80")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    run_dir = resolve_run_dir(cfg, args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    device = torch.device(cfg["target"].get("device", "cuda:0"))
    dtype = getattr(torch, str(cfg["target"].get("dtype", "bfloat16")))
    block_size = int(cfg["drafter"]["block_size"])
    mask_token_id = int(cfg["drafter"]["mask_token_id"])
    image_token_id = int(contract["tokens"]["image_token_id"])
    # Extract exactly the layers the algorithm's draft model consumes (the
    # routing for DFlash; the explicit context list for DFlare/DFly).
    algorithm = get_algorithm(str(cfg.get("algorithm", "dflash")))
    draft_config = algorithm.build_draft_config(contract, cfg)
    dflash_config = dict(getattr(draft_config, "dflash_config", {}) or {})
    layer_ids = [
        int(layer)
        for layer in (dflash_config.get("context_layer_ids") or contract["layers"]["draft_layer_ids"])
    ]
    merge_size = int(contract["image"]["merge_size"])

    model, processor = load_target(cfg, contract)
    model.eval()
    rows, media_dir = load_sharegpt(run_dir / "regen" / "target_data.jsonl", meta_path=run_dir / "data" / "meta.json")
    row = rows[args.sample_index]
    image = Image.open(row_image_path(row, media_dir)).convert("RGB")
    inputs = encode_prompt(processor, row_question(row), image)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    stored_prompt = list(row["regen"]["prompt_ids"])
    answer_ids = list(row["regen"]["answer_ids"])
    prompt_len = len(stored_prompt)
    full_ids = torch.tensor(stored_prompt + answer_ids, dtype=torch.long)
    seq_len = full_ids.numel()
    print(
        f"sample {row.get('id')} prompt={prompt_len} answer={len(answer_ids)} seq={seq_len}",
        flush=True,
    )

    with torch.inference_mode():
        record = forward_and_capture(
            model, inputs, full_ids, layer_ids, image_token_id, dtype, merge_size=merge_size
        )
    hidden = record["hidden"].to(device=device, dtype=torch.bfloat16)
    position3 = record["position3"].to(device=device, dtype=torch.long)
    print(f"capture hidden {tuple(hidden.shape)} positions {tuple(position3.shape)}", flush=True)

    # ---- positions: prompt-only builder (the exact call spec_generate makes) ----
    adapter = adapter_for_model(model)
    prompt_mm = adapter.call_position_builder(
        model,
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=None,
        inputs_embeds=None,
        attention_mask=inputs.get("attention_mask"),
        past_key_values=None,
        second_per_grid_ts=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    print(f"prompt mm {tuple(prompt_mm.shape)} max={int(prompt_mm.max())}", flush=True)
    prompt_vs_capture = (prompt_mm[:, :, :prompt_len] - position3[:, :prompt_len]).abs().max().item()
    future_offsets = torch.arange(1, block_size + 1, device=device).view(1, 1, -1)
    future_mm = prompt_mm[:, :, -1:] + future_offsets
    future_vs_capture = (
        future_mm - position3[:, prompt_len : prompt_len + block_size].unsqueeze(0)
    ).abs().max().item()
    print(
        f"mm check: prompt vs capture max|diff|={prompt_vs_capture}  "
        f"future(last+1..) vs capture max|diff|={future_vs_capture}",
        flush=True,
    )
    print(f"prompt_mm last={prompt_mm[:, :, -1].tolist()} capture[L]={position3[:, prompt_len].tolist()}", flush=True)

    # ---- draft / trainer module ----
    target_head = torch.load(run_dir / "features" / "target_head.pt", map_location="cpu", weights_only=False)
    module = build_trainer_module(
        draft_config, contract, cfg, target_head["embed_tokens"], lm_head_weight=target_head.get("lm_head")
    )
    module.to(device)
    module.lm_head.to(device)
    module.embed_tokens.to(device)
    module.draft_model.to(torch.bfloat16)
    checkpoint = torch.load(run_dir / "train" / args.checkpoint, map_location="cpu", weights_only=False)
    module.draft_model.load_state_dict(checkpoint["model"])
    module.eval()
    module.draft_model.eval()
    print(
        f"draft spatial_rope={module.draft_model.spatial_rope_enabled} "
        f"mode={module.draft_model.spatial_rope_mode} interleaved={module.draft_model.spatial_rope_interleaved} "
        f"sections={module.draft_model.spatial_rope_sections}",
        flush=True,
    )

    offsets = [int(part) for part in args.anchor_offsets.split(",") if part != ""]
    anchors = torch.tensor([[prompt_len + off for off in offsets]], device=device)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    input_ids_batch = full_ids.unsqueeze(0).to(device)
    loss_mask_batch = torch.cat(
        (
            torch.zeros(prompt_len, dtype=torch.float32, device=device),
            torch.ones(seq_len - prompt_len, dtype=torch.float32, device=device),
        )
    ).unsqueeze(0)
    image_mask = (input_ids_batch == image_token_id).to(device)

    with torch.inference_mode():
        anchor_positions, block_keep_mask, noise_embedding, full_position_ids, attn_mask, _ = (
            module._prepare_block_inputs(input_ids_batch, loss_mask_batch, position_ids=None)
        )
        anchor_mm = torch.gather(
            position3.unsqueeze(1), 2, anchor_positions.clamp(min=0).unsqueeze(0).expand(3, -1, -1)
        )
        draft_mm = anchor_mm.unsqueeze(-1) + torch.arange(block_size, device=device).view(1, 1, 1, -1)
        full_mm = torch.cat([position3.unsqueeze(1), draft_mm.flatten(2)], dim=-1)
        train_out = module.draft_model(
            position_ids=full_position_ids,
            multimodal_position_ids=full_mm,
            noise_embedding=noise_embedding,
            target_hidden=hidden.unsqueeze(0),
            attention_mask=attn_mask,
            image_mask=image_mask,
        )
        # Same pre-head hook as the training forward (identity for DFlash/DFly
        # without correction; DFly's TreeFlash correction otherwise), with
        # teacher-forced previous tokens. Inference applies the correction with
        # the previously *proposed* tokens instead (documented local
        # adaptation); this diagnostic keeps the train/eval comparison on the
        # same conditioning convention.
        label_indices, _, _ = module._build_block_targets(
            input_ids_batch,
            loss_mask_batch,
            anchor_positions,
            block_keep_mask,
            seq_len,
            label_start=int(module.input_shift),
        )
        prev_token_ids = module._teacher_forced_prev_tokens(
            input_ids_batch, label_indices, anchor_positions.size(1)
        )
        train_out = module._correct_draft_hidden(
            train_out,
            prev_token_ids=prev_token_ids,
            label_indices=label_indices,
            bsz=1,
            n=anchor_positions.size(1),
            bs=block_size,
        )
        train_logits = module.lm_head(train_out)  # [1, N*bs, V]
    print(f"train anchors {anchor_positions.tolist()} keep={block_keep_mask.tolist()}", flush=True)

    # ---- eval-style round 1 (spec_generate replication) ----
    bonus_id = int(input_ids_batch[0, prompt_len])
    block_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
    block_ids[0, 0] = bonus_id
    eval_positions = torch.arange(prompt_len + block_size, device=device).unsqueeze(0)
    eval_mm = torch.cat([prompt_mm, future_mm], dim=-1)
    with torch.inference_mode():
        eval_out = module.draft_model(
            target_hidden=hidden[:prompt_len].unsqueeze(0),
            noise_embedding=module.embed_tokens(block_ids),
            position_ids=eval_positions,
            multimodal_position_ids=eval_mm,
            past_key_values=DynamicCache(),
            use_cache=True,
            image_mask=image_mask[:, :prompt_len],
        )
        # Teacher-forced correction for the same-conditioning comparison; the
        # real inference path corrects slot k with the token proposed at k-1
        # (see ``Qwen3DFlyDraftModel._sample_draft_tokens``). Block position k
        # sits at input position prompt_len + k and conditions on the token at
        # prompt_len + k - 1, matching the trainer's label convention.
        eval_prev = input_ids_batch[:, prompt_len - 1 : prompt_len + block_size - 1]
        eval_out = module._correct_draft_hidden(
            eval_out,
            prev_token_ids=eval_prev,
            label_indices=eval_positions[:, -block_size:],
            bsz=1,
            n=1,
            bs=block_size,
        )
        eval_logits = module.lm_head(eval_out)  # [1, bs, V]

    # 1D positions: training block slice vs eval slice.
    train_block_positions = full_position_ids[0, seq_len : seq_len + block_size]
    print(
        f"1D positions: train block {train_block_positions.tolist()} "
        f"eval last {eval_positions[0, -block_size:].tolist()}",
        flush=True,
    )

    print("\n--- per-anchor predictions (slot 1..7) ---", flush=True)
    for index, off in enumerate(offsets):
        anchor = prompt_len + off
        truth = input_ids_batch[0, anchor + 1 : anchor + block_size].tolist()
        train_pred = train_logits[0, index * block_size + 1 : (index + 1) * block_size, :].argmax(-1).tolist()
        train_match = [pred == target for pred, target in zip(train_pred, truth)]
        line = f"anchor {anchor} (L+{off}) truth {truth}\n  train pred {train_pred} match {sum(train_match)}/{len(truth)}"
        if off == 0:  # the eval path has no block at this anchor other than round 1
            eval_pred = eval_logits[0, 1:, :].argmax(-1).tolist()
            diff = (train_logits[0, 1:block_size, :].float() - eval_logits[0, 1:, :].float()).abs().max().item()
            eval_match = [pred == target for pred, target in zip(eval_pred, truth)]
            line += (
                f"\n  eval  pred {eval_pred} match {sum(eval_match)}/{len(truth)}"
                f"\n  train-vs-eval logits max|diff| = {diff:.4f}"
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
