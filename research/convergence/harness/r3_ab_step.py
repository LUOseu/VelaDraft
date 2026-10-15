#!/usr/bin/env python3
"""D1 R3: within-logical-batch physical regrouping A/B on a frozen feature store.

For every logical batch (``--batch-size`` samples; one optimizer update per
logical batch in every arm) the harness freezes ONE anchor table, drawn with
the real sampler under the canonical padded layout, then replays it in every
arm so the supervision set cannot differ between arms.

Arms:
  canonical : shipped physical layout -- reference microbatches of
              ``--ref-micro-batch`` rows, per-microbatch decay-weighted MEAN,
              ``(loss / accumulation).backward()``.
  cp        : same layout as canonical but the decay weights computed in fp32
              (PreciseDecayLoss, mean normalisation) -- isolates the shipped
              loss's bf16 weight quantisation from any regrouping effect.
  ref       : the reference layout but with the preserved per-sample weights
              (sanity arm: must equal canonical).
  split     : same physical count as the sorted arms (n / --sub-batch groups)
              but UNSORTED positional split + preserved weights -- the
              equal-k control that isolates sorting from splitting.
  token     : plan_groups(mode="token")  + preserved per-sample weights
              g_i = 1 / (R * Z_ref_mb(i)).
  dual      : plan_groups(mode="dual")   + same weights.
  cost      : plan_groups(mode="cost")   + same weights (fitted C + w*A with
              w = d1_bucketing.COST_ANCHOR_WEIGHT from the R2 tensor probe).
  grid      : plan_groups(mode="grid")   + same weights (2D quantile partition).

Weighted arms swap the loss to ``normalize="tokens"`` and scale the block mask
by g_i (group row order), so the accumulated objective equals the canonical
reference exactly (proof: tests/oracle_regroup_parity.py; this harness
re-checks loss/grad/update parity on real store data, in bf16).

Measured per (logical batch, arm): fwd+bwd+grad-clip wall with CUDA sync, peak
allocated bytes, padding slots (context + block), loss and parity vs canonical
on the same batch. Arms alternate per batch; the first ``--warmup`` batches are
untimed. Weights are reset to the initial state before every arm run.

Usage:
  python r3_ab_step.py --store .../features --run-dir ... --config configs/qwen2_5_vl_3b_2k.yaml \\
    --device cuda:0 --batches 6 --warmup 1 --batch-size 8 --sub-batch 4 \\
    --arms canonical,token,dual,cost --out .../r3_ab_w0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

import torch

_HERE = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start.parents[2]
sys.path.insert(0, str(_HERE))  # d1_bucketing
_REPO_DEFAULT = _find_repo_root(_HERE)
sys.path.insert(0, os.environ.get("REPO", str(_REPO_DEFAULT)))

import d1_bucketing as d1b  # noqa: E402

from vlm_dflash.algorithms import get_algorithm  # noqa: E402
from vlm_dflash.config import load_config  # noqa: E402
from vlm_dflash.contract import contract_capture_layer_ids, ensure_contract  # noqa: E402
from vlm_dflash.dflash.decay_loss import DFlashDecayLoss  # noqa: E402
from vlm_dflash.dflash.features import (  # noqa: E402
    FeatureDataset,
    capture_layer_layout,
    collate_features,
)
from vlm_dflash.utils import read_json  # noqa: E402


def log(message: str) -> None:
    print(f"[r3] {message}", flush=True)


class PreciseDecayLoss(DFlashDecayLoss):
    """Weighted reduction with the decay*scale product applied in fp32.

    The stock `_reduce` casts the (already scaled) block mask to `token_nll.dtype`
    (bf16 in training), which quantizes the preserved per-sample weights g_i to
    ~0.4% -- enough to show up as a spurious half-percent loss gap between two
    mathematically identical arms (measured in smoke2: ref(1x8 weighted) vs
    canonical = 0.51%). Casting to fp32 before the product removes that artifact
    without changing any semantics; the remaining difference is real bf16 feature
    drift from the physical-composition change.
    """

    def _reduce(self, token_nll, block_mask, num_tokens, block_size,
                draft_correct_per_pos=None, draft_count_per_pos=None):
        return super()._reduce(token_nll.float(), block_mask.float(), num_tokens, block_size,
                               draft_correct_per_pos=draft_correct_per_pos,
                               draft_count_per_pos=draft_count_per_pos)


def build_module(args):
    cfg = load_config(args.config)
    algorithm = get_algorithm(cfg.get("algorithm", "dflash"))
    contract = ensure_contract(cfg, args.run_dir)
    manifest = read_json(Path(args.store) / "manifest.json")
    target_head = torch.load(Path(args.store) / "target_head.pt", map_location="cpu", weights_only=False)
    if target_head["model_fingerprint"] != contract["target"]["model_fingerprint"]:
        raise RuntimeError("target_head.pt fingerprint mismatch with contract")
    lm_head_weight = target_head.get("lm_head")
    embed_weight = target_head["embed_tokens"]
    draft_config = algorithm.build_draft_config(contract, cfg)
    module = algorithm.build_trainer_module(
        draft_config, contract, cfg, embed_weight, lm_head_weight=lm_head_weight
    )
    store_capture, store_hidden = capture_layer_layout(
        manifest, contract_hidden_size=int(contract["text"]["hidden_size"])
    )
    if store_capture != contract_capture_layer_ids(contract):
        raise RuntimeError("store capture layers differ from contract")
    module.configure_feature_layout(store_capture, store_hidden)
    image_token_id = int(contract["tokens"]["image_token_id"])
    spatial = bool(getattr(module, "uses_multimodal_position_ids",
                           getattr(module.draft_model, "spatial_rope_enabled", False)))
    return module, cfg, contract, image_token_id, spatial


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def pad_to(batch: dict, target_len: int) -> dict:
    """Right-pad a collated batch (zeros, as collate_features does)."""
    current = batch["input_ids"].shape[1]
    extra = target_len - current
    if extra <= 0:
        return batch
    out = dict(batch)
    for key in ("input_ids", "loss_mask", "hidden", "final_hidden"):
        if key in out:
            shape = list(out[key].shape)
            shape[1] = extra
            out[key] = torch.cat([out[key], torch.zeros(shape, dtype=out[key].dtype, device=out[key].device)], dim=1)
    position3 = out["position3"]
    out["position3"] = torch.cat(
        [position3, torch.zeros(3, position3.shape[1], extra, dtype=position3.dtype, device=position3.device)], dim=2)
    return out


def make_group_batch(records: list[dict], group: list[int], table: dict[int, list[int]],
                     block_size: int, image_token_id: int, device) -> dict:
    batch = collate_features([records[pos] for pos in group], image_token_id)
    needed = 0
    for pos in group:
        anchors = table[pos]
        if anchors:
            needed = max(needed, max(anchors) + block_size)
    batch = pad_to(batch, max(batch["input_ids"].shape[1], needed))
    batch["image_mask"] = batch["input_ids"] == image_token_id
    lengths = batch["lengths"]
    return to_device(batch, device), lengths


def replay_anchors(table: dict[int, list[int]], group: list[int], num_anchors: int, device):
    counts = [len(table[pos]) for pos in group]
    max_n = min(num_anchors, max(counts)) if counts else 0
    anchors = torch.zeros(len(group), max_n, dtype=torch.long, device=device)
    keep = torch.zeros(len(group), max_n, dtype=torch.bool, device=device)
    for row, pos in enumerate(group):
        positions = table[pos][:max_n]
        anchors[row, : len(positions)] = torch.tensor(positions, dtype=torch.long, device=device)
        keep[row, : len(positions)] = True
    return anchors, keep


def wrapped_order(dataset_len: int, n_batches: int, warmup: int, batch_size: int, seed: int,
                  shuffle: bool = True) -> list:
    """Sample order covering ``(warmup + n_batches) * batch_size`` rows, cycling the
    dataset with one permuted epoch per pass (epoch 0 = legacy single randperm, so
    runs that fit in one epoch keep their exact historical row order)."""
    needed = (warmup + n_batches) * batch_size
    order = []
    epoch = 0
    while len(order) < needed:
        if shuffle:
            generator = torch.Generator().manual_seed(seed + 1000 * epoch)
            order.extend(torch.randperm(dataset_len, generator=generator).tolist())
        else:
            order.extend(range(dataset_len))
        epoch += 1
    return order[:needed]


def snapshot_grads(module):
    return {name: p.grad.detach().clone() for name, p in module.draft_model.named_parameters() if p.grad is not None}


def tensor_rel_err(a: dict, b: dict) -> float:
    worst = 0.0
    for name in a:
        if name not in b:
            return float("inf")
        denom = b[name].float().norm().item() + 1e-12
        worst = max(worst, (a[name].float() - b[name].float()).norm().item() / denom)
    return worst


class ArmHarness:
    def __init__(self, module, device, block_size, first_slot, gamma, num_anchors, max_grad_norm, lr):
        self.module = module
        self.device = device
        self.block_size = block_size
        self.first_slot = first_slot
        self.gamma = gamma
        self.num_anchors = num_anchors
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.orig_build = module._build_block_targets
        self.orig_sample = module._sample_anchor_positions
        self.orig_loss_fn = module.loss_fn

    def run_arm(self, arm, batch_records, table, groups, weights, image_token_id, spatial, time_it):
        module = self.module
        # reset weights + hooks, same starting point for every arm
        module._build_block_targets = self.orig_build
        module.loss_fn = self.orig_loss_fn
        for parameter in module.draft_model.parameters():
            parameter.grad = None
        master = {name: p.detach().float().clone() for name, p in module.draft_model.named_parameters()}
        total_loss = 0.0
        z_per_group = []
        shapes = []
        accum = len(groups)
        weighted = weights is not None
        if time_it:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
        for group in groups:
            batch, lengths = make_group_batch(batch_records, group, table, self.block_size, image_token_id, self.device)
            anchors, keep = replay_anchors(table, group, self.num_anchors, self.device)

            def sample(_self, seq_len, loss_mask, device, doc_remaining=None, _a=anchors, _k=keep):
                return _a, _k

            module._sample_anchor_positions = MethodType(sample, module)
            if arm == "cp":
                module.loss_fn = PreciseDecayLoss(loss_gamma=self.gamma, normalize="mean",
                                                  first_slot=self.first_slot)
            if weighted:
                weight_row = torch.tensor([weights[pos] for pos in group], dtype=torch.float32)
                orig_build = self.orig_build

                def patched(input_ids, loss_mask, anchor_positions, block_keep_mask, seq_len,
                            label_start=0, doc_remaining=None, _orig=orig_build, _w=weight_row):
                    labels, targets, block_mask = _orig(
                        input_ids, loss_mask, anchor_positions, block_keep_mask, seq_len, label_start, doc_remaining)
                    return labels, targets, block_mask * _w.to(block_mask.device).view(-1, 1, 1).to(block_mask.dtype)

                module._build_block_targets = MethodType(lambda _self, *a, **k: patched(*a, **k), module)
                module.loss_fn = PreciseDecayLoss(loss_gamma=self.gamma, normalize="tokens",
                                                  first_slot=self.first_slot)
            metrics = module(
                input_ids=batch["input_ids"],
                hidden_states=batch["hidden"],
                loss_mask=batch["loss_mask"],
                image_mask=batch["image_mask"],
                multimodal_position_ids=batch["position3"] if spatial else None,
                teacher_hidden=batch.get("final_hidden"),
            )
            z_per_group.append(float(metrics.loss_weight))
            if weighted:
                total_loss += float(metrics.loss.detach())
                metrics.loss.backward()
            else:
                total_loss += float(metrics.loss.detach()) / accum
                (metrics.loss / accum).backward()
            shapes.append({
                "group": [int(pos) for pos in group],
                "S": int(batch["input_ids"].shape[1]),
                "N": int(anchors.shape[1]),
                "context_slots": int(batch["input_ids"].shape[0] * batch["input_ids"].shape[1]),
                "block_slots": int(anchors.shape[0] * anchors.shape[1] * self.block_size),
                "real_tokens": int(lengths.sum().item()),
            })
        grads = snapshot_grads(module)
        for name, parameter in module.draft_model.named_parameters():
            master[name].grad = None if parameter.grad is None else parameter.grad.detach().float().clone()
            parameter.grad = None
        grad_norm = torch.nn.utils.clip_grad_norm_(list(master.values()), self.max_grad_norm)
        updated = {name: (param - self.lr * param.grad).clone() if param.grad is not None else param.clone()
                   for name, param in master.items()}
        if time_it:
            torch.cuda.synchronize(self.device)
            wall_ms = (time.perf_counter() - start) * 1000.0
            peak = torch.cuda.max_memory_allocated(self.device)
        else:
            wall_ms, peak = None, None
        # restore hooks
        module._build_block_targets = self.orig_build
        module.loss_fn = self.orig_loss_fn
        module._sample_anchor_positions = self.orig_sample
        return {
            "arm": arm, "groups": [list(group) for group in groups], "loss": total_loss,
            "z_per_group": z_per_group, "grad_norm": float(grad_norm), "wall_ms": wall_ms,
            "peak_alloc_bytes": peak, "shapes": shapes, "grads": grads, "updated": updated,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sub-batch", type=int, default=4)
    parser.add_argument("--ref-micro-batch", type=int, default=8)
    parser.add_argument("--arms", default="canonical,cp,ref,token,dual,cost")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="iterate the store index in order (for interleaved mixed stores); "
                             "default = seeded randperm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity", type=int, default=1,
                        help="max logical batches with full grad/update parity (0=none); records vs canonical AND vs ref (ref must be in --arms, else vs-ref is null)")
    parser.add_argument("--parity-batches", default=None,
                        help="comma-separated logical batch indices with full grad/update parity "
                             "(overrides --parity), e.g. '0,50,99' for first/mid/last; "
                             "loss-vs-ref is recorded on every batch regardless")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    module, cfg, contract, image_token_id, spatial = build_module(args)
    train_cfg = cfg.get("train", {})
    harness = ArmHarness(
        module, device, block_size=int(cfg["drafter"]["block_size"]), first_slot=1,
        gamma=cfg["drafter"].get("loss_decay_gamma"),
        num_anchors=int(cfg["drafter"]["num_anchors"]),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        lr=float(train_cfg.get("lr", 6e-4)),
    )
    module.to(device)
    module.lm_head.to(device)
    module.embed_tokens.to(device)
    draft_dtype = str(train_cfg.get("draft_dtype", "bfloat16"))
    module.draft_model.to(torch.bfloat16 if draft_dtype == "bfloat16" else torch.float32)
    module.train()
    state0 = {name: value.detach().clone() for name, value in module.draft_model.state_dict().items()}

    dataset = FeatureDataset(args.store)
    log(f"store: {len(dataset)} rows; module built ({sum(p.numel() for p in module.parameters())} params)")
    if args.build_only:
        return 0
    order = wrapped_order(len(dataset), args.batches, args.warmup, args.batch_size,
                          args.seed, shuffle=not args.no_shuffle)
    logical_batches = [order[i:i + args.batch_size]
                       for i in range(0, (args.warmup + args.batches) * args.batch_size, args.batch_size)]
    records = [dataset[index] for index in order[: (args.warmup + args.batches) * args.batch_size]]
    parity_batches = None
    if args.parity_batches is not None:
        parity_batches = {int(tok) for tok in str(args.parity_batches).split(",") if tok.strip() != ""}
    pos_records = records  # aligned to positions inside each logical batch

    arms = [arm for arm in args.arms.split(",") if arm]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("REPO", str(_REPO_DEFAULT))
    git_sha = os.environ.get("D1_GIT_SHA", "unknown")
    results = []
    for b, rows in enumerate(logical_batches):
        warmup = b < args.warmup
        batch_records = records[b * args.batch_size:(b + 1) * args.batch_size]
        positions = list(range(len(rows)))
        batch_c, lengths = make_group_batch(batch_records, positions, {p: [] for p in positions}, 1,
                                            image_token_id, device)
        torch.manual_seed(args.seed * 100003 + b)
        anchors, keep = harness.orig_sample(batch_c["input_ids"].shape[1], batch_c["loss_mask"], device)
        table = {}
        for pos in positions:
            count = int(keep[pos].sum().item())
            table[pos] = anchors[pos, :count].tolist()
        z_rows = d1b.sample_z_rows(batch_c["loss_mask"], anchors.to(device), keep,
                                   harness.block_size, harness.first_slot, harness.gamma)
        real_lengths = batch_c["lengths"].tolist()
        ref_groups = d1b.canonical_microbatches(len(positions), args.ref_micro_batch)
        keys = [f"pos{pos}" for pos in positions]  # positional keys: unique even if
        # duplicate sample_ids appear (the W1 pool contains one duplicated row)
        weights = d1b.row_weights(keys, ref_groups, z_rows, accumulation=len(ref_groups))
        weights_pos = {positions[i]: weights[keys[i]] for i in positions}
        stats = [d1b.SampleStat(key=keys[i], C=int(real_lengths[i]), A=len(table[i]),
                                Z=float(z_rows[i]), index=i) for i in positions]
        canonical_loss = None
        canonical_grads = None
        canonical_updated = None
        parity = (b in parity_batches) if parity_batches is not None else (b < args.parity)
        # canonical always runs first (its loss/grads are the parity reference);
        # the other arms rotate order per batch to spread order/thermal bias.
        others = [arm for arm in arms if arm != "canonical"]
        shift = b % len(others) if others else 0
        batch_arms = (["canonical"] if "canonical" in arms else []) + others[shift:] + others[:shift]
        batch_rows = []
        parity_store = {}
        batch_losses = {}
        for arm in batch_arms:
            module.draft_model.load_state_dict(state0)
            if arm == "canonical":
                groups, weights_arg = ref_groups, None
            elif arm == "cp":
                groups, weights_arg = ref_groups, None
            elif arm == "ref":
                groups, weights_arg = ref_groups, weights_pos
            elif arm == "split":
                groups = d1b.canonical_microbatches(len(positions), args.sub_batch)
                weights_arg = weights_pos
            else:
                groups = d1b.plan_groups(stats, arm, args.sub_batch)
                weights_arg = weights_pos
            try:
                result = harness.run_arm(arm, batch_records, table, groups, weights_arg,
                                         image_token_id, spatial, time_it=not warmup)
            except RuntimeError as exc:  # capacity runs may legitimately OOM on the control arm
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                oom_row = {
                    "arm": arm, "groups": [list(g) for g in groups], "loss": None,
                    "z_per_group": [], "grad_norm": None, "wall_ms": None, "peak_alloc_bytes": None,
                    "shapes": [], "logical_batch": b, "warmup": warmup, "run_error": "cuda_oom",
                    "error": str(exc)[:300], "rows": [int(r) for r in rows],
                    "table_A": [len(table[p]) for p in positions],
                    "C": [int(c) for c in real_lengths],
                    "ref_groups": ref_groups, "ref_micro_batch": args.ref_micro_batch,
                    "sub_batch": args.sub_batch, "tag": args.tag, "git_sha": git_sha,
                    "store": str(args.store), "device": str(device),
                    "batch_seed": args.seed * 100003 + b,
                    "num_anchors": harness.num_anchors, "block_size": harness.block_size,
                    "loss_decay_gamma": harness.gamma, "draft_dtype": draft_dtype,
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                batch_rows.append(oom_row)
                log(f"b{b} {arm}: CUDA OOM recorded (capacity evidence), continuing")
                continue
            result.update({
                "logical_batch": b, "warmup": warmup, "rows": [int(r) for r in rows],
                "table_A": [len(table[p]) for p in positions],
                "C": [int(c) for c in real_lengths],
                "ref_groups": ref_groups, "ref_micro_batch": args.ref_micro_batch,
                "sub_batch": args.sub_batch, "tag": args.tag, "git_sha": git_sha,
                "store": str(args.store), "device": str(device), "batch_seed": args.seed * 100003 + b,
                "num_anchors": harness.num_anchors, "block_size": harness.block_size,
                "loss_decay_gamma": harness.gamma, "draft_dtype": draft_dtype,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            })
            if arm == "canonical":
                canonical_loss = result["loss"]
                z_canon = sum(result["z_per_group"])
                z_rows_sum = sum(z_rows)
                result["z_check_rel"] = abs(z_canon - z_rows_sum) / (abs(z_rows_sum) + 1e-9)
                if parity:
                    canonical_grads = {name: value.detach().cpu() for name, value in result["grads"].items()}
                    canonical_updated = {name: value.detach().cpu() for name, value in result["updated"].items()}
            else:
                if canonical_loss is not None:  # canonical itself may OOM in capacity runs
                    result["loss_rel_err"] = abs(result["loss"] - canonical_loss) / (abs(canonical_loss) + 1e-12)
                if parity and canonical_grads is not None:
                    canonical_grads_dev = {name: value.to(device) for name, value in canonical_grads.items()}
                    canonical_updated_dev = {name: value.to(device) for name, value in canonical_updated.items()}
                    result["grad_rel_err"] = tensor_rel_err(result["grads"], canonical_grads_dev)
                    result["update_rel_err"] = tensor_rel_err(result["updated"], canonical_updated_dev)
            batch_losses[arm] = float(result["loss"])
            if parity:
                parity_store[arm] = (
                    float(result["loss"]),
                    {name: value.detach().cpu() for name, value in result["grads"].items()},
                    {name: value.detach().cpu() for name, value in result["updated"].items()},
                )
            out = {key: value for key, value in result.items() if key not in ("grads", "updated")}
            batch_rows.append(out)
        # whole-trajectory loss vs ref (every batch): ref arm = fp32-weighted loss on
        # the canonical physical grouping -> arm-vs-ref isolates pure regrouping drift.
        if "ref" in batch_losses:
            ref_loss_all = batch_losses["ref"]
            for out in batch_rows:
                if out["arm"] == "ref" or out.get("loss") is None:
                    continue
                out["loss_rel_err_vs_ref"] = abs(out["loss"] - ref_loss_all) / (abs(ref_loss_all) + 1e-12)
        # backfill grad/update vs-ref parity (parity batches only; needs CPU copies)
        if parity and "ref" in parity_store:
            ref_loss, ref_grads_cpu, ref_updated_cpu = parity_store["ref"]
            for out in batch_rows:
                if out["arm"] == "ref":
                    continue
                if out["arm"] not in parity_store:  # OOM'd arms have no parity entry
                    continue
                loss_i, grads_i, updated_i = parity_store[out["arm"]]
                out["loss_rel_err_vs_ref"] = abs(loss_i - ref_loss) / (abs(ref_loss) + 1e-12)
                out["grad_rel_err_vs_ref"] = tensor_rel_err(grads_i, ref_grads_cpu)
                out["update_rel_err_vs_ref"] = tensor_rel_err(updated_i, ref_updated_cpu)
        del parity_store
        for out in batch_rows:
            with open(out_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(out) + "\n")
            loss_txt = f"{out['loss']:.6f}" if out.get("loss") is not None else None
            log(f"b{b}{'[warm]' if warmup else ''} {out['arm']}: loss={loss_txt} wall={out['wall_ms']} "
                f"loss_rel={out.get('loss_rel_err')} grad_rel={out.get('grad_rel_err')} "
                f"grad_rel_vs_ref={out.get('grad_rel_err_vs_ref')} "
                f"groups={[len(g) for g in out['groups']]}")
            results.append(out)

    def _max_rel(rows, key):
        vals = [row[key] for row in rows if row.get(key) is not None]
        return max(vals) if vals else None

    timed = [row for row in results if not row["warmup"] and row["wall_ms"] is not None]
    recorded_parity = sorted({int(row["logical_batch"]) for row in results
                              if row.get("grad_rel_err") is not None
                              or row.get("grad_rel_err_vs_ref") is not None})
    requested_parity = (str(args.parity_batches) if args.parity_batches is not None
                        else f"first {args.parity}")
    summary = {"schema": "d1_r3_summary_v2",
               "git_sha": git_sha, "store": args.store, "batch_size": args.batch_size,
               "sub_batch": args.sub_batch, "ref_micro_batch": args.ref_micro_batch,
               "seed": args.seed,
               "parity_batches": recorded_parity,
               "parity_batches_requested": requested_parity,
               "parity_semantics": (
                   "rel err = worst-case ||a-b||/||b|| over draft_model grads/updated params. "
                   "*_vs_canonical: arm vs shipped canonical path on the same logical batch (includes "
                   "canonical's bf16 weight quantisation). *_vs_ref: arm vs ref arm (fp32-weighted loss, "
                   "canonical physical grouping) = pure regrouping drift. loss_rel_err_vs_ref is recorded on "
                   "EVERY batch (whole trajectory); grad/update parity only on --parity-batches / logical_batch "
                   "< --parity (see parity_batches; may include warmup batch 0). null = metric not recorded; "
                   "0.0 is a real value. run_errors counts per-arm failed batches (e.g. cuda_oom)."),
               "arms": {}}
    for arm in arms:
        rows = [row for row in timed if row["arm"] == arm]
        if not rows:
            continue
        parity_rows = [row for row in results if row["arm"] == arm]
        summary["arms"][arm] = {
            "n": len(rows),
            "wall_ms_median": statistics.median(row["wall_ms"] for row in rows),
            "wall_ms_all": [round(row["wall_ms"], 2) for row in rows],
            "peak_alloc_max": max(row["peak_alloc_bytes"] for row in rows),
            "loss_rel_err_max_vs_canonical": _max_rel(rows, "loss_rel_err"),
            "loss_rel_err_max_vs_ref": _max_rel(rows, "loss_rel_err_vs_ref"),
            "grad_rel_err_max_vs_canonical": _max_rel(parity_rows, "grad_rel_err"),
            "grad_rel_err_max_vs_ref": _max_rel(parity_rows, "grad_rel_err_vs_ref"),
            "update_rel_err_max_vs_canonical": _max_rel(parity_rows, "update_rel_err"),
            "update_rel_err_max_vs_ref": _max_rel(parity_rows, "update_rel_err_vs_ref"),
            "parity_batches": sorted({int(row["logical_batch"]) for row in parity_rows
                                      if row.get("grad_rel_err") is not None}),
            "run_errors": sum(1 for row in parity_rows if row.get("run_error")),
        }
    base = summary["arms"].get("canonical", {}).get("wall_ms_median")
    if base:
        for arm, entry in summary["arms"].items():
            entry["speedup_vs_canonical"] = base / entry["wall_ms_median"]
    print(json.dumps(summary, indent=2))
    Path(str(out_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
