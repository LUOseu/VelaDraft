#!/usr/bin/env python3
"""Fixed Experiment B: full DFlash update capacity, with fair attention checkpoints.

Run one cell per process to contain OOM/allocator effects. Shape-only diagnostics
are explicitly distinguished from natural cached-feature training. Checkpoint
scope is mask materialization and SDPA only (not projections, MLP, or loss).
"""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SHAPES = ((1024, 64), (4096, 256), (4096, 512))
MODES = ('dense', 'dense_checkpoint', 'chunked_checkpoint')


def attention_for(mode):
    from vlm_dflash.dflash.chunked_attention import sdpa_chunked_attention, _repeat_kv
    if mode == 'chunked_checkpoint':
        return sdpa_chunked_attention
    if mode not in MODES:
        raise ValueError(mode)

    def attention(query, key, value, spec, *, dropout_p, scale):
        repeats = query.shape[1] // key.shape[1]
        key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
        def run(q, k, v):
            mask = spec.materialize_additive(0, spec.query_len, dtype=q.dtype)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 dropout_p=dropout_p, is_causal=False, scale=scale)
        enabled = mode == 'dense_checkpoint' and torch.is_grad_enabled() and query.requires_grad
        output = checkpoint(run, query, key, value, use_reentrant=False) if enabled else run(query, key, value)
        return output, dict(requested_backend=mode, effective_backend=mode, chunks=1,
                            query_chunk_size=spec.query_len, activation_checkpointing=enabled,
                            fallback_reason=None)
    return attention



def create_audit_optimizer(module, horizon):
    from scripts.npu.audit_common import create_optimizer
    # Formal Qwen2.5-VL-3B DFlash optimizer settings; diagnostic horizon fixed
    # before timing, shared by all three attention execution paths.
    cfg = {'train': {'lr': 6e-4, 'betas': [.9, .95], 'weight_decay': 0.,
                     'warmup_ratio': .04, 'min_lr_ratio': .1}}
    return create_optimizer(module, cfg, horizon)


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + '\n')



def compare_artifacts(reference_path, candidate_path):
    """Compare every output and trainable gradient using frozen H1 BF16 tolerances."""
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    if reference['grads'].keys() != candidate['grads'].keys():
        raise ValueError('Trainable gradient key sets differ')
    comparisons = {}
    pairs = [('output', reference['output'], candidate['output']),
             ('loss', reference['loss'], candidate['loss'])]
    pairs.extend((f'grad:{name}', value, candidate['grads'][name])
                 for name, value in reference['grads'].items())
    for name, left, right in pairs:
        left, right = left.float(), right.float()
        delta = (left - right).abs()
        comparisons[name] = {'max_abs_error': float(delta.max()),
                             'relative_l2_error': float(torch.linalg.vector_norm(left-right) / torch.linalg.vector_norm(left).clamp_min(1e-30)),
                             'pass': bool(torch.all(delta <= .02 + .02 * right.abs()))}
    return {'reference': str(reference_path), 'candidate': str(candidate_path),
            'rtol': .02, 'atol': .02, 'tolerance_source': 'scripts/npu/operator_smoke.py BF16',
            'pass': all(item['pass'] for item in comparisons.values()), 'tensors': comparisons}


def build(args):
    from vlm_dflash.algorithms import get_algorithm
    target = Path(args.target)
    contract = json.loads(Path(args.contract).read_text()) if args.contract else {
        'target': {'model_path': str(target)},
        'layers': {'num_target_layers': 36, 'draft_layer_ids': [1, 17, 33],
                   'num_draft_layers': 3, 'capture_layer_ids': [1, 17, 33]},
        'rope': {'sections': [16, 24, 24], 'interleaved': False},
        'text': {'hidden_size': 2048}, 'tokens': {'image_token_id': 151655},
    }
    contract['target']['model_path'] = str(target)
    cfg = {'algorithm': {'name': 'dflash'}, 'drafter': {
        'num_hidden_layers': 3, 'block_size': 8, 'num_anchors': args.anchors,
        'mask_token_id': 151669, 'loss_decay_gamma': 7., 'attention_backend': 'sdpa_chunked',
        'attention_chunk_size': 128, 'spatial_rope_enabled': True,
        'spatial_rope_mode': 'replace'}, 'train': {'micro_batch_size': 1}}
    alg = get_algorithm('dflash')
    config = alg.build_draft_config(contract, cfg)
    expected = (2048, 11008, 16, 2, 3)
    actual = (config.hidden_size, config.intermediate_size, config.num_attention_heads,
              config.num_key_value_heads, config.num_hidden_layers)
    if actual != expected:
        raise ValueError(f'Formal geometry mismatch: {actual} != {expected}')
    head = torch.load(args.head, map_location='cpu', weights_only=False)
    torch.manual_seed(20261021)
    module = alg.build_trainer_module(config, contract, cfg, head['embed_tokens'], head.get('lm_head'))
    device = torch.device(args.device)
    module.to(device=device, dtype=torch.bfloat16)
    module.embed_tokens.to(device=device, dtype=torch.bfloat16)
    module.lm_head.to(device=device, dtype=torch.bfloat16)
    module.configure_feature_layout(contract['layers']['capture_layer_ids'], config.hidden_size)
    module.train()
    return module, config, contract


def batch_for(args, module, config, contract):
    device = next(module.parameters()).device
    from vlm_dflash import runtime
    runtime.seed_all(197, device)
    source = 'shape_only_synthetic_features_real_frozen_target_head'
    sample_id = None
    actual_length = args.context
    if args.features_dir:
        from vlm_dflash.dflash.features import FeatureDataset, collate_features
        dataset = FeatureDataset(args.features_dir)
        choices = []
        for i in range(len(dataset)):
            row = dataset[i]
            length = row['input_ids'].numel()
            if length <= args.context:
                choices.append((length, str(row['sample_id']), i))
        if not choices:
            raise ValueError('No complete natural sample fits bucket; visual spans will not be truncated.')
        _, sample_id, index = max(choices)
        row = dataset[index]
        batch = collate_features([row], contract['tokens']['image_token_id'])
        actual_length = row['input_ids'].numel()
        ids, hidden, loss, pos3 = batch['input_ids'], batch['hidden'], batch['loss_mask'], batch['position3']
        source = 'natural_cached_features_complete_sample_zero_padded_context'
        pad = args.context - actual_length
        ids = F.pad(ids, (0, pad))
        hidden = F.pad(hidden, (0, 0, 0, pad))
        loss = F.pad(loss, (0, pad))
        pos3 = F.pad(pos3, (0, pad))
    else:
        ids = torch.randint(2, config.vocab_size - 1000, (1, args.context))
        hidden = torch.randn(1, args.context, config.hidden_size * len(contract['layers']['capture_layer_ids'])) * .1
        loss = torch.ones(1, args.context)
        loss[:, :args.context // 4] = 0
        pos3 = torch.arange(args.context).view(1, 1, -1).repeat(3, 1, 1)
    ids, hidden, loss, pos3 = ids.to(device), hidden.to(device, torch.bfloat16), loss.to(device), pos3.to(device)
    # Preserve the real boundary when selecting anchors, even for zero-padded contexts.
    anchors, keep = module._sample_anchor_positions(actual_length, loss[:, :actual_length], device)
    _, _, block_mask = module._build_block_targets(ids, loss, anchors, keep, args.context, label_start=0)
    rows = {'source': source, 'sample_id': sample_id, 'actual_context_length': actual_length,
            'context_bucket': args.context, 'context_padding_fraction': 1 - actual_length / args.context,
            'requested_anchors': args.anchors, 'allocated_anchors': anchors.shape[1],
            'effective_anchors': int(keep.sum()), 'allocated_query_rows': anchors.numel() * 8,
            'effective_supervision_rows': int((block_mask[..., module.first_supervised_slot:] > 0).sum()),
            'dataset_records_considered': len(dataset) if args.features_dir else 0,
            'unique_samples_replayed': 1,
            'anchor_padding_fraction': 1 - float(keep.float().mean()),
            'requested_anchor_count_achieved': anchors.shape[1] == args.anchors,
            'requested_natural_context_extent_achieved': actual_length == args.context,
            'requested_shape_achieved': anchors.shape[1] == args.anchors and actual_length == args.context}
    batch = dict(input_ids=ids, hidden_states=hidden, loss_mask=loss,
                 multimodal_position_ids=pos3, anchor_table=(anchors, keep))
    return batch, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--contract')
    parser.add_argument('--features-dir')
    parser.add_argument('--device', default='npu:1')
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--anchors', type=int, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--measured-updates', type=int, default=3)
    parser.add_argument('--budget-seconds', type=float, default=5400)
    parser.add_argument('--correctness-output')
    args = parser.parse_args()
    if (args.context, args.anchors) not in SHAPES:
        parser.error('Shape must belong to frozen protocol.')
    from vlm_dflash import runtime
    from vlm_dflash.dflash import draft_model
    runtime.resolve_device(args.device)
    torch.npu.set_device(args.device)
    draft_model.sdpa_chunked_attention = attention_for(args.mode)
    record = dict(protocol='reviewer_audit_B_v1', mode=args.mode, context=args.context, anchors=args.anchors,
                  frozen_shapes=SHAPES, dtype='bfloat16', device=args.device,
                  checkpoint_scope='mask materialization plus SDPA only; K/V repeat outside; no MLP/layer/head checkpoint',
                  initial_seed=20261021, optimizer='Production AdamW FP32 masters/moments, BF16 forward/backward; lr=6e-4 betas=(.9,.95) weight_decay=0 clip=1 warmup=.04 min_lr_ratio=.1; shared fixed diagnostic horizon',
                  scope='offline draft full forward/backward/clip/optimizer update; excludes teacher capture',
                  status='running', warmup_updates=[], measured_updates=[])
    record['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    record['git_head'] = revision.stdout.strip() if revision.returncode == 0 else None
    record['git_metadata_note'] = None if revision.returncode == 0 else 'Deployed source tree has no .git; per-file hashes are authoritative.'
    record['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / 'vlm_dflash/dflash').glob('*.py')}
    save(args.output, record)
    begin = time.monotonic()
    try:
        module, config, contract = build(args)
        record['geometry'] = config.to_dict()
        record['hardware'] = runtime.device_report(args.device)
        record['software'] = {'torch': torch.__version__, 'torch_npu': getattr(torch.npu, '__version__', None)}
        record['contract'] = contract
        batch, record['rows'] = batch_for(args, module, config, contract)
        from scripts.npu.audit_common import optimizer_step
        masters, optimizer, scheduler = create_audit_optimizer(module, 5 + args.measured_updates)
        runtime.synchronize(args.device)
        record['setup_seconds'] = time.monotonic() - begin
        for index in range(5 + args.measured_updates):
            if time.monotonic() - begin > args.budget_seconds:
                record['status'] = 'budget_exhausted'
                break
            optimizer.zero_grad(set_to_none=True)
            module.zero_grad(set_to_none=True)
            runtime.synchronize(args.device)
            runtime.reset_peak_memory_stats(args.device)
            start = time.monotonic()
            captured = []
            handle = module.draft_model.register_forward_hook(lambda m, a, out: captured.append(out.detach())) if index == 0 and args.correctness_output else None
            metrics = module(**batch)
            metrics.loss.backward()
            if handle:
                handle.remove()
            # Correctness artifacts are excluded from timing; first update is a pilot/warmup.
            if index == 0 and args.correctness_output:
                torch.save({'output': captured[0].cpu(), 'loss': metrics.loss.detach().cpu(),
                            'grads': {n: p.grad.detach().cpu() for n,p in module.named_parameters() if p.grad is not None}}, args.correctness_output)
            if not bool(torch.isfinite(metrics.loss)):
                raise FloatingPointError('Non-finite loss')
            grad_norm = optimizer_step(module, masters, optimizer, scheduler, max_grad_norm=1.)
            if not math.isfinite(grad_norm):
                raise FloatingPointError('Non-finite gradient norm')
            runtime.synchronize(args.device)
            entry = dict(seconds=time.monotonic() - start,
                         correctness_artifact_io_included=bool(index == 0 and args.correctness_output),
                         peak_hbm_bytes=runtime.max_memory_allocated(args.device),
                         loss=float(metrics.loss.detach()), grad_norm=float(grad_norm),
                         attention=module.attention_runtime_report())
            record['warmup_updates' if index < 5 else 'measured_updates'].append(entry)
            if index == 4:
                durations = [r['seconds'] for r in record['warmup_updates']]
                record['first5_estimate'] = {'seconds': durations, 'mean_seconds': sum(durations)/5,
                    'projected_remaining_seconds': sum(durations)/5 * args.measured_updates}
            save(args.output, record)
        else:
            record['status'] = 'pass'
    except Exception as exc:
        record['status'] = 'oom' if 'out of memory' in str(exc).lower() else 'error'
        record['error'] = str(exc)
        record['traceback'] = traceback.format_exc()
    record['total_seconds'] = time.monotonic() - begin
    save(args.output, record)
    print(json.dumps({'output': args.output, 'status': record['status'], 'seconds': record['total_seconds']}), flush=True)
    return 0 if record['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
