#!/usr/bin/env python3
"""Derive candidate paired training configs from the original 8B recipe.

These are candidates only: the campaign must independently validate dataset identity,
exact step-0 state, training plan, numerical equivalence and the A100 node before launch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

SWITCHES = {
    'dflash2': {'ref': {'dflash2_teacher_projection_reuse': False, 'dflash2_scoring_row_compaction': False,
                        'dflash2_fused_teacher_kl': False},
                'opt': {'dflash2_teacher_projection_reuse': True, 'dflash2_scoring_row_compaction': True,
                        'dflash2_fused_teacher_kl': True}},
    'dfly': {'ref': {'recipe_row_compaction': False, 'recipe_compile_tv_kl': False},
             'opt': {'recipe_row_compaction': True, 'recipe_compile_tv_kl': True}},
}


def derive(bases: dict) -> dict:
    out = {}
    for method in ('dflash2', 'dfly'):
        base = bases[method]
        train, drafter = base['train'], base['drafter']
        checks = [base['algorithm']['name'] == method, base['target']['dtype'] in ('bfloat16', 'bf16'),
                  base['capture']['seq_length'] == 4096, drafter['block_size'] == 8,
                  drafter['num_anchors'] == 512, train['num_epochs'] == 3,
                  train['micro_batch_size'] * train['grad_accumulation_steps'] == 8,
                  train.get('online_feature_cache_gb', 0) == 0, train.get('seed', 42) not in (17, 29)]
        if not all(checks):
            raise ValueError(f'{method}: source config does not match verified 8B recipe invariants')
        for seed in (17, 29):
            for arm in ('ref', 'opt'):
                cfg = copy.deepcopy(base)
                name = f'{method}_s{seed}_{arm}'
                cfg['train']['seed'] = seed
                cfg['train']['device'] = cfg['target']['device'] = 'cuda:0'
                cfg['train']['save_every_steps'] = 1250
                # The trainer filters intermediate snapshots at total_steps; the
                # 75k endpoint is represented by its final checkpoint instead.
                cfg['train']['snapshot_steps'] = [25000, 50000]
                cfg['drafter'].update(SWITCHES[method][arm])
                cfg['run'] = {'name': name, 'dir': name}
                out[name] = cfg
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dflash2', type=Path, required=True)
    p.add_argument('--dfly', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('output directory must be new or empty')
    source = {m: getattr(args, m) for m in ('dflash2', 'dfly')}
    configs = derive({m: json.loads(path.read_text()) for m, path in source.items()})
    args.out.mkdir(parents=True, exist_ok=True)
    index = {'status': 'DRAFT_NOT_LAUNCH_READY', 'source_sha256': {m: hashlib.sha256(path.read_bytes()).hexdigest() for m, path in source.items()},
             'missing': ['dataset and target identity', 'dev/test paths', 'paired initial state', 'numerical checks', '8xA100 preflight', 'dispatcher and resume validation'], 'configs': {}}
    for name, cfg in configs.items():
        content = (json.dumps(cfg, indent=2, sort_keys=True) + '\n').encode()
        (args.out / f'{name}.json').write_bytes(content)
        index['configs'][name] = hashlib.sha256(content).hexdigest()
    (args.out / 'DRAFT_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    print(f'wrote {len(configs)} DRAFT configs to {args.out}; NOT launch-ready')

if __name__ == '__main__':
    main()
