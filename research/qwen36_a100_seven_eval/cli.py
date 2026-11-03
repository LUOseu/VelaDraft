"""Entry point behind scripts/run_qwen36_a100_benchmarks.sh.

  run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full \
      --methods dflash,dspark,domino,jetspec,dflash2,dflare,dfly --draft-length 3 --publish --resume

``--preflight`` only checks the environment and lists anything that needs a person (data, weights,
HF write access, GPUs) without starting an engine. A new run freezes the target, checkpoints, data and
code into ``manifest.json`` first; ``--resume`` continues the latest run of the same suite (or
``--run-id``) and refuses if the code commit changed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from research.qwen36_a100_seven_eval import data as data_mod
from research.qwen36_a100_seven_eval.settings import (
    METHODS, REPO, SUITES, apply_overrides, load_config, read_json, save_json)


def git(*args) -> str:
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def gpus_info(indices) -> list[dict]:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,memory.used,driver_version',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(('index', 'name', 'uuid', 'memory_total_mib', 'memory_used_mib', 'driver'),
                     [x.strip() for x in line.split(',')])) for line in out.strip().splitlines()]
    return [r for r in rows if int(r['index']) in set(indices)]


def preflight(config: dict, suite: str, publish: bool) -> dict:
    checks = []

    def check(name, ok, detail='', action=None):
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail)[:300], action=action))

    try:
        version = importlib.metadata.version('vllm')
        check('vllm', version == config['engine']['vllm'], version, f'install vllm=={config["engine"]["vllm"]}')
    except importlib.metadata.PackageNotFoundError:
        check('vllm', False, 'not installed', 'create the environment described in the README')
    try:
        gpus = gpus_info(config['run']['gpus'])
        want = config['run'].get('gpu_name_contains')
        names = sorted({g['name'] for g in gpus})
        check('gpus', gpus and (not want or all(want in g['name'] for g in gpus)), names,
              f'run on {want} GPUs or change run.gpu_name_contains')
        busy = [g['index'] for g in gpus if int(g['memory_used_mib']) > 500]
        check('gpus_free', True, f'busy now: {busy}' if busy else 'all free',
              'lanes wait for, or the site hook frees, busy GPUs' if busy else None)
    except Exception as error:
        check('gpus', False, error, 'nvidia-smi must be available')
    cuda_home = config['engine'].get('cuda_home') or os.environ.get('CUDA_HOME', '')
    nvcc = Path(cuda_home) / 'bin' / 'nvcc'
    check('cuda_toolkit', nvcc.exists(), cuda_home or 'CUDA_HOME unset',
          'point engine.cuda_home at a complete CUDA 13.0 toolkit (FlashInfer JIT)')
    target = Path(config['target']['path'])
    check('target', (target / 'config.json').exists(), target,
          f'download {config["target"]["repo"]}@{config["target"]["revision"]} to target.path')
    repo = Path(config['data']['clone'])
    check('mmspec', (repo / 'dataset/MMSpec/test/mmspec.jsonl').exists() or shutil.which('git'), repo,
          'git clone is attempted automatically at run start')
    if publish:
        try:
            from research.qwen36_a100_seven_eval.publish import check_write_access
            access = check_write_access(config['publish']['repo'])
            check('hf_write', access['ok'], access, 'log in with a token that can write the results repo')
        except Exception as error:
            check('hf_write', False, error, 'hf auth login with write access to ' + config['publish']['repo'])
    free = shutil.disk_usage(Path(config['run']['root']).parent if Path(config['run']['root']).parent.exists() else '/').free
    check('disk', free > 50e9, f'{free / 1e9:.0f} GB free', 'need at least 50 GB for exports and raw results')
    ok = all(c['ok'] for c in checks)
    return dict(ok=ok, suite=suite, checks=checks)


def latest_run(root: Path, suite: str) -> Path | None:
    link = root / f'latest_{suite}'
    return link.resolve() if link.exists() else None


def create_run(config: dict, suite: str) -> Path:
    from research.qwen36_a100_seven_eval.prepare import freeze_checkpoints, target_identity
    root = Path(config['run']['root'])
    commit = git('rev-parse', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('refusing to freeze a run from a checkout with uncommitted changes; commit first')
    run_id = time.strftime('%Y%m%dT%H%M%S') + f'_{suite}_{commit[:8]}'
    run = root / 'runs' / run_id
    run.mkdir(parents=True)
    save_json(run / 'status.json', dict(status='preparing', suite=suite, run_id=run_id))
    cache = root / 'sha256_cache.json'
    clone = data_mod.fetch(config['data']['repo'], config['data']['commit'], Path(config['data']['clone']))
    sessions, data_manifest = data_mod.build_sessions(clone, suite)
    manifest = dict(run_id=run_id, suite=suite, methods=config['methods'], created=time.strftime('%F %T'),
                    code=dict(commit=commit, dirty=False),
                    versions={n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers')},
                    target=target_identity(Path(config['target']['path']), config['target']['revision'], cache),
                    exports=freeze_checkpoints(config['methods'], config, cache),
                    data=data_manifest, protocol=config['protocol'], engine=config['engine'],
                    site=config.get('site', {}), gpus=gpus_info(config['run']['gpus']),
                    publish=dict(repo=config['publish']['repo'], path=config['publish']['path']))
    save_json(run / 'sessions.json', sessions)
    save_json(run / 'manifest.json', manifest)
    link = root / f'latest_{suite}'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run)
    return run


def write_status(run: Path, lanes: dict, state: str) -> dict:
    from research.qwen36_a100_seven_eval.lane import job_list, job_name
    manifest = read_json(run / 'manifest.json')
    jobs = [job_name(m, p) for m, p in job_list(manifest)]
    done = [j for j in jobs if (run / 'done' / j).exists()]
    failed = [j for j in jobs if (run / 'failed' / j).exists()]
    shards = len(list((run / 'raw').glob('**/*.jsonl'))) if (run / 'raw').exists() else 0
    status = dict(status=state, suite=manifest['suite'], run_id=manifest['run_id'], updated=time.strftime('%F %T'),
                  jobs=len(jobs), jobs_done=len(done), jobs_failed=failed, raw_shards=shards,
                  lanes={str(g): ('running' if p.poll() is None else f'exit {p.returncode}') for g, p in lanes.items()},
                  formal=manifest['data']['formal'])
    save_json(run / 'status.json', status)
    return status


def run_all(run: Path, gpus: list[int], publish: bool) -> int:
    from research.qwen36_a100_seven_eval import summarize
    manifest = read_json(run / 'manifest.json')
    for claim in (run / 'claims').glob('*') if (run / 'claims').exists() else []:
        name = claim.name
        if not (run / 'done' / name).exists() and not (run / 'failed' / name).exists():
            shutil.rmtree(claim)  # stale claim from an interrupted launch; its finished shards are kept
    env = dict(os.environ, PYTHONPATH=str(REPO))
    if manifest['engine'].get('cuda_home'):
        cuda = manifest['engine']['cuda_home']
        env.update(CUDA_HOME=cuda, CUDA_PATH=cuda, CUDACXX=f'{cuda}/bin/nvcc', PATH=f'{cuda}/bin:' + env['PATH'])
    for key, value in manifest.get('site', {}).get('environment', {}).items():
        env[key] = str(value)
    (run / 'logs').mkdir(exist_ok=True)
    lanes = {g: subprocess.Popen([sys.executable, '-u', '-m', 'research.qwen36_a100_seven_eval.lane',
                                  '--run-dir', str(run), '--gpu', str(g)], env=env, cwd=REPO,
                                 stdout=(run / 'logs' / f'lane_gpu{g}.out').open('a'), stderr=subprocess.STDOUT)
             for g in gpus}
    publisher = None
    if publish:
        online = {k: v for k, v in env.items() if k not in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        publisher = subprocess.Popen([sys.executable, '-u', '-m', 'research.qwen36_a100_seven_eval.publish',
                                      '--run-dir', str(run), '--repo', manifest['publish']['repo'],
                                      '--path', manifest['publish']['path'],
                                      '--interval', str(manifest['site'].get('publish_interval', 300))],
                                     env=online, cwd=REPO, stdout=(run / 'logs' / 'publish.out').open('a'),
                                     stderr=subprocess.STDOUT)
    last_summary = 0.0
    while any(p.poll() is None for p in lanes.values()):
        write_status(run, lanes, 'running')
        if time.monotonic() - last_summary > 600:
            save_json(run / 'summary.json', summarize.summarize(run))
            last_summary = time.monotonic()
        time.sleep(60)
    result = summarize.summarize(run)
    save_json(run / 'summary.json', result)
    summarize.write_csv(result, run / 'summary.csv')
    final = 'complete' if result['status'] == 'complete' and not result['failed_jobs'] else 'partial'
    write_status(run, lanes, final)
    if publisher is not None:
        publisher.wait()
    return 0 if final == 'complete' else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--suite', choices=SUITES, default='mmspec-full')
    parser.add_argument('--methods', default=','.join(METHODS), help='comma-separated subset of ' + ','.join(METHODS))
    parser.add_argument('--draft-length', type=int, default=3, help='must equal the frozen K=3')
    parser.add_argument('--gpus', help='comma-separated GPU indices (default: run.gpus in the config)')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--publish', action='store_true', help='mirror results to Hugging Face while running')
    parser.add_argument('--resume', action='store_true', help='continue the latest run of this suite')
    parser.add_argument('--run-id', help='with --resume: continue this run instead of the latest')
    parser.add_argument('--preflight', action='store_true', help='check the environment and exit')
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), methods=args.methods.split(','),
                             draft_length=args.draft_length, output_root=args.output_root)
    if args.gpus:
        config['run']['gpus'] = [int(g) for g in args.gpus.split(',')]
    report = preflight(config, args.suite, args.publish)
    for c in report['checks']:
        print(f"[{'ok' if c['ok'] else 'NEEDS ACTION'}] {c['name']}: {c['detail']}" +
              ('' if c['ok'] or not c['action'] else f"  -> {c['action']}"))
    if args.preflight:
        return 0 if report['ok'] else 2
    if not report['ok']:
        print('preflight failed; fix the items above (or run --preflight for the list)', file=sys.stderr)
        return 2
    root = Path(config['run']['root'])
    run = None
    if args.resume:
        run = (root / 'runs' / args.run_id) if args.run_id else latest_run(root, args.suite)
        if run is not None and run.exists():
            frozen = read_json(run / 'manifest.json')
            if frozen['code']['commit'] != git('rev-parse', 'HEAD'):
                raise SystemExit(f'code commit changed since {run.name} was frozen; check out {frozen["code"]["commit"]}')
            if frozen['methods'] != config['methods']:
                raise SystemExit('--methods differs from the frozen run')
        else:
            run = None
    if run is None:
        run = create_run(config, args.suite)
    print('RUN', run, flush=True)
    return run_all(run, config['run']['gpus'], args.publish)


if __name__ == '__main__':
    raise SystemExit(main())
