"""Wait for a real TP2 pair; optionally preempt only user-owned 8B jobs when a pair is obtainable."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

TASK = Path('/home/lyh/qwen36_k_sweep_20261101')
REPO = TASK / 'repo'
PY = TASK / 'runtime/bin/python'
UID = os.getuid()

def log(*parts): print(time.strftime('%Y-%m-%d %H:%M:%S'), *parts, flush=True)

def processes():
    out = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            st = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = [x.decode(errors='replace') for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
            out[int(path.name)] = dict(uid=path.stat().st_uid, parent=int(st[1]), start=st[19], cmd=' '.join(argv), argv=argv)
        except (OSError, IndexError): pass
    return out

def owned_8b_tree(ps):
    # Only our named 8B evaluation processes, never other accounts' processes.
    protected = {os.getpid()}
    parent = ps.get(os.getpid(), {}).get('parent', 0)
    while parent and parent not in protected:
        protected.add(parent); parent = ps.get(parent, {}).get('parent', 0)
    def is_job(x):
        if x['uid'] != UID: return False
        try: argv = x.get('argv') or shlex.split(x['cmd'])
        except ValueError: return False
        if not argv: return False
        exe = Path(argv[0]).name
        if exe.upper() == 'SCREEN':
            return any(a.startswith('qwen8b_') and 'summary' not in a for a in argv[1:])
        # Never treat source text passed to a shell/Python -c as process identity.
        if '-c' in argv: return False
        if exe in ('sh', 'bash'):
            return any(a.startswith('/home/lyh/qwen3vl8b_eval_20261027/') and
                       any(s in Path(a).name for s in ('queue','runner','keepalive','supervisor')) for a in argv[1:])
        if exe.startswith('python'):
            return any(a.startswith('/home/lyh/qwen3vl8b_eval_20261027/') for a in argv) and any(
                a in ('native-eval','vlm_dflash.eval.native_runtime') or 'quality_' in a for a in argv[1:])
        return False
    roots = {p for p, x in ps.items() if p not in protected and is_job(x)}
    while True:
        added = {p for p, x in ps.items() if p not in protected and x['uid'] == UID and x['parent'] in roots}
        if added <= roots: return roots
        roots |= added

def gpu_state():
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    state = {int(i.strip()): dict(uuid=u.strip(), used=int(mem), pids=[]) for i, u, mem in csv.reader(io.StringIO(query))}
    for p, u in csv.reader(io.StringIO(apps)):
        for v in state.values():
            if v['uuid'] == u.strip(): v['pids'].append(int(p))
    return state

def gpu_lock_free(uuid, root=Path('/tmp/veladraft_gpu_locks')):
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / (uuid + '.lock')).open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, PermissionError): return False

def terminate_owned_8b(ps, victims):
    record = {str(p): ps[p] for p in victims}
    dest = TASK / f'preempt_8b_{int(time.time())}.json'
    dest.write_text(json.dumps({'reason': 'user authorized prioritizing Qwen3.6 K sweep; second GPU is available', 'processes': record}, indent=2))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        current = processes()
        for pid in victims:
            x = current.get(pid)
            if x and x['uid'] == UID and x['start'] == ps[pid]['start']:
                try: os.kill(pid, sig)
                except ProcessLookupError: pass
        if sig == signal.SIGTERM: time.sleep(12)
    log('OWNED_8B_PREEMPTED', dest)

def wait_pair(allowed):
    last = 0
    candidate = None
    since = 0
    while True:
        try:
            state = {g: x for g, x in gpu_state().items() if g in allowed}; ps = processes(); own = owned_8b_tree(ps)
            free = [g for g, x in state.items() if x['used'] < 500 and not x['pids'] and gpu_lock_free(x['uuid'])]
            eligible = [g for g, x in state.items() if x['pids'] and set(x['pids']) <= own]
            proposal = ('free', *free[:2]) if len(free) >= 2 else (('preempt', free[0], eligible[0]) if free and eligible else None)
            if proposal != candidate:
                candidate, since = proposal, time.monotonic()
            stable = candidate is not None and time.monotonic() - since >= 30
            if stable and candidate[0] == 'free': return list(candidate[1:])
            if stable and candidate[0] == 'preempt':
                log('PAIR_OBTAINABLE_PREEMPT_OWN_8B', free[0], eligible[0])
                terminate_owned_8b(ps, own)
                time.sleep(10); continue
            if time.monotonic() - last > 60:
                log('WAIT_TWO_GPUS', {'free': free, 'preemptible_8b': eligible})
                last = time.monotonic()
        except (OSError, ValueError, subprocess.SubprocessError) as e: log('GPU_QUERY_RETRY', str(e))
        time.sleep(10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('dflash','dflash2','dfly'), required=True)
    parser.add_argument('--then', nargs='*', choices=('dflash','dflash2','dfly'), default=[])
    parser.add_argument('--profile-after-first', action='store_true')
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    allowed = [int(x) for x in args.gpus.split(',')]
    assert len(allowed) == 2 and len(set(allowed)) == 2
    TASK.mkdir(exist_ok=True)
    lock = open(TASK / f'{args.method}_queue.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while not (TASK / 'PREPARED.json').exists():
        if (TASK / 'PREPARE_EXIT').exists() and (TASK / 'PREPARE_EXIT').read_text().strip() != '0':
            log('PREPARATION_FAILED'); return 1
        log('WAIT_PREPARATION'); time.sleep(30)
    pair = wait_pair(allowed)
    (TASK / f'{args.method}_selected_gpus.json').write_text(json.dumps({'gpus': pair, 'selected_at': time.time()}))
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(TASK / 'tmp'), TRITON_CACHE_DIR=str(TASK / 'cache/triton'),
               XDG_CACHE_HOME=str(TASK / 'cache/xdg'), HF_HOME=str(TASK / 'cache/hf'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               QWEN8B_DIGEST_CACHE=str(TASK / 'cache/digests.sqlite'),
               VLLM_BATCH_INVARIANT='0', VELADRAFT_GPU_OWNERSHIP='pid',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    turns = json.loads((TASK / 'PREPARED.json').read_text())['turns']
    for method in [args.method] + args.then:
        for k in (1, 3, 5, 7):
            stem = TASK / f'{method}_k{k}_r1'
            result = stem.with_suffix('.json')
            if result.exists():
                previous = json.loads(result.read_text())
                if previous.get('status') == 'complete' and previous.get('resources_verified'):
                    log('EXISTING_COMPLETE_PRESERVED', result); continue
                retry = 2
                while (TASK / f'{method}_k{k}_r1_a{retry}.json').exists() or (TASK / f'{method}_k{k}_r1_a{retry}.evidence').exists(): retry += 1
                stem = TASK / f'{method}_k{k}_r1_a{retry}'
                result = stem.with_suffix('.json')
                log('FAILED_ATTEMPT_PRESERVED_RETRY', result)
            while True:
                state = gpu_state()
                if all(state[g]['used'] < 500 and not state[g]['pids'] and gpu_lock_free(state[g]['uuid']) for g in pair): break
                log('WAIT_SELECTED_PAIR', pair); time.sleep(30)
            cmd = [str(PY), '-m', 'vlm_dflash.cli', 'native-eval', '--target', '/home/lyh/models/Qwen3.6-35B-A3B',
                   '--draft', str(TASK / 'exports' / method), '--source', str(TASK / 'diagnostic12.jsonl'),
                   '--images-dir', '/home/lyh/a6000_3b_v2_L300/shared/benchmarks/mmspec/images',
                   '--method', method, '--gpus', ','.join(map(str, pair)), '--num-prompts', '12', '--repeats', '1',
                   '--max-new-tokens', '1024', '--max-model-len', '8192', '--speculative-tokens', str(k),
                   '--gpu-memory-utilization', '.85', '--min-pixels', '65536', '--max-pixels', '1048576',
                   '--chat-template-kwargs', '{"enable_thinking": false}', '--sd-cudagraph-mode', 'FULL_AND_PIECEWISE',
                   '--system-prompt', "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                   '--runtime-manifest', str(TASK / 'exports' / f'{method}_k{k}_runtime.json'), '--out', str(result)]
            stem.with_suffix('.command.json').write_text(json.dumps(cmd, indent=2))
            log('RUN_START', method, k, pair)
            contaminated = False
            with open(stem.with_suffix('.launch.log'), 'x') as stream:
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                resource = Path(str(stem) + '.evidence/resources.jsonl')
                offset = 0
                while proc.poll() is None:
                    if resource.exists():
                        with resource.open() as f:
                            f.seek(offset)
                            for line in f:
                                try: row = json.loads(line)
                                except ValueError: continue
                                if row.get('foreign_pids') or row.get('error'): contaminated = True
                            offset = f.tell()
                    if contaminated:
                        log('RESOURCE_ABORT', method, k)
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        break
                    time.sleep(5)
            if proc.returncode != 0 or contaminated:
                log('RUN_FAILED_STOP', method, k, proc.returncode); return 1
            report = json.loads(result.read_text())
            assert report['status'] == 'complete' and report['resources_verified'] and report['timed_pairs'] == turns
            log('RUN_COMPLETE', method, k, report['speedup_sum_ar_over_sd'])
            if args.profile_after_first and method == 'dflash' and k == 1:
                log('SEPARATE_PROFILE_START', method)
                with (TASK/'profile_driver.log').open('a') as stream:
                    code = subprocess.call([str(PY), str(REPO/'research/qwen36_k_sweep/profile_driver.py'), str(result)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                log('SEPARATE_PROFILE_EXIT', code, 'continuing uninstrumented K sweep')
    log('SWEEP_COMPLETE')
    return 0

if __name__ == '__main__': sys.exit(main())
