"""Wait for isolated installation, then execute a resource-audited native arm."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--arm',choices=('ar','sd'),required=True);p.add_argument('--out',required=True);p.add_argument('--full',action='store_true');args=p.parse_args()
    task=Path('/root/autodl-tmp/qwen36_backend_20261101');repo=task/'repo';root=Path(args.out);root.mkdir(exist_ok=False)
    deadline=time.monotonic()+45*60
    while True:
        state=task/'v032_install_mirror_STATUS.json'
        status=json.loads(state.read_text())['status'] if state.exists() else 'resolving'
        (root/'STATUS.json').write_text(json.dumps({'status':'waiting_install','install':status,'gpu':args.gpu,'arm':args.arm}))
        if status=='complete':break
        if status not in ('running','resolving') or time.monotonic()>deadline:raise RuntimeError('Isolated installation did not finish: '+status)
        time.sleep(10)
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    lockroot=Path(os.environ['VELADRAFT_GPU_LOCK_DIR'])
    while True:
        state=gpu_state()[args.gpu]
        if state['used']<500 and not state['pids'] and gpu_lock_free(state['uuid'],lockroot):break
        if time.monotonic()>deadline:raise RuntimeError('GPU remained occupied past queue deadline')
        time.sleep(10)
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        compiled_cache=Path(os.environ.get('Q36_V032_REUSE_CACHE',str(root)))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',Q36_V032_ARM=args.arm,
                 VLLM_USE_V2_MODEL_RUNNER='1',PYTHONPATH=str(repo),VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_BATCH_INVARIANT='0',
                 VLLM_CACHE_ROOT=str(compiled_cache/'cache'),TORCHINDUCTOR_CACHE_DIR=str(compiled_cache/'inductor'),TRITON_CACHE_DIR=str(compiled_cache/'triton'),
                 CUDA_HOME='/root/autodl-tmp/qwen36_tp1_20261101/runtime/lib/python3.12/site-packages/nvidia/cu13')
        env['CUDA_PATH']=env['CUDA_HOME']
        env['CUDACXX']=env['CUDA_HOME']+'/bin/nvcc'
        env['FLASHINFER_WORKSPACE_BASE']=str(task/'flashinfer_v032_cuda130')
        env['PATH']=env['CUDA_HOME']+'/bin:'+str(task/'runtime_v032/bin')+':'+env['PATH']
        for key in ('VELADRAFT_NATIVE_PLAN','VELADRAFT_NATIVE_PLAN_SHA256','VELADRAFT_NATIVE_ARM','VLLM_TUNED_CONFIG_FOLDER','Q36_SHARED_GATE_FUSION','Q36_CONTEXT_KV_GRAPH','VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD'):env.pop(key,None)
        hashes={str(f):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in (repo/'research/qwen36_v032').glob('*.py')}
        (root/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2))
        (root/'TOOLCHAIN.json').write_text(json.dumps({'cuda_home':env['CUDA_HOME'],'compiled_cache':str(compiled_cache),
            'gdn_decode_kernel':env.get('VLLM_GDN_DECODE_KERNEL','cuda (default)'),
            'nvcc':subprocess.check_output([env['CUDACXX'],'--version'],text=True),
            'nvcc_sha256':__import__('hashlib').sha256(Path(env['CUDACXX']).read_bytes()).hexdigest()},indent=2))
        cmd=[str(task/'runtime_v032/bin/python'),'-u','-m','research.qwen36_v032.stage','--arm',args.arm,'--out',str(root/'stage')]
        if args.full:cmd+=['--all-sessions','--max-tokens','1024']
        with (root/'stage.log').open('x') as stream:
            child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            (root/'STATUS.json').write_text(json.dumps({'status':'running','pid':child.pid,'gpu':args.gpu,'arm':args.arm}))
            while child.poll() is None:
                if lease.rows and any(lease.rows[-1].get(k) for k in ('foreign_pids','over_bound','error')):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(5)
        if child.returncode:
            (root/'STATUS.json').write_text(json.dumps({'status':'failed','returncode':child.returncode,'arm':args.arm}));return 1
    assert all(__import__('hashlib').sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))
    (root/'STATUS.json').write_text(json.dumps({'status':'complete','arm':args.arm}));return 0


if __name__=='__main__':sys.exit(main())
