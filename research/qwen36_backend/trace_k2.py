"""Independent whole-model CUDA kernel traces: native AR and actual DFlash K1."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(os.environ.get('Q36_ROOTCAUSE_TASK','/home/lyh/qwen36_rootcause_20261101'))
REPO=TASK/'repo';OLD=Path(os.environ.get('Q36_EXISTING_TASK','/home/lyh/qwen36_k_sweep_20261101'))
PLAN=Path(os.environ.get('Q36_EXISTING_PLAN','/home/lyh/qwen36_verifylen_20261101/attempt2_g12/plans/k1.evidence/plan.json'))
REFERENCE=Path(os.environ.get('Q36_EXISTING_REFERENCE',str(OLD/'dfly_k1_r1_a2.evidence/reference.json')))
GPUS=[int(x) for x in os.environ.get('Q36_ROOTCAUSE_GPUS','1,2').split(',')]

def main():
    root=TASK/'k2_trace_a1';root.mkdir(exist_ok=False)
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    while True:
        state=gpu_state()
        if all(state[g]['used']<500 and not state[g]['pids'] and gpu_lock_free(state[g]['uuid'],Path(os.environ['VELADRAFT_GPU_LOCK_DIR'])) for g in GPUS):break
        print('WAIT_GPU',GPUS,time.strftime('%F %T'),flush=True);time.sleep(20)
    with GPULease(GPUS,root/'resources.jsonl') as lease:
        for arm in ('ar','sd'):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                     PYTHONPATH=str(REPO/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(REPO),PYTHONDONTWRITEBYTECODE='1',
                     VELADRAFT_NATIVE_PLAN=str(PLAN),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),VELADRAFT_NATIVE_ARM=arm,
                     QWEN8B_DIGEST_CACHE=str(OLD/'cache/digests.sqlite'),VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_BATCH_INVARIANT='0',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(OLD/'tmp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     TRITON_CACHE_DIR=str(OLD/'cache/triton'),VLLM_CACHE_ROOT=str(TASK/'cache/vllm'),TORCHINDUCTOR_CACHE_DIR=str(TASK/'cache/inductor'),Q36_TRACE_DIR=str(root/arm/'traces'))
            cmd=[sys.executable,'-m','research.qwen36_k_sweep.profile_stage','--plan',str(PLAN),'--reference',str(REFERENCE),'--arm',arm,'--out',str(root/arm),'--context-index','2','--max-tokens','64','--repetitions','1']
            (root/'STATUS.json').write_text(json.dumps({'status':'running','arm':arm,'diagnostic_only':True}))
            print('TRACE_START',arm,time.strftime('%F %T'),flush=True)
            with (root/(arm+'.log')).open('x') as stream:
                p=subprocess.Popen(cmd,env=env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while p.poll() is None:
                    if lease.rows and (lease.rows[-1].get('foreign_pids') or lease.rows[-1].get('over_bound') or lease.rows[-1].get('error')):
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        break
                    time.sleep(5)
                if p.returncode:
                    (root/'STATUS.json').write_text(json.dumps({'status':'failed','arm':arm,'code':p.returncode}));return 1
            print('TRACE_COMPLETE',arm,time.strftime('%F %T'),flush=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2));(root/'STATUS.json').write_text(json.dumps({'status':'complete'}));return 0

if __name__=='__main__':sys.exit(main())
