"""One leased TP2 lane, native AR followed by real DFlash SD K1..K7 profiles."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE=Path('/home/lyh/qwen36_verifylen_20261101')
TASK=Path(os.environ.get('Q36_VERIFY_TASK',str(BASE)))
OLD=Path('/home/lyh/qwen36_k_sweep_20261101')
REPO=BASE/'repo'
PY=OLD/'runtime/bin/python'
GPUS=[int(x) for x in os.environ.get('Q36_VERIFY_GPUS','0,1').split(',')]
assert len(GPUS)==2 and len(set(GPUS))==2

def log(*args):print(time.strftime('%F %T'),*args,flush=True)
def save(name,value):
    p=TASK/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def main():
    TASK.mkdir(exist_ok=True)
    for d in ['plans','manifests','stages','logs','cache']: (TASK/d).mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',
             QWEN8B_DIGEST_CACHE=str(OLD/'cache/digests.sqlite'),TMPDIR=str(OLD/'tmp'),
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_BATCH_INVARIANT='0',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VELADRAFT_GPU_OWNERSHIP='pid',
             TRITON_CACHE_DIR=str(OLD/'cache/triton'),XDG_CACHE_HOME=str(OLD/'cache/xdg'))
    adapter=REPO/'research/qwen36_native_vllm'
    ref=OLD/'dfly_k1_r1_a2.evidence/reference.json'
    reference=json.loads(ref.read_text());assert reference['status']=='complete' and len(reference['rows'])==16
    save('REFERENCE.json',{'path':str(ref),'sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),'sessions':12,'turns':16})
    for k in range(1,8):
        manifest=TASK/'manifests'/f'dflash_k{k}.json'
        if not manifest.exists():
            with (TASK/'logs'/f'manifest_k{k}.log').open('x') as stream:
                subprocess.run([str(PY),str(adapter/'make_runtime_manifest.py'),'--method','dflash','--checkpoint',str(OLD/'checkpoints/dflash_final.pt'),'--sweep-k',str(k),'--out',str(manifest)],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        out=TASK/'plans'/f'k{k}.json'
        if out.exists():continue
        command=[str(PY),'-m','vlm_dflash.cli','native-eval','--target','/home/lyh/models/Qwen3.6-35B-A3B',
                 '--draft',str(OLD/'exports/dflash'),'--source',str(OLD/'diagnostic12.jsonl'),
                 '--images-dir','/home/lyh/a6000_3b_v2_L300/shared/benchmarks/mmspec/images',
                 '--method','dflash','--gpus',','.join(map(str,GPUS)),'--num-prompts','12','--repeats','1',
                 '--max-new-tokens','128','--max-model-len','8192','--speculative-tokens',str(k),
                 '--gpu-memory-utilization','.85','--min-pixels','65536','--max-pixels','1048576',
                 '--chat-template-kwargs','{"enable_thinking": false}','--sd-cudagraph-mode','FULL_AND_PIECEWISE',
                 '--system-prompt',"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
                 '--runtime-manifest',str(manifest),'--out',str(out),'--dry-run']
        with (TASK/'logs'/f'plan_k{k}.log').open('x') as stream:subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        log('PLAN_READY',k)
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    from research.qwen36_verifylen.summary import collect
    while not (BASE/'CPU_READY').exists():
        save('STATUS.json',{'status':'waiting_cpu_validation'});log('WAIT_CPU_VALIDATION');time.sleep(10)
    while True:
        state=gpu_state()
        if all(state[g]['used']<500 and not state[g]['pids'] and gpu_lock_free(state[g]['uuid']) for g in GPUS):break
        save('STATUS.json',{'status':'waiting_gpu_pair','gpus':GPUS});log('WAIT_GPU_PAIR',GPUS);time.sleep(20)
    save('TOPOLOGY.json',{'gpus':GPUS,'nvidia_smi_topo':subprocess.check_output(['nvidia-smi','topo','-m'],text=True),'note':'Physical GPUs 1,2 span NUMA sockets; do not mix with the failed 0,1 attempt.'})
    resource_log=TASK/'resources.jsonl'
    if resource_log.exists():
        index=1
        while (TASK/f'resources_resume{index}.jsonl').exists():index+=1
        resource_log=TASK/f'resources_resume{index}.jsonl'
    with GPULease(GPUS,resource_log) as lease:
        stages=[('ar','ar',1)]+[(f'sd_k{k}','sd',k) for k in range(1,8)]
        for name,arm,k in stages:
            out=TASK/'stages'/name
            if (out/'profile.json').exists() and json.loads((out/'profile.json').read_text()).get('status') in ('complete','recovered_complete'):continue
            if out.exists():raise FileExistsError(f'Preserve failed stage {out}; use a new attempt directory')
            path=TASK/'plans'/f'k{k}.evidence/plan.json'
            stage_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           VELADRAFT_NATIVE_PLAN=str(path),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           VELADRAFT_NATIVE_ARM=arm,VLLM_WORKER_MULTIPROC_METHOD='spawn',
                           PYTHONPATH=str(REPO/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(REPO),
                           VLLM_CACHE_ROOT=str(TASK/'cache/vllm'),TORCHINDUCTOR_CACHE_DIR=str(TASK/'cache/inductor'))
            cmd=[str(PY),'-m','research.qwen36_k_sweep.profile_stage','--plan',str(path),'--reference',str(ref),'--arm',arm,'--out',str(out),'--repetitions','3']
            save('STATUS.json',{'status':'running','stage':name,'draft_length':k if arm=='sd' else 0,'expected_positions':k+1 if arm=='sd' else 1,'gpus':GPUS})
            log('PROFILE_STAGE_START',name)
            with (TASK/'logs'/f'{name}.log').open('x') as stream:
                proc=subprocess.Popen(cmd,cwd=REPO,env=stage_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    if lease.rows and (lease.rows[-1].get('foreign_pids') or lease.rows[-1].get('error')):
                        log('RESOURCE_ABORT',name);os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
                    time.sleep(5)
                if proc.returncode!=0:
                    save('STATUS.json',{'status':'failed','stage':name,'returncode':proc.returncode,'log':str(TASK/'logs'/f'{name}.log')});return 1
            collect(TASK);log('PROFILE_STAGE_COMPLETE',name)
    # Validate after the lease has appended its closing sample.
    save('resources_verified.json',{'latest_lease':lease.validate(),'latest_resource_log':str(resource_log),'prior_leases':'see recovery receipt and previous resource logs'});collect(TASK)
    save('STATUS.json',{'status':'complete','gpus':GPUS,'draft_lengths':list(range(1,8))});log('VERIFY_LENGTH_SWEEP_COMPLETE')
    return 0

if __name__=='__main__':sys.exit(main())
