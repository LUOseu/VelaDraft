"""Predeclared K1/K3 development pilots after the fixed-K2 control completes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

task=Path('/root/autodl-tmp/qwen36_backend_20261101');repo=task/'repo';root=task/'k13_fixed512_pilot_a1';root.mkdir(exist_ok=False)
after=task/'fixed512_v022_gpu0_a1/SUMMARY.json';deadline=time.monotonic()+3600
while not after.exists():
    s=after.parent/'STATUS.json'
    if s.exists() and json.loads(s.read_text()).get('status')=='failed':raise RuntimeError('Fixed-K2 dependency failed')
    if time.monotonic()>deadline:raise RuntimeError('Dependency wait exceeded one hour')
    (root/'STATUS.json').write_text(json.dumps({'status':'waiting_dependency','path':str(after)}));time.sleep(10)
assert json.loads(after.read_text())['status']=='complete'
from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
from vlm_dflash.eval.native_resources import GPULease
while True:
    s=gpu_state()[0]
    if s['used']<500 and not s['pids'] and gpu_lock_free(s['uuid'],Path(os.environ['VELADRAFT_GPU_LOCK_DIR'])):break
    if time.monotonic()>deadline:raise RuntimeError('GPU wait exceeded one hour')
    time.sleep(10)
files=list((repo/'research/qwen36_budget_v032').glob('*.py'))+list((repo/'research/qwen36_v032').glob('*.py'))
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(root/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2))
conditions=[('ar','ar',2),('sd_k1','sd',1),('sd_k3','sd',3)]
with GPULease(0,root/'resources.jsonl') as lease:
    for label,arm,k in conditions:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(repo),
                 Q36_V032_ARM=arm,VLLM_USE_V2_MODEL_RUNNER='1',VLLM_GDN_DECODE_KERNEL='cuda',
                 VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_BATCH_INVARIANT='0',
                 CUDA_HOME='/root/autodl-tmp/qwen36_tp1_20261101/runtime/lib/python3.12/site-packages/nvidia/cu13',
                 FLASHINFER_WORKSPACE_BASE=str(task/'flashinfer_v032_cuda130'),
                 VLLM_CACHE_ROOT=str(root/'cache'/label),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'/label),
                 TRITON_CACHE_DIR=str(task/'v032_sd_a2/triton'))
        env['CUDA_PATH']=env['CUDA_HOME'];env['CUDACXX']=env['CUDA_HOME']+'/bin/nvcc';env['PATH']=env['CUDA_HOME']+'/bin:'+str(task/'runtime_v032/bin')+':'+env['PATH']
        for key in ('VELADRAFT_NATIVE_PLAN','VELADRAFT_NATIVE_PLAN_SHA256','VELADRAFT_NATIVE_ARM','VLLM_TUNED_CONFIG_FOLDER'):env.pop(key,None)
        cmd=[str(task/'runtime_v032/bin/python'),'-u','-m','research.qwen36_budget_v032.stage','--arm',arm,'--k',str(k),'--max-tokens','512','--out',str(root/label)]
        (root/'STATUS.json').write_text(json.dumps({'status':'running','stage':label,'gpu':0}))
        with (root/(label+'.log')).open('x') as stream:
            child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if lease.rows and any(lease.rows[-1].get(x) for x in ('foreign_pids','over_bound','error')):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                time.sleep(5)
        if child.returncode:
            (root/'STATUS.json').write_text(json.dumps({'status':'failed','stage':label,'returncode':child.returncode}));raise SystemExit(1)
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
resource=lease.validate();(root/'resources_verified.json').write_text(json.dumps(resource,indent=2))
reports={label:json.loads((root/label/'result.json').read_text()) for label,_,_ in conditions}
assert all(x['status']=='complete' and len(x['rows'])==9 and x['output_tokens']==4608 for x in reports.values())
for report in reports.values():assert all(x['prompt_token_ids']==y['prompt_token_ids'] for x,y in zip(reports['ar']['rows'],report['rows']))
result={'status':'complete','development_pilot_only':True,'contexts':3,'repeats':3,'fixed_output_tokens':512,'k_values':[1,3],
        'ar_seconds':reports['ar']['wall_seconds'],'sd_seconds':{str(k):reports[f'sd_k{k}']['wall_seconds'] for k in (1,3)},
        'speedup':{str(k):reports['ar']['wall_seconds']/reports[f'sd_k{k}']['wall_seconds'] for k in (1,3)},
        'limitation':'K selection pilot; not a formal result or a reason to replace fixed-K2 controls without validation.'}
(root/'SUMMARY.json').write_text(json.dumps(result,indent=2));(root/'STATUS.json').write_text(json.dumps({'status':'complete'}));print(json.dumps(result,indent=2),flush=True)
