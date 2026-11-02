"""After the first valid DFlash K1 run, profile AR vs SD K1/K7 on the same three contexts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

TASK=Path('/home/lyh/qwen36_k_sweep_20261101')
REPO=TASK/'repo'

def main():
    result_path=Path(sys.argv[1]);result=json.loads(result_path.read_text())
    assert result['status']=='complete' and result['resources_verified'] and result['method']=='dflash'
    evidence=Path(result['evidence']);plan1=evidence/'plan.json';reference=evidence/'reference.json'
    plan=json.loads(plan1.read_text());assert plan['speculative_tokens']==1
    root=TASK/'profile_dflash_k1_k7'
    if root.exists():raise FileExistsError('Preserve existing profile')
    root.mkdir()
    cmd=json.loads(result_path.with_suffix('.command.json').read_text())
    for flag,value in [('--speculative-tokens','7'),('--runtime-manifest',str(TASK/'exports/dflash_k7_runtime.json')),('--out',str(root/'plan7.json'))]:
        cmd[cmd.index(flag)+1]=value
    cmd.append('--dry-run')
    env=dict(os.environ,PYTHONPATH=str(REPO),PYTHONDONTWRITEBYTECODE='1',QWEN8B_DIGEST_CACHE=str(TASK/'cache/digests.sqlite'),
             TMPDIR=str(TASK/'tmp'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_BATCH_INVARIANT='0',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_CACHE_DIR=str(TASK/'cache/triton'))
    subprocess.run(cmd,cwd=REPO,env=env,stdout=open(root/'plan7.log','x'),stderr=subprocess.STDOUT,check=True)
    plan7=root/'plan7.evidence/plan.json'
    from vlm_dflash.eval.native_resources import GPULease
    with GPULease(plan['gpus'],root/'resources.jsonl') as lease:
        for name,arm,path in [('ar','ar',plan1),('sd_k1','sd',plan1),('sd_k7','sd',plan7)]:
            stage_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           VELADRAFT_NATIVE_PLAN=str(path),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           VELADRAFT_NATIVE_ARM=arm,VLLM_WORKER_MULTIPROC_METHOD='spawn',
                           PYTHONPATH=str(REPO/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(REPO),
                           VLLM_CACHE_ROOT=str(root/name/'cache'),TORCHINDUCTOR_CACHE_DIR=str(root/name/'inductor'))
            command=[sys.executable,'-m','research.qwen36_k_sweep.profile_stage','--plan',str(path),'--reference',str(reference),'--arm',arm,'--out',str(root/name)]
            with (root/(name+'.log')).open('x') as stream:
                subprocess.run(command,env=stage_env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))
    print('PROFILE_COMPLETE',root,flush=True)

if __name__=='__main__':main()
