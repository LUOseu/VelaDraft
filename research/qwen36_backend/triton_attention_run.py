import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from vlm_dflash.eval.native_resources import GPULease

task=Path('/root/autodl-tmp/qwen36_backend_20261101');root=task/'triton_attention_sd_a1';root.mkdir(exist_ok=False)
repo=task/'repo';plan=Path('/root/autodl-tmp/qwen36_rootcause_20261101/k2_setup/plan.evidence/plan.json')
old=Path('/root/autodl-tmp/qwen36_tp1_20261101');ref=old/'results/dflash_tp1_a3.evidence/reference.json'
with GPULease(0,root/'resources.jsonl') as lease:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
             PYTHONPATH=str(repo/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(repo),
             VELADRAFT_NATIVE_PLAN=str(plan),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(plan.read_bytes()).hexdigest(),
             VELADRAFT_NATIVE_ARM='sd',VLLM_WORKER_MULTIPROC_METHOD='spawn',
             VLLM_TUNED_CONFIG_FOLDER='/root/autodl-tmp/qwen36_rootcause_20261101/moe_tune_k2_a1/tuned_configs',
             VLLM_CACHE_ROOT=str(root/'cache'),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'))
    cmd=[sys.executable,'-u','-m','research.qwen36_backend.triton_attention_stage','--plan',str(plan),'--reference',str(ref),
         '--arm','sd','--label','triton_attention_sd','--out',str(root/'stage')]
    with (root/'stage.log').open('x') as stream:
        child=subprocess.run(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if child.returncode:
        (root/'STATUS.json').write_text(json.dumps({'status':'failed','code':child.returncode}));sys.exit(1)
(root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))
(root/'STATUS.json').write_text(json.dumps({'status':'complete'}))
