"""ABBA process order, unchanged native AR, and optional official tuned MoE settings on both arms."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(os.environ.get('Q36_ROOTCAUSE_TASK','/home/lyh/qwen36_rootcause_20261101'));REPO=TASK/'repo';OLD=Path(os.environ.get('Q36_EXISTING_TASK','/home/lyh/qwen36_k_sweep_20261101'))
PLAN=Path(os.environ.get('Q36_EXISTING_PLAN','/home/lyh/qwen36_verifylen_20261101/attempt2_g12/plans/k1.evidence/plan.json'))
REFERENCE=Path(os.environ.get('Q36_EXISTING_REFERENCE',str(OLD/'dfly_k1_r1_a2.evidence/reference.json')))
GPUS=[int(x) for x in os.environ.get('Q36_ROOTCAUSE_GPUS','1,2').split(',')]

def main():
    root=TASK/'moe_e2e_ab_a1';root.mkdir(exist_ok=False)
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    while True:
        state=gpu_state()
        if all(state[g]['used']<500 and not state[g]['pids'] and gpu_lock_free(state[g]['uuid']) for g in GPUS):break
        print('WAIT_GPU',GPUS,time.strftime('%F %T'),flush=True);time.sleep(20)
    labels=[('default_ar','ar',False),('default_sd','sd',False),('tuned_sd','sd',True),('tuned_ar','ar',True)]
    with GPULease(GPUS,root/'resources.jsonl') as lease:
        for name,arm,tuned in labels:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,GPUS)),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(REPO/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(REPO),
                     PYTHONDONTWRITEBYTECODE='1',VELADRAFT_NATIVE_PLAN=str(PLAN),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),VELADRAFT_NATIVE_ARM=arm,
                     QWEN8B_DIGEST_CACHE=str(OLD/'cache/digests.sqlite'),VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_BATCH_INVARIANT='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     TMPDIR=str(OLD/'tmp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_CACHE_DIR=str(OLD/'cache/triton'),
                     VLLM_CACHE_ROOT=str(root/'compile_cache'/name),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'/name))
            env.pop('Q36_TRACE_DIR',None);env.pop('VLLM_TUNED_CONFIG_FOLDER',None)
            if tuned:env['VLLM_TUNED_CONFIG_FOLDER']=str(TASK/'moe_tune_a1/tuned_configs')
            cmd=[sys.executable,'-m','research.qwen36_rootcause.e2e_stage','--plan',str(PLAN),'--reference',str(REFERENCE),'--arm',arm,'--label',name,'--out',str(root/name)]
            (root/'STATUS.json').write_text(json.dumps({'status':'running','stage':name,'gpus':GPUS}));print('E2E_START',name,time.strftime('%F %T'),flush=True)
            with (root/(name+'.log')).open('x') as stream:
                p=subprocess.Popen(cmd,env=env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while p.poll() is None:
                    if lease.rows and (lease.rows[-1].get('foreign_pids') or lease.rows[-1].get('over_bound') or lease.rows[-1].get('error')):
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        break
                    time.sleep(5)
                if p.returncode:(root/'STATUS.json').write_text(json.dumps({'status':'failed','stage':name,'code':p.returncode}));return 1
            print('E2E_STAGE_COMPLETE',name,time.strftime('%F %T'),flush=True)
    (root/'resources_verified.json').write_text(json.dumps(lease.validate(),indent=2))
    reports={name:json.loads((root/name/'result.json').read_text()) for name,_,_ in labels}
    assert all(x['status']=='complete' and len(x['rows'])==9 for x in reports.values())
    key=lambda r:(r['id'],r['turn'],r['repeat'])
    for r in reports.values():assert [key(x) for x in r['rows']]==[key(x) for x in reports['default_ar']['rows']]
    comparisons={}
    for left,right in [('default_ar','default_sd'),('tuned_ar','tuned_sd'),('default_ar','tuned_ar'),('default_sd','tuned_sd'),('default_ar','tuned_sd')]:
        a,b=reports[left],reports[right]
        assert all(x['prompt_token_ids']==y['prompt_token_ids'] for x,y in zip(a['rows'],b['rows']))
        comparisons[left+'_over_'+right]={'ratio':a['wall_seconds']/b['wall_seconds'],'left_seconds':a['wall_seconds'],'right_seconds':b['wall_seconds'],'left_tokens':a['output_tokens'],'right_tokens':b['output_tokens'],'exact_outputs':sum(x['tokens']==y['tokens'] for x,y in zip(a['rows'],b['rows'])),'pairs':9}
    result={'status':'complete','small_diagnostic_only':True,'process_order':[x[0] for x in labels],'repetitions_per_context':3,'independent_contexts':3,'comparisons':comparisons,'resources_verified':True,
            'limitation':'One process per condition, three fixed contexts. No broad significance or strict losslessness claim.'}
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2));(root/'STATUS.json').write_text(json.dumps({'status':'complete'}));print(json.dumps(result,indent=2),flush=True);return 0

if __name__=='__main__':sys.exit(main())
