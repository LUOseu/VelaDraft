"""Confirm native AR/SD paired E2E over all 12 fixed diagnostic sessions, 3 warm-engine repetitions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);p.add_argument('--tuned-folder');p.add_argument('--after');p.add_argument('--order',default='ar,sd');args=p.parse_args()
    root=Path(args.out);root.mkdir(exist_ok=False);repo=Path(os.environ['Q36_ROOTCAUSE_TASK'])/'repo';old=Path(os.environ['Q36_EXISTING_TASK']);plan=Path(args.plan);reference=Path(os.environ['Q36_EXISTING_REFERENCE'])
    if args.after:
        while not Path(args.after).exists():print('WAIT_SMALL_AB',args.after,flush=True);time.sleep(30)
        assert json.loads(Path(args.after).read_text())['status']=='complete'
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    from vlm_dflash.eval.native import bootstrap_ci
    lock_root=Path(os.environ.get('VELADRAFT_GPU_LOCK_DIR','/tmp/veladraft_gpu_locks'))
    while True:
        s=gpu_state()[args.gpu]
        if s['used']<500 and not s['pids'] and gpu_lock_free(s['uuid'],lock_root):break
        print('WAIT_GPU',args.gpu,flush=True);time.sleep(20)
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        for arm in args.order.split(','):
            assert arm in ('ar','sd')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(repo/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(repo),
                     VELADRAFT_NATIVE_PLAN=str(plan),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(plan.read_bytes()).hexdigest(),VELADRAFT_NATIVE_ARM=arm,VLLM_WORKER_MULTIPROC_METHOD='spawn',
                     QWEN8B_DIGEST_CACHE=str(old/'cache/digests.sqlite'),VLLM_CACHE_ROOT=str(root/'cache'/arm),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'/arm),
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_BATCH_INVARIANT='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            env.pop('Q36_TRACE_DIR',None);env.pop('VLLM_TUNED_CONFIG_FOLDER',None)
            if args.tuned_folder:env['VLLM_TUNED_CONFIG_FOLDER']=args.tuned_folder
            cmd=[sys.executable,'-m','research.qwen36_rootcause.confirmation_stage','--plan',str(plan),'--reference',str(reference),'--arm',arm,'--label',root.name+'_'+arm,'--out',str(root/arm)]
            print('CONFIRM_START',arm,time.strftime('%F %T'),flush=True)
            with (root/(arm+'.log')).open('x') as stream:
                child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if lease.rows and (lease.rows[-1].get('foreign_pids') or lease.rows[-1].get('over_bound') or lease.rows[-1].get('error')):
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    time.sleep(5)
                if child.returncode:print('CONFIRM_FAILED',arm,child.returncode,flush=True);return 1
            print('CONFIRM_ARM_COMPLETE',arm,time.strftime('%F %T'),flush=True)
    resource=lease.validate();(root/'resources_verified.json').write_text(json.dumps(resource,indent=2))
    a=json.loads((root/'ar/result.json').read_text());b=json.loads((root/'sd/result.json').read_text())
    assert a['status']==b['status']=='complete' and len(a['rows'])==len(b['rows'])==48
    groups={'ar':{},'sd':{}};exact=0
    for x,y in zip(a['rows'],b['rows']):
        assert (x['id'],x['turn'],x['repeat'])==(y['id'],y['turn'],y['repeat']) and x['prompt_token_ids']==y['prompt_token_ids']
        for arm,row in [('ar',x),('sd',y)]:groups[arm].setdefault(row['id'],[]).append(row['wall_seconds'])
        exact+=x['tokens']==y['tokens']
    assert len(groups['ar'])==len(groups['sd'])==12
    result={'status':'complete','diagnostic_subset':True,'sessions':12,'turns':16,'repetitions_per_context':3,'independent_process_repeats':1,'timed_pairs':48,
            'k':json.loads(plan.read_text())['speculative_tokens'],'ar_seconds':a['wall_seconds'],'sd_seconds':b['wall_seconds'],'speedup':a['wall_seconds']/b['wall_seconds'],
            'session_cluster_CI95':bootstrap_ci(groups['ar'],groups['sd']),'ar_output_tokens':a['output_tokens'],'sd_output_tokens':b['output_tokens'],'exact_pairs':exact,
            'tuning_folder':args.tuned_folder,'gpu':args.gpu,'resource_evidence':resource,'limitations':['Fixed diagnostic subset, not full benchmark; warm-engine repeats, one process per arm.','Output and quality equivalence not established by speed.']}
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True);return 0

if __name__=='__main__':sys.exit(main())
