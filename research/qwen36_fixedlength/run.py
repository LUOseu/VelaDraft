"""Same-card paired fixed-output benchmark, preserving the natural-output runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--runtime',choices=('v022','v032_cuda','v032_triton'),required=True);p.add_argument('--out',required=True);p.add_argument('--after',required=True);args=p.parse_args()
    task=Path('/root/autodl-tmp/qwen36_backend_20261101');old=Path('/root/autodl-tmp/qwen36_tp1_20261101');repo=task/'repo'
    root=Path(args.out);root.mkdir(exist_ok=False);after=Path(args.after);deadline=time.monotonic()+3600
    while not after.exists():
        for state in after.parent.glob('*/STATUS.json'):
            if json.loads(state.read_text()).get('status')=='failed':raise RuntimeError('Dependency failed: '+str(state))
        if time.monotonic()>deadline:raise RuntimeError('Dependency did not complete within one hour')
        (root/'STATUS.json').write_text(json.dumps({'status':'waiting_dependency','path':str(after)}));time.sleep(10)
    assert json.loads(after.read_text())['status']=='complete'
    from research.qwen36_k_sweep.queue import gpu_state,gpu_lock_free
    from vlm_dflash.eval.native_resources import GPULease
    from vlm_dflash.eval.native import bootstrap_ci
    while True:
        s=gpu_state()[args.gpu]
        if s['used']<500 and not s['pids'] and gpu_lock_free(s['uuid'],Path(os.environ['VELADRAFT_GPU_LOCK_DIR'])):break
        if time.monotonic()>deadline:raise RuntimeError('GPU remained occupied')
        time.sleep(10)
    order=('ar','sd') if args.gpu==0 else ('sd','ar')
    files=list((repo/'research/qwen36_fixedlength').glob('*.py'))+list((repo/'research/qwen36_v032').glob('*.py'))
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (root/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2))
    with GPULease(args.gpu,root/'resources.jsonl') as lease:
        for arm in order:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_WORKER_MULTIPROC_METHOD='spawn')
            for key in ('VELADRAFT_NATIVE_PLAN','VELADRAFT_NATIVE_PLAN_SHA256','VELADRAFT_NATIVE_ARM','VLLM_TUNED_CONFIG_FOLDER','Q36_SHARED_GATE_FUSION','Q36_CONTEXT_KV_GRAPH','VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD','VLLM_GDN_DECODE_KERNEL'):env.pop(key,None)
            env['CUDA_HOME']=str(old/'runtime/lib/python3.12/site-packages/nvidia/cu13');env['CUDA_PATH']=env['CUDA_HOME'];env['CUDACXX']=env['CUDA_HOME']+'/bin/nvcc'
            env['FLASHINFER_WORKSPACE_BASE']=str(task/'flashinfer_v032_cuda130')
            if args.runtime=='v022':
                python=old/'runtime/bin/python';module='research.qwen36_fixedlength.stage_v022'
                plan=Path('/root/autodl-tmp/qwen36_rootcause_20261101/k2_setup/plan.evidence/plan.json')
                env.update(PYTHONPATH=str(repo/'vlm_dflash/eval/native_bootstrap')+os.pathsep+str(repo),
                           VELADRAFT_NATIVE_PLAN=str(plan),VELADRAFT_NATIVE_PLAN_SHA256=hashlib.sha256(plan.read_bytes()).hexdigest(),VELADRAFT_NATIVE_ARM=arm,
                           VLLM_TUNED_CONFIG_FOLDER='/root/autodl-tmp/qwen36_rootcause_20261101/moe_tune_k2_a1/tuned_configs')
                extra=['--plan',str(plan),'--reference',str(old/'results/dflash_tp1_a3.evidence/reference.json'),'--label',root.name+'_'+arm]
                cache=task/'moe_e2e_a1';env.update(VLLM_CACHE_ROOT=str(cache/'compile_cache'/('default_'+arm)),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'/('default_'+arm)),TRITON_CACHE_DIR=str(old/'cache/triton'))
                env.pop('VLLM_USE_V2_MODEL_RUNNER',None);env.pop('Q36_V032_ARM',None)
            else:
                python=task/'runtime_v032/bin/python';module='research.qwen36_fixedlength.stage_v032'
                kernel='triton' if args.runtime.endswith('triton') else 'cuda'
                env.update(PYTHONPATH=str(repo),Q36_V032_ARM=arm,VLLM_USE_V2_MODEL_RUNNER='1',VLLM_GDN_DECODE_KERNEL=kernel)
                cache=task/(f'v032_gdn_triton_cache_{arm}' if kernel=='triton' else f'v032_{arm}_a2')
                env.update(VLLM_CACHE_ROOT=str(cache/'cache'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
                extra=['--all-sessions','--max-tokens','512']
            env['PATH']=env['CUDA_HOME']+'/bin:'+str(python.parent)+':'+env['PATH']
            cmd=[str(python),'-u','-m',module,'--arm',arm,'--out',str(root/arm),*extra]
            (root/'STATUS.json').write_text(json.dumps({'status':'running','arm':arm,'gpu':args.gpu,'runtime':args.runtime,'fixed_output_length':512}))
            with (root/(arm+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    if lease.rows and any(lease.rows[-1].get(k) for k in ('foreign_pids','over_bound','error')):
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    time.sleep(5)
                if child.returncode:
                    (root/'STATUS.json').write_text(json.dumps({'status':'failed','arm':arm,'returncode':child.returncode}));return 1
    resource=lease.validate();(root/'resources_verified.json').write_text(json.dumps(resource,indent=2))
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    a=json.loads((root/'ar/result.json').read_text());b=json.loads((root/'sd/result.json').read_text());groups={'ar':{},'sd':{}}
    assert a['status']==b['status']=='complete' and len(a['rows'])==len(b['rows'])==48
    for x,y in zip(a['rows'],b['rows']):
        assert (x['id'],x['turn'],x['repeat'])==(y['id'],y['turn'],y['repeat']) and x['prompt_token_ids']==y['prompt_token_ids']
        assert len(x['tokens'])==len(y['tokens'])==512
        for arm,row in [('ar',x),('sd',y)]:groups[arm].setdefault(row['id'],[]).append(row['wall_seconds'])
    result=dict(status='complete',protocol='fixed-512-output-token E2E latency; separate from natural completion',runtime=args.runtime,
                k=2,sessions=12,turns=16,repeats=3,pairs=48,output_tokens_per_arm=24576,ignore_eos=True,
                ar_seconds=a['wall_seconds'],sd_seconds=b['wall_seconds'],speedup=a['wall_seconds']/b['wall_seconds'],
                session_cluster_CI95=bootstrap_ci(groups['ar'],groups['sd']),gpu=args.gpu,process_order=order,resource_evidence=resource,
                limitations=['Fixed output budget, including tokens after EOS if any; no natural completion or quality claim.','Diagnostic subset, not untouched paper evaluation.'])
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2));(root/'STATUS.json').write_text(json.dumps({'status':'complete'}));print(json.dumps(result,indent=2),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
