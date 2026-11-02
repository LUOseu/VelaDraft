"""Uninstrumented matched-context validation of an official MoE tuning file."""
import argparse
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
from vlm_dflash.eval.native_runtime import WorkerMetadata

class MoEAuditWorker(WorkerMetadata):
    def moe_config_snapshot(self):
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        n=512//self.vllm_config.parallel_config.tensor_parallel_size
        return {'rank':self.rank,'folder':os.environ.get('VLLM_TUNED_CONFIG_FOLDER'),
                'configs':{str(m):try_get_optimal_moe_config((256,2*n,2048),(256,2048,n),8,None,m) for m in (1,2,3,4,8,32,128,4096)}}

def main():
    p=argparse.ArgumentParser()
    for name in ('plan','reference','arm','label','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.plan).read_text());ref=json.loads(Path(args.reference).read_text());assert ref['status']=='complete'
    refs={(x['id'],x['turn']):x for x in ref['rows']};multi=next(s for s in plan['sessions'] if len(s['questions'])>=2)
    selected=[(plan['sessions'][0],0),(multi,0),(multi,len(multi['questions'])-1)]
    from vlm_dflash.eval.native import engine_args,verify_plan,verify_preservation,save
    from vlm_dflash.eval.native_runtime import request,worker_metadata,numeric_state
    verify_plan(plan);preserved=verify_preservation(plan)
    tuned=os.environ.get('VLLM_TUNED_CONFIG_FOLDER');config_file=None
    if tuned:
        n=512//plan.get('tensor_parallel_size',1);files=list(Path(tuned).glob(f'E=256,N={n},device_name=*.json'))
        if len(files)!=1:raise ValueError('Expected exactly one matched MoE tuning file')
        config_file=files[0]
    tuning_sha=hashlib.sha256(config_file.read_bytes()).hexdigest() if config_file else None
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    kwargs=engine_args(plan,args.arm);kwargs['worker_extension_cls']='research.qwen36_rootcause.e2e_stage.MoEAuditWorker'
    requested=copy.deepcopy(kwargs);llm=LLM(**copy.deepcopy(kwargs))
    def shutdown():
        try:llm.llm_engine.engine_core.shutdown()
        except Exception:pass
    atexit.register(shutdown)
    before=worker_metadata(llm,args.arm,plan);config_before=llm.collective_rpc('moe_config_snapshot')
    proc=AutoProcessor.from_pretrained(plan['target'],local_files_only=True,min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels'])
    sampling=SamplingParams(temperature=0,max_tokens=128,seed=plan['seed'],stop_token_ids=plan['stop_token_ids'])
    result={'status':'running','label':args.label,'arm':args.arm,'observer_hooks':False,'max_new_tokens':128,'rows':[],
            'engine_args':requested,'tuning_sha256':tuning_sha,'tuning_folder':tuned,'moe_snapshot':config_before,'worker_before':before,'preservation':preserved,'numeric':numeric_state()}
    save(out/'result.json',result)
    for session,turn in selected:request(llm,proc,sampling,torch,Image,session,turn,refs,plan)
    for repeat in range(3):
        for session,turn in selected:
            row=request(llm,proc,sampling,torch,Image,session,turn,refs,plan)
            assert row['prompt_token_ids']==refs[session['id'],turn]['prompt_token_ids']
            row['repeat']=repeat;result['rows'].append(row);save(out/'result.json',result)
    after=worker_metadata(llm,args.arm,plan);assert after==before
    config_after=llm.collective_rpc('moe_config_snapshot');assert config_after==config_before
    if config_file:assert hashlib.sha256(config_file.read_bytes()).hexdigest()==tuning_sha
    llm.llm_engine.engine_core.shutdown();atexit.unregister(shutdown);verify_plan(plan)
    result.update(status='complete',worker_after=after,wall_seconds=sum(x['wall_seconds'] for x in result['rows']),output_tokens=sum(len(x['tokens']) for x in result['rows']))
    save(out/'result.json',result);print('E2E_COMPLETE',args.label,result['wall_seconds'],result['output_tokens'],flush=True)

if __name__=='__main__':main()
