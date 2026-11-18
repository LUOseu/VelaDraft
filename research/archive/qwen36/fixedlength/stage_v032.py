"""Public native MRV2 runtime comparison; separate from the v0.22 main protocol."""
import argparse
import atexit
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('ar','sd'),required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-tokens',type=int,default=128);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--all-sessions',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('Q36_V032_ARM')!=args.arm:raise ValueError('Worker bootstrap arm mismatch')
    if importlib.metadata.version('vllm')!='0.32.0':raise ValueError('Wrong isolated runtime')
    planpath=Path('/root/autodl-tmp/qwen36_rootcause_20261101/k2_setup/plan.evidence/plan.json')
    refpath=Path('/root/autodl-tmp/qwen36_tp1_20261101/results/dflash_tp1_a3.evidence/reference.json')
    plan=json.loads(planpath.read_text());reference=json.loads(refpath.read_text());assert reference['status']=='complete'
    refs={(r['id'],r['turn']):r for r in reference['rows']}
    multi=next(s for s in plan['sessions'] if len(s['questions'])>=2)
    selected=[(plan['sessions'][0],0),(multi,0),(multi,len(multi['questions'])-1)]
    if args.all_sessions:
        selected=[(s,t) for s in plan['sessions'] for t in range(len(s['questions']))]
        assert len(plan['sessions'])==12 and len(selected)==16
    from vlm_dflash.eval.native import save,verify_preservation
    from vlm_dflash.eval.native_runtime import request,numeric_state
    preserved=verify_preservation(plan)
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    kwargs=dict(model=plan['target'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=8192,
                max_num_seqs=1,max_num_batched_tokens=4096,gpu_memory_utilization=plan['gpu_memory_utilization'],
                enable_prefix_caching=False,mm_processor_cache_gb=0,limit_mm_per_prompt={'image':1},
                enforce_eager=False,generation_config='vllm',disable_log_stats=True,
                mm_processor_kwargs=dict(min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels']),
                attention_config={'backend':'FLASH_ATTN'},worker_extension_cls='research.qwen36_v032.worker.AuditWorker')
    if args.arm=='sd':
        kwargs['speculative_config']=dict(model=plan['draft'],method='dflash',num_speculative_tokens=2,
                                        draft_tensor_parallel_size=1,attention_backend='FLASH_ATTN')
        kwargs['compilation_config']=dict(cudagraph_mode='FULL_AND_PIECEWISE',cudagraph_capture_sizes=[3],max_cudagraph_capture_size=3)
    requested=copy.deepcopy(kwargs)
    llm=LLM(**kwargs)
    def shutdown():
        try:llm.llm_engine.engine_core.shutdown()
        except Exception:pass
    atexit.register(shutdown)
    before=llm.collective_rpc('backend_metadata',timeout=120)
    if len(before)!=1 or not before[0]['use_v2'] or before[0]['target_noncausal']:raise ValueError('Unexpected target runtime')
    if args.arm=='sd':
        if before[0]['speculative_tokens']!=2 or not before[0]['draft_graphs'] or 'FULL' not in before[0]['draft_graph_mode']:
            raise ValueError('DFlash did not capture the official full graph')
    proc=AutoProcessor.from_pretrained(plan['target'],local_files_only=True,min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels'])
    sampling=SamplingParams(temperature=0,max_tokens=args.max_tokens,seed=plan['seed'],ignore_eos=True)
    result=dict(status='running',protocol='q36-fixed-output-v032-mrv2',fixed_output_length=True,ignore_eos=True,arm=args.arm,k=2 if args.arm=='sd' else 0,
                versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers')},
                worker_before=before,engine_args=requested,preservation=preserved,rows=[],numeric=numeric_state(),
                input_plan_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),observer_hooks=False,
                max_new_tokens=args.max_tokens,repetitions=args.repeats)
    save(out/'result.json',result)
    for session,turn in selected:request(llm,proc,sampling,torch,Image,session,turn,refs,plan)
    for repeat in range(args.repeats):
        for session,turn in selected:
            row=request(llm,proc,sampling,torch,Image,session,turn,refs,plan)
            if row['prompt_token_ids']!=refs[session['id'],turn]['prompt_token_ids']:
                raise ValueError('New runtime physical prompt IDs differ from the frozen diagnostic inputs')
            assert len(row['tokens'])==args.max_tokens and row['finish_reason']=='length', 'Fixed output-length contract failed'
            row['repeat']=repeat;result['rows'].append(row);save(out/'result.json',result)
    after=llm.collective_rpc('backend_metadata',timeout=120);assert after==before
    shutdown();atexit.unregister(shutdown);assert verify_preservation(plan)==preserved
    result.update(status='complete',worker_after=after,wall_seconds=sum(r['wall_seconds'] for r in result['rows']),output_tokens=sum(len(r['tokens']) for r in result['rows']))
    save(out/'result.json',result);print('V032_COMPLETE',args.arm,result['wall_seconds'],result['output_tokens'],flush=True)


if __name__=='__main__':main()
