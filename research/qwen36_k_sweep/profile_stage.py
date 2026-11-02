"""Separate observer-affected profile run on identical canonical contexts, not E2E evidence."""
import argparse
import atexit
import copy
import hashlib
import json
import os
from pathlib import Path

def freeze_engine_args(value):
    frozen=copy.deepcopy(value)
    json.dumps(frozen,allow_nan=False)
    return frozen

def save_json(path,value):
    encoded=json.dumps(value,indent=2,allow_nan=False)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(encoded)
    os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','reference','arm','out'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--repetitions',type=int,default=1)
    parser.add_argument('--max-tokens',type=int,default=128)
    parser.add_argument('--context-index',type=int,choices=(0,1,2))
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.plan).read_text());reference=json.loads(Path(args.reference).read_text())
    assert reference['status']=='complete'
    refs={(r['id'],r['turn']):r for r in reference['rows']}
    multi=next(s for s in plan['sessions'] if len(s['questions'])>=2)
    selected=[(plan['sessions'][0],0),(multi,0),(multi,len(multi['questions'])-1)]
    if args.context_index is not None:selected=[selected[args.context_index]]
    from vlm_dflash.eval.native import engine_args,verify_plan
    from vlm_dflash.eval.native_runtime import request,metrics,metric_delta
    verify_plan(plan)
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    kwargs=engine_args(plan,args.arm)
    kwargs.update(worker_extension_cls='research.qwen36_k_sweep.profile_worker.ProfileWorker',disable_log_stats=False)
    frozen_kwargs=freeze_engine_args(kwargs)
    llm=LLM(**copy.deepcopy(kwargs))
    def shutdown():
        try: llm.llm_engine.engine_core.shutdown()
        except Exception: pass
    atexit.register(shutdown)
    proc=AutoProcessor.from_pretrained(plan['target'],local_files_only=True,min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels'])
    params=SamplingParams(temperature=0,max_tokens=args.max_tokens,seed=plan['seed'],stop_token_ids=plan['stop_token_ids'],ignore_eos=False)
    for session,turn in selected:request(llm,proc,params,torch,Image,session,turn,refs,plan)
    wiring=llm.collective_rpc('q36_profile_install',timeout=120)
    rows=[]
    for repetition,(session,turn) in enumerate(selected*args.repetitions):
        before=metrics(llm)
        llm.collective_rpc('q36_profile_begin',timeout=120,args=(f'{session["id"]}_t{turn}_r{repetition//len(selected)}',))
        record=request(llm,proc,params,torch,Image,session,turn,refs,plan)
        spans=llm.collective_rpc('q36_profile_end',timeout=120)
        after=metrics(llm)
        assert record['prompt_token_ids']==refs[session['id'],turn]['prompt_token_ids']
        rows.append({'id':session['id'],'turn':turn,'repetition':repetition//len(selected),'prompt_tokens':len(record['prompt_token_ids']),
                     'actual_total_output_tokens':len(record['tokens']),'finish_reason':record['finish_reason'],
                     'observed_request_wall_s':record['wall_seconds'],'accepted_counters':metric_delta(before,after),
                     'actual_committed_tokens_per_spec_round':None,
                     'committed_count_note':'Output token count is measured per request; no guessed +1 progress count or attribution after EOS truncation.',
                     'ranks':spans})
        save_json(out/'profile.json',{'status':'running','diagnostic_only':True,'rows':rows})
    llm.llm_engine.engine_core.shutdown();atexit.unregister(shutdown);verify_plan(plan)
    result={'status':'complete','diagnostic_only':True,'headline_speedup_claim':False,'arm':args.arm,'k':plan['speculative_tokens'],
            'engine_args':frozen_kwargs,'max_new_tokens':args.max_tokens,'wiring':wiring,'rows':rows,
            'reference_sha256':hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
            'limitations':['CPU spans and CUDA stream elapsed spans include instrumentation overhead.',
                          'Inclusive spans contain named children; do not add them together.',
                          'Report per-rank times; TP collective cost is included, not independently isolated.',
                          'Use decode phase q=1 for AR target forward, verify phase actual q for SD.',
                          'First and later-turn contexts are fixed across arms and K; output tokens need not match.']}
    result['limitations'].append('Initial prompts are identical, but subsequent decoding prefixes can diverge; this is not an exact fixed-prefix kernel microbenchmark.')
    save_json(out/'profile.json',result);print('PROFILE_STAGE_COMPLETE',args.arm,plan['speculative_tokens'],flush=True)

if __name__=='__main__':main()
