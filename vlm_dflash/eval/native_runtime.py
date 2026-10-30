"""One fresh native LLM per process. No private verifier or timing hooks."""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time
import traceback
import uuid

from .native import digest,engine_args,save,verify_plan,verify_preservation,validate_environment
from .native_resources import normalize_uuid


def numeric_state():
    import torch
    matmul=torch.backends.cuda.matmul
    return dict(bf16_reduced=matmul.allow_bf16_reduced_precision_reduction,
        bf16_split_k=getattr(matmul,'allow_bf16_reduced_precision_reduction_split_k',None),
        blas=str(torch.backends.cuda.preferred_blas_library()),
        fp32_matmul=getattr(matmul,'fp32_precision',None))


class WorkerMetadata:
    """Read-only public collective_rpc extension, with no sampler installation."""
    def native_eval_metadata(self):
        context=self.vllm_config.compilation_config.static_forward_context
        entries=[dict(name=name,impl=type(layer.impl).__module__+'.'+type(layer.impl).__name__)
                 for name,layer in sorted(context.items()) if hasattr(layer,'impl')]
        drafter=getattr(self.model_runner,'drafter',None)
        draft_config=drafter._create_draft_vllm_config() if drafter is not None else None
        spec=self.vllm_config.speculative_config
        return dict(numeric=numeric_state(),attention=entries,
                    speculative_tokens=spec.num_speculative_tokens if spec else None,
                    draft_noncausal=draft_config.attention_config.use_non_causal if draft_config else None,
                    target_noncausal=self.vllm_config.attention_config.use_non_causal,
                    diagnostic_hooks=bool(getattr(self.model_runner,'_diagnostic_hooks_installed',False)))


def worker_metadata(llm,arm,plan):
    values=llm.collective_rpc('native_eval_metadata',timeout=90,args=(),kwargs={})
    if len(values)!=plan.get('tensor_parallel_size',1):raise ValueError('Actual worker count differs from tensor parallel size')
    for value in values:check_worker(value,arm,plan)
    if any(value!=values[0] for value in values[1:]):raise ValueError('Tensor-parallel ranks report different numeric/backend state')
    return dict(values[0],ranks=len(values))


def check_worker(value,arm,plan):
    if value['target_noncausal'] or value['diagnostic_hooks']:raise ValueError('Noncausal target or diagnostic sampler hooks')
    if not any(row['name'].startswith('language_model.model.layers.') for row in value['attention']):
        raise ValueError('Actual target language attention backend evidence unavailable')
    if arm=='sd' and (value['speculative_tokens']!=plan['speculative_tokens'] or
                      value['draft_noncausal'] is not (plan['method']!='jetspec')):
        raise ValueError('Actual draft speculative budget/causality differs from admitted method')
    if arm=='ar' and value['speculative_tokens'] is not None:raise ValueError('AR unexpectedly mounted a drafter')
    if arm=='sd' and (not any(row['name'].startswith('model.layers.') for row in value['attention']) or
                      any(not row['impl'].endswith('.FlashAttentionImpl') for row in value['attention'])):
        raise ValueError('Actual SD target/draft attention is not FlashAttention')


def metrics(llm):
    names={'vllm:spec_decode_num_drafts':'rounds','vllm:spec_decode_num_draft_tokens':'proposed',
           'vllm:spec_decode_num_accepted_tokens':'accepted'}
    try:
        values={names[m.name]:float(m.value) for m in llm.get_metrics() if m.name in names}
        if set(values)!=set(names.values()):return None
        if any(not v.is_integer() or v<0 for v in values.values()):return None
        return values
    except Exception:return None


def metric_delta(before,after):
    if before is None or after is None:return dict(status='unavailable')
    values={k:after[k]-before[k] for k in before}
    if min(values.values())<0 or values['accepted']>values['proposed']:return dict(status='unavailable')
    return dict(status='complete',**values,
                accepted_draft_tokens_per_round=values['accepted']/values['rounds'] if values['rounds'] else None,
                accepted_draft_fraction=values['accepted']/values['proposed'] if values['proposed'] else None,
                definition='accepted draft tokens only; no synthetic +1 bonus')


def messages_for(session,turn,reference,system_prompt=None):
    messages=[{'role':'system','content':[{'type':'text','text':system_prompt}]}] if system_prompt else []
    for index in range(turn+1):
        content=[{'type':'text','text':session['questions'][index]}]
        if index==0:content.insert(0,{'type':'image'})
        messages.append({'role':'user','content':content})
        if index<turn:messages.append({'role':'assistant','content':reference[(session['id'],index)]['text']})
    return messages


def request(llm,processor,params,torch,Image,session,turn,reference,plan):
    torch.cuda.synchronize();start=time.perf_counter_ns();wall_start=time.time_ns();image=None
    try:
        data=Path(session['image']).read_bytes()
        import hashlib
        image_sha=hashlib.sha256(data).hexdigest()
        if image_sha!=session['image_sha256']:raise ValueError('Physical input image changed')
        with Image.open(io.BytesIO(data)) as original:
            image=original.convert('RGB');image.load()
        messages=messages_for(session,turn,reference,plan.get('system_prompt'))
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**plan.get('chat_template_kwargs',{}))
        image_uuid=str(uuid.uuid4())
        outputs=llm.generate([dict(prompt=prompt,multi_modal_data={'image':image},
                                  multi_modal_uuids={'image':[image_uuid]})],params,use_tqdm=False)
        torch.cuda.synchronize()
        if len(outputs)!=1 or len(outputs[0].outputs)!=1:raise ValueError('Batch1 single-output contract changed')
        output=outputs[0].outputs[0]
        tokens=list(output.token_ids);inputs=list(outputs[0].prompt_token_ids)
        text=output.text;finish=output.finish_reason;stop=output.stop_reason
        end=time.perf_counter_ns();wall_end=time.time_ns()
        return dict(id=session['id'],turn=turn,status='complete',prompt_token_ids=inputs,tokens=tokens,text=text,
                    finish_reason=finish,stop_reason=stop,image_uuid=image_uuid,image_sha256=image_sha,
                    messages=messages,prompt=prompt,began_ns=start,ended_ns=end,wall_seconds=(end-start)/1e9,
                    began_wall_ns=wall_start,ended_wall_ns=wall_end)
    finally:
        if image is not None:image.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());verify_plan(plan);validate_environment(os.environ)
    if digest(args.plan)!=os.environ.get('VELADRAFT_NATIVE_PLAN_SHA256'):raise ValueError('Plan/bootstrap SHA mismatch')
    arm='sd' if args.stage.startswith('sd') else 'ar'
    if arm!=os.environ.get('VELADRAFT_NATIVE_ARM'):raise ValueError('Stage/bootstrap arm mismatch')
    if importlib.metadata.version('vllm')!='0.22.1':raise ValueError('Native protocol pins vLLM0.22.1')
    out=args.directory/(args.stage+'.json')
    if out.exists():raise FileExistsError('Existing stage evidence preserved')
    report=dict(stage=args.stage,status='running',rows=[],GPU_executed=False)
    save(out,report);llm=None
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import LLM,SamplingParams
        report['versions']={k:importlib.metadata.version(k) for k in ('torch','transformers','vllm')}
        if torch.cuda.is_initialized():raise ValueError('Adapter initialized CUDA before model construction')
        report['numeric_parent']=numeric_state();report['preservation']=verify_preservation(plan)
        processor=AutoProcessor.from_pretrained(plan['target'],local_files_only=True,
            min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels'])
        kwargs=engine_args(plan,arm);report['engine_args']=kwargs
        llm=LLM(**copy.deepcopy(kwargs));torch.cuda.synchronize();report['GPU_executed']=True
        report['gpu_uuids']=[normalize_uuid(torch.cuda.get_device_properties(i).uuid) for i in range(plan.get('tensor_parallel_size',1))]
        report['gpu_uuid']=report['gpu_uuids'][0]
        if report['gpu_uuids']!=plan.get('gpu_uuids',[plan['gpu_uuid']]):raise ValueError('Actual CUDA devices differ from locked GPUs')
        report['worker_before']=worker_metadata(llm,arm,plan)
        params=SamplingParams(temperature=0,max_tokens=plan['max_new_tokens'],seed=plan['seed'],
                              stop_token_ids=plan['stop_token_ids'],ignore_eos=False)
        reference={}
        if args.stage!='reference':
            prior=json.loads((args.directory/'reference.json').read_text())
            if prior['status']!='complete':raise ValueError('A complete native reference is required')
            reference={(r['id'],r['turn']):r for r in prior['rows']}
        # Per-request official counters only in a separate stats-enabled supplement
        # (disable_log_stats=False); primary timing runs never poll metrics per request.
        per_request=arm=='sd' and kwargs.get('disable_log_stats') is False
        for phase in (('reference',) if args.stage=='reference' else ('warmup','timed')):
            before=metrics(llm) if arm=='sd' and phase=='timed' else None
            for session in plan['sessions']:
                for turn in range(len(session['questions'])):
                    try:
                        start=metrics(llm) if per_request and phase=='timed' else None
                        row=request(llm,processor,params,torch,Image,session,turn,reference,plan)
                        row['phase']=phase
                        if start is not None:row['official_counters']=metric_delta(start,metrics(llm))
                        if phase!='reference' and row['prompt_token_ids']!=reference[(session['id'],turn)]['prompt_token_ids']:
                            raise ValueError('Physical input tokens differ from canonical history')
                    except Exception as error:
                        report['rows'].append(dict(id=session['id'],turn=turn,phase=phase,status='failed',error=str(error)))
                        save(out,report);raise
                    report['rows'].append(row)
                    if phase=='reference':reference[(session['id'],turn)]=row
                    save(out,report)
            if arm=='sd' and phase=='timed':report['timed_official_counters']=metric_delta(before,metrics(llm))
        report['worker_after']=worker_metadata(llm,arm,plan)
        if report['worker_after']!=report['worker_before']:raise ValueError('Worker state changed during collection')
        verify_plan(plan);report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                core=getattr(getattr(llm,'llm_engine',None),'engine_core',None)
                if core is not None:core.shutdown()
            except Exception as error:report.update(status='failed',shutdown_error=str(error))
            llm=None;gc.collect()
        save(out,report)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
