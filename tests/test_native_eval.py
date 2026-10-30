"""CPU protocol tests; synthetic rows are never GPU performance evidence."""
import copy
import importlib
import json
from pathlib import Path

import pytest


def native():
    try:
        return importlib.import_module('vlm_dflash.eval.native')
    except ModuleNotFoundError:
        pytest.fail('Native vLLM evaluation protocol has not been implemented')


def plan():
    return dict(target='/target', draft='/draft', speculative_tokens=7, max_new_tokens=64,
                max_model_len=8192, gpu_memory_utilization=.45, min_pixels=200704,
                max_pixels=1003520, repeats=3, seed=43, stop_token_ids=[151645,151643],
                sessions=[dict(id='a',questions=['q']),dict(id='b',questions=['q','q2'])])


def reports():
    p=plan();reference=[]
    for session in p['sessions']:
        for turn in range(len(session['questions'])):
            reference.append(dict(id=session['id'],turn=turn,prompt_token_ids=[1,2],tokens=[3],text='a'))
    stages=[]
    for stage in native().stage_order(3):
        rows=[]
        for phase in ('warmup','timed'):
            for i,ref in enumerate(reference):
                value=copy.deepcopy(ref);duration=200 if stage.startswith('ar') else 100
                value.update(phase=phase,status='complete',image_uuid=stage+phase+str(i),
                             began_ns=1000,ended_ns=1000+duration,wall_seconds=duration/1e9,
                             finish_reason='stop',stop_reason=None)
                rows.append(value)
        stages.append(dict(stage=stage,status='complete',rows=rows))
    return p,reference,stages


def test_native_ar_retains_public_automatic_defaults():
    kwargs=native().engine_args(plan(),'ar')
    assert not {'speculative_config','attention_config','compilation_config','block_size'} & kwargs.keys()
    assert kwargs['enable_prefix_caching'] is False and kwargs['mm_processor_cache_gb']==0


def test_sd_uses_public_flash_linear_speculative_config():
    kwargs=native().engine_args(plan(),'sd')
    assert kwargs['speculative_config']==dict(model='/draft',method='dflash',num_speculative_tokens=7,
                                             draft_tensor_parallel_size=1,attention_backend='FLASH_ATTN')
    assert kwargs['attention_config']=={'backend':'FLASH_ATTN'}
    assert kwargs['compilation_config']['cudagraph_capture_sizes']==[8]


def test_process_order_is_interleaved_and_complete():
    assert native().stage_order(3)==['ar_r0','sd_r0','sd_r1','ar_r1','ar_r2','sd_r2']


def test_sum_of_walls_and_raw_tokens_are_primary():
    p,ref,stages=reports();out=native().summarize(p,ref,stages)
    assert out['status']=='complete' and out['speedup_sum_ar_over_sd']==pytest.approx(2.)
    assert out['timed_pairs']==9 and out['ar_output_tokens']==out['sd_output_tokens']==9
    assert out['confidence_interval_scope']=='session clusters, fixed process sequence; not run/card/seed variance'
    assert out['paper_ready'] is False


def test_output_numerics_are_reported_without_suppressing_speed():
    p,ref,stages=reports();stages[1]['rows'][3]['tokens']=[4,5]
    out=native().summarize(p,ref,stages)
    assert out['status']=='complete' and out['exact_output_pairs']==8
    assert out['sd_output_tokens']==10 and out['quality_evaluation_required'] is True
    assert out['all_measured_output_tokens_exact'] is False


@pytest.mark.parametrize('mutation',['missing','duplicate','failed','input','duration','reorder'])
def test_incomplete_or_wrong_input_cannot_produce_speed(mutation):
    p,ref,stages=reports()
    if mutation=='missing':stages.pop()
    elif mutation=='duplicate':stages[0]['rows'][-1]=copy.deepcopy(stages[0]['rows'][-2])
    elif mutation=='failed':stages[0]['rows'][0]['status']='failed'
    elif mutation=='input':stages[1]['rows'][-1]['prompt_token_ids']=[99]
    elif mutation=='duration':stages[1]['rows'][-1]['wall_seconds']=float('nan')
    elif mutation=='reorder':stages.reverse()
    out=native().summarize(p,ref,stages)
    assert out['status']=='failed' and 'speedup_sum_ar_over_sd' not in out


def test_bootstrap_keeps_entire_session_with_all_repeats():
    # Three turns in one session must behave as one cluster, never three trials.
    ci=native().bootstrap_ci({'a':[2.,4.,6.]},{'a':[1.,2.,3.]},samples=100,seed=42)
    assert ci==pytest.approx([2.,2.])


@pytest.mark.parametrize('key',['FAIR_TORCH_PRECISION_PROFILE','TORCH_BLAS_PREFER_CUBLASLT',
    'VLLM_BATCH_INVARIANT','VLLM_ATTENTION_BACKEND','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'])
def test_native_rejects_legacy_numeric_environment(key):
    value='1' if key!='FAIR_TORCH_PRECISION_PROFILE' else 'P2'
    with pytest.raises(ValueError,match='override'):
        native().validate_environment({key:value})


def test_source_loader_keeps_all_turns_and_excludes_gold_assistants(tmp_path):
    image=tmp_path/'a.png';image.write_bytes(b'image')
    source=tmp_path/'questions.jsonl';source.write_text(json.dumps(dict(id='a',images=['a.png'],conversations=[
        {'from':'human','value':'<image>Q1'},{'from':'gpt','value':'secret gold'},
        {'from':'human','value':'Q2'},{'from':'gpt','value':'other gold'}]))+'\n')
    sessions=native().load_sessions(source,tmp_path,1)
    assert sessions[0]['questions']==['Q1','Q2'] and 'secret gold' not in json.dumps(sessions)


def test_wrong_uuid_remains_rejected():
    module=importlib.import_module('vlm_dflash.eval.native_resources')
    actual='864585d7-9789-e033-7781-e773f55178c6'
    assert module.normalize_uuid(actual)=='GPU-'+actual
    for bad in ('MIG-'+actual, ' '+actual,actual.replace('-','')):
        with pytest.raises(ValueError):module.normalize_uuid(bad)


def test_native_flags_are_observations_not_a_lossless_certificate():
    p,ref,stages=reports();out=native().summarize(p,ref,stages)
    assert 'bitwise_lossless' not in out
    assert out['all_measured_output_tokens_exact'] is True


def test_late_preservation_failure_removes_headline(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from vlm_dflash.eval import native_resources
    module=native();p,ref,stages=reports();p['method']='dflash';p['gpus']=[0]
    fake_uuid='GPU-864585d7-9789-e033-7781-e773f55178c6'
    class Lease:
        uuid=fake_uuid;uuids=[fake_uuid]
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def boundary(self):pass
        def validate(self,timed_windows=None):return {'status':'complete','began_wall_ns':0,'ended_wall_ns':10**20}
    monkeypatch.setattr(native_resources,'GPULease',Lease)
    monkeypatch.setattr(module,'make_plan',lambda args:p)
    monkeypatch.setattr(module,'verify_plan',lambda p:None)
    calls=[]
    def preservation(p):
        calls.append(1)
        if len(calls)==2:raise ValueError('late checkpoint identity failure')
        return {'unit_test_only':True}
    monkeypatch.setattr(module,'verify_preservation',preservation)
    def process(command,**kwargs):
        stage=command[command.index('--stage')+1];directory=Path(command[-1])
        report={'stage':'reference','status':'complete','rows':ref} if stage=='reference' else copy.deepcopy(next(r for r in stages if r['stage']==stage))
        for row in report['rows']:row.update(began_wall_ns=1000,ended_wall_ns=2000)
        report.update(numeric_parent={},worker_before={'numeric':{}},worker_after={'numeric':{}},gpu_uuid=fake_uuid,
                      engine_args=module.engine_args(p,'sd' if stage.startswith('sd') else 'ar'))
        module.save(directory/(stage+'.json'),report);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',process)
    args=SimpleNamespace(out=str(tmp_path/'result.json'),dry_run=False,gpu=0,method='dflash',repeats=3)
    assert module.run(args)==1
    out=json.loads((tmp_path/'result.json').read_text())
    assert out['status']=='failed' and 'speedup_sum_ar_over_sd' not in out
    # The run must get as far as the late identity check, not fail earlier.
    assert len(calls)==2 and 'late checkpoint identity failure' in out['error']



def test_complete_preservation_supports_real_scalar_visual_gate(tmp_path):
    import torch
    from safetensors.torch import save_file
    checkpoint=tmp_path/'weights.pt';draft=tmp_path/'export';draft.mkdir()
    tensors={'visual_gate':torch.tensor(.25,dtype=torch.bfloat16),'projection':torch.arange(4,dtype=torch.bfloat16)}
    torch.save({'algorithm':'dflash','model':tensors},checkpoint)
    save_file(tensors,str(draft/'model.safetensors'))
    result=native().verify_preservation({'method':'dflash','draft':str(draft),'adapter':{'checkpoint':str(checkpoint)}})
    assert result['all_values_bitwise_equal'] and result['tensor_count']==2


def test_stock_dflash_keeps_native_argmax_optimization_but_heads_get_full_logits():
    p=plan();p['method']='dflash'
    assert 'use_local_argmax_reduction' not in native().engine_args(p,'sd')['speculative_config']
    p['method']='dfly'
    assert native().engine_args(p,'sd')['speculative_config']['use_local_argmax_reduction'] is False


def test_vllm_config_mutation_does_not_corrupt_raw_constructor_snapshot(tmp_path,monkeypatch):
    """CPU fake API for the real vLLM constructor's in-place ModelConfig insertion."""
    import sys
    from types import SimpleNamespace
    from vlm_dflash.eval import native_runtime as runtime
    p,refs,_=reports();p.update(method='dflash',gpu_uuid='GPU-864585d7-9789-e033-7781-e773f55178c6')
    file=tmp_path/'plan.json';file.write_text(json.dumps(p))
    (tmp_path/'reference.json').write_text(json.dumps({'status':'complete','rows':refs}))
    class LLM:
        def __init__(self,**kwargs):
            kwargs['speculative_config']['draft_model_config']=object()
        llm_engine=None
    cuda=SimpleNamespace(is_initialized=lambda:False,synchronize=lambda:None,
        get_device_properties=lambda index:SimpleNamespace(uuid=p['gpu_uuid']))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:kw))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:None)))
    monkeypatch.setattr(runtime,'verify_plan',lambda plan:None)
    monkeypatch.setattr(runtime,'verify_preservation',lambda plan:{'CPU_fake_API_only':True})
    monkeypatch.setattr(runtime,'numeric_state',lambda:{})
    monkeypatch.setattr(runtime,'worker_metadata',lambda *a:{'CPU_fake_API_only':True})
    monkeypatch.setattr(runtime,'metrics',lambda llm:None)
    monkeypatch.setattr(runtime.importlib.metadata,'version',lambda name:'0.22.1')
    monkeypatch.setattr(runtime,'request',lambda llm,processor,params,torch,Image,s,t,r,plan:
        dict(id=s['id'],turn=t,prompt_token_ids=r[s['id'],t]['prompt_token_ids'],tokens=[3],text='a',status='complete'))
    monkeypatch.setenv('VELADRAFT_NATIVE_PLAN_SHA256',native().digest(file));monkeypatch.setenv('VELADRAFT_NATIVE_ARM','sd')
    monkeypatch.setattr(sys,'argv',['native_runtime','--plan',str(file),'--stage','sd_r0','--directory',str(tmp_path)])
    assert runtime.main()==0
    out=json.loads((tmp_path/'sd_r0.json').read_text())
    assert out['status']=='complete' and out['engine_args']==native().engine_args(p,'sd')


def test_tensor_parallel_lane_sets_target_and_draft_tp():
    p=plan();p['tensor_parallel_size']=2
    kwargs=native().engine_args(p,'sd')
    assert kwargs['tensor_parallel_size']==2 and kwargs['speculative_config']['draft_tensor_parallel_size']==2
    assert native().engine_args(p,'ar')['tensor_parallel_size']==2


def test_gpu_list_resolution():
    from types import SimpleNamespace
    mod=native()
    assert mod.resolve_gpus(SimpleNamespace(gpu=3))==[3]
    assert mod.resolve_gpus(SimpleNamespace(gpu=0,gpus='4,5'))==[4,5]
    for bad in ('1,1','0,1,2'):
        with pytest.raises(ValueError):mod.resolve_gpus(SimpleNamespace(gpu=0,gpus=bad))


def test_stop_ids_come_from_target_generation_config(tmp_path):
    (tmp_path/'generation_config.json').write_text(json.dumps({'eos_token_id':[248046,248044]}))
    assert native().target_stop_token_ids(tmp_path)==[248046,248044]
    (tmp_path/'generation_config.json').write_text(json.dumps({'eos_token_id':151645}))
    assert native().target_stop_token_ids(tmp_path)==[151645]
    (tmp_path/'generation_config.json').write_text(json.dumps({}))
    with pytest.raises(ValueError):native().target_stop_token_ids(tmp_path)


def test_system_prompt_is_first_message_and_image_stays_on_first_user_turn():
    from vlm_dflash.eval.native_runtime import messages_for
    session=dict(id='a',questions=['q0','q1'])
    reference={('a',0):dict(text='a0')}
    messages=messages_for(session,1,reference,'sys')
    assert [m['role'] for m in messages]==['system','user','assistant','user']
    assert messages[1]['content'][0]=={'type':'image'} and messages[3]['content']==[{'type':'text','text':'q1'}]
    assert [m['role'] for m in messages_for(session,0,reference)]==['user']


def test_worker_metadata_requires_every_tensor_parallel_rank():
    from vlm_dflash.eval import native_runtime as runtime
    rank=dict(numeric={},attention=[dict(name='language_model.model.layers.3.self_attn.attn',impl='x.FlashAttentionImpl'),
                                     dict(name='model.layers.40.self_attn.attn',impl='x.FlashAttentionImpl')],
              speculative_tokens=7,draft_noncausal=True,target_noncausal=False,diagnostic_hooks=False)
    class LLM:
        def __init__(self,values):self.values=values
        def collective_rpc(self,*a,**k):return self.values
    p=plan();p.update(method='dflash',tensor_parallel_size=2)
    assert runtime.worker_metadata(LLM([rank,copy.deepcopy(rank)]),'sd',p)['ranks']==2
    with pytest.raises(ValueError,match='worker count'):runtime.worker_metadata(LLM([rank]),'sd',p)
    other=copy.deepcopy(rank);other['numeric']={'bf16_reduced':True}
    with pytest.raises(ValueError,match='different'):runtime.worker_metadata(LLM([rank,other]),'sd',p)


def test_sd_graph_mode_defaults_to_piecewise_and_admits_native_full():
    p=plan()
    assert native().engine_args(p,'sd')['compilation_config']['cudagraph_mode']=='PIECEWISE'
    p['sd_cudagraph_mode']='FULL_AND_PIECEWISE'
    config=native().engine_args(p,'sd')['compilation_config']
    assert config['cudagraph_mode']=='FULL_AND_PIECEWISE' and config['cudagraph_capture_sizes']==[8]
    assert 'compilation_config' not in native().engine_args(p,'ar')
