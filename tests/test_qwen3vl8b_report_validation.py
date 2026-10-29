import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'research' / 'qwen3vl8b_native_vllm' / 'report_validation.py'


def module():
    assert MODULE.exists(), 'result-to-audit validation is missing'
    spec=importlib.util.spec_from_file_location('qwen_report_validation', MODULE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def make_evidence(tmp_path):
    d=tmp_path/'run.evidence';d.mkdir()
    stages=('reference','ar_r0','sd_r0','sd_r1','ar_r1','ar_r2','sd_r2')
    for name in ('plan.json','resources.jsonl',*(s+'.log' for s in stages)):
        (d/name).write_text(name)
    for stage in stages:
        count=39 if stage.startswith('sd_') else 36
        worker={'attention':[{'name':str(i),'impl':'vllm.v1.attention.backends.flash_attn.FlashAttentionImpl'} for i in range(count)],'numeric':{'mode':'native'},'target_noncausal':False,'diagnostic_hooks':False,'speculative_tokens':7 if stage.startswith('sd_') else None,'draft_noncausal':True if stage.startswith('sd_') else None};row={'status':'complete','numeric_parent':{'mode':'native'},'worker_before':worker,'worker_after':worker}
        (d/(stage+'.json')).write_text(json.dumps(row))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    result={'status':'complete','resources_verified':True,'method':'dflare','sessions':120,'repeats':3,'timed_pairs':435,
            'resource_evidence':{'status':'complete'},'smoke':False,'exact_output_pairs':50,'ar_seconds':200.0,'sd_seconds':100.0,'speedup_sum_ar_over_sd':2.0,
            'session_cluster_CI95':[1.5,2.5],'source_plan_sha256':hashes['plan.json'],'evidence':str(d)}
    result_path=tmp_path/'result.json';result_path.write_text(json.dumps(result))
    audit={'status':'complete','method':'dflare','result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
           'ar_seconds':200.0,'sd_seconds':100.0,'speedup':2.0,'session_CI95':[1.5,2.5],
           'source_sha256':hashes,'comparisons':{'paired_r0':{'exact':17},'paired_r1':{'exact':17},'paired_r2':{'exact':16}}}
    backend={'status':'complete','stages':{s:{'all_flash':True,'count':39 if s.startswith('sd_') else 36,
                                               'source_sha256':hashes[s+'.json']} for s in stages}}
    return result_path,audit,backend,d


def test_accepts_only_bound_result_and_evidence(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    m.validate_speed(path,audit,'dflare',backend)
    result=json.loads(path.read_text());result['speedup_sum_ar_over_sd']=3;path.write_text(json.dumps(result))
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)


def test_rejects_reused_audit_or_changed_backend_stage(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    with pytest.raises(ValueError):m.validate_speed(path,audit,'domino',backend)
    backend['stages']['ar_r0']['source_sha256']='0'*64
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)
    backend['stages']['ar_r0']['source_sha256']=audit['source_sha256']['ar_r0.json']
    (evidence/'sd_r0.json').write_text('changed')
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)


def test_report_does_not_display_speed_if_result_no_longer_matches_audit(tmp_path,monkeypatch):
    path,audit,backend,evidence=make_evidence(tmp_path)
    result=json.loads(path.read_text());result['method']='dflash';result['evidence']=str(evidence)
    primary=tmp_path/'dflash_formal_v5.json';primary.write_text(json.dumps(result))
    audit['method']='dflash';audit['result_sha256']=hashlib.sha256(primary.read_bytes()).hexdigest()
    (tmp_path/'dflash_formal_audit_v1.json').write_text(json.dumps(audit))
    folder=MODULE.parent;monkeypatch.syspath_prepend(str(folder))
    p=folder/'build_qwen3vl8b_paper_results.py';spec=importlib.util.spec_from_file_location('qwen_report_builder',p)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    assert '| dflash | 2.000× |' in builder.build(tmp_path)
    result['speedup_sum_ar_over_sd']=9.0;primary.write_text(json.dumps(result))
    text=builder.build(tmp_path)
    assert '| dflash | 9.000× |' not in text
    assert '结果与审计绑定未通过' in text


def test_quality_summary_is_bound_to_raw_paired_rows(tmp_path):
    m=module();quality=tmp_path/'quality';quality.mkdir()
    tasks=('textvqa','docvqa','chartqa');raw_hashes={};report_hashes={}
    for arm in ('ar','dflare'):
        directory=quality/arm;directory.mkdir();rows=[]
        for task in tasks:
            for i in range(256):
                rows.append({'phase':'quality','status':'complete','task':task,'id':f'{task}-{i}',
                             'group_id':str(i),'image_sha256':'abc','gold_answers':['gold'],
                             'prompt_token_ids':[1,2],'tokens':[3],'score_full':1.0,
                             'image_uuid':f'{arm}-{task}-{i}'})
        raw=directory/'predictions.jsonl';raw.write_text(''.join(json.dumps(x)+'\n' for x in rows));raw_hashes[arm]=m.sha(raw)
        engine={key:0 for key in ('model','dtype','tensor_parallel_size','max_model_len','max_num_seqs',
            'max_num_batched_tokens','gpu_memory_utilization','enable_prefix_caching','mm_processor_cache_gb',
            'limit_mm_per_prompt','enforce_eager','generation_config','disable_log_stats','mm_processor_kwargs')}
        report={'status':'complete','arm':'ar' if arm=='ar' else 'sd','method':'dflare',
                'failures':0,'resource_evidence':{'status':'complete','samples':2,'began_wall_ns':10,'ended_wall_ns':20},'raw_sha256':raw_hashes[arm],
                'task_counts':{task:256 for task in tasks},'gpu_uuid':'GPU-test','temperature':0,'seed':43,'max_new_tokens':128,
                'numeric_parent':{'a':1},'worker_before':{'numeric':{'a':1}},'worker_after':{'numeric':{'a':1}},'engine_args':engine,
                'preservation':{'all_values_bitwise_equal':True},
                'quality_source_manifest_sha256':'a'*64,'quality_protocol_sha256':'b'*64,'data_manifest_sha256':'d'*64}
        (directory/'resources.jsonl').write_text(json.dumps({'phase':'start','gpu_uuid':'GPU-test','foreign_pids':[],'began_wall_ns':10})+'\n'+json.dumps({'phase':'end','gpu_uuid':'GPU-test','foreign_pids':[],'ended_wall_ns':20})+'\n')
        rp=directory/'quality.json';rp.write_text(json.dumps(report));report_hashes[arm]=m.sha(rp)
    entry={'raw_sha256':raw_hashes['dflare'],'report_sha256':report_hashes['dflare'],'tasks':{}}
    for task in tasks:entry['tasks'][task]={'full':{'ar_score':100.,'sd_score':100.,'delta_percentage_points':0.},'exact_token_outputs':256}
    pair={'status':'complete','real_gold_scoring':True,'ar_raw_sha256':raw_hashes['ar'],'methods':{'dflare':entry}}
    m.validate_quality(quality,pair,'dflare')
    ar_report=quality/'ar'/'quality.json';altered=json.loads(ar_report.read_text());altered['temperature']=0.7;ar_report.write_text(json.dumps(altered))
    with pytest.raises(ValueError):m.validate_quality(quality,pair,'dflare')
    altered['temperature']=0;ar_report.write_text(json.dumps(altered))
    resources=quality/'ar'/'resources.jsonl';saved=resources.read_text();resources.write_text(saved+json.dumps({'phase':'sample','gpu_uuid':'GPU-test','foreign_pids':[123]})+'\n')
    with pytest.raises(ValueError):m.validate_quality(quality,pair,'dflare')
    resources.write_text(saved)
    records=[json.loads(line) for line in saved.splitlines()];records[0]['began_wall_ns']=11
    resources.write_text(''.join(json.dumps(x)+'\n' for x in records))
    with pytest.raises(ValueError):m.validate_quality(quality,pair,'dflare')
    resources.write_text(saved)
    pair['methods']['dflare']['tasks']['docvqa']['full']['sd_score']=101.
    with pytest.raises(ValueError):m.validate_quality(quality,pair,'dflare')


def test_empty_root_does_not_claim_live_results(tmp_path,monkeypatch):
    folder=MODULE.parent;monkeypatch.syspath_prepend(str(folder))
    p=folder/'build_qwen3vl8b_paper_results.py';spec=importlib.util.spec_from_file_location('qwen_report_empty',p)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    text=builder.build(tmp_path)
    assert '已完成速度与质量' not in text
    assert '均已按上传 SHA-256 校验' not in text
    assert 'GPU 队列运行' not in text
    assert '的质量配对在 GPU3 完成' not in text
    assert '/home/lyh/qwen3vl8b_eval_20261027/' not in text


def test_al_supplement_must_match_primary_workload_and_checkpoint():
    m=module()
    plan={'target':'/model','draft':'/draft','method':'dfly','speculative_tokens':7,
          'max_new_tokens':1024,'max_model_len':8192,'gpu_memory_utilization':0.78,
          'min_pixels':200704,'max_pixels':1003520,'seed':43,'stop_token_ids':[151645,151643],
          'sessions':[{'id':'x','questions':['q'],'image':'/img','image_sha256':'abc'}],
          'adapter':{'checkpoint':'/ckpt','method':'dfly','scope':'test','bootstrap':'/adapter.py','environment':{},'files':{'/adapter.py':'pin'}},'files':{'/ckpt':'hash','/model/config.json':'targethash',
                                                  '/draft/model.safetensors':'drafthash'},'repeats':3,'gpu':1}
    other=json.loads(json.dumps(plan));other.update(repeats=1,gpu=3)
    m.validate_al_pair(plan,other)
    other['sessions'][0]['questions']=['different prompt']
    with pytest.raises(ValueError):m.validate_al_pair(plan,other)
    other=json.loads(json.dumps(plan));other['files']['/ckpt']='different'
    with pytest.raises(ValueError):m.validate_al_pair(plan,other)
    other=json.loads(json.dumps(plan));other['adapter']['bootstrap']='/different-adapter.py'
    with pytest.raises(ValueError):m.validate_al_pair(plan,other)


def test_output_exact_must_match_three_audited_rounds(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    data=json.loads(path.read_text());data['exact_output_pairs']=51;path.write_text(json.dumps(data))
    audit['result_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)


def test_speed_rejects_nonflash_even_when_audit_hashes_are_rebound(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    stage=evidence/'ar_r0.json';data=json.loads(stage.read_text());data['worker_before']['attention'][0]['impl']='FlexAttentionImpl';stage.write_text(json.dumps(data))
    digest=m.sha(stage);audit['source_sha256']['ar_r0.json']=digest;backend['stages']['ar_r0']['source_sha256']=digest
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)


def test_speed_requires_worker_metadata_after_run(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    stage=evidence/'ar_r0.json';data=json.loads(stage.read_text());data.pop('worker_after');stage.write_text(json.dumps(data))
    digest=m.sha(stage);audit['source_sha256']['ar_r0.json']=digest;backend['stages']['ar_r0']['source_sha256']=digest
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)


def test_speed_rejects_changed_causal_or_numeric_configuration(tmp_path):
    m=module();path,audit,backend,evidence=make_evidence(tmp_path)
    stage=evidence/'sd_r0.json';data=json.loads(stage.read_text())
    data['worker_before']['draft_noncausal']=False;data['worker_after']['draft_noncausal']=False
    stage.write_text(json.dumps(data));digest=m.sha(stage);audit['source_sha256']['sd_r0.json']=digest;backend['stages']['sd_r0']['source_sha256']=digest
    with pytest.raises(ValueError):m.validate_speed(path,audit,'dflare',backend)
