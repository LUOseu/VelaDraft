"""Bind reported speed figures to their audit and recorded source evidence."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

STAGES = ('reference', 'ar_r0', 'sd_r0', 'sd_r1', 'ar_r1', 'ar_r2', 'sd_r2')
REQUIRED = {'plan.json', 'resources.jsonl'} | {s + suffix for s in STAGES for suffix in ('.json', '.log')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _same_number(left, right):
    return (type(left) in (int,float) and type(right) in (int,float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12))


def validate_speed(result_path, audit, method, backend=None):
    result_path=Path(result_path);result=json.loads(result_path.read_text())
    if (result.get('status')!='complete' or result.get('method')!=method or audit.get('status')!='complete'
            or audit.get('method')!=method or audit.get('result_sha256')!=sha(result_path)
            or result.get('resources_verified') is not True
            or result.get('resource_evidence',{}).get('status')!='complete'
            or result.get('sessions')!=120 or result.get('repeats')!=3 or result.get('timed_pairs')!=435
            or result.get('smoke') is not False):
        raise ValueError('Speed result, audit identity, resource status or sample scope differs')
    ar=result.get('ar_seconds');sd=result.get('sd_seconds');speed=result.get('speedup_sum_ar_over_sd')
    if (not _same_number(ar,audit.get('ar_seconds')) or not _same_number(sd,audit.get('sd_seconds'))
            or not _same_number(speed,audit.get('speedup')) or not sd or not _same_number(speed,ar/sd)
            or ar<=0 or sd<=0):
        raise ValueError('Speed values do not match paired-time audit')
    ci=result.get('session_cluster_CI95');other=audit.get('session_CI95')
    if not isinstance(ci,list) or not isinstance(other,list) or len(ci)!=2 or len(other)!=2 or any(not _same_number(x,y) for x,y in zip(ci,other)):
        raise ValueError('Confidence interval differs from audit')
    exact=result.get('exact_output_pairs')
    comparisons=audit.get('comparisons',{})
    per_repeat=[comparisons.get(f'paired_r{i}',{}).get('exact') for i in range(3)]
    if (type(exact) is not int or not 0<=exact<=result['timed_pairs']
            or any(type(value) is not int or not 0<=value<=145 for value in per_repeat)
            or sum(per_repeat)!=exact):
        raise ValueError('Output-exact count differs from audited repeats')
    recorded=audit.get('source_sha256')
    if not isinstance(recorded,dict) or set(recorded)!=REQUIRED:
        raise ValueError('Audit source file list differs from required evidence')
    evidence=Path(result['evidence'])
    if result.get('source_plan_sha256')!=recorded['plan.json']:
        raise ValueError('Plan SHA differs from result or audit')
    for name,expected in recorded.items():
        if sha(evidence/name)!=expected:
            raise ValueError('Current evidence differs from audit: '+name)
    reference_worker_numeric=None;reference_parent_numeric=None
    for stage in STAGES:
        payload=json.loads((evidence/(stage+'.json')).read_text())
        worker=payload.get('worker_before',{});modules=worker.get('attention',[]);is_sd=stage.startswith('sd_')
        expected_count=39 if is_sd else 36
        if (payload.get('status')!='complete' or len(modules)!=expected_count
                or any(not isinstance(item.get('impl'),str) or not item['impl'].endswith('.flash_attn.FlashAttentionImpl') for item in modules)
                or payload.get('worker_after')!=worker):
            raise ValueError('Observed stage is incomplete or not entirely FlashAttention: '+stage)
        required=('numeric','target_noncausal','draft_noncausal','speculative_tokens','diagnostic_hooks')
        if (any(key not in worker for key in required) or worker['target_noncausal'] is not False
                or worker['diagnostic_hooks'] is not False
                or worker['speculative_tokens']!=(7 if is_sd else None)
                or worker['draft_noncausal'] is not ((method!='jetspec') if is_sd else None)):
            raise ValueError('Observed stage verifier/causality configuration differs: '+stage)
        if stage=='reference':
            reference_worker_numeric=worker['numeric'];reference_parent_numeric=payload.get('numeric_parent')
        if (not isinstance(reference_worker_numeric,dict) or not reference_worker_numeric
                or not isinstance(reference_parent_numeric,dict) or not reference_parent_numeric
                or worker['numeric']!=reference_worker_numeric or payload.get('numeric_parent')!=reference_parent_numeric):
            raise ValueError('Observed stage numeric configuration differs: '+stage)
    if backend is not None:
        entries=backend.get('stages',{})
        if backend.get('status')!='complete' or set(entries)!=set(STAGES):
            raise ValueError('Backend audit is incomplete')
        for stage in STAGES:
            item=entries[stage];count=39 if stage.startswith('sd_') else 36
            if item.get('all_flash') is not True or item.get('count')!=count or item.get('source_sha256')!=recorded[stage+'.json']:
                raise ValueError('Backend audit is not bound to the same evidence: '+stage)
    return result


def validate_quality(root, pair, method):
    """Validate summary against same-run raw scored rows and recorded SHA hashes.

    This checks recorded row scores and pair identity, not independent gold
    reconstruction; the campaign compare script performs that separate audit.
    """
    root=Path(root);tasks=('textvqa','docvqa','chartqa')
    entry=pair.get('methods',{}).get(method,{})
    if pair.get('status')!='complete' or pair.get('real_gold_scoring') is not True or set(entry.get('tasks',{}))!=set(tasks):
        raise ValueError('Quality paired report is incomplete')
    loaded={};all_uuids=[]
    for label,arm in (('ar','ar'),(method,'sd')):
        directory=root/label;report_path=directory/'quality.json';raw_path=directory/'predictions.jsonl'
        report=json.loads(report_path.read_text());raw_sha=sha(raw_path)
        expected=pair.get('ar_raw_sha256') if arm=='ar' else entry.get('raw_sha256')
        if (report.get('status')!='complete' or report.get('arm')!=arm or report.get('failures')!=0
                or report.get('resource_evidence',{}).get('status')!='complete'
                or report.get('raw_sha256')!=raw_sha or raw_sha!=expected
                or (arm=='sd' and (report.get('method')!=method or sha(report_path)!=entry.get('report_sha256')))):
            raise ValueError('Quality arm, resource, method or raw/report SHA differs')
        resource_path=directory/'resources.jsonl'
        resource=[json.loads(line) for line in resource_path.read_text().splitlines() if line]
        resource_evidence=report['resource_evidence']
        if (len(resource)<2 or resource[0].get('phase')!='start' or resource[-1].get('phase')!='end'
                or resource_evidence.get('samples')!=len(resource)
                or resource_evidence.get('began_wall_ns')!=resource[0].get('began_wall_ns')
                or resource_evidence.get('ended_wall_ns')!=resource[-1].get('ended_wall_ns')
                or any(row.get('error') or row.get('foreign_pids') or row.get('gpu_uuid')!=report.get('gpu_uuid') for row in resource)):
            raise ValueError('Quality resource evidence is incomplete or contaminated')
        if (report.get('temperature')!=0 or report.get('seed')!=43 or report.get('max_new_tokens')!=128
                or report.get('task_counts')!={task:256 for task in tasks}
                or report.get('worker_before')!=report.get('worker_after')
                or (arm=='sd' and report.get('preservation',{}).get('all_values_bitwise_equal') is not True)):
            raise ValueError('Quality sample/configuration or checkpoint preservation differs')
        rows=[json.loads(line) for line in raw_path.read_text().splitlines() if line]
        uuids=[row.get('image_uuid') for row in rows]
        if len(set(uuids))!=len(uuids) or any(not isinstance(uid,str) or not uid for uid in uuids):
            raise ValueError('Physical quality request UUIDs are missing or duplicated')
        all_uuids.extend(uuids);by_task={}
        for task in tasks:
            selected=[row for row in rows if row.get('task')==task and row.get('phase')=='quality']
            ids=[row.get('id') for row in selected]
            if len(selected)!=256 or len(set(ids))!=256 or any(row.get('status')!='complete' for row in selected):
                raise ValueError('Incomplete or duplicate quality rows: '+task)
            by_task[task]={row['id']:row for row in selected}
        loaded[arm]=(report,by_task)
    ar_report=loaded['ar'][0];sd_report=loaded['sd'][0]
    if len(set(all_uuids))!=len(all_uuids) or ar_report.get('gpu_uuid')!=sd_report.get('gpu_uuid'):
        raise ValueError('Quality arms share UUIDs or use different GPUs')
    for key in ('quality_source_manifest_sha256','quality_protocol_sha256','data_manifest_sha256'):
        value=ar_report.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) or value!=sd_report.get(key):
            raise ValueError('Quality source/protocol digest differs: '+key)
    if (ar_report.get('numeric_parent') is None or ar_report.get('numeric_parent')!=sd_report.get('numeric_parent')
            or ar_report.get('worker_before',{}).get('numeric') is None
            or ar_report['worker_before']['numeric']!=sd_report.get('worker_before',{}).get('numeric')):
        raise ValueError('Quality numeric configuration differs')
    common=('model','dtype','tensor_parallel_size','max_model_len','max_num_seqs','max_num_batched_tokens',
            'gpu_memory_utilization','enable_prefix_caching','mm_processor_cache_gb','limit_mm_per_prompt',
            'enforce_eager','generation_config','disable_log_stats','mm_processor_kwargs')
    left_args=ar_report.get('engine_args',{});right_args=sd_report.get('engine_args',{})
    if any(key not in left_args or key not in right_args or left_args[key]!=right_args[key] for key in common):
        raise ValueError('Quality common engine configuration differs')
    for task in tasks:
        left=loaded['ar'][1][task];right=loaded['sd'][1][task]
        if set(left)!=set(right):raise ValueError('Quality sample IDs differ: '+task)
        ids=sorted(left);a=[left[key] for key in ids];b=[right[key] for key in ids]
        for x,y in zip(a,b):
            if any(x.get(field)!=y.get(field) for field in ('id','group_id','image_sha256','gold_answers','prompt_token_ids')):
                raise ValueError('Quality pair input differs: '+task)
        scores=[]
        for rows in (a,b):
            values=[row.get('score_full') for row in rows]
            if not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values):
                raise ValueError('Invalid recorded quality score: '+task)
            scores.append(100*sum(values)/len(values))
        item=entry['tasks'][task];full=item.get('full',{})
        if (not _same_number(full.get('ar_score'),scores[0]) or not _same_number(full.get('sd_score'),scores[1])
                or not _same_number(full.get('delta_percentage_points'),scores[1]-scores[0])
                or item.get('exact_token_outputs')!=sum(x['tokens']==y['tokens'] for x,y in zip(a,b))):
            raise ValueError('Quality aggregate differs from raw paired rows: '+task)
    return entry


def validate_al_pair(primary, supplemental):
    """Require the supplemental counters to use the primary input/model scope."""
    keys=('target','draft','method','speculative_tokens','max_new_tokens','max_model_len',
          'gpu_memory_utilization','min_pixels','max_pixels','seed','stop_token_ids','sessions')
    if any(key not in primary or key not in supplemental or primary[key]!=supplemental[key] for key in keys):
        raise ValueError('AL supplement differs from the primary workload/configuration')
    if primary.get('speculative_tokens')!=7:
        raise ValueError('Expected S7 for this AL report')
    a=primary.get('adapter') or {};b=supplemental.get('adapter') or {}
    checkpoint=a.get('checkpoint')
    required_adapter={'method','scope','bootstrap','checkpoint','environment','files'}
    if not isinstance(checkpoint,str) or not required_adapter.issubset(a) or a!=b:
        raise ValueError('AL supplement uses a different adapter or checkpoint')
    first=primary.get('files',{});second=supplemental.get('files',{})
    relevant={key:value for key,value in first.items()
              if key==checkpoint or key.startswith(primary['target']+'/') or key.startswith(primary['draft']+'/')}
    if checkpoint not in relevant or not any(k.startswith(primary['target']+'/') for k in relevant) or not any(k.startswith(primary['draft']+'/') for k in relevant):
        raise ValueError('Primary plan lacks target/draft/checkpoint file pins')
    if any(second.get(key)!=value for key,value in relevant.items()):
        raise ValueError('AL supplement target/draft/checkpoint hashes differ')
