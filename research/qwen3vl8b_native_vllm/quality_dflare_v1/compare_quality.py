"""Recompute paired scores against fixed real gold, never use AR outputs as gold."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from .data import load_task
from .scoring import score
from .legacy_scoring import extract_short_answer

TASKS=('textvqa','docvqa','chartqa')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def paired_delta(ar,sd,groups,samples=2000,seed=42):
    if not groups or len(ar)!=len(sd) or len(sd)!=len(groups):raise ValueError('Bad paired scope')
    keys=sorted(set(groups));sums=np.array([sum(b-a for a,b,g in zip(ar,sd,groups) if g==key) for key in keys]);counts=np.array([groups.count(key) for key in keys]);selected=np.random.default_rng(seed).integers(0,len(keys),size=(samples,len(keys)))
    values=100*sums[selected].sum(axis=1)/counts[selected].sum(axis=1)
    return {'delta_percentage_points':100*(sum(sd)-sum(ar))/len(ar),'paired_cluster_percentile_CI95':np.quantile(values,[.025,.975]).tolist(),
            'questions':len(groups),'clusters':len(keys),'bootstrap_samples':samples,'seed':seed,'noninferiority_or_equivalence_claim':False}

def validate_pair(a,b):
    if a['status']!='complete' or b['status']!='complete':raise ValueError('Failed request')
    for key in ('id','group_id','prompt_token_ids','image_sha256','gold_answers'):
        if a[key]!=b[key]:raise ValueError('AR/SD identity mismatch: '+key)

def load_arm(directory,gold,expected_arm):
    report=json.loads((directory/'quality.json').read_text());path=directory/'predictions.jsonl'
    if (report['status']!='complete' or report['arm']!=expected_arm or digest(path)!=report['raw_sha256']
            or report.get('resource_evidence',{}).get('status')!='complete'):
        raise ValueError('Incomplete arm, resource record, or raw SHA mismatch')
    rows=[json.loads(s) for s in path.read_text().splitlines()];uids=[x['image_uuid'] for x in rows]
    if len(rows)!=780 or len(set(uids))!=780:raise ValueError('Physical scope/UUID mismatch')
    by={}
    for task in TASKS:
        selected=[r for r in rows if r['task']==task and r['phase']=='quality']
        if len(selected)!=256 or len({r['id'] for r in selected})!=256 or {r['id'] for r in selected}!=set(gold[task]):raise ValueError('Incomplete task: '+task)
        by[task]={}
        for r in selected:
            g=gold[task][r['id']]
            if r['status']!='complete' or r['gold_answers']!=g['answers'] or r['image_sha256']!=g['image_sha256'] or r['group_id']!=g['group_id'] or r['question']!=g['question']:raise ValueError('Real gold/question/image mismatch')
            raw=score(task,r['prediction'],g['answers']);extracted=extract_short_answer(r['prediction']);secondary=score(task,extracted,g['answers'])
            if abs(raw-r['score_full'])>1e-12 or extracted!=r['extracted'] or abs(secondary-r['score_extracted'])>1e-12:raise ValueError('Score mismatch')
            by[task][r['id']]=r
        if abs(100*sum(x['score_full'] for x in selected)/256-report['task_scores_full'][task])>1e-9:raise ValueError('Aggregate full score mismatch')
    return report,by,uids

def compare(root):
    gold={}
    for task in TASKS:
        _,records=load_task(task);gold[task]={r['id']:r for r in records}
    ar_report,ar,uuids=load_arm(root/'ar',gold,'ar')
    result={'schema':'qwen3vl8b-quality-paired-v1','status':'complete','performance_claim':False,'real_gold_scoring':True,
      'scope':'256 samples each, one execution per system, not the full benchmark','primary_scoring':'full decoded text',
      'secondary_scoring':'first nonempty line; pinned legacy extraction', 'methods':{},'limitations':['Score similarity does not prove theoretical losslessness, output equality, equivalence or noninferiority.',
        'Formal speed workload long-answer quality is not evaluated by these subsets.',
        'Training/evaluation overlap for the 8B drafters has not been audited.',
        'Intervals are per comparison and are not multiplicity-adjusted.']}
    for method in ('dflare',):
        report,sd,uids=load_arm(root/method,gold,'sd');uuids+=uids
        if report['method']!=method or report['gpu_uuid']!=ar_report['gpu_uuid']:
            raise ValueError('Method or physical GPU identity differs')
        for key in ('quality_source_manifest_sha256','quality_protocol_sha256','data_manifest_sha256','temperature','seed','max_new_tokens'):
            if report[key]!=ar_report[key]:raise ValueError('Quality source/protocol differs: '+key)
        if report['numeric_parent']!=ar_report['numeric_parent'] or report['worker_before']['numeric']!=ar_report['worker_before']['numeric']:
            raise ValueError('Recorded numeric configuration differs from AR')
        common=('model','dtype','tensor_parallel_size','max_model_len','max_num_seqs','max_num_batched_tokens',
                'gpu_memory_utilization','enable_prefix_caching','mm_processor_cache_gb','limit_mm_per_prompt',
                'enforce_eager','generation_config','disable_log_stats','mm_processor_kwargs')
        if any(report['engine_args'][key]!=ar_report['engine_args'][key] for key in common):
            raise ValueError('Common AR/SD engine configuration differs')
        if report['preservation'].get('all_values_bitwise_equal') is not True:
            raise ValueError('Trained tensor preservation was not established')
        entry={'tasks':{},'raw_sha256':report['raw_sha256'],'report_sha256':digest(root/method/'quality.json')}
        for task in TASKS:
            ids=sorted(ar[task]);left=[ar[task][x] for x in ids];right=[sd[task][x] for x in ids]
            for a,b in zip(left,right):validate_pair(a,b)
            groups=[x['group_id'] for x in left];item={}
            for field,label in (('score_full','full'),('score_extracted','extracted')):
                item[label]={'ar_score':100*sum(x[field] for x in left)/len(ids),'sd_score':100*sum(x[field] for x in right)/len(ids),
                   **paired_delta([x[field] for x in left],[x[field] for x in right],groups)}
            item.update(exact_token_outputs=sum(a['tokens']==b['tokens'] for a,b in zip(left,right)),
                exact_text_outputs=sum(a['prediction']==b['prediction'] for a,b in zip(left,right)),
                changed_full_scores=sum(a['score_full']!=b['score_full'] for a,b in zip(left,right)),
                ar_output_tokens=sum(len(a['tokens']) for a in left),sd_output_tokens=sum(len(b['tokens']) for b in right),
                ar_finish_reasons=dict(collections.Counter(str(a['finish_reason']) for a in left)),
                sd_finish_reasons=dict(collections.Counter(str(b['finish_reason']) for b in right)),
                finish_reason_differences=sum(a['finish_reason']!=b['finish_reason'] for a,b in zip(left,right)),
                changed_examples=[{'id':a['id'],'ar_prediction':a['prediction'],'sd_prediction':b['prediction'],'gold_answers':a['gold_answers'],
                   'ar_score':a['score_full'],'sd_score':b['score_full']} for a,b in zip(left,right) if a['score_full']!=b['score_full']])
            entry['tasks'][task]=item
        result['methods'][method]=entry
    if len(uuids)!=1560 or len(set(uuids))!=1560:raise ValueError('Cross-arm physical UUID reused')
    result['physical_calls']=len(uuids);result['ar_raw_sha256']=ar_report['raw_sha256'];return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=compare(a.root)
    with a.out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'physical_calls':report['physical_calls']}))
if __name__=='__main__':main()
