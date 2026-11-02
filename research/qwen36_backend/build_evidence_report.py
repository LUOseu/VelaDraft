"""Recompute reported timing and session CIs from copied raw requests, using stdlib only."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random


def interval(a,b):
    ids=sorted(a);rng=random.Random(42);values=[]
    for _ in range(2000):
        counts=Counter(rng.randrange(len(ids)) for _ in range(len(ids)))
        values.append(sum(count*a[ids[i]] for i,count in counts.items()) /
                      sum(count*b[ids[i]] for i,count in counts.items()))
    values.sort()
    def q(p):
        where=p*(len(values)-1);lo=int(where);t=where-lo
        return (1-t)*values[lo]+t*values[min(lo+1,len(values)-1)]
    return [q(.025),q(.975)]


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    root=args.directory;ev=root/'evidence';canonical={};rows=[];audit=[];manifest={}
    cases=[
        ('baseline_confirm_tuned_k2_gpu0','natural','0.22.1 tuned','triton',0,False),
        ('baseline_confirm_tuned_k2_gpu1_repeat','natural','0.22.1 tuned','triton',1,False),
        ('confirm_v032_gpu0_a1','natural','0.32.0 MRV2','cuda',0,True),
        ('confirm_v032_gpu1_a1','natural','0.32.0 MRV2','cuda',1,True),
        ('confirm_v032_gdn_triton_gpu0_a1','natural','0.32.0 MRV2','triton',0,True),
        ('confirm_v032_gdn_triton_gpu1_a1','natural','0.32.0 MRV2','triton',1,True),
        ('fixed512_v022_gpu0_a1','fixed512','0.22.1 tuned','triton',0,False),
        ('fixed512_v032_cuda_gpu1_a1','fixed512','0.32.0 MRV2','cuda',1,False),
        ('fixed512_v032_triton_gpu1_a1','fixed512','0.32.0 MRV2','triton',1,False),
        ('fixed512_v032_cuda_gpu0_a2','fixed512','0.32.0 MRV2','cuda',0,False),
    ]
    pending=[]
    for name,regime,runtime,gdn,gpu,nested in cases:
        case=ev/name;summary=case/'SUMMARY.json'
        if not summary.exists():pending.append(name);continue
        s=json.loads(summary.read_text());assert s['status']=='complete'
        data={arm:json.loads((case/arm/('stage/result.json' if nested else 'result.json')).read_text()) for arm in ('ar','sd')}
        totals={};groups={};tokens={};resources={};gpu_ids=[]
        for arm,d in data.items():
            assert d['status']=='complete' and len(d['rows'])==48 and not d['observer_hooks']
            assert d['preservation']['all_values_bitwise_equal']
            assert {x['repeat'] for x in d['rows']}=={0,1,2}
            assert len({(x['id'],x['turn'],x['repeat']) for x in d['rows']})==48
            assert len({(x['id'],x['turn']) for x in d['rows']})==16
            groups[arm]=defaultdict(float)
            for x in d['rows']:
                assert x['status']=='complete' and x['wall_seconds']>0
                assert math.isclose(x['wall_seconds'],(x['ended_ns']-x['began_ns'])/1e9,rel_tol=1e-12,abs_tol=1e-9)
                key=(x['id'],x['turn']);inputs=(x['prompt_token_ids'],x['image_sha256'])
                if key in canonical:assert canonical[key]==inputs,(name,arm,key)
                else:canonical[key]=inputs
                groups[arm][x['id']]+=x['wall_seconds']
                if regime=='fixed512':assert len(x['tokens'])==512 and x['finish_reason']=='length'
            assert len(groups[arm])==12
            totals[arm]=sum(x['wall_seconds'] for x in d['rows']);tokens[arm]=sum(len(x['tokens']) for x in d['rows'])
            assert math.isclose(totals[arm],d['wall_seconds'],abs_tol=1e-8)
            assert tokens[arm]==d['output_tokens']
            worker=d['worker_before'][0] if isinstance(d['worker_before'],list) else d['worker_before']
            assert not worker['target_noncausal']
            assert worker['speculative_tokens']==(2 if arm=='sd' else None)
            assert all(x['impl'].endswith('.FlashAttentionImpl') for x in worker['attention'])
            resource_path=case/arm/'resources_verified.json' if nested else case/'resources_verified.json'
            resource=json.loads(resource_path.read_text());assert resource['status']=='complete'
            assert resource['began_wall_ns']<=min(x['began_wall_ns'] for x in d['rows'])
            assert resource['ended_wall_ns']>=max(x['ended_wall_ns'] for x in d['rows'])
            assert resource.get('extra_app_samples_outside_timed',0)==0
            resources[arm]=resource;gpu_ids.append(resource['gpu_uuids'])
        assert gpu_ids[0]==gpu_ids[1] and len(gpu_ids[0])==1
        assert [(x['id'],x['turn'],x['repeat']) for x in data['ar']['rows']]==[(x['id'],x['turn'],x['repeat']) for x in data['sd']['rows']]
        common=('model','dtype','tensor_parallel_size','max_model_len','max_num_seqs','max_num_batched_tokens','gpu_memory_utilization','enable_prefix_caching','mm_processor_cache_gb','limit_mm_per_prompt','generation_config','mm_processor_kwargs','disable_log_stats')
        for key in common:assert data['ar']['engine_args'][key]==data['sd']['engine_args'][key],(name,key)
        assert data['ar']['engine_args']['dtype']=='bfloat16'
        assert data['ar']['numeric']==data['sd']['numeric']
        speed=totals['ar']/totals['sd'];ci=interval(groups['ar'],groups['sd'])
        assert math.isclose(speed,s['speedup'],abs_tol=1e-12)
        assert all(math.isclose(x,y,abs_tol=1e-10) for x,y in zip(ci,s['session_cluster_CI95']))
        exact=sum(x['tokens']==y['tokens'] for x,y in zip(data['ar']['rows'],data['sd']['rows']))
        rows.append(dict(regime=regime,runtime=runtime,gdn=gdn,k=2,gpu=gpu,sessions=12,turns=16,repeats=3,
                         ar_seconds=totals['ar'],sd_seconds=totals['sd'],speedup=speed,ci95_low=ci[0],ci95_high=ci[1],
                         ar_output_tokens=tokens['ar'],sd_output_tokens=tokens['sd'],token_exact_pairs=exact,pairs=48,
                         ar_tokens_per_second=tokens['ar']/totals['ar'],sd_tokens_per_second=tokens['sd']/totals['sd'],
                         quality_scored=False,status='valid_diagnostic_timing',evidence=name))
        audit.append(dict(case=name,status='verified',raw_rows=96,session_clusters=12,coverage=True,
                          prompt_ids_and_image_bytes_equal_across_all_cases=True,timers_recomputed=True,ci_recomputed=True,
                          draft_export_bitwise_preserved=True,same_gpu_within_pair=True,resource_window_covers_requests=True))
        for file in case.rglob('*.json'):manifest[str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
    report=dict(status='verified_completed_cases',target='Qwen3.6-35B-A3B',method='DFlash',hardware='A800-SXM4-80GB, TP1, BF16',
                rows=rows,pending=pending,bootstrap={'unit':'session','samples':2000,'seed':42,'scope':'Fixed 12 diagnostic sessions, with warm repeats clustered inside sessions; excludes card/run variance.'},
                limitations=['Diagnostic data participated in development; not an untouched paper benchmark.','Natural-completion and fixed512 results must be reported separately.','Fixed512 ignores EOS; tokens after EOS are included, if present.','Quality scores are not collected here. No strict token-exact lossless claim.','GPU evidence is sampled process-count monitoring in a container, not proof of continuous exclusivity.'])
    (root/'RESULTS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    with (root/'RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (root/'EVIDENCE_AUDIT.json').write_text(json.dumps({'status':'verified','cases':audit,'pending':pending},indent=2)+'\n')
    (root/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'verified_cases':len(rows),'paired_requests':len(rows)*48,'pending':pending}))


if __name__=='__main__':main()
