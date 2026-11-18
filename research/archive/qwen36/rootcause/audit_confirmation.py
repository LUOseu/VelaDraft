"""Independent recomputation and coverage checks for a completed confirmation."""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

def main():
    root=Path(sys.argv[1]);s=json.loads((root/'SUMMARY.json').read_text())
    assert s['status']=='complete' and s['timed_pairs']==48 and s['sessions']==12
    reports={a:json.loads((root/a/'result.json').read_text()) for a in ('ar','sd')}
    grouped={a:{} for a in reports}; totals={}; token_totals={};hashes={}
    expected=None;uuids=[]
    for arm,d in reports.items():
        assert d['status']=='complete' and d['observer_hooks'] is False and d['max_new_tokens']==1024
        assert d['worker_before']==d['worker_after'] and not d['worker_before']['diagnostic_hooks']
        assert d['preservation']['all_values_bitwise_equal']
        assert not d['engine_args']['enable_prefix_caching'] and d['engine_args']['mm_processor_cache_gb']==0
        rows=d['rows'];assert len(rows)==48
        keys=[(r['id'],r['turn'],r['repeat']) for r in rows];assert len(set(keys))==48
        if expected is None:expected=keys
        else:assert keys==expected
        assert {r['repeat'] for r in rows}=={0,1,2}
        assert all(len([r for r in rows if r['repeat']==j])==16 for j in range(3))
        for row in rows:
            assert row['status']=='complete'
            wall=(row['ended_ns']-row['began_ns'])/1e9
            assert wall>0 and math.isclose(wall,row['wall_seconds'],rel_tol=1e-10)
            grouped[arm].setdefault(row['id'],[]).append(wall);uuids.append(row['image_uuid'])
        assert len(grouped[arm])==12
        totals[arm]=sum(r['wall_seconds'] for r in rows);token_totals[arm]=sum(len(r['tokens']) for r in rows)
        assert math.isclose(totals[arm],s[arm+'_seconds'],rel_tol=1e-12) and token_totals[arm]==s[arm+'_output_tokens']
        hashes[arm]=hashlib.sha256((root/arm/'result.json').read_bytes()).hexdigest()
    assert len(set(uuids))==96
    for a,b in zip(reports['ar']['rows'],reports['sd']['rows']):assert a['prompt_token_ids']==b['prompt_token_ids'] and a['image_sha256']==b['image_sha256']
    assert reports['ar']['numeric']==reports['sd']['numeric']
    assert reports['ar']['tuning_sha256']==reports['sd']['tuning_sha256']
    resources=[json.loads(x) for x in (root/'resources.jsonl').read_text().splitlines() if x]
    assert resources[0]['phase']=='start' and resources[-1]['phase']=='end'
    assert not any(x.get('foreign_pids') or x.get('over_bound') or x.get('error') for x in resources)
    begin,end=resources[0]['began_wall_ns'],resources[-1]['ended_wall_ns']
    assert all(begin<=r['began_wall_ns']<=r['ended_wall_ns']<=end for d in reports.values() for r in d['rows'])
    ids=sorted(grouped['ar']);a=[sum(grouped['ar'][i]) for i in ids];b=[sum(grouped['sd'][i]) for i in ids];rng=random.Random(42);samples=[]
    for _ in range(2000):
        idx=[rng.randrange(12) for _ in range(12)];samples.append(sum(a[i] for i in idx)/sum(b[i] for i in idx))
    samples.sort()
    def q(p):
        x=(len(samples)-1)*p;i=int(x);return samples[i]*(1-(x-i))+samples[min(i+1,len(samples)-1)]*(x-i)
    ci=[q(.025),q(.975)];assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(ci,s['session_cluster_CI95']))
    per_repeat=[]
    for repeat in range(3):
        durations={arm:sum(r['wall_seconds'] for r in d['rows'] if r['repeat']==repeat) for arm,d in reports.items()};per_repeat.append(durations['ar']/durations['sd'])
    result={'status':'verified','speedup':totals['ar']/totals['sd'],'CI95':ci,'per_warm_engine_repeat_speedup':per_repeat,
            'ar_e2e_tokens_per_s':token_totals['ar']/totals['ar'],'sd_e2e_tokens_per_s':token_totals['sd']/totals['sd'],
            'token_count_change_pct':100*(token_totals['sd']/token_totals['ar']-1),'sha256':hashes,'samples':48,'sessions':12,
            'limitations':'Diagnostic subset; count-mode sampled GPU ownership; CI covers sessions, not all deployment or training variance.'}
    (root/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
