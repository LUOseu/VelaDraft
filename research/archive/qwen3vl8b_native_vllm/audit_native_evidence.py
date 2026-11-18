"""Independent CPU audit of completed native-eval artifacts; never infer causality."""
import argparse,collections,hashlib,json,math,random
from pathlib import Path

STAGES=('ar_r0','sd_r0','sd_r1','ar_r1','ar_r2','sd_r2')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def audit(evidence,result):
    plan=json.loads((evidence/'plan.json').read_text());main=json.loads(result.read_text())
    if main.get('status')!='complete' or main.get('paper_ready') is not False:raise ValueError('Complete raw native result required')
    sessions=plan['sessions'];keys=[(s['id'],t) for s in sessions for t in range(len(s['questions']))]
    if len(sessions)!=120 or len(keys)!=145 or plan['repeats']!=3:raise ValueError('Expected all 120 sessions/145 turns/3 repeats')
    ref=json.loads((evidence/'reference.json').read_text());refrows=ref['rows']
    if ref['status']!='complete' or [(x['id'],x['turn']) for x in refrows]!=keys:raise ValueError('Incomplete reference')
    refs={(x['id'],x['turn']):x for x in refrows};rows={};uuids=[x['image_uuid'] for x in refrows];stats={};ar_groups=collections.defaultdict(list);sd_groups=collections.defaultdict(list)
    for stage in STAGES:
        raw=json.loads((evidence/(stage+'.json')).read_text());rr=raw['rows']
        if raw['status']!='complete' or [(x['phase'],x['id'],x['turn']) for x in rr]!=[(p,*k) for p in ('warmup','timed') for k in keys]:raise ValueError('Incomplete stage '+stage)
        for row in rr:
            if row['status']!='complete' or row['prompt_token_ids']!=refs[row['id'],row['turn']]['prompt_token_ids'] or row['image_sha256']!=refs[row['id'],row['turn']]['image_sha256']:raise ValueError('Input mismatch '+stage)
            sec=(row['ended_ns']-row['began_ns'])/1e9
            if sec<=0 or not math.isclose(sec,row['wall_seconds'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Timer mismatch '+stage)
            uuids.append(row['image_uuid'])
        rows[stage]=[x for x in rr if x['phase']=='timed']
        if stage.startswith('ar'):
            stats[stage]={'same_as_reference':sum(x['tokens']==refs[x['id'],x['turn']]['tokens'] for x in rows[stage]),'compared':len(keys)}
        for x in rows[stage]:(ar_groups if stage.startswith('ar') else sd_groups)[x['id']].append(x['wall_seconds'])
    if len(set(uuids))!=len(uuids):raise ValueError('Reused physical image UUID')
    for i in range(3):
        left,right=rows[f'ar_r{i}'],rows[f'sd_r{i}']
        stats[f'paired_r{i}']={'exact':sum(a['tokens']==b['tokens'] for a,b in zip(left,right)),
          'sd_longer':sum(len(b['tokens'])>len(a['tokens']) for a,b in zip(left,right)),
          'sd_shorter':sum(len(b['tokens'])<len(a['tokens']) for a,b in zip(left,right)),
          'finish_diff':sum(a['finish_reason']!=b['finish_reason'] for a,b in zip(left,right))}
    stats['ar_repeat_pairs']={f'{i}-{j}':sum(a['tokens']==b['tokens'] for a,b in zip(rows[f'ar_r{i}'],rows[f'ar_r{j}'])) for i,j in ((0,1),(0,2),(1,2))}
    ar=sum(sum(v) for v in ar_groups.values());sd=sum(sum(v) for v in sd_groups.values());ratio=ar/sd
    if not math.isclose(ratio,main['speedup_sum_ar_over_sd'],rel_tol=1e-11):raise ValueError('Reported ratio disagrees with rows')
    ordered=sorted(ar_groups);rng=random.Random(42);draws=[]
    for _ in range(2000):
        indices=[rng.randrange(len(ordered)) for _ in ordered]
        draws.append(sum(sum(ar_groups[ordered[i]]) for i in indices)/sum(sum(sd_groups[ordered[i]]) for i in indices))
    draws.sort()
    def q(f):
        x=(len(draws)-1)*f;i=int(x);return draws[i]*(1-(x-i))+draws[min(i+1,len(draws)-1)]*(x-i)
    ci=[q(.025),q(.975)]
    if any(not math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11) for a,b in zip(ci,main['session_cluster_CI95'])):raise ValueError('Bootstrap mismatch')
    resources=[json.loads(x) for x in (evidence/'resources.jsonl').read_text().splitlines() if x]
    if not resources or resources[0].get('phase')!='start' or resources[-1].get('phase')!='end' or any(x.get('error') or x.get('foreign_pids') for x in resources):raise ValueError('Resource samples incomplete/contaminated')
    paths=['plan.json','reference.json','resources.jsonl']+[s+'.json' for s in STAGES]+[s+'.log' for s in ('reference',*STAGES)]
    return {'status':'complete','paper_ready':False,'scope':'120 sessions/145 turns/3 paired repeats; correctness and gold quality remain separate',
      'method':plan['method'],'gpu':plan['gpu_uuid'],'physical_calls_including_reference':len(uuids),'unique_uuids':len(set(uuids)),
      'ar_seconds':ar,'sd_seconds':sd,'speedup':ratio,'session_CI95':ci,'comparisons':stats,'resource_samples':len(resources),
      'limits':['sampled GPU ownership, not whole-host exclusivity','AR repeat equality is measured, not presumed','does not establish strict losslessness or quality equivalence'],
      'source_sha256':{name:digest(evidence/name) for name in paths},'result_sha256':digest(result)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('evidence',type=Path);ap.add_argument('result',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args()
    report=audit(args.evidence,args.result)
    with args.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'status':report['status'],'method':report['method'],'speedup':report['speedup']}))
if __name__=='__main__':main()
