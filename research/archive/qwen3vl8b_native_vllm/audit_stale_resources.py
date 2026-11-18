"""Independently check recorded GPU PID history; no GPU or sampler hooks."""
import argparse,hashlib,json
from pathlib import Path


def audit(rows, expected_uuid):
    if not rows or rows[0].get('phase')!='start' or rows[-1].get('phase')!='end':
        raise ValueError('Missing resource boundaries')
    previous_own={};stale_count=0
    for row in rows:
        if row.get('gpu_uuid')!=expected_uuid or row.get('error') or row.get('foreign_pids'):
            raise ValueError('Wrong GPU, sampling error, or foreign GPU process')
        reported=set(row.get('reported_pids',[]));own=set(row.get('observed_own_gpu_pids',[]));stale=set(row.get('stale_observed_own_pids',[]));foreign=set(row.get('foreign_pids',[]))
        if own & stale or own & foreign or stale & foreign or own|stale|foreign!=reported:
            raise ValueError('GPU PID partition is incomplete or overlapping')
        for pid in stale:
            prior=previous_own.get(pid)
            if prior is None or row['began_wall_ns']-prior['ended_wall_ns']>10_000_000_000:
                raise ValueError('Stale PID lacks a recent recorded own-process observation')
            if row['ended_wall_ns']-prior['began_wall_ns']>12_000_000_000:
                raise ValueError('Sampling timestamps cannot independently bound the stale interval')
            stale_count+=1
        for pid in own:previous_own[pid]=row
    return {'status':'complete','gpu_uuid':expected_uuid,'resource_samples':len(rows),'stale_pid_observations':stale_count,
            'scope':'Recorded PID partition and recent-own history; relies on pinned sampler for contemporaneous /proc status, not continuous exclusivity'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('evidence',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    plan=json.loads((a.evidence/'plan.json').read_text());rows=[json.loads(x) for x in (a.evidence/'resources.jsonl').read_text().splitlines() if x]
    paths=[(Path(k),v) for k,v in plan['files'].items() if k.endswith('/vlm_dflash/eval/native_resources.py')]
    if len(paths)!=1:raise ValueError('Resource sampler path is not uniquely pinned')
    path,expected=paths[0];actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise ValueError('Resource sampler source changed')
    out=audit(rows,plan['gpu_uuid']);out['sampler_sha256']=actual
    with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
