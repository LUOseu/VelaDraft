"""Recover only the fully sampled K1 stage after its specific final-JSON serialization error."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from summary import summarize_stage

def validate_partial(raw, contexts, resources):
    if raw.get('status')!='running' or len(raw.get('rows',[]))!=9:raise ValueError('Incomplete raw request journal')
    expected={(sid,turn,r) for sid,turn in contexts for r in range(3)}
    actual=[(r['id'],r['turn'],r['repetition']) for r in raw['rows']]
    if set(actual)!=expected or len(actual)!=len(set(actual)):raise ValueError('Missing or duplicate context/repetition')
    if not resources or resources[0].get('phase')!='start' or resources[-1].get('phase')!='end':raise ValueError('Resource epoch not bracketed')
    if any(x.get('foreign_pids') or x.get('error') for x in resources):raise ValueError('Resource epoch contaminated')
    rows=summarize_stage({'arm':'sd','k':1,'rows':raw['rows']})
    measured={(x['session_id'],x['turn']) for x in rows if x['component']=='target_forward' and x['calls']>0}
    if measured!=set(contexts):raise ValueError('Not all contexts have paired full-width verify steps')
    return len(measured)

def main():
    root=Path(sys.argv[1]).resolve();p=root/'stages/sd_k1/profile.json';log=root/'logs/sd_k1.log'
    if 'TypeError: Object of type ModelConfig is not JSON serializable' not in log.read_text():raise ValueError('Unexpected failure; do not recover')
    raw_bytes=p.read_bytes();raw=json.loads(raw_bytes)
    plan=json.loads((root/'plans/k1.evidence/plan.json').read_text())
    first=plan['sessions'][0];multi=next(s for s in plan['sessions'] if len(s['questions'])>=2)
    contexts=[(first['id'],0),(multi['id'],0),(multi['id'],len(multi['questions'])-1)]
    resources=[json.loads(line) for line in (root/'resources.jsonl').read_text().splitlines() if line]
    count=validate_partial(raw,contexts,resources)
    archive=p.with_name('profile.raw_running.json')
    if archive.exists():raise FileExistsError('Recovery archive exists')
    os.link(p,archive)
    result={'status':'recovered_complete','arm':'sd','k':1,'diagnostic_only':True,'headline_speedup_claim':False,
            'recovered_after':'Final JSON serialization failed after all 9 requests and rank events completed; raw original is preserved unchanged.',
            'resource_scope':'Sampled owner checks, no foreign PID/error; does not claim continuous exclusivity.',
            'rows':raw['rows']}
    encoded=json.dumps(result,indent=2,allow_nan=False);tmp=p.with_suffix('.tmp');tmp.write_text(encoded);os.replace(tmp,p)
    receipt={'status':'recovered_complete','raw_archive':str(archive),'raw_sha256':hashlib.sha256(raw_bytes).hexdigest(),
             'derived_sha256':hashlib.sha256(encoded.encode()).hexdigest(),'contexts':contexts,'context_count':count,
             'request_count':9,'rank_count':2,'resource_samples':len(resources),'resource_log':str(root/'resources.jsonl'),
             'original_error':'ModelConfig not JSON serializable', 'source_stage_sha256':hashlib.sha256((Path(__file__).resolve().parent.parent/'qwen36_k_sweep/profile_stage.py').read_bytes()).hexdigest()}
    (root/'stages/sd_k1/recovery_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
