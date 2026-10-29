"""Fail closed if the 3-task paired smoke is incomplete or mismatched."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);data={}
for arm in ('ar','dflash','dfly'):
    folder=root/('smoke_'+arm);report=json.loads((folder/'quality.json').read_text())
    rows=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
    if report['status']!='complete' or report['smoke'] is not True or len(rows)!=15 or any(x['status']!='complete' for x in rows):raise ValueError('Failed/incomplete smoke '+arm)
    data[arm]={x['task']:x for x in rows if x['phase']=='quality'}
    if set(data[arm])!={'textvqa','docvqa','chartqa'}:raise ValueError('Missing smoke task')
for method in ('dflash','dfly'):
    for task in data['ar']:
        a,b=data['ar'][task],data[method][task]
        for key in ('id','prompt_token_ids','image_sha256','gold_answers','group_id'):
            if a[key]!=b[key]:raise ValueError('Smoke paired identity mismatch: '+method+'/'+task+'/'+key)
print('Smoke identities passed; single-question scores are not quality claims')
