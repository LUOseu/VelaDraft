import json
from pathlib import Path
import sys
from vlm_dflash.eval.native import bootstrap_ci

root=Path(sys.argv[1]);reports={}
for arm in ('ar','sd'):
    r=json.loads((root/arm/'stage/result.json').read_text())
    assert r['status']=='complete' and len(r['rows'])==48
    resource=json.loads((root/arm/'resources_verified.json').read_text());assert resource['status']=='complete'
    reports[arm]=r
a,b=reports['ar'],reports['sd'];groups={'ar':{},'sd':{}};exact=0
for x,y in zip(a['rows'],b['rows']):
    assert (x['id'],x['turn'],x['repeat'])==(y['id'],y['turn'],y['repeat']) and x['prompt_token_ids']==y['prompt_token_ids']
    for arm,row in [('ar',x),('sd',y)]:groups[arm].setdefault(row['id'],[]).append(row['wall_seconds'])
    exact+=x['tokens']==y['tokens']
assert len(groups['ar'])==12
for key in ('numeric','target_dtype','target_quantization','runner','use_v2'):
    assert a['worker_before'][0][key]==b['worker_before'][0][key]
resources={arm:json.loads((root/arm/'resources_verified.json').read_text()) for arm in ('ar','sd')}
assert resources['ar']['gpu_uuids']==resources['sd']['gpu_uuids']
result=dict(status='complete',scope='fixed diagnostic subset, not paper main table',k=2,vllm='0.32.0',
            ar_seconds=a['wall_seconds'],sd_seconds=b['wall_seconds'],speedup=a['wall_seconds']/b['wall_seconds'],
            session_cluster_CI95=bootstrap_ci(groups['ar'],groups['sd']),sessions=12,turns=16,repeats=3,pairs=48,
            ar_tokens=a['output_tokens'],sd_tokens=b['output_tokens'],exact_pairs=exact,
            ar_tokens_per_second=a['output_tokens']/a['wall_seconds'],sd_tokens_per_second=b['output_tokens']/b['wall_seconds'],
            per_repeat_speedup=[sum(x['wall_seconds'] for x in a['rows'] if x['repeat']==i)/sum(x['wall_seconds'] for x in b['rows'] if x['repeat']==i) for i in range(3)],
            resource_evidence=resources,limitations=['Diagnostic subset used during development; quality equivalence not scored.','One independent process per arm on this GPU, three warm-engine repeats; session-cluster CI is not run/card variance.'])
(root/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
