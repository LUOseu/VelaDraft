import json,hashlib,torch
from pathlib import Path
root=Path('/home/lyh/veladraft-pr30-audit-20261021');r=root/'results';rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def code_sha(path):
 h=hashlib.sha256()
 for p in sorted(path.rglob('*.py')):h.update(str(p.relative_to(path)).encode());h.update(p.read_bytes())
 return h.hexdigest()
for method in ['dflash','dspark','dfly','domino']:
 a=json.loads((r/f'{method}_base_3.json').read_text());b=json.loads((r/f'{method}_head_3.json').read_text())
 pa=torch.load(r/f'{method}_base_3.probes.pt',map_location='cpu',weights_only=True);pb=torch.load(r/f'{method}_head_3.probes.pt',map_location='cpu',weights_only=True)
 checks={'initial_weights_exact':a['initial_sha256']==b['initial_sha256'],'final_weights_exact':a['final_sha256']==b['final_sha256'],
 'all_losses_gradients_and_metrics_exact':a['history']==b['history'],'all_gradient_probes_exact':pa.keys()==pb.keys() and all(torch.equal(pa[k],pb[k]) for k in pa),
 'same_config':a['config']==b['config'],'no_torch_npu':not a['torch_npu_imported'] and not b['torch_npu_imported']}
 rows.append({'method':method,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
 'losses':[x['loss'] for x in a['history']],'valid_tokens':[x['valid_tokens'] for x in a['history']],
 'input_shapes':[x['input_shape'] for x in a['history']],'query_shapes':[x['draft_shapes'] for x in a['history']],
 'base_json_sha256':sha(r/f'{method}_base_3.json'),'head_json_sha256':sha(r/f'{method}_head_3.json')})
guards=[json.loads(p.read_text()) for p in r.glob('*.guard.json')]
previous_path=r/'regression_summary.before_weight_report_fix.json'
previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
for row in rows:
 row['head_source_tree_sha256']=code_sha(root/'head/vlm_dflash') if row['method']=='dfly' or previous is None else previous['source_tree_sha256']['head']
result={'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL','rows':rows,
 'scope':'Contended A6000 functional base/head regression only; no throughput, hardware speed, or memory performance comparison',
 'initialization':'same seed42 freshly initialized model; equality verified by full state tensor SHA; not a trained-quality check',
 'data':'first3 of fixed first8 natural-image teacher-feature subset; same reconstructed teacher embedding',
 'geometry':'3 layers, hidden2048, FFN11008, heads16, KVheads2, head_dim128, block8, requested anchors256; actual query counts recorded',
 'optimizer':'production FP32-master AdamW helper; diagnostic microbatch1/accum1,3updates,horizon100,stage75..77; source LR/betas retained',
 'head_base_revision':'4625ef676cd77b37a7fa9ed1f0e9f3864ee5f5ca',
 'base_revision':'8cacc9db4cee10de4a7c7aedfe3b3d461b8d9716',
 'source_tree_sha256':{leg:code_sha(root/leg/'vlm_dflash') for leg in ['base','head']},
 'harness_sha256':{name:sha(root/'head/scripts/npu'/name) for name in ['audit_cuda.py','audit_common.py']},
 'bundle_sha256':sha(root/'bundle.pt'),'materialization':json.loads((root/'bundle.materialization.json').read_text()),
 'environment':{'python':'/home/lyh/veladraft-pr30-audit-20261021/env/bin/python','torch':b['torch'],'transformers':b['transformers'],
 'isolation':'new system-site-packages venv with transformers5.12.1 installed only in overlay; existing c1vllm unchanged'},
 'gpu_reservation_wall_seconds_including_pilot_failure':sum(g['wall_seconds'] for g in guards),
 'minimum_recorded_free_MiB':min(free for g in guards for _,free in g['free_MiB_samples']),
 'pilot_failure':'Preserved probe_rounding_failure files: float32 linspace rounded out-of-bounds on large matrices; integer probe indexing fixed and covered by six CPU tests; no production-model failure.'}
(r/'regression_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
