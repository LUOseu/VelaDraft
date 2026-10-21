import sys,json,torch
from pathlib import Path
r=Path('/home/lyh/veladraft-pr30-audit-20261021');sys.path.insert(0,str(r/'head'))
from scripts.npu.audit_common import load_bundle,build_module
from scripts.npu.audit_compaction import configuration,select_batches,plans_for,tree_digest
b=load_bundle(r/'bundle64.pt');b['contract']['target']['model_path']='/home/lyh/models/Qwen2.5-VL-3B-Instruct'
cfg=configuration(b,'cpu')
m,g=build_module(b,cfg,'cpu',checkpoint='/home/lyh/vlm-scale/runs/r3r5_n20000/r3/train/ckpt_final.pt')
plans=plans_for(m,b,select_batches(b['records']))
frozen=torch.load(r/'A_cuda/anchor_plan.pt',map_location='cpu',weights_only=False)
assert tree_digest(frozen)=='713a1dec4d24ac996de64abd0dd889851c9eba0fc0743ed4b5d216ed232226da'
assert tree_digest(plans)==tree_digest(frozen),'CPU-generated plans differ across backend hosts'
for a,c in zip(plans,frozen):
 assert torch.equal(a['anchors'],c['anchors']) and torch.equal(a['keep'],c['keep'])
# Predeclared capacity proxy, independent of timings: allocated attention entries.
worst=max(range(len(plans)),key=lambda i:plans[i]['allocated_rows']*(plans[i]['context']+plans[i]['query_rows_per_sample']))
report={'status':'PASS','plan_tree_digest':tree_digest(plans),'all_plan_fields_exact':True,
        'worst_batch':worst,'selection':'max allocated_rows * (context + query_rows_per_sample), before timings',
        'worst_plan':{k:v for k,v in plans[worst].items() if k not in ('anchors','keep')}}
(r/'A_cuda/plan_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
