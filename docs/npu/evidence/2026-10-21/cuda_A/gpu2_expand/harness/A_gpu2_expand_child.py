import sys,torch
from pathlib import Path
r=Path('/home/lyh/veladraft-pr30-audit-20261021');sys.path.insert(0,str(r/'head'))
torch.cuda.set_per_process_memory_fraction(40*2**30/torch.cuda.get_device_properties(0).total_memory,0)
from scripts.npu import audit_compaction as audit
original_plans=audit.plans_for
def checked_plans(*args,**kwargs):
 plans=original_plans(*args,**kwargs)
 assert audit.tree_digest(plans)=='713a1dec4d24ac996de64abd0dd889851c9eba0fc0743ed4b5d216ed232226da','frozen NPU plans changed'
 return plans
audit.plans_for=checked_plans
sys.argv=['audit_compaction.py','--bundle',str(r/'bundle64.pt'),'--checkpoint','/home/lyh/vlm-scale/runs/r3r5_n20000/r3/train/ckpt_final.pt','--device','cuda:0','--out',str(r/'A_cuda/gpu2_expand'),'--target','/home/lyh/models/Qwen2.5-VL-3B-Instruct','--phase','all','--max-seconds','600']
raise SystemExit(audit.main())
