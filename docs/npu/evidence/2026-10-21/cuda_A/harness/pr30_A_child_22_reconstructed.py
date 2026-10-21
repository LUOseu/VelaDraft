import os,sys,json,time,traceback,torch
from pathlib import Path
r=Path('/home/lyh/veladraft-pr30-audit-20261021');sys.path.insert(0,str(r/'head'))
torch.cuda.set_per_process_memory_fraction(22*2**30/torch.cuda.get_device_properties(0).total_memory,0)
torch.use_deterministic_algorithms(True)
from scripts.npu.audit_common import load_bundle
from scripts.npu.audit_compaction import configuration,run_leg,tree_digest,main
mode=sys.argv[1]
checkpoint='/home/lyh/vlm-scale/runs/r3r5_n20000/r3/train/ckpt_final.pt'
if mode=='qualification':
 started=time.monotonic();result={'status':'FAILED'}
 try:
  bundle=load_bundle(r/'bundle64.pt');bundle['contract']['target']['model_path']='/home/lyh/models/Qwen2.5-VL-3B-Instruct'
  cfg=configuration(bundle,'cuda:0');plans=torch.load(r/'A_cuda/anchor_plan.pt',map_location='cpu',weights_only=False)
  verification=json.loads((r/'A_cuda/plan_verification.json').read_text())
  assert verification['status']=='PASS' and tree_digest(plans)==verification['plan_tree_digest']
  worst=verification['worst_batch']
  leg=run_leg(bundle,cfg,checkpoint,[plans[worst]],'cuda:0',42,False,1,0)
  result={'status':'PASS','leg':leg,'plan_verification':verification}
 except Exception as exc:
  result.update(error=str(exc),traceback=traceback.format_exc(),oom='out of memory' in str(exc).lower())
 result.update(scope='fixed worst formal microbatch8 OFF full optimizer update under shared 22GiB allocator cap; not whole-device capacity or performance',allocator_cap_gib=22,elapsed_seconds=time.monotonic()-started)
 (r/'A_cuda/qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 raise SystemExit(0 if result['status']=='PASS' else 1)
else:
 remaining=sys.argv[2]
 sys.argv=['audit_compaction.py','--bundle',str(r/'bundle64.pt'),'--checkpoint',checkpoint,'--device','cuda:0','--out',str(r/'A_cuda/run'),'--target','/home/lyh/models/Qwen2.5-VL-3B-Instruct','--phase','all','--max-seconds',remaining]
 raise SystemExit(main())
