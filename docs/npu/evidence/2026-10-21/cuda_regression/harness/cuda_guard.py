import subprocess,time,json,sys,os
from pathlib import Path
root=Path('/home/lyh/veladraft-pr30-audit-20261021')
method,leg,steps=sys.argv[1:4]
out=root/'results'/f'{method}_{leg}_{steps}'
q=['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits']
free=int(subprocess.check_output(q,text=True).strip())
if free<30000: raise SystemExit(f'Not enough shared headroom: {free} MiB')
cmd=[str(root/'env/bin/python'),str(root/'head/scripts/npu/audit_cuda.py'),'--repo',str(root/leg),'--bundle',str(root/'bundle.pt'),'--config',str(root/'corpus/r3/config.json'),'--output',str(out.with_suffix('.json')),'--method',method,'--steps',steps,'--memory-budget-gib','22']
start=time.monotonic();samples=[];reason=None
with out.with_suffix('.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
 while p.poll() is None:
  free=int(subprocess.check_output(q,text=True).strip());samples.append([round(time.monotonic()-start,2),free])
  if free<3072 or time.monotonic()-start>900:
   reason='headroom_below_3GiB' if free<3072 else 'leg_15min_timeout'
   p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   break
  time.sleep(5)
 result={'method':method,'leg':leg,'steps':int(steps),'pid':p.pid,'returncode':p.returncode,'reason':reason,'wall_seconds':time.monotonic()-start,'gpu_index':1,'scope':'contended functional only; no performance inference','free_MiB_samples':samples,'command':cmd}
 out.with_suffix('.guard.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
 sys.exit(p.returncode or (1 if reason else 0))
