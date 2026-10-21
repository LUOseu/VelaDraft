import subprocess,time,json,sys,os,signal
from pathlib import Path
r=Path('/home/lyh/veladraft-pr30-audit-20261021');mode=sys.argv[1]
prior=0
if mode!='qualification':
 assert json.loads((r/'A_cuda/qualification.json').read_text())['status']=='PASS'
 prior=json.loads((r/'A_cuda/qualification.guard.json').read_text())['wall_seconds']
remaining=600-prior
assert remaining>0
query=['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits']
assert int(subprocess.check_output(query,text=True).strip())>=30000
cmd=[str(r/'env/bin/python'),str(r/'A_child.py'),mode,str(remaining)]
start=time.monotonic();samples=[];reason=None
with (r/f'A_cuda/{mode}.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 try:
  while p.poll() is None:
   free=int(subprocess.check_output(query,text=True).strip());samples.append([round(time.monotonic()-start,2),free])
   if free<3072 or time.monotonic()-start>remaining:
    reason='memory_headroom' if free<3072 else 'A_total600s_budget';break
   time.sleep(5)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 result={'mode':mode,'returncode':p.returncode,'reason':reason,'wall_seconds':time.monotonic()-start,'prior_A_seconds':prior,'gpu':1,'free_MiB_samples':samples,'command':cmd,'scope':'contended diagnostic; no exclusive GPU performance claim'}
 (r/f'A_cuda/{mode}.guard.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(p.returncode or (1 if reason else 0))
