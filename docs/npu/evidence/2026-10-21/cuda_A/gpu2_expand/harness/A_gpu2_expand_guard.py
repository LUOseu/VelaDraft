import subprocess,time,json,os,signal,fcntl,hashlib
from pathlib import Path
r=Path('/home/lyh/veladraft-pr30-audit-20261021');out=r/'A_cuda/gpu2_expand';out.mkdir(parents=True,exist_ok=True)
uuid='GPU-1d7c6b74-f5df-a297-6796-3986c3701ffa'
lock_path=Path('/tmp/veladraft_gpu_locks')/(uuid+'.lock')
def query():
 actual,free=subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert actual.strip()==uuid
 apps=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
  if line.startswith(uuid):apps.append(int(line.split(',')[1].strip()))
 return int(free.strip()),apps
with lock_path.open('a+') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  (out/'guard.json').write_text(json.dumps({'status':'NOT_LAUNCHED','reason':'existing queue GPU lock busy'}));raise SystemExit(2)
 free,apps=query()
 if free<45*1024 or apps:
  (out/'guard.json').write_text(json.dumps({'status':'NOT_LAUNCHED','reason':'GPU2 not empty with >=45GiB free','free_MiB':free,'apps':apps}));raise SystemExit(2)
 snapshot={'gpu_index':2,'gpu_uuid':uuid,'lock_path':str(lock_path),'preflight_free_MiB':free,'preflight_apps':apps,'allocator_cap_gib':40,'max_seconds':600,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'A_gpu2_expand_child.py',r/'A_gpu2_expand_guard.py']}}
 snapshot['allocator_environment']={'PYTORCH_ALLOC_CONF':'expandable_segments:True','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 snapshot['reason']='One process-allocator retry after default allocator backward OOM:29.13GiB live allocated,7.18GiB cached-unallocated,requested4.06GiB; same40GiB cap and unchanged full workload.'
 (out/'preflight.json').write_text(json.dumps(snapshot,indent=2))
 aborted=[]
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,lambda signum,frame:aborted.append(signum))
 cmd=[str(r/'env/bin/python'),str(r/'A_gpu2_expand_child.py')]
 start=time.monotonic();samples=[];reason=None
 with (out/'run.log').open('w') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_ALLOC_CONF':'expandable_segments:True'})
  try:
   while child.poll() is None:
    free,apps=query();foreign=[]
    for pid in apps:
     try:
      if os.getpgid(pid)!=child.pid:foreign.append(pid)
     except ProcessLookupError:pass
    samples.append([round(time.monotonic()-start,2),free,foreign])
    if free<3072 or foreign or aborted or time.monotonic()-start>600:
     reason='foreign_compute_process' if foreign else ('memory_headroom' if free<3072 else ('signal' if aborted else '600s_budget'));break
    time.sleep(1)
  finally:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  result={**snapshot,'status':'FINISHED','pid':child.pid,'returncode':child.returncode,'reason':reason,'wall_seconds':time.monotonic()-start,'samples':samples,'command':cmd,'scope':'GPU2 observed exclusive compute and queue advisory lock held throughout; host CPU still shared'}
  (out/'guard.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(child.returncode or (1 if reason else 0))
