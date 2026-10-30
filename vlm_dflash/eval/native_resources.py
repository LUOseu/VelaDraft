"""Advisory GPU lease and sampled NVML process ownership; no model imports."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


def normalize_uuid(value):
    text=str(value)
    if not re.fullmatch(r'(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',text):
        raise ValueError('Invalid complete GPU UUID')
    return 'GPU-'+text.removeprefix('GPU-').lower()


def query(*args):
    return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader,nounits'],text=True,timeout=10).strip()


def descendants():
    rows=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True,timeout=10)
    edges=[tuple(map(int,line.split())) for line in rows.splitlines()];allowed={os.getpid()}
    while True:
        more={pid for pid,parent in edges if parent in allowed}
        if more<=allowed:return allowed
        allowed|=more


def process_visibility_complete():
    """Require an ordinary procfs mount before identifying a vanished PID."""
    try:
        with open('/proc/mounts', encoding='utf-8') as stream:
            for line in stream:
                fields=line.split()
                if len(fields)>=4 and fields[1]=='/proc':
                    return not any(x.startswith('hidepid=') and x!='hidepid=0' for x in fields[3].split(','))
    except OSError:
        pass
    return False


def process_status(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        return 'zombie' if state=='Z' else 'running'
    except FileNotFoundError:
        return 'missing'
    except (OSError,IndexError):
        return 'unknown'


def classify_gpu_pids(reported, descendants_now, known_recent_own, status):
    foreign=[];stale=[]
    for pid in sorted(reported-descendants_now):
        if pid in known_recent_own and status(pid) in ('missing','zombie'):
            stale.append(pid)
        else:
            foreign.append(pid)
    return foreign,stale


def ownership_mode():
    """'pid' (default): reported GPU PIDs are resolved against this process tree.
    'count': the container's PID namespace hides host PIDs (nvidia-smi reports host
    PIDs, /proc only container PIDs), so ownership cannot be resolved. Then every
    leased GPU must be empty at each stage boundary; while a stage runs the first leased
    GPU carries at most VELADRAFT_GPU_MAX_OWN_APPS compute apps (the client context plus
    TP rank 0) and every other leased GPU at most one (its TP worker)."""
    mode=os.environ.get('VELADRAFT_GPU_OWNERSHIP','pid')
    if mode not in ('pid','count'):raise ValueError('VELADRAFT_GPU_OWNERSHIP must be pid or count')
    return mode


class GPULease:
    """Lease one GPU index or a list of indices (tensor parallel); all are sampled together."""
    def __init__(self,index,path):
        self.indices=[index] if isinstance(index,int) else list(index)
        self.index=self.indices[0];self.path=Path(path);self.rows=[];self.stop=threading.Event();self.locks=[];self.recent_own_gpu_pids={};self.proc_complete=process_visibility_complete();self.watcher_error=None
        self.mode=ownership_mode();self.max_own_apps=int(os.environ.get('VELADRAFT_GPU_MAX_OWN_APPS','2'))
    def app_bound(self,uuid,uuids,phase):
        if phase in ('start','boundary'):return 0
        return self.max_own_apps if uuid==uuids[0] else 1
    def per_gpu_pids(self,uuids):
        return {uuid:sorted({int(v) for v in query('-i',uuid,'--query-compute-apps=pid').splitlines() if v.strip()}) for uuid in uuids}
    def sample(self,phase):
        uuids=getattr(self,'uuids',None) or [self.uuid]
        row={'phase':phase,'began_wall_ns':time.time_ns(),'gpu_uuid':self.uuid,'gpu_uuids':uuids}
        try:
            if getattr(self,'mode','pid')=='count':
                row['ownership']='count';row['per_gpu_pids']=self.per_gpu_pids(uuids)
                row['per_gpu_memory_mib']={u:int(query('-i',u,'--query-gpu=memory.used')) for u in uuids}
                row['reported_pids']=sorted({p for v in row['per_gpu_pids'].values() for p in v})
                row['over_bound']=[u for u,v in row['per_gpu_pids'].items() if len(v)>self.app_bound(u,uuids,phase)]
                row['foreign_pids']=[];row['stale_observed_own_pids']=[]
                row['ended_wall_ns']=time.time_ns();self.rows.append(row)
                with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')
                return row
            pids={int(v) for uuid in uuids for v in query('-i',uuid,'--query-compute-apps=pid').splitlines() if v.strip()}
            live=descendants();now=time.monotonic()
            for pid in pids & live:self.recent_own_gpu_pids[pid]=now
            recent={pid for pid,seen in self.recent_own_gpu_pids.items() if now-seen<=10}
            status=process_status if self.proc_complete else lambda pid:'unknown'
            row['reported_pids']=sorted(pids)
            row['observed_own_gpu_pids']=sorted(pids & live)
            row['foreign_pids'],row['stale_observed_own_pids']=classify_gpu_pids(pids,live,recent,status)
        except Exception as error:row['error']=str(error)
        row['ended_wall_ns']=time.time_ns();self.rows.append(row)
        with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')
        return row
    def __enter__(self):
        self.uuids=[normalize_uuid(query('-i',str(i),'--query-gpu=uuid')) for i in self.indices]
        self.uuid=self.uuids[0]
        root=Path(os.environ.get('VELADRAFT_GPU_LOCK_DIR','/tmp/veladraft_gpu_locks'));root.mkdir(parents=True,exist_ok=True)
        try:
            for uuid in self.uuids:
                lock=(root/(uuid+'.lock')).open('a');self.locks.append(lock)
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if query('-i',uuid,'--query-compute-apps=pid') or int(query('-i',uuid,'--query-gpu=memory.used'))>=500:
                    raise ValueError('GPU is occupied; no foreign process is stopped')
            self.sample('start')
            def poll():
                try:
                    while not self.stop.wait(2):self.sample('sample')
                except BaseException as error:
                    # Keep an in-memory failure even if the log becomes writable again.
                    self.watcher_error=f'{type(error).__name__}: {error}'
            self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
        except BaseException:
            for lock in self.locks:lock.close()
            raise
    def __exit__(self,*exc):
        try:
            self.stop.set();self.thread.join(timeout=30)
            if self.thread.is_alive():self.rows.append({'error':'GPU watcher did not finish'})
            self.sample('end')
        finally:
            for lock in self.locks:
                try:
                    fcntl.flock(lock,fcntl.LOCK_UN)
                finally:
                    lock.close()
    def boundary(self,timeout=180):
        """Count mode: wait for the finished stage's processes to release every leased GPU."""
        if getattr(self,'mode','pid')!='count':return
        deadline=time.monotonic()+timeout
        while True:
            row=self.sample('boundary_wait')
            if not row.get('error') and not row['reported_pids']:break
            if time.monotonic()>deadline:break
            time.sleep(2)
        self.sample('boundary')
    def validate(self,timed_windows=None):
        """timed_windows: [(began_wall_ns, ended_wall_ns)] of the timed phases. In count mode
        extra apps inside a window invalidate the result; outside (startup, compile, warmup,
        stage boundaries) they are counted and reported, e.g. a site utilisation floor
        that starts while an engine loads."""
        if self.watcher_error is not None:raise ValueError('GPU resource watcher failed: '+self.watcher_error)
        outside=0
        if getattr(self,'mode','pid')=='count':
            flagged=[row for row in self.rows if row.get('over_bound') and row['phase']!='boundary_wait']
            def inside(row):
                return timed_windows is None or any(row['began_wall_ns']<=end and row['ended_wall_ns']>=begin for begin,end in timed_windows)
            if any(inside(row) for row in flagged):
                raise ValueError('GPU carried unexpected compute apps (count ownership mode); whole result invalid')
            outside=len(flagged)
        if not self.rows or self.rows[0]['phase']!='start' or self.rows[-1]['phase']!='end':raise ValueError('GPU resource coverage incomplete')
        if any(row.get('foreign_pids') or row.get('error') for row in self.rows):raise ValueError('GPU resource evidence contaminated; whole result invalid')
        memory=[row['per_gpu_memory_mib'] for row in self.rows if 'per_gpu_memory_mib' in row]
        peak={u:max(m[u] for m in memory) for u in memory[0]} if memory else None
        return dict(status='complete',samples=len(self.rows),continuous_exclusivity_claim=False,gpu_uuids=getattr(self,'uuids',None) or [self.uuid],
                    peak_memory_used_mib=peak,
                    ownership_mode=getattr(self,'mode','pid'),max_own_apps_per_gpu=({u:self.app_bound(u,self.uuids,'sample') for u in self.uuids} if getattr(self,'mode','pid')=='count' else None),
                    extra_app_samples_outside_timed=outside,
                    began_wall_ns=self.rows[0]['began_wall_ns'],ended_wall_ns=self.rows[-1]['ended_wall_ns'])
