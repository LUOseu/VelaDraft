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


class GPULease:
    def __init__(self,index,path):
        self.index=index;self.path=Path(path);self.rows=[];self.stop=threading.Event();self.lock=None;self.recent_own_gpu_pids={};self.proc_complete=process_visibility_complete();self.watcher_error=None
    def sample(self,phase):
        row={'phase':phase,'began_wall_ns':time.time_ns(),'gpu_uuid':self.uuid}
        try:
            pids={int(v) for v in query('-i',self.uuid,'--query-compute-apps=pid').splitlines() if v.strip()}
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
    def __enter__(self):
        self.uuid=normalize_uuid(query('-i',str(self.index),'--query-gpu=uuid'))
        root=Path('/tmp/veladraft_gpu_locks');root.mkdir(exist_ok=True)
        self.lock=(root/(self.uuid+'.lock')).open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if query('-i',self.uuid,'--query-compute-apps=pid') or int(query('-i',self.uuid,'--query-gpu=memory.used'))>=500:
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
            self.lock.close();raise
    def __exit__(self,*exc):
        try:
            self.stop.set();self.thread.join(timeout=30)
            if self.thread.is_alive():self.rows.append({'error':'GPU watcher did not finish'})
            self.sample('end')
        finally:
            try:
                fcntl.flock(self.lock,fcntl.LOCK_UN)
            finally:
                self.lock.close()
    def validate(self):
        if self.watcher_error is not None:raise ValueError('GPU resource watcher failed: '+self.watcher_error)
        if not self.rows or self.rows[0]['phase']!='start' or self.rows[-1]['phase']!='end':raise ValueError('GPU resource coverage incomplete')
        if any(row.get('foreign_pids') or row.get('error') for row in self.rows):raise ValueError('GPU resource evidence contaminated; whole result invalid')
        return dict(status='complete',samples=len(self.rows),continuous_exclusivity_claim=False,
                    began_wall_ns=self.rows[0]['began_wall_ns'],ended_wall_ns=self.rows[-1]['ended_wall_ns'])
