import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def resources():
    spec=importlib.util.spec_from_file_location('native_resources',Path(__file__).resolve().parents[1]/'vlm_dflash'/'eval'/'native_resources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_dead_previously_observed_child_is_stale(resources):
    foreign,stale=resources.classify_gpu_pids({101},{1},{1,101},lambda pid:'missing')
    assert foreign==[] and stale==[101]

def test_live_non_descendant_is_foreign_even_if_pid_was_previously_ours(resources):
    foreign,stale=resources.classify_gpu_pids({101},{1},{1,101},lambda pid:'running')
    assert foreign==[101] and stale==[]

def test_unknown_missing_pid_is_not_whitelisted(resources):
    foreign,stale=resources.classify_gpu_pids({202},{1},{1},lambda pid:'missing')
    assert foreign==[202] and stale==[]

def test_observed_zombie_is_stale_but_unknown_zombie_remains_foreign(resources):
    foreign,stale=resources.classify_gpu_pids({101,202},{1},{1,101},lambda pid:'zombie')
    assert foreign==[202] and stale==[101]

def test_inaccessible_proc_is_conservatively_foreign(resources):
    foreign,stale=resources.classify_gpu_pids({101},{1},{1,101},lambda pid:'unknown')
    assert foreign==[101] and stale==[]

def test_live_descendant_is_never_foreign(resources):
    foreign,stale=resources.classify_gpu_pids({101},{1,101},{1},lambda pid:'running')
    assert foreign==[] and stale==[]

def test_sampling_records_seen_own_then_dead_pid_as_stale(resources,tmp_path,monkeypatch):
    monkeypatch.setattr(resources,'process_visibility_complete',lambda:True)
    monkeypatch.setattr(resources,'query',lambda *args:'101')
    states=iter([{1,101},{1}]);monkeypatch.setattr(resources,'descendants',lambda:next(states))
    monkeypatch.setattr(resources,'process_status',lambda pid:'missing')
    lease=resources.GPULease(0,tmp_path/'gpu.jsonl');lease.uuid='GPU-test'
    lease.sample('start');lease.sample('end')
    assert lease.rows[0]['observed_own_gpu_pids']==[101]
    assert lease.rows[1]['foreign_pids']==[]
    assert lease.rows[1]['stale_observed_own_pids']==[101]
    assert lease.validate()['status']=='complete'


def test_transient_watcher_log_write_failure_invalidates_lease(resources, tmp_path, monkeypatch):
    import threading
    path = tmp_path / 'gpu.jsonl'
    monkeypatch.setattr(resources, 'process_visibility_complete', lambda: False)
    monkeypatch.setattr(resources, 'descendants', lambda: {resources.os.getpid()})
    def fake_query(*args):
        if '--query-gpu=uuid' in args:
            return 'GPU-864585d7-9789-e033-7781-e773f55178c6'
        if '--query-gpu=memory.used' in args:
            return '0'
        return ''
    monkeypatch.setattr(resources, 'query', fake_query)
    original_open = Path.open
    failed = threading.Event()
    calls = 0
    def flaky_open(self, *args, **kwargs):
        nonlocal calls
        if self == path:
            calls += 1
            if calls == 2:
                failed.set()
                raise OSError('transient resource log write failure')
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', flaky_open)
    lease = resources.GPULease(0, path)
    with lease:
        assert failed.wait(5), 'background sampler did not run'
    with pytest.raises(ValueError, match='resource|watcher'):
        lease.validate()


def test_lease_lock_released_even_if_watcher_join_interrupted(resources, tmp_path):
    class InterruptedThread:
        def join(self, timeout):
            raise KeyboardInterrupt('interrupted while joining watcher')
    lock_path = tmp_path/'lease.lock'
    lease = resources.GPULease(0, tmp_path/'gpu.jsonl')
    lease.lock = lock_path.open('a')
    resources.fcntl.flock(lease.lock, resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB)
    lease.thread = InterruptedThread()
    with pytest.raises(KeyboardInterrupt):
        lease.__exit__(None, None, None)
    assert lease.lock.closed
    with lock_path.open('a') as other:
        resources.fcntl.flock(other, resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB)
