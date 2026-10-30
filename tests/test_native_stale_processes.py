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
    lease.locks = [lock_path.open('a')]
    resources.fcntl.flock(lease.locks[0], resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB)
    lease.thread = InterruptedThread()
    with pytest.raises(KeyboardInterrupt):
        lease.__exit__(None, None, None)
    assert lease.locks[0].closed
    with lock_path.open('a') as other:
        resources.fcntl.flock(other, resources.fcntl.LOCK_EX | resources.fcntl.LOCK_NB)


def test_count_ownership_requires_empty_boundaries_and_bounded_apps(resources, tmp_path, monkeypatch):
    monkeypatch.setenv('VELADRAFT_GPU_OWNERSHIP', 'count')
    monkeypatch.setenv('VELADRAFT_GPU_MAX_OWN_APPS', '2')
    seen = {'GPU-a': '', 'GPU-b': ''}
    monkeypatch.setattr(resources, 'query',
                        lambda *args: '1234' if '--query-gpu=memory.used' in args else seen[args[1]])
    lease = resources.GPULease([0, 1], tmp_path/'gpu.jsonl')
    lease.uuids = ['GPU-a', 'GPU-b']; lease.uuid = 'GPU-a'
    lease.sample('start')
    seen.update({'GPU-a': '900001\n900002', 'GPU-b': '900003'})
    lease.sample('sample')
    seen.update({'GPU-a': '', 'GPU-b': ''})
    lease.boundary(timeout=0)
    lease.sample('end')
    result = lease.validate()
    assert result['ownership_mode'] == 'count' and result['gpu_uuids'] == ['GPU-a', 'GPU-b']
    assert result['peak_memory_used_mib'] == {'GPU-a': 1234, 'GPU-b': 1234}
    seen.update({'GPU-a': '900001\n900002\n900009'})
    lease.sample('sample')
    with pytest.raises(ValueError, match='count ownership'):
        lease.validate()


def test_count_ownership_bounds_non_client_gpus_to_one_worker(resources, tmp_path, monkeypatch):
    monkeypatch.setenv('VELADRAFT_GPU_OWNERSHIP', 'count')
    monkeypatch.setenv('VELADRAFT_GPU_MAX_OWN_APPS', '2')
    seen = {'GPU-a': '', 'GPU-b': ''}
    monkeypatch.setattr(resources, 'query',
                        lambda *args: '10' if '--query-gpu=memory.used' in args else seen[args[1]])
    lease = resources.GPULease([0, 1], tmp_path/'gpu.jsonl')
    lease.uuids = ['GPU-a', 'GPU-b']; lease.uuid = 'GPU-a'
    lease.sample('start')
    # Client context + rank 0 on the first GPU, rank 1 plus a foreign app on the second.
    seen.update({'GPU-a': '1\n2', 'GPU-b': '3\n4'})
    assert lease.sample('sample')['over_bound'] == ['GPU-b']
    seen.update({'GPU-a': '', 'GPU-b': ''})
    lease.sample('end')
    with pytest.raises(ValueError, match='count ownership'):
        lease.validate()


def test_count_ownership_flags_process_left_at_stage_boundary(resources, tmp_path, monkeypatch):
    monkeypatch.setenv('VELADRAFT_GPU_OWNERSHIP', 'count')
    monkeypatch.setattr(resources, 'query', lambda *args: '777')
    lease = resources.GPULease(0, tmp_path/'gpu.jsonl')
    lease.uuids = ['GPU-a']; lease.uuid = 'GPU-a'
    lease.boundary(timeout=0)
    assert lease.rows[-1]['phase'] == 'boundary' and lease.rows[-1]['over_bound'] == ['GPU-a']


def test_count_ownership_tolerates_extra_apps_outside_timed_windows(resources, tmp_path, monkeypatch):
    monkeypatch.setenv('VELADRAFT_GPU_OWNERSHIP', 'count')
    monkeypatch.setenv('VELADRAFT_GPU_MAX_OWN_APPS', '1')
    seen = {'GPU-a': ''}
    monkeypatch.setattr(resources, 'query',
                        lambda *args: '10' if '--query-gpu=memory.used' in args else seen[args[1]])
    lease = resources.GPULease(0, tmp_path/'gpu.jsonl')
    lease.uuids = ['GPU-a']; lease.uuid = 'GPU-a'
    lease.sample('start')
    seen['GPU-a'] = '1\n2'
    loading = lease.sample('sample')
    seen['GPU-a'] = '1'
    timed = lease.sample('sample')
    seen['GPU-a'] = ''
    lease.sample('end')
    window = [(timed['began_wall_ns'], timed['ended_wall_ns'])]
    assert lease.validate(timed_windows=window)['extra_app_samples_outside_timed'] == 1
    with pytest.raises(ValueError, match='count ownership'):
        lease.validate(timed_windows=[(loading['began_wall_ns'], loading['ended_wall_ns'])])
