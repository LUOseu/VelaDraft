import importlib.util
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sweep_queue', Path(__file__).with_name('queue.py'))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

class OwnershipTests(unittest.TestCase):
    def test_management_shell_mentions_are_not_job_identity(self):
        ps={101:dict(uid=q.UID,parent=0,start='a',cmd='bash -c "python3 manage.py # /home/lyh/qwen3vl8b_eval_20261027/queue.sh"'),
            102:dict(uid=q.UID,parent=101,start='b',cmd='python3 -'),
            103:dict(uid=q.UID,parent=0,start='c',cmd='python3 -c "print(\"/home/lyh/qwen3vl8b_eval_20261027/native-eval\")"')}
        self.assertEqual(q.owned_8b_tree(ps),set())

    def test_empty_gpu_with_live_reservation_is_not_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with (root/'GPU-test.lock').open('a') as holder:
                fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertFalse(q.gpu_lock_free('GPU-test',root))
            self.assertTrue(q.gpu_lock_free('GPU-test',root))

    def test_only_named_owned_8b_trees_are_preemptible(self):
        u=q.UID
        ps={1:dict(uid=u,parent=0,start='a',cmd='bash /home/lyh/qwen3vl8b_eval_20261027/queue.sh'),
            2:dict(uid=u,parent=1,start='b',cmd='VLLM::EngineCore'),
            3:dict(uid=u+1,parent=1,start='c',cmd='other user worker'),
            4:dict(uid=u,parent=0,start='d',cmd='python /home/lyh/qwen36_k_sweep_20261101/repo/research/qwen36_k_sweep/queue.py'),
            5:dict(uid=u+1,parent=0,start='e',cmd='bash /home/lyh/qwen3vl8b_eval_20261027/queue.sh'),
            6:dict(uid=u,parent=0,start='f',cmd='python unrelated_train.py')}
        self.assertEqual(q.owned_8b_tree(ps),{1,2})

    def test_recycled_pid_is_never_signaled(self):
        before={42:dict(uid=q.UID,parent=1,start='old',cmd='owned')}
        now={42:dict(uid=q.UID,parent=1,start='new',cmd='unrelated')}
        with tempfile.TemporaryDirectory() as tmp, patch.object(q,'TASK',Path(tmp)), patch.object(q,'processes',return_value=now), patch.object(q.time,'sleep'), patch.object(q.os,'kill') as kill:
            q.terminate_owned_8b(before,{42})
            kill.assert_not_called()

if __name__=='__main__':unittest.main()
