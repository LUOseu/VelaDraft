import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from profile_worker import Recorder, ProfileWorker
from profile_stage import freeze_engine_args, save_json
from pathlib import Path
import tempfile

class Event:
    def __init__(self, **kwargs):pass
    def record(self):pass
    def elapsed_time(self, other):return 1.25

class RecorderTests(unittest.TestCase):
    def test_engine_config_snapshot_survives_vllm_nested_mutation(self):
        original={'speculative_config':{'model':'draft','num_speculative_tokens':1}}
        frozen=freeze_engine_args(original)
        original['speculative_config']['draft_model_config']=object()
        self.assertNotIn('draft_model_config',frozen['speculative_config'])

    def test_failed_serialization_preserves_existing_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json';path.write_text('{"status":"running"}')
            with self.assertRaises(TypeError):save_json(path,{'bad':object()})
            self.assertEqual(path.read_text(),'{"status":"running"}')

    def test_target_spans_use_current_scheduler_step(self):
        runner=NS(drafter=None, _model_forward=lambda:3, _sample=lambda:None, sample_tokens=lambda:None)
        runner.execute_model=lambda output:runner._model_forward()
        worker=ProfileWorker();worker.model_runner=runner;worker.rank=0
        with patch.dict('sys.modules',{'torch':NS(cuda=NS(Event=Event,synchronize=lambda:None))}):
            installed=worker.q36_profile_install()
        self.assertIn('_sample',installed['components'])
        worker.q36_profile_begin()
        runner.execute_model(NS(scheduled_spec_decode_tokens={'r':list(range(7))},total_num_scheduled_tokens=8))
        spans=worker.q36_profile_end()['spans']
        self.assertTrue(all(x.get('phase')=='verify' and x.get('scheduled_tokens')==8 for x in spans))

    def test_disabled_wrapper_preserves_result_without_records(self):
        obj=NS(f=lambda x:x+2);r=Recorder(NS(Event=Event,synchronize=lambda:None));r.wrap(obj,'f','test')
        self.assertEqual(obj.f(3),5);self.assertEqual(r.records,[])

    def test_nested_spans_are_kept_separate_and_calls_not_duplicated(self):
        seen=[];obj=NS(child=lambda x:seen.append(x) or x+1)
        obj.parent=lambda x:obj.child(x)*2
        r=Recorder(NS(Event=Event,synchronize=lambda:None));r.wrap(obj,'child','child');r.wrap(obj,'parent','parent_inclusive')
        r.active=True;r.current={'phase':'verify','scheduled_tokens':8}
        self.assertEqual(obj.parent(2),6);self.assertEqual(seen,[2])
        spans=r.finish();self.assertEqual([x['component'] for x in spans],['child','parent_inclusive'])
        self.assertTrue(all(x['scheduled_tokens']==8 for x in spans));self.assertFalse(r.active)

if __name__=='__main__':unittest.main()
