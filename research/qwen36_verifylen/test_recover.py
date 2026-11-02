import unittest
from recover import validate_partial

def sample():
    rows=[]
    for repeat in range(3):
        for sid,turn in [('a',0),('b',0),('b',2)]:
            spans=[{'component':'target_forward','step':1,'phase':'verify','scheduled_tokens':2,'stream_elapsed_ms':1.,'host_ms':1.}]
            rows.append({'id':sid,'turn':turn,'repetition':repeat,'ranks':[{'rank':0,'spans':spans},{'rank':1,'spans':spans}]})
    return {'status':'running','rows':rows}

class RecoveryTests(unittest.TestCase):
    def test_accepts_complete_raw_samples_with_clean_resource_epoch(self):
        resources=[{'phase':'start','foreign_pids':[]},{'phase':'end','foreign_pids':[]}]
        self.assertEqual(validate_partial(sample(),[('a',0),('b',0),('b',2)],resources),3)
    def test_missing_request_is_not_recovered(self):
        raw=sample();raw['rows'].pop()
        with self.assertRaises(ValueError):validate_partial(raw,[('a',0),('b',0),('b',2)],[{'phase':'start'},{'phase':'end'}])
    def test_foreign_gpu_process_is_not_recovered(self):
        with self.assertRaises(ValueError):validate_partial(sample(),[('a',0),('b',0),('b',2)],[{'phase':'start'},{'phase':'sample','foreign_pids':[123]},{'phase':'end'}])

if __name__=='__main__':unittest.main()
