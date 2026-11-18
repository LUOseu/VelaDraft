import hashlib
import json
from pathlib import Path
import subprocess
import sys
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

def test_recovery_receipt_uses_the_archived_stage_source(tmp_path):
    stage = tmp_path / "stages" / "sd_k1"
    plan = tmp_path / "plans" / "k1.evidence"
    logs = tmp_path / "logs"
    for directory in (stage, plan, logs):
        directory.mkdir(parents=True)
    raw = json.dumps(sample()).encode()
    (stage / "profile.json").write_bytes(raw)
    sessions = [{"id": "a", "questions": ["one"]}, {"id": "b", "questions": ["one", "two", "three"]}]
    (plan / "plan.json").write_text(json.dumps({"sessions": sessions}))
    (logs / "sd_k1.log").write_text("TypeError: Object of type ModelConfig is not JSON serializable")
    (tmp_path / "resources.jsonl").write_text('{"phase":"start"}\n{"phase":"end"}\n')

    script = Path(__file__).with_name("recover.py")
    subprocess.run([sys.executable, str(script), str(tmp_path)], check=True)

    source = script.resolve().parents[1] / "k_sweep" / "profile_stage.py"
    receipt = json.loads((stage / "recovery_receipt.json").read_text())
    assert receipt["source_stage_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (stage / "profile.raw_running.json").read_bytes() == raw


if __name__=='__main__':unittest.main()
