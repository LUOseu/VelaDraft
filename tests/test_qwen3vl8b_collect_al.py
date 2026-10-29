import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'research'/'qwen3vl8b_native_vllm'/'collect_al_supplement_v1.py';spec=importlib.util.spec_from_file_location('al',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ALTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.r=Path(self.tmp.name);self.e=self.r/'ev';self.e.mkdir();self.rp=self.r/'result.json';self.sp=self.e/'sd_r0.json'
  sessions=[{'id':str(i),'questions':['q']*(2 if i<25 else 1)} for i in range(120)]
  plan={'method':'dfly','sessions':sessions,'speculative_tokens':7};plan_path=self.e/'plan.json';plan_path.write_text(json.dumps(plan))
  self.rp.write_text(json.dumps({'status':'complete','resources_verified':True,'timed_pairs':145,'sessions':120,'repeats':1,'source_plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'evidence':str(self.e),'method':'dfly'}))
  rows=[{'phase':'timed','status':'complete','id':s['id'],'turn':i} for s in sessions for i in range(len(s['questions']))]
  self.stage={'status':'complete','rows':rows,'engine_args':{'disable_log_stats':False},'timed_official_counters':{'status':'complete','rounds':100,'accepted':175,'proposed':700}}
 def test_counts_convention_and_rejects_disabled_or_invalid_counters(self):
  self.sp.write_text(json.dumps(self.stage));r=m.summarize(self.rp);self.assertEqual(r['acceptance_length'],2.75);self.assertEqual(r['draft_rounds'],100)
  self.stage['engine_args']['disable_log_stats']=True;self.sp.write_text(json.dumps(self.stage))
  with self.assertRaises(ValueError):m.summarize(self.rp)
  self.stage['engine_args']['disable_log_stats']=False;self.stage['timed_official_counters']['rounds']=0;self.sp.write_text(json.dumps(self.stage))
  with self.assertRaises(ValueError):m.summarize(self.rp)
 def test_rejects_more_than_seven_drafts_per_round(self):
  self.stage['timed_official_counters'].update(rounds=1,accepted=100,proposed=100)
  self.sp.write_text(json.dumps(self.stage))
  with self.assertRaises(ValueError):m.summarize(self.rp)
if __name__=='__main__':unittest.main()

def test_rejects_145_turns_from_wrong_session_scope(tmp_path):
    evidence=tmp_path/'evidence';evidence.mkdir()
    (evidence/'sd_r0.json').write_text(json.dumps({'status':'complete','rows':[{'phase':'timed'}]*145,'engine_args':{'disable_log_stats':False},'timed_official_counters':{'status':'complete','rounds':10,'accepted':5,'proposed':70}}))
    result=tmp_path/'result.json'
    result.write_text(json.dumps({'status':'complete','resources_verified':True,'timed_pairs':145,'sessions':145,'repeats':1,'evidence':str(evidence),'method':'dfly'}))
    with unittest.TestCase().assertRaises(ValueError):
        m.summarize(result)
