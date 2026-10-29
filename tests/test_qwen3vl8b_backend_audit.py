import importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'research'/'qwen3vl8b_native_vllm'/'audit_attention_backends_v1.py';spec=importlib.util.spec_from_file_location('audit_attention_backends',p)
class AuditTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
  for s,n in [('reference',36),('ar_r0',36),('sd_r0',39)]:
   obj={'status':'complete','worker_before':{'attention':[{'name':str(i),'impl':'vllm.v1.attention.backends.flash_attn.FlashAttentionImpl'} for i in range(n)]}}
   (self.d/(s+'.json')).write_text(json.dumps(obj))
 def test_rejects_nonflash_ar_and_incomplete_stages(self):
  self.assertTrue(p.exists(),'independent backend auditor is missing')
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  self.assertEqual(m.check(self.d,smoke=True)['status'],'complete')
  q=self.d/'ar_r0.json';x=json.loads(q.read_text());x['worker_before']['attention'][0]['impl']='FlexAttentionImpl';q.write_text(json.dumps(x))
  with self.assertRaises(ValueError):m.check(self.d,smoke=True)
if __name__=='__main__':unittest.main()
