"""Fake-only orchestration regression, not GPU or quality evidence."""
import contextlib,hashlib,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from . import runner

class FakeRunnerTest(unittest.TestCase):
 def test_smoke_writes_complete_three_task_journal_without_gpu(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);image=root/'x.jpg';image.write_bytes(b'synthetic-image');h=hashlib.sha256(image.read_bytes()).hexdigest()
   records=[{'id':str(i),'group_id':str(i),'question':'q','image':str(image),'image_sha256':h,'answers':['yes']} for i in range(256)]
   plan={'method':'dflash2','gpu_uuid':'GPU-00000000-0000-0000-0000-000000000001','target':'fake','min_pixels':1,'max_pixels':2,'stop_token_ids':[1]}
   p=root/'plan.json';p.write_text(json.dumps(plan));ph=hashlib.sha256(p.read_bytes()).hexdigest()
   native=types.ModuleType('vlm_dflash.eval.native');native.verify_plan=lambda p:None;native.verify_preservation=lambda p:{'all_values_bitwise_equal':True};native.engine_args=lambda p,a:{'model':'fake'}
   class Lease:
    def __init__(self,index,path):self.uuid=plan['gpu_uuid']
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def validate(self):return {'status':'complete'}
   resources=types.ModuleType('vlm_dflash.eval.native_resources');resources.GPULease=Lease;resources.normalize_uuid=lambda x:x
   runtime=types.ModuleType('vlm_dflash.eval.native_runtime');runtime.worker_metadata=lambda *a:{'numeric':{}};runtime.numeric_state=lambda:{}
   class FakeImage:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def convert(self,*args):return self
    def load(self):pass
    def close(self):pass
   pil=types.ModuleType('PIL');pil.Image=types.SimpleNamespace(open=lambda *a:FakeImage())
   class Processor:
    @classmethod
    def from_pretrained(cls,*a,**kw):return cls()
    def apply_chat_template(self,*a,**kw):return 'prompt'
   tf=types.ModuleType('transformers');tf.AutoProcessor=Processor
   class Engine:
    def __init__(self,**kwargs):self.llm_engine=types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=lambda:None))
    def generate(self,*args,**kwargs):return [types.SimpleNamespace(prompt_token_ids=[4],outputs=[types.SimpleNamespace(text='yes',token_ids=[5],finish_reason='stop',stop_reason=None)])]
   vllm=types.ModuleType('vllm');vllm.LLM=Engine;vllm.SamplingParams=lambda **kw:kw
   torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_initialized=lambda:False,synchronize=lambda:None,get_device_properties=lambda n:types.SimpleNamespace(uuid=plan['gpu_uuid']))
   modules={'vlm_dflash':types.ModuleType('vlm_dflash'),'vlm_dflash.eval':types.ModuleType('vlm_dflash.eval'),'vlm_dflash.eval.native':native,'vlm_dflash.eval.native_resources':resources,'vlm_dflash.eval.native_runtime':runtime,'torch':torch,'PIL':pil,'transformers':tf,'vllm':vllm}
   args=['runner','--arm','ar','--method','dflash2','--directory',str(root/'out'),'--smoke']
   with patch.dict(sys.modules,modules),patch.dict('os.environ',{'VELADRAFT_NATIVE_PLAN':str(p),'VELADRAFT_NATIVE_PLAN_SHA256':ph,'VELADRAFT_NATIVE_ARM':'ar'}),patch.object(sys,'argv',args),patch.object(runner,'load_task',side_effect=lambda t:({'count':256},records)),patch.object(runner,'score',return_value=1.),patch.object(runner,'extract_short_answer',side_effect=lambda x:x):
    with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(),0)
   report=json.loads((root/'out/quality.json').read_text());rows=[json.loads(x) for x in (root/'out/predictions.jsonl').read_text().splitlines()]
   self.assertEqual(report['status'],'complete');self.assertTrue(report['smoke']);self.assertEqual(len(rows),15)
   self.assertEqual(report['task_counts'],{t:1 for t in runner.TASKS});self.assertEqual(len({x['image_uuid'] for x in rows}),15)
if __name__=='__main__':unittest.main()
