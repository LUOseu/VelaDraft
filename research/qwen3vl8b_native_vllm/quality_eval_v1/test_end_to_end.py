"""CPU-only synthetic coverage test; never a performance or quality measurement."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import compare_quality as q

class EndToEndTest(unittest.TestCase):
 def test_three_arms_full_scope_and_input_mismatch_fail_closed(self):
  tasks={t:[{'id':t+str(i),'answers':['yes'],'image_sha256':'hash'+str(i),'group_id':'g'+str(i),'question':'q'+str(i)} for i in range(256)] for t in q.TASKS}
  def fake_score(task,text,answers):return float(text==answers[0])
  with tempfile.TemporaryDirectory() as tmp,patch.object(q,'load_task',side_effect=lambda task:({},tasks[task])),patch.object(q,'score',side_effect=fake_score),patch.object(q,'extract_short_answer',side_effect=lambda x:x):
   root=Path(tmp)
   common={'model':'target','dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':8192,'max_num_seqs':1,'max_num_batched_tokens':4096,'gpu_memory_utilization':.78,'enable_prefix_caching':False,'mm_processor_cache_gb':0,'limit_mm_per_prompt':{'image':1},'enforce_eager':False,'generation_config':'vllm','disable_log_stats':True,'mm_processor_kwargs':{}}
   for arm in ('ar','dflash','dfly'):
    folder=root/arm;folder.mkdir();rows=[]
    for task in q.TASKS:
     for phase,items in (('warmup',tasks[task][:4]),('quality',tasks[task])):
      for i,g in enumerate(items):rows.append({'task':task,'phase':phase,'status':'complete','id':g['id'],'group_id':g['group_id'],'prompt_token_ids':[1,2],
       'image_sha256':g['image_sha256'],'gold_answers':g['answers'],'question':g['question'],'prediction':'yes','tokens':[3],
       'score_full':1.,'score_extracted':1.,'extracted':'yes','image_uuid':arm+task+phase+str(i),'finish_reason':'stop'})
    raw=folder/'predictions.jsonl';raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));report={'status':'complete','arm':'ar' if arm=='ar' else 'sd','method':'dflash' if arm=='ar' else arm,
     'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'resource_evidence':{'status':'complete'},'task_scores_full':{t:100. for t in q.TASKS},
     'gpu_uuid':'GPU-test','numeric_parent':{},'worker_before':{'numeric':{}},'engine_args':common,'preservation':{'all_values_bitwise_equal':True},
     'quality_source_manifest_sha256':'source','quality_protocol_sha256':'protocol','data_manifest_sha256':'data','temperature':0,'seed':43,'max_new_tokens':128}
    (folder/'quality.json').write_text(json.dumps(report))
   result=q.compare(root);self.assertEqual(result['physical_calls'],2340);self.assertEqual(result['methods']['dfly']['tasks']['docvqa']['full']['delta_percentage_points'],0)
   folder=root/'dfly';raw=folder/'predictions.jsonl';rows=[json.loads(x) for x in raw.read_text().splitlines()];rows[4]['prompt_token_ids']=[99];raw.write_text(''.join(json.dumps(r)+'\n' for r in rows));report=json.loads((folder/'quality.json').read_text());report['raw_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest();(folder/'quality.json').write_text(json.dumps(report))
   with self.assertRaisesRegex(ValueError,'prompt_token_ids'):q.compare(root)
if __name__=='__main__':unittest.main()
