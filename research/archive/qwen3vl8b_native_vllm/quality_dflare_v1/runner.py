"""Separate Qwen3-VL-8B native AR/SD real-gold scoring, not timed benchmarking."""
import argparse,copy,gc,hashlib,io,json,os,time,traceback,uuid
from pathlib import Path
from .data import load_task
from .scoring import score
from .legacy_scoring import extract_short_answer

TASKS=('textvqa','docvqa','chartqa')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('ar','sd'),required=True);parser.add_argument('--method',choices=('dflare',),required=True);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--gpu',type=int,default=3);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 from vlm_dflash.eval.native import verify_plan,verify_preservation,engine_args
 from vlm_dflash.eval.native_resources import GPULease,normalize_uuid
 from vlm_dflash.eval.native_runtime import worker_metadata,numeric_state
 plan_path=Path(os.environ['VELADRAFT_NATIVE_PLAN']);plan=json.loads(plan_path.read_text())
 if digest(plan_path)!=os.environ.get('VELADRAFT_NATIVE_PLAN_SHA256') or plan['method']!=args.method or os.environ.get('VELADRAFT_NATIVE_ARM')!=args.arm:raise ValueError('Plan SHA or arm/method mismatch')
 root=args.directory.resolve();root.mkdir(parents=True,exist_ok=False)
 report={'schema':'qwen3vl8b-quality-arm-v1','status':'preparing','arm':args.arm,'method':args.method,
   'performance_claim':False,'paper_ready':False,'gold_source':'dataset human reference','primary_scoring':'full decoded text',
   'secondary_scoring':'first nonempty line; fixed legacy extractor','max_new_tokens':128,'temperature':0,'seed':43,'smoke':args.smoke,
   'plan_sha256':digest(plan_path),'quality_protocol_sha256':digest(Path(__file__).with_name('QUALITY_PROTOCOL_V1.md')),
   'quality_source_manifest_sha256':digest(Path(__file__).with_name('SOURCE_SHA256.txt')),
   'data_manifest_sha256':digest(Path(__file__).with_name('DATA_MANIFEST.json')),'task_counts':{},'task_scores_full':{},
   'task_scores_extracted':{},'failures':0,'started_unix':time.time()}
 out=root/'quality.json';raw=root/'predictions.jsonl';save(out,report);llm=None
 try:
  verify_plan(plan)
  tasks={task:load_task(task) for task in TASKS}
  if any(pins['count']!=256 or len(rows)!=256 for pins,rows in tasks.values()):raise ValueError('Frozen quality scope differs')
  import torch
  from PIL import Image
  from transformers import AutoProcessor
  from vllm import LLM,SamplingParams
  if torch.cuda.is_initialized():raise ValueError('CUDA initialized before construction')
  report['preservation']=verify_preservation(plan);report['numeric_parent']=numeric_state()
  with GPULease(args.gpu,root/'resources.jsonl') as lease:
   if lease.uuid!=plan['gpu_uuid']:raise ValueError('Quality card differs from pinned native plan')
   kwargs=engine_args(plan,args.arm);report['engine_args']=copy.deepcopy(kwargs)
   proc=AutoProcessor.from_pretrained(plan['target'],local_files_only=True,min_pixels=plan['min_pixels'],max_pixels=plan['max_pixels'])
   report['status']='loading';save(out,report);llm=LLM(**copy.deepcopy(kwargs));torch.cuda.synchronize()
   if normalize_uuid(str(torch.cuda.get_device_properties(0).uuid))!=lease.uuid:raise ValueError('Actual GPU differs from lease')
   report['gpu_uuid']=lease.uuid;report['worker_before']=worker_metadata(llm,args.arm,plan)
   sampling=SamplingParams(temperature=0,max_tokens=128,seed=43,stop_token_ids=plan['stop_token_ids'],ignore_eos=False)
   with raw.open('x') as journal:
    def generate(task,record,phase):
     prompt=proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':record['question']}]}],tokenize=False,add_generation_prompt=True)
     image_bytes=Path(record['image']).read_bytes()
     if hashlib.sha256(image_bytes).hexdigest()!=record['image_sha256']:raise ValueError('Image bytes changed during quality request')
     with Image.open(io.BytesIO(image_bytes)) as src:
      image=src.convert('RGB');image.load()
      response=llm.generate([{'prompt':prompt,'multi_modal_data':{'image':image},'multi_modal_uuids':{'image':[uid:=str(uuid.uuid4())]}}],sampling,use_tqdm=False)[0]
      image.close()
     torch.cuda.synchronize();v=response.outputs[0];text=v.text;extracted=extract_short_answer(text)
     row={'task':task,'id':record['id'],'group_id':record['group_id'],'phase':phase,'status':'complete',
       'prediction':text,'tokens':list(v.token_ids),'prompt_token_ids':list(response.prompt_token_ids),
       'gold_answers':record['answers'],'score_full':score(task,text,record['answers']),
       'extracted':extracted,'score_extracted':score(task,extracted,record['answers']),
       'image_sha256':record['image_sha256'],'question':record['question'],'image_uuid':uid,
       'finish_reason':v.finish_reason,'stop_reason':v.stop_reason,'method':args.method,'arm':args.arm}
     journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno());return row
    for task,(pins,records) in tasks.items():
     report['status']='warming_'+task;save(out,report)
     for record in records[:4]:generate(task,record,'warmup')
     report['status']='evaluating_'+task;save(out,report);results=[]
     for record in (records[:1] if args.smoke else records):
      results.append(generate(task,record,'quality'))
      if len(results)%16==0:report['task_counts'][task]=len(results);save(out,report)
     count=1 if args.smoke else 256
     if len(results)!=count:raise ValueError('Incomplete quality task')
     report['task_counts'][task]=len(results);report['task_scores_full'][task]=100*sum(x['score_full'] for x in results)/count
     report['task_scores_extracted'][task]=100*sum(x['score_extracted'] for x in results)/count;save(out,report)
   report['worker_after']=worker_metadata(llm,args.arm,plan)
   if report['worker_after']!=report['worker_before']:raise ValueError('Worker state changed')
   verify_plan(plan);llm.llm_engine.engine_core.shutdown();llm=None
  report['resource_evidence']=lease.validate();report['raw_sha256']=digest(raw);report['status']='complete'
 except BaseException as error:
  report.update(status='failed',error=str(error),traceback=traceback.format_exc());report['failures']+=1
 finally:
  if llm is not None:
   try:llm.llm_engine.engine_core.shutdown()
   except Exception as error:report['shutdown_error']=str(error)
   llm=None;gc.collect()
  report['completed_unix']=time.time();save(out,report)
 print(json.dumps({'status':report['status'],'arm':args.arm,'method':args.method,'scores':report['task_scores_full'],'error':report.get('error')}))
 return 0 if report['status']=='complete' else 1
if __name__=='__main__':raise SystemExit(main())
