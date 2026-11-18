import hashlib,importlib,json,os
from pathlib import Path
root=Path('/home/lyh/qwen3vl8b_eval_20261027/jetspec_8b_work_20261029');p=root/'jetspec_adapter'
log=root/'cpu_test_v3.log';last=[]
for line in log.read_text(errors='replace').splitlines():
 try:
  x=json.loads(line)
  if isinstance(x,dict) and x.get('scope','').startswith('local shared-block JetSpec'):last.append(x)
 except (ValueError,AttributeError):pass
assert len(last)==1 and last[0]['tests_run']>=25 and last[0]['failures']==last[0]['errors']==0
assert last[0]['cuda_initialized'] is False and last[0]['model_construction'] is False
receipt=json.loads((p/'hf_export_step75000_v1/export_receipt.json').read_text())
assert receipt['algorithm']=='jetspec' and receipt['all_36_tensors_bitwise_equal'] and receipt['tensor_count']==36
assert receipt['source_sha256']=='afb8fdb1d14df3b3d003003fa12a30d29e2d6b08fa7cc45d24728be929437391'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
frozen=json.loads((p/'source_bindings.json').read_text());paths={root/name for name in frozen['frozen']}
paths.update(p/name for name in ('official_jetspec_adapter.py','export_jetspec.py','test_jetspec_adapter.py','checkpoint_metadata.json','source_bindings.json','REMOTE_FROZEN_BEFORE.json','bootstrap/sitecustomize.py','source_run/config.json','source_run/contract.json','source_evidence/draft_model.py','source_evidence/gpu_model_runner.py'))
paths.update(p/'hf_export_step75000_v1'/name for name in ('config.json','model.safetensors','export_receipt.json'))
checkpoint=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/jetspec/ckpt_final.pt');paths.update((checkpoint,log,Path(__file__).resolve()))
for entry in frozen['upstream'].values():
 module=importlib.import_module(entry['module']);path=Path(module.__file__)
 if 'relative_path' in entry:path=path.parent/entry['relative_path']
 raw=path.read_bytes();content=raw.rstrip(b'\n') if entry['strip_final_newlines'] else raw
 assert hashlib.sha256(content).hexdigest()==entry['sha256'],str(path)
 paths.add(path)
for name,expected in frozen['frozen'].items():assert sha(root/name)==expected,name
actual={str(path.resolve()):sha(path) for path in sorted(paths)}
assert actual[str(checkpoint)]==receipt['source_sha256']
assert actual[str(p/'hf_export_step75000_v1/model.safetensors')]==receipt['model_safetensors_sha256']
assert actual[str(p/'hf_export_step75000_v1/config.json')]==receipt['config_sha256']
manifest={'method':'jetspec','scope':'Qwen3-VL-8B step75000 local causal shared-block JetSpec, no tree; official Flash and stock greedy verifier','bootstrap':str(p/'bootstrap/sitecustomize.py'),'checkpoint':str(checkpoint),'environment':{'VLLM_OFFICIAL_JETSPEC_LOCAL_BLOCK_ADAPTER':'1'},'files':actual}
out=root/'runtime_manifest_v3.json'
with out.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps({'status':'complete','files':len(actual),'CPU_tests':last[0]['tests_run']}))
