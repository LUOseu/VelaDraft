"""Independently inspect actual measured attention implementations, no GPU calls."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def check(directory,smoke=False):
 directory=Path(directory)
 stages=['reference','ar_r0','sd_r0'] if smoke else ['reference','ar_r0','sd_r0','sd_r1','ar_r1','ar_r2','sd_r2']
 checks={}
 for stage in stages:
  path=directory/(stage+'.json');data=json.loads(path.read_text())
  if data.get('status')!='complete':raise ValueError(f'{stage} incomplete')
  info=data.get('worker_before',{});modules=info.get('attention',[])
  expected=39 if stage.startswith('sd_') else 36
  if len(modules)!=expected:raise ValueError(f'{stage}: expected {expected} recorded language attention modules, found {len(modules)}')
  invalid=[x for x in modules if not isinstance(x.get('impl'),str) or not x['impl'].endswith('.flash_attn.FlashAttentionImpl')]
  if invalid:raise ValueError(f'{stage}: non-Flash attention implementations: {invalid}')
  if 'worker_after' in data and data['worker_after']!=info:raise ValueError(f'{stage}: worker metadata changed')
  checks[stage]={'count':len(modules),'all_flash':True,'source_sha256':sha(path)}
 return {'status':'complete','scope':'observed vLLM language attention module implementations, not continuous GPU exclusivity or whole-model numerical equality','stages':checks}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 try:result=check(a.directory,a.smoke)
 except Exception as error:
  with a.output.open('x') as f:json.dump({'status':'failed','error':str(error)},f,indent=2);f.write('\n')
  raise
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':result['status'],'stages':len(result['stages'])}))
