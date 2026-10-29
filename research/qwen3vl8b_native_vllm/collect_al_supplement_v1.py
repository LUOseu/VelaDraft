"""Collect actual official speculative counters from an untimed supplemental run."""
import argparse,hashlib,json,math
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def summarize(result):
 result=Path(result);data=json.loads(result.read_text())
 if (data.get('status')!='complete' or not data.get('resources_verified') or data.get('timed_pairs')!=145
     or data.get('sessions')!=120 or data.get('repeats')!=1):raise ValueError('Supplemental scope incomplete or GPU resources not verified')
 evidence=Path(data['evidence']);stage=evidence/'sd_r0.json';sd=json.loads(stage.read_text());plan_path=evidence/'plan.json'
 if sha(plan_path)!=data.get('source_plan_sha256'):raise ValueError('Supplemental plan hash mismatch')
 plan=json.loads(plan_path.read_text());sessions=plan.get('sessions',[])
 expected=[(str(s['id']),turn) for s in sessions for turn in range(len(s['questions']))]
 rows=[x for x in sd.get('rows',[]) if x.get('phase')=='timed']
 actual=[(str(row.get('id')),row.get('turn')) for row in rows]
 if (plan.get('method')!=data.get('method') or plan.get('speculative_tokens')!=7 or len(sessions)!=120 or len(expected)!=145 or len(set(expected))!=145
     or actual!=expected or any(row.get('status')!='complete' for row in rows)
     or sd.get('status')!='complete' or sd.get('engine_args',{}).get('disable_log_stats') is not False):raise ValueError('Supplemental source/row scope differs or official stats were not enabled')
 counts=sd.get('timed_official_counters',{});rounds=counts.get('rounds');accepted=counts.get('accepted');proposed=counts.get('proposed')
 if counts.get('status')!='complete' or not all(type(x) in (int,float) and math.isfinite(x) and x>=0 and x==int(x) for x in (rounds,accepted,proposed)) or not rounds or accepted>proposed or proposed>7*rounds:raise ValueError('Official acceptance counters unavailable or inconsistent')
 return {'status':'complete','method':data['method'],'acceptance_length':1+accepted/rounds,'accepted_draft_tokens_per_round':accepted/rounds,
  'draft_rounds':int(rounds),'accepted_draft_tokens':int(accepted),'proposed_draft_tokens':int(proposed),'scored_turns':145,'sessions':120,'repeats':1,
  'definition':'vLLM mean acceptance length = 1 + accepted draft tokens / number of drafts (convention includes a bonus token)',
  'measurement_scope':'separate stats-enabled 145-turn run using same benchmark protocol; not a timed-main-run counter and not a speed measurement',
  'evidence':str(evidence),'result_sha256':sha(result),'sd_stage_sha256':sha(stage)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 try:d=summarize(a.result)
 except Exception as e:
  with a.output.open('x') as f:json.dump({'status':'unavailable','error':str(e)},f,indent=2);f.write('\n')
  raise
 with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps({'status':d['status'],'method':d['method'],'acceptance_length':d['acceptance_length']}))
