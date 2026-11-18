import hashlib,json
from pathlib import Path
from dflare_8b_native_adapter import contracts as c
root=Path('/home/lyh/qwen3vl8b_eval_20261027/dflare_8b_work_20261029')
receipt=root/'CPU_PREFLIGHT_8B_V3.json'
r=json.loads(receipt.read_text())
assert r['status']=='complete' and r['algorithm']=='dflare' and r['step']==75000
assert r['CPU_test_count']>=24 and r['CPU_failures']==r['CPU_errors']==r['CPU_skips']==0
assert r['all42_values_bitwise_equal'] and r['tensor_count']==42 and r['own_source_sha256']==c.own_sources()
paths={Path(c.CHECKPOINT),receipt}
paths.update(c.HERE/name for name in c.own_sources())
paths.update(Path(entry['remote_path']) for entry in json.loads((c.HERE/'SOURCE_PROVENANCE.json').read_text()).values())
paths.update(Path(entry['path']) for entry in json.loads((c.HERE/'OFFICIAL_SOURCE_PINS.json').read_text()).values())
paths.update(Path(path) for path in r['export_artifacts'])
bootstrap=c.HERE/'bootstrap/sitecustomize.py';assert bootstrap in paths
manifest={'method':'dflare','scope':'Qwen3-VL-8B step75000 DFlare, learned 3-layer fusion and dedicated context KV, interleaved spatial MRoPE, stock target verifier',
 'bootstrap':str(bootstrap),'checkpoint':c.CHECKPOINT,
 'environment':{'DFLARE_NATIVE_ENABLE':'1','DFLARE_NATIVE_CPU_RECEIPT':str(receipt),'DFLARE_NATIVE_CPU_RECEIPT_SHA256':c.sha(receipt)},
 'files':{str(p):c.sha(p) for p in sorted(paths)}}
out=root/'runtime_manifest_v1.json'
with out.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps({'status':'complete','files':len(paths),'receipt_sha256':c.sha(receipt)}))
