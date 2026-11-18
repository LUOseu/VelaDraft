import json
from pathlib import Path
import sqlite3

root=Path('/root/autodl-tmp/qwen36_rootcause_20261101')
old=Path('/root/autodl-tmp/qwen36_tp1_20261101')
expected=json.loads((root/'target_weight_manifest.json').read_text())
plan=json.loads((old/'results/dflash_tp1_a3.evidence/plan.json').read_text())
db=sqlite3.connect('file:'+str(old/'cache/digests.sqlite')+'?mode=ro',uri=True)
count=0
for name,value in expected.items():
    path=Path(plan['target'])/name;stat=path.stat()
    assert plan['files'][str(path)]==value['sha256'] and stat.st_size==value['size']
    row=db.execute('SELECT sha FROM digests WHERE dev=? AND ino=? AND size=? AND mtime=? AND ctime=?',
                   (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)).fetchone()
    assert row and row[0]==value['sha256'];count+=1
from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
file=next((root/'moe_tune_k2_a1/tuned_configs').glob('*.json'))
table=json.loads(file.read_text())
changed=[m for m in range(1,4097) if table[str(m)]!=get_default_config(m,256,512,2048,8,None)]
assert changed==[1,3],changed
report={'status':'verified','target_revision':'995ad96eacd98c81ed38be0c5b274b04031597b0','weight_shards':count,
        'changed_moe_batch_sizes':changed,'all_other_batch_sizes_1_to_4096_default':True}
(root/'FINAL_CONFIG_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
