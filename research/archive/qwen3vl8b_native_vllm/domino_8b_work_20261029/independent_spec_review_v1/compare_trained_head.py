"""Real 8B Domino head weights vs local canonical code, CPU-only synthetic input.

This is an independent component check, not target-model/GPU/full-decoder parity.
"""
import ast,hashlib,json,os
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
import torch
from torch import nn
from domino_logic import sample_domino

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
source=Path('/home/lyh/VelaDraft-a6000-v2/vlm_dflash/dflash/draft_model.py')
checkpoint=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/domino/ckpt_final.pt')
adapter=Path(__import__('domino_logic').__file__).resolve()
checkpoint_hash=sha(checkpoint)
assert checkpoint_hash=='f3fdfcd68f4eeb1aab74649c71c9cdb5c117f5c5822cf54b9c0f749b1dde85f7'
tree=ast.parse(source.read_text());cl=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='Qwen3DFlashDraftModel')
def get(name):
 node=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name==name)
 env={'torch':torch,'nn':nn,'F':torch.nn.functional,'sample':lambda logits:logits.argmax(-1)}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<canonical>','exec'),env)
 return env[name]
payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False);s=payload['model']
assert payload['algorithm']=='domino' and payload['step']==75000
wi,wh=s['prefix_gru.weight_ih_l0'],s['prefix_gru.weight_hh_l0']
p0,p2=s['embed_proj.0.weight'],s['embed_proj.2.weight']
assert tuple(wi.shape)==(3072,4096) and tuple(wh.shape)==(3072,1024) and tuple(p0.shape)==(256,5120) and tuple(p2.shape)==(151936,256)
torch.manual_seed(20261029);hidden=torch.randn(8,4096,dtype=torch.bfloat16);base=torch.randn(8,151936,dtype=torch.bfloat16)
ids=torch.full((8,),151669,dtype=torch.long);ids[0]=151643;vec=torch.arange(4096,dtype=torch.float32)
def embed(ids):return torch.sin(ids.float().unsqueeze(-1)*0.0001+vec*0.01).to(torch.bfloat16)
def correct(x):return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,p0)),p2)
owner=NS(block_size=8,projector_type='domino',sample_from_anchor=False,pure_draft_prefix_len=1,shift_label=True,gru_hidden_dim=1024,prefix_gru=NS(weight_ih_l0=wi,weight_hh_l0=wh),embed_proj=correct)
owner._domino_gru_step=MethodType(get('_domino_gru_step'),owner)
target=NS(lm_head=lambda _:base.unsqueeze(0),get_input_embeddings=lambda:embed)
expected=get('_sample_draft_tokens')(owner,target,hidden.unsqueeze(0),ids.unsqueeze(0))
actual=sample_domino(hidden,ids,compute_logits=lambda _:base,embed_tokens=embed,weight_ih=wi,weight_hh=wh,projector_in=p0,projector_out=p2)
assert torch.equal(expected,actual) and not torch.cuda.is_initialized()
result={'status':'pass','method':'domino','seed':20261029,'dtype':'bfloat16','checkpoint_sha256':checkpoint_hash,
 'canonical_source_path':str(source),'canonical_source_sha256':sha(source),'adapter_path':str(adapter),'adapter_sha256':sha(adapter),
 'script_sha256':sha(Path(__file__)),'torch_version':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),
 'trained_tensor_shapes':{n:list(s[n].shape) for n in ('prefix_gru.weight_ih_l0','prefix_gru.weight_hh_l0','embed_proj.0.weight','embed_proj.2.weight')},
 'actual_tokens':actual.tolist(),'reference_tokens':expected.tolist(),'exact_match':torch.equal(expected,actual),
 'inputs':'one deterministic synthetic hidden-state matrix, synthetic target logits and synthetic embedding; full 8B dimensions and full vocabulary',
 'limitations':['Single synthetic CPU fixture; no target forward, GPU model loading, cache, attention kernel, target head alias or end-to-end behavior checked.',
 'Local reference source snapshot is hashed but its identity to the original training machine source is not established.']}
out=Path(__file__).with_name('TRAINED_HEAD_CPU_COMPARISON_V1.json')
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
