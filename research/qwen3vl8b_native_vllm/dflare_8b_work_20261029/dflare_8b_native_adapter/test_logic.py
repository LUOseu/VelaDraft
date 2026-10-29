"""Canonical source execution on CPU; no serving model, target forward or CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent

def canonical_method(cls,name):
    tree=ast.parse((HERE/'canonical_source/dflare/model.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name)
    env={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<canonical-dflare>','exec'),env)
    return env[name]

class LogicTests(unittest.TestCase):
    def test_raw_fusion_matches_canonical_live_bf16_softmax(self):
        self.assertTrue((HERE/'context.py').is_file(),'missing native per-layer raw feature context implementation')
        from .context import build_layer_contexts
        prepare=canonical_method('Qwen3DFlareDraftModel','_prepare_target_hidden')
        probabilities=canonical_method('Qwen3DFlareDraftModel','fusion_probabilities')
        build=canonical_method('Qwen3DFlareDraftModel','_build_layer_contexts')
        torch.manual_seed(72)
        for dtype in (torch.float32,torch.bfloat16):
            raw=torch.randn(11,12288,dtype=dtype)
            logits=nn.Parameter(torch.tensor([[2.123,-3.17,.523],[-.8,.91,2.1],[1.01,-1.42,.32]],dtype=dtype))
            norm=nn.RMSNorm(4096,eps=1e-6,dtype=dtype)
            ref=NS(target_hidden_size=4096,num_context_layers=3,layers=[0,1,2],hidden_norm=norm,layer_fusion_weights=logits)
            ref.fusion_probabilities=lambda:probabilities(ref)
            expected=build(ref,prepare(ref,raw.unsqueeze(0)))
            actual=build_layer_contexts(raw,logits,norm)
            for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b.squeeze(0)))
            altered=raw.clone();altered[:,4096:8192]+=1
            self.assertFalse(torch.equal(actual[0],build_layer_contexts(altered,logits,norm)[0]))
        with self.assertRaises(ValueError):build_layer_contexts(torch.zeros(3,4096),logits,norm)

    def test_dedicated_context_kv_and_noise_weights_match_canonical(self):
        self.assertTrue((HERE/'context.py').is_file(),'missing native dual-source K/V implementation')
        from .context import project_context
        torch.manual_seed(831)
        method=canonical_method('Qwen3DFlareAttention','_context_noise_kv')
        for dtype in (torch.float32,torch.bfloat16):
            source=NS(separate_context_kv=True,head_dim=128)
            for kind in ('k','v','k_proj_target','v_proj_target'):
                name=kind if kind.endswith('target') else kind+'_proj'
                setattr(source,name,nn.Linear(4096,1024,bias=False,dtype=dtype))
            context=torch.randn(1,5,4096,dtype=dtype);query=torch.randn(1,8,4096,dtype=dtype)
            keys,values=method(source,query,context)
            k,v=project_context(context.squeeze(0),source.k_proj_target,source.v_proj_target)
            self.assertTrue(torch.equal(k,keys[0,:5]));self.assertTrue(torch.equal(v,values[0,:5]))
            self.assertTrue(torch.equal(keys[0,5:],source.k_proj(query)[0].view(8,8,128)))
            self.assertFalse(torch.equal(k,source.k_proj(context)[0].view(5,8,128)))

    def test_schema_all42_no_fc_and_exact_runtime_parameter_packing(self):
        self.assertTrue((HERE/'contracts.py').is_file(),'missing strict42 contracts')
        from . import contracts as c
        state={k:torch.empty(v['shape'],dtype=torch.bfloat16,device='meta') for k,v in c.SCHEMA.items()}
        c.validate_state(state)
        self.assertEqual(len(state),42);self.assertNotIn('fc.weight',state)
        for changed in (dict(state,**{'fc.weight':torch.empty(4096,12288,device='meta')}),
                        {k:v for k,v in state.items() if k!='layer_fusion_weights'},
                        dict(state,**{'layers.1.self_attn.k_proj_target.weight':torch.empty(512,4096,device='meta')})):
            with self.assertRaises(ValueError):c.validate_state(changed)
        runtime=c.runtime_parameter_schema()
        self.assertEqual(len(runtime),33)
        self.assertEqual(runtime['model.layers.0.self_attn.k_proj_target.weight'],[1024,4096])
        self.assertEqual(runtime['model.layers.0.self_attn.qkv_proj.weight'],[6144,4096])
        self.assertFalse(any('fc' in k for k in runtime))
        params={k:torch.empty(v,device='meta',dtype=torch.bfloat16) for k,v in runtime.items()}
        c.validate_runtime_parameters(params)
        params['model.layer_fusion_weights']=torch.empty(3,3)
        with self.assertRaises(ValueError):c.validate_runtime_parameters(params)

if __name__=='__main__':unittest.main(verbosity=2)

class ExportTests(unittest.TestCase):
    def test_export_bytes_and_duplicates_are_strict(self):
        from .export_checkpoint import verify_tensor_values
        src={'x':torch.tensor([1,2],dtype=torch.bfloat16)}
        self.assertEqual(len(verify_tensor_values(src,{'x':src['x'].clone()})),1)
        for bad in ({'x':torch.tensor([1,3],dtype=torch.bfloat16)},{'x':src['x'].float()},{'x':src['x'][None]},{}):
            with self.assertRaises(ValueError):verify_tensor_values(src,bad)
