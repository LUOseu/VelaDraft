import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, HERE / relative)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canonical_walk():
    tree = ast.parse((HERE / 'canonical_source/dflash2/model.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3DFlash2DraftModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_sample_draft_tokens')
    env = {'torch': torch, 'nn': nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<canonical-walk>', 'exec'), env)
    return env['_sample_draft_tokens']


class LogicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(143)

    def require(self, filename):
        self.assertTrue((HERE / filename).is_file(), 'missing DFlash2 implementation: ' + filename)
        return source_module('tested_' + filename.replace('.', '_'), filename)

    def test_dynamic_conv_matches_canonical_nonidentity_and_blocks(self):
        impl = self.require('conv.py')
        ref = source_module('canonical_conv', 'canonical_source/dflash2/conv.py')
        for dtype in (torch.float32, torch.bfloat16):
            for n in (1, 8, 16):
                m = impl.GroupedDynamicCausalConv(32, block_size=8, kernel_size=2, group_size=16).to(dtype)
                r = ref.GroupedDynamicCausalConv(32, block_size=8, kernel_size=2, group_size=16).to(dtype)
                with torch.no_grad():
                    m.base_kernel.normal_();m.kernel_projection.weight.normal_(std=.1)
                r.load_state_dict(m.state_dict())
                x = torch.randn(n,32).to(dtype)
                a,k = m.prepare(x);b,j = r.prepare(x)
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(k,j))
                self.assertTrue(torch.equal(m.finish(a,k),r.finish(b,j)))
            x = torch.zeros(16,32,dtype=dtype);x[7]=10
            delta = torch.ones(16,2,2,dtype=dtype)
            base = torch.ones(2,32,dtype=dtype)
            y = impl.grouped_dynamic_conv(x,delta,base,block_size=8,group_size=16)
            self.assertTrue(torch.equal(y[8],torch.zeros_like(y[8])), 'tap crossed anchor block')

    def test_selector_math_and_walk_match_canonical(self):
        impl = self.require('selector.py');logic = self.require('logic.py')
        ref = source_module('canonical_selector','canonical_source/dflash2/selector.py')
        for dtype in (torch.float32,torch.bfloat16):
            s=impl.CandidateSelector(vocab_size=24,hidden_size=32,rank=8,top_k=16).to(dtype)
            r=ref.CandidateSelector(vocab_size=24,hidden_size=32,rank=8,top_k=16).to(dtype)
            r.load_state_dict(s.state_dict())
            hidden=torch.randn(1,8,32).to(dtype);weight=torch.randn(24,32).to(dtype)
            ids=torch.randint(0,24,(1,8));anchor=ids[:,0]
            head=nn.Linear(32,24,bias=False).to(dtype);head.weight.data.copy_(weight)
            expected=canonical_walk()(SimpleNamespace(block_size=8,candidate_selector=r),SimpleNamespace(lm_head=head),hidden,ids)
            actual=logic.selector_walk(s,head(hidden[:,1:]),hidden[:,1:],anchor)
            self.assertTrue(torch.equal(actual,expected))

    def test_selector_uses_selected_predecessor_and_unary_topk_only(self):
        s=self.require('selector.py').CandidateSelector(vocab_size=4,hidden_size=1,rank=1,top_k=2)
        logic=self.require('logic.py')
        with torch.no_grad():
            s.hidden_projection.weight.fill_(1)
            s.predecessor_codebook[:,0]=torch.tensor([2.,-2.,2.,-2.])
            s.successor_codebook[:,0]=torch.tensor([-1.,1.,100.,0.])
        logits=torch.tensor([[[1.,.9,-100.,-100.],[1.,.9,-100.,-100.]]])
        result=logic.selector_walk(s,logits,torch.ones(1,2,1),torch.tensor([0]))
        self.assertEqual(result.tolist(),[[1,0]])

    def test_four_conv_stages_and_residuals_match_reference(self):
        forward=self.require('layer_logic.py').decoder_forward
        conv=self.require('conv.py').GroupedDynamicCausalConv
        class Norm(nn.Module):
            def forward(self,x,residual=None):
                if residual is None:return torch.nn.functional.layer_norm(x,(32,))
                total=x+residual
                return torch.nn.functional.layer_norm(total,(32,)),total
        class Attention(nn.Linear):
            def forward(self,positions,hidden_states):return super().forward(hidden_states)
        layer=SimpleNamespace(input_layernorm=Norm(),post_attention_layernorm=Norm(),
            self_attn=Attention(32,32,bias=False),mlp=nn.Linear(32,32,bias=False),
            attention_conv=conv(32,block_size=8,kernel_size=2,group_size=16),
            mlp_conv=conv(32,block_size=8,kernel_size=2,group_size=16))
        for m in (layer.attention_conv,layer.mlp_conv):
            with torch.no_grad():m.base_kernel.normal_();m.kernel_projection.weight.normal_(std=.02)
        x=torch.randn(8,32);previous=torch.randn(8,32);pos=torch.arange(8)
        for residual in (None,previous):
            original=x if residual is None else x+residual
            h,k=layer.attention_conv.prepare(layer.input_layernorm(original))
            h=original+layer.attention_conv.finish(layer.self_attn(pos,h),k)
            z,k=layer.mlp_conv.prepare(layer.post_attention_layernorm(h))
            expected=h+layer.mlp_conv.finish(layer.mlp(z),k)
            actual,res=forward(layer,pos,x,residual)
            torch.testing.assert_close(actual+res,expected,atol=1e-5,rtol=1e-5)

    def test_strict_schema_rejects_missing_unknown_or_wrong_dtype(self):
        c=self.require('contracts.py')
        state={k:torch.empty(tuple(v['shape']),device='meta',dtype=torch.bfloat16) for k,v in c.SCHEMA.items()}
        c.validate_state(state)
        for wrong in ({k:v for k,v in state.items() if k!='candidate_selector.predecessor_codebook'},
                      dict(state,unknown=torch.empty(1)),dict(state,**{'norm.weight':torch.empty(4096)})):
            with self.assertRaises(ValueError):c.validate_state(wrong)

    def test_export_comparison_rejects_changed_bytes_dtype_shape_and_keys(self):
        verify=self.require('export_checkpoint.py').verify_tensor_values
        source={'x':torch.tensor([1.,2.],dtype=torch.bfloat16)}
        verify(source,{'x':source['x'].clone()})
        for state in ({'x':torch.tensor([1.,3.],dtype=torch.bfloat16)}, {'x':source['x'].float()},
                      {'x':source['x'].view(1,2)}, {}):
            with self.assertRaises(ValueError):verify(source,state)

    def test_runtime_parameter_dtype_and_packed_layout_fail_closed(self):
        c=self.require('contracts.py')
        self.assertTrue(hasattr(c,'runtime_parameter_schema'),'Missing runtime packed-parameter validation')
        schema=c.runtime_parameter_schema()
        self.assertEqual(len(schema),42)
        self.assertEqual(schema['model.layers.0.self_attn.qkv_proj.weight'],[6144,4096])
        self.assertEqual(schema['model.layers.2.mlp.gate_up_proj.weight'],[24576,4096])
        params={k:torch.empty(v,device='meta',dtype=torch.bfloat16) for k,v in schema.items()}
        c.validate_runtime_parameters(params)
        for name in ('model.layers.1.attention_conv.kernel_projection.weight','candidate_selector.hidden_projection.weight'):
            wrong=dict(params);wrong[name]=torch.empty(schema[name],device='meta',dtype=torch.float32)
            with self.assertRaises(ValueError):c.validate_runtime_parameters(wrong)

if __name__=='__main__':unittest.main(verbosity=2)
