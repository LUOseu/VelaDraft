"""Independent canonical CPU oracles for the learned DFly paths."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
from unittest import mock
import torch
from torch import nn
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent

def canonical_method(cls,name,file='dfly/model.py'):
    tree=ast.parse((HERE/'canonical_source'/file).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name)
    env={'torch':torch,'nn':nn,'F':F,'sample':lambda logits:logits.argmax(-1)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(HERE/'canonical_source'/file),'exec'),env)
    return env[name]

def cpu_vllm_config(function):
    def wrapped(*args,**kwargs):
        from vllm.config import VllmConfig,set_current_vllm_config
        from vllm.distributed.parallel_state import (init_distributed_environment,initialize_model_parallel,
            destroy_model_parallel,destroy_distributed_environment)
        # Real CPU-only TP1 metadata: official Linear/ModelWeightParameter need
        # initialized distributed state even when no collective math is required.
        with tempfile.TemporaryDirectory() as temp,set_current_vllm_config(VllmConfig()),mock.patch(
            'torch.cuda._lazy_init',side_effect=AssertionError('CUDA forbidden in CPU fixture')):
            try:
                init_distributed_environment(world_size=1,rank=0,local_rank=0,backend='gloo',
                    distributed_init_method='file://'+temp+'/rendezvous')
                initialize_model_parallel(1,1,backend='gloo')
                return function(*args,**kwargs)
            finally:
                destroy_model_parallel();destroy_distributed_environment()
    return wrapped

class LogicTests(unittest.TestCase):
    def test_fc_plus_live_fusion_matches_canonical_and_is_not_plain_dflash(self):
        self.assertTrue((HERE/'context.py').is_file(),'missing DFly FC-plus-fusion context')
        from .context import build_layer_contexts
        build=canonical_method('Qwen3DFlyDraftModel','_build_layer_contexts')
        project=canonical_method('Qwen3DFlyDraftModel','_project_base_context')
        probs=canonical_method('Qwen3DFlareDraftModel','fusion_probabilities','dflare/model.py')
        torch.manual_seed(281)
        for dtype in (torch.float32,torch.bfloat16):
            raw=torch.randn(5,12288,dtype=dtype)
            fusion=nn.Parameter(torch.randn(3,3,dtype=dtype))
            fc=nn.Linear(12288,4096,bias=False,dtype=dtype);norm=nn.RMSNorm(4096,eps=1e-6,dtype=dtype)
            ref=NS(fc=fc,layer_fusion_weights=fusion,layers=[0,1,2],hidden_norm=norm)
            ref.fusion_probabilities=lambda:probs(ref)
            ref._project_base_context=lambda x:project(ref,x)
            expected=build(ref,raw.reshape(1,5,3,4096))
            actual=build_layer_contexts(raw,fusion,fc,norm)
            for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b.squeeze(0)))
            self.assertFalse(torch.equal(actual[0],norm(fc(raw))))
        with self.assertRaises(ValueError):build_layer_contexts(torch.zeros(5,4096),fusion,fc,norm)

    @cpu_vllm_config
    def test_official_two_norm_three_matrix_correction_matches_canonical(self):
        self.assertTrue((HERE/'correction.py').is_file(),'missing original hidden correction')
        from .correction import HiddenStatesCorrection
        forward=canonical_method('HiddenStatesCorrection','forward')
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        torch.manual_seed(611)
        for dtype in (torch.float32,torch.bfloat16):
            owned=HiddenStatesCorrection(32,32,dtype=dtype)
            ref=NS(hidden_norm=Qwen3RMSNorm(32,eps=1e-6).to(dtype),embed_norm=Qwen3RMSNorm(32,eps=1e-6).to(dtype))
            for n in ('hidden_norm','embed_norm'):getattr(ref,n).weight.data.copy_(getattr(owned,n).weight)
            for n in ('gate_proj','up_proj','down_proj'):
                setattr(ref,n,nn.Linear(64 if n!='down_proj' else 32,32,bias=False,dtype=dtype))
                getattr(owned,n).weight.data.copy_(getattr(ref,n).weight)
            h=torch.randn(3,32,dtype=dtype);e=torch.randn(3,32,dtype=dtype)
            expected=forward(ref,h,e);actual=owned(h,e)
            self.assertTrue(torch.equal(expected,actual));self.assertFalse(torch.equal(actual,h))
            for n in ('hidden_norm','embed_norm','gate_proj','up_proj','down_proj'):
                param=getattr(owned,n).weight;old=param.data.clone();param.data.zero_()
                self.assertFalse(torch.equal(actual,owned(h,e)),n+' is not on the live path');param.data.copy_(old)

    def test_sampler_canonical_slots1to7_and_actual_selected_predecessor(self):
        self.assertTrue((HERE/'sampling.py').is_file(),'missing predecessor-conditioned DFly sampler')
        from .sampling import sequential_greedy
        canonical=canonical_method('Qwen3DFlyDraftModel','_sample_draft_tokens')
        torch.manual_seed(726)
        embedding=nn.Embedding(97,32);head=nn.Linear(32,97,bias=False)
        hidden=torch.randn(1,8,32);anchor=torch.tensor([19]);calls=[]
        def correction(h,e):return h+2*e
        def lookup(ids):calls.append(ids.clone());return embedding(ids)
        target=NS(get_input_embeddings=lambda:embedding,lm_head=head)
        ref=NS(hidden_correction=correction,block_size=8)
        block=torch.zeros(1,8,dtype=torch.long);block[:,0]=anchor
        expected=canonical(ref,target,hidden,block)
        model=NS(model=NS(hidden_correction=correction),embed_input_ids=lookup,compute_logits=head)
        actual=sequential_greedy(model,hidden[0,1:8],anchor)
        self.assertTrue(torch.equal(expected,actual));self.assertEqual(len(calls),7)
        self.assertEqual(calls[0].tolist(),[19])
        for i in range(1,7):self.assertTrue(torch.equal(calls[i],actual[:,i-1]))
        self.assertTrue(torch.equal(block[:,0],anchor));self.assertFalse(torch.equal(actual,head(hidden[:,1:]).argmax(-1)))

    def test_exact42_schema_retains_fc_and_all_six_added_tensors(self):
        self.assertTrue((HERE/'contracts.py').is_file(),'missing strict42 DFly contracts')
        from . import contracts as c
        state={k:torch.empty(v['shape'],device='meta',dtype=torch.bfloat16) for k,v in c.SCHEMA.items()}
        c.validate_state(state);self.assertEqual(len(state),42);self.assertIn('fc.weight',state)
        for key in ['fc.weight','layer_fusion_weights']+[x for x in state if x.startswith('hidden_correction.')]:
            with self.assertRaises(ValueError):c.validate_state({k:v for k,v in state.items() if k!=key})
        runtime=c.runtime_parameter_schema();self.assertEqual(len(runtime),33)
        self.assertEqual(runtime['model.hidden_correction.gate_proj.weight'],[4096,8192])
        self.assertNotIn('model.hidden_correction.gate_up_proj.weight',runtime)
        params={k:torch.empty(v,device='meta',dtype=torch.bfloat16) for k,v in runtime.items()};c.validate_runtime_parameters(params)
        params['model.fc.weight']=torch.empty(4096,12288,device='meta')
        with self.assertRaises(ValueError):c.validate_runtime_parameters(params)

if __name__=='__main__':unittest.main(verbosity=2)
