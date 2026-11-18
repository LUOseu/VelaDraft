"""Actual trained conv/selector CPU checks on synthetic inputs; no target forward."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch
from torch import nn
from safetensors.torch import load_file

from . import contracts as c
from .conv import GroupedDynamicCausalConv
from .selector import CandidateSelector
from .logic import selector_walk
from .test_logic import source_module,canonical_walk


@torch.inference_mode()
def actual_weight_functional_proof():
    torch.manual_seed(915)
    state=load_file(str(c.HERE/'hf_step75000_v1/model.safetensors'),device='cpu')
    c.validate_state(state)
    ref_conv=source_module('actual_reference_conv','canonical_source/dflash2/conv.py').GroupedDynamicCausalConv
    convs={}
    for layer in range(3):
        for kind in ('attention_conv','mlp_conv'):
            prefix=f'layers.{layer}.{kind}.'
            tensors={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            actual=GroupedDynamicCausalConv(4096,block_size=8,kernel_size=2,group_size=16).bfloat16()
            reference=ref_conv(4096,block_size=8,kernel_size=2,group_size=16).bfloat16()
            actual.load_state_dict(tensors,strict=True);reference.load_state_dict(tensors,strict=True)
            x=torch.randn(16,4096,dtype=torch.bfloat16);out=torch.randn_like(x)
            a,k=actual.prepare(x);b,j=reference.prepare(x)
            c.require(torch.equal(a,b) and torch.equal(k,j),'Actual trained input conv differs: '+prefix)
            c.require(torch.equal(actual.finish(out,k),reference.finish(out,j)),'Actual trained output conv differs: '+prefix)
            identity=torch.zeros_like(actual.base_kernel);identity[:,0]=1
            convs[prefix]={'prepared_values_bitwise_equal':True,'output_values_bitwise_equal':True,
                'input_output_kernel_projection_values_used':True,
                'base_identity_max_deviation':float((actual.base_kernel.float()-identity.float()).abs().max()),
                'projection_nonzero_count':int(actual.kernel_projection.weight.count_nonzero())}
    selector=CandidateSelector(vocab_size=151936,hidden_size=4096,rank=256,top_k=16).bfloat16()
    ref_type=source_module('actual_reference_selector','canonical_source/dflash2/selector.py').CandidateSelector
    reference=ref_type(vocab_size=151936,hidden_size=4096,rank=256,top_k=16).bfloat16()
    tensors={k[len('candidate_selector.'):]:v for k,v in state.items() if k.startswith('candidate_selector.')}
    selector.load_state_dict(tensors,strict=True);reference.load_state_dict(tensors,strict=True)
    hidden=torch.randn(1,8,4096,dtype=torch.bfloat16)
    unary=torch.randn(1,7,151936,dtype=torch.bfloat16)
    anchor=torch.tensor([920])
    class PreparedUnaryHead:
        def __call__(self,x):return unary
    block=torch.full((1,8),151669,dtype=torch.long);block[:,0]=anchor
    expected=canonical_walk()(NS(block_size=8,candidate_selector=reference),NS(lm_head=PreparedUnaryHead()),hidden,block)
    chosen=selector_walk(selector,unary,hidden[:,1:],anchor)
    c.require(torch.equal(chosen,expected),'Actual trained selector differs from canonical walk')
    ids,scores=selector.select(unary[:,:1],hidden[:,1:2],anchor[:,None])
    ref_ids,ref_scores=reference.select(unary[:,:1],hidden[:,1:2],anchor[:,None])
    c.require(torch.equal(ids,ref_ids) and torch.equal(scores,ref_scores),'Actual trained rank256/top16 selector scores differ')
    c.require(not torch.cuda.is_initialized(),'Actual CPU component proof initialized CUDA')
    return {'input_kind':'synthetic BF16 hidden/unary inputs with actual trained tensors',
        'trained_conv_tensor_count':12,'trained_selector_tensor_count':3,'convolutions':convs,
        'actual_rank':256,'actual_top_k':16,'selector_scores_bitwise_equal':True,
        'canonical_predecessor_walk_bitwise_equal':True,'selected_tokens':chosen.tolist(),
        'full_backbone_forward_executed':False,'official_target_or_head_forward_executed':False,
        'cuda_initialized':False,'GPU_numerical_or_performance_gate_passed':False}


class ActualWeightsTests(unittest.TestCase):
    def test_actual_all51_export_values_match_original_checkpoint(self):
        from .export_checkpoint import verify_tensor_values
        original=torch.load(c.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
        restored=load_file(str(c.HERE/'hf_step75000_v1/model.safetensors'),device='cpu')
        c.validate_state(original['model']);c.validate_state(restored)
        proof=verify_tensor_values(original['model'],restored)
        self.assertEqual(len(proof),51);self.assertFalse(torch.cuda.is_initialized())

    def test_actual_trained_conv_and_selector_match_canonical_source(self):
        result=actual_weight_functional_proof()
        self.assertEqual(result['trained_conv_tensor_count']+result['trained_selector_tensor_count'],15)
        self.assertTrue(result['canonical_predecessor_walk_bitwise_equal'])

if __name__=='__main__':unittest.main(verbosity=2)
