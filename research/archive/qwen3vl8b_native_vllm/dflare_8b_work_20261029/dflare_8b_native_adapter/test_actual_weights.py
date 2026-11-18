"""Actual saved fusion/dedicated projections versus canonical source on CPU."""
from types import SimpleNamespace as NS
import unittest
import torch
from torch import nn
from safetensors.torch import load_file
from . import contracts as c
from .test_logic import canonical_method
from .test_integration import cpu_vllm_config
from .context import build_layer_contexts,project_context

@torch.inference_mode()
@cpu_vllm_config
def actual_weight_functional_proof():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm
    torch.manual_seed(191)
    state=load_file(str(c.HERE/'hf_step75000_v1/model.safetensors'),device='cpu');c.validate_state(state)
    raw=torch.randn(9,12288,dtype=torch.bfloat16);query=torch.randn(1,8,4096,dtype=torch.bfloat16)
    fusion=nn.Parameter(state['layer_fusion_weights'],requires_grad=False)
    norm=RMSNorm(4096,eps=1e-6,dtype=torch.bfloat16)
    norm.weight.data.copy_(state['hidden_norm.weight'])
    canonical_norm=Qwen3RMSNorm(4096,eps=1e-6).bfloat16();canonical_norm.weight.data.copy_(state['hidden_norm.weight'])
    ref=NS(target_hidden_size=4096,num_context_layers=3,layers=[0,1,2],layer_fusion_weights=fusion,hidden_norm=canonical_norm)
    prepare=canonical_method('Qwen3DFlareDraftModel','_prepare_target_hidden')
    probs=canonical_method('Qwen3DFlareDraftModel','fusion_probabilities')
    ref.fusion_probabilities=lambda:probs(ref)
    build=canonical_method('Qwen3DFlareDraftModel','_build_layer_contexts')
    reference_contexts=build(ref,prepare(ref,raw[None]))
    actual_contexts=build_layer_contexts(raw,fusion,norm)
    kv=canonical_method('Qwen3DFlareAttention','_context_noise_kv')
    details={}
    for i in range(3):
        expected=reference_contexts[i].squeeze(0);actual=actual_contexts[i]
        c.require(torch.equal(expected,actual),'Actual saved fusion/hidden norm differs from canonical layer'+str(i))
        attention=NS(separate_context_kv=True,head_dim=128)
        for kind in ('k_proj','v_proj','k_proj_target','v_proj_target'):
            projection=nn.Linear(4096,1024,bias=False,dtype=torch.bfloat16)
            projection.weight.data.copy_(state[f'layers.{i}.self_attn.{kind}.weight']);setattr(attention,kind,projection)
        ref_k,ref_v=kv(attention,query,expected[None])
        actual_k,actual_v=project_context(actual,attention.k_proj_target,attention.v_proj_target)
        c.require(torch.equal(ref_k[0,:9],actual_k) and torch.equal(ref_v[0,:9],actual_v),'Actual dedicated targetKV differs')
        c.require(torch.equal(ref_k[0,9:],attention.k_proj(query).view(8,8,128)),'Actual noise key route differs')
        c.require(torch.equal(ref_v[0,9:],attention.v_proj(query).view(8,8,128)),'Actual noise value route differs')
        c.require(not torch.equal(actual_k,attention.k_proj(actual).view(9,8,128)),'Saved independent key projection was lost')
        details[str(i)]={'context_bitwise_equal':True,'target_key_value_bitwise_equal':True,
            'noise_key_value_bitwise_equal':True,'saved_target_and_noise_K_differ_on_input':True}
    c.require(not torch.cuda.is_initialized(),'Actual CPU proof initialized CUDA')
    return {'actual_trained_extra_tensor_count':7,'actual_live_fusion_dtype':str(fusion.dtype),
        'actual_live_fusion_probabilities':probs(ref).float().tolist(),'per_layer':details,
        'canonical_source_methods_executed':['_prepare_target_hidden','fusion_probabilities','_build_layer_contexts','_context_noise_kv'],
        'raw_feature_shape':list(raw.shape),'query_shape':list(query.shape),
        'input_kind':'synthetic BF16 raw/query input with actual saved BF16 tensors',
        'official_hidden_RMSNorm_matches_canonical_HF_RMSNorm_bitwise_on_fixture':True,
        'target_or_head_forward_executed':False,'full_backbone_forward_executed':False,'cuda_initialized':False}

class ActualWeightsTests(unittest.TestCase):
    def test_actual_all42_export_values_match_checkpoint(self):
        from .export_checkpoint import verify_tensor_values
        original=torch.load(c.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
        restored=load_file(str(c.HERE/'hf_step75000_v1/model.safetensors'),device='cpu')
        c.validate_state(original['model']);c.validate_state(restored)
        self.assertEqual(len(verify_tensor_values(original['model'],restored)),42)
    def test_actual_saved_fusion_and_six_context_matrices_execute(self):
        proof=actual_weight_functional_proof()
        self.assertEqual(proof['actual_trained_extra_tensor_count'],7)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
