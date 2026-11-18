"""Saved 42-tensor CPU export and learned-path numerical proof against canonical."""
from types import SimpleNamespace as NS
import unittest
import torch
from torch import nn
from .test_logic import HERE,canonical_method,cpu_vllm_config

@torch.inference_mode()
@cpu_vllm_config
def actual_weight_functional_proof():
    from . import contracts as c
    from .context import build_layer_contexts,project_shared_context
    from .correction import HiddenStatesCorrection
    from .sampling import sequential_greedy
    from safetensors.torch import load_file
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm
    state=load_file(str(c.EXPORT/'model.safetensors'),device='cpu');c.validate_state(state)
    torch.manual_seed(927)
    raw=torch.randn(9,12288,dtype=torch.bfloat16);query=torch.randn(1,8,4096,dtype=torch.bfloat16)
    fc=nn.Linear(12288,4096,bias=False,dtype=torch.bfloat16);fc.weight.data.copy_(state['fc.weight'])
    norm=RMSNorm(4096,eps=1e-6,dtype=torch.bfloat16);norm.weight.data.copy_(state['hidden_norm.weight'])
    canonical_norm=Qwen3RMSNorm(4096,eps=1e-6).bfloat16();canonical_norm.weight.data.copy_(state['hidden_norm.weight'])
    fusion=nn.Parameter(state['layer_fusion_weights'],requires_grad=False)
    ref=NS(fc=fc,hidden_norm=canonical_norm,layers=[0,1,2],layer_fusion_weights=fusion)
    probs=canonical_method('Qwen3DFlareDraftModel','fusion_probabilities','dflare/model.py')
    ref.fusion_probabilities=lambda:probs(ref)
    project=canonical_method('Qwen3DFlyDraftModel','_project_base_context')
    ref._project_base_context=lambda x:project(ref,x)
    expected=canonical_method('Qwen3DFlyDraftModel','_build_layer_contexts')(ref,raw.reshape(1,9,3,4096))
    actual=build_layer_contexts(raw,fusion,fc,norm);details={}
    kv=canonical_method('Qwen3DFlashAttention','_context_noise_kv','dflash/draft_model.py')
    for i in range(3):
        c.require(torch.equal(actual[i],expected[i][0]),'Actual saved FC/fusion context differs on layer '+str(i))
        attention=NS(head_dim=128,q_size=4096,kv_size=1024)
        for kind in ('k','v'):
            layer=nn.Linear(4096,1024,bias=False,dtype=torch.bfloat16)
            layer.weight.data.copy_(state[f'layers.{i}.self_attn.{kind}_proj.weight']);setattr(attention,kind+'_proj',layer)
        attention.qkv_proj=nn.Linear(4096,6144,bias=False,dtype=torch.bfloat16)
        attention.qkv_proj.weight.data.copy_(torch.cat([state[f'layers.{i}.self_attn.{kind}_proj.weight'] for kind in ('q','k','v')]))
        keys,values=kv(attention,query,expected[i]);ak,av=project_shared_context(actual[i],attention)
        c.require(torch.equal(ak,keys[0,:9]) and torch.equal(av,values[0,:9]),'Saved shared context KV differs')
        c.require(torch.equal(keys[0,9:],attention.k_proj(query).view(8,8,128))
            and torch.equal(values[0,9:],attention.v_proj(query).view(8,8,128)),'Shared query/noise KV differs')
        c.require(not torch.equal(actual[i],norm(fc(raw))),'Fusion residual was lost')
        details[str(i)]={'FC_plus_fusion_context_bitwise_equal':True,'shared_context_KV_bitwise_equal':True,
            'shared_query_KV_bitwise_equal':True,'fusion_residual_is_live':True}
    correction=HiddenStatesCorrection(4096,4096,dtype=torch.bfloat16)
    correction_state={k.removeprefix('hidden_correction.'):v for k,v in state.items() if k.startswith('hidden_correction.')}
    correction.load_state_dict(correction_state,strict=True)
    canonical_correction=NS(hidden_norm=Qwen3RMSNorm(4096,eps=1e-6).bfloat16(),embed_norm=Qwen3RMSNorm(4096,eps=1e-6).bfloat16())
    for name in ('hidden_norm','embed_norm'):getattr(canonical_correction,name).weight.data.copy_(correction_state[name+'.weight'])
    for name in ('gate_proj','up_proj','down_proj'):
        layer=nn.Linear(4096 if name=='down_proj' else 8192,4096,bias=False,dtype=torch.bfloat16)
        layer.weight.data.copy_(correction_state[name+'.weight']);setattr(canonical_correction,name,layer)
    forward=canonical_method('HiddenStatesCorrection','forward')
    embed=nn.Embedding(97,4096,dtype=torch.bfloat16);head=nn.Linear(4096,97,bias=False,dtype=torch.bfloat16)
    h=query[:,1:8].squeeze(0);e=embed(torch.arange(7))
    corrected=correction(h,e);reference=forward(canonical_correction,h,e)
    c.require(torch.equal(corrected,reference),'Actual correction differs from canonical')
    c.require(not torch.equal(corrected,h),'Trained correction was replaced by identity')
    participation={}
    for name,param in correction.named_parameters():
        old=param.data.clone();param.data.zero_();changed=not torch.equal(corrected,correction(h,e));param.data.copy_(old)
        c.require(changed,'Saved correction parameter not on live path: '+name);participation[name]=True
    anchor=torch.tensor([29]);block=torch.zeros(1,8,dtype=torch.long);block[:,0]=anchor
    ref_model=NS(hidden_correction=lambda x,y:forward(canonical_correction,x,y),block_size=8)
    target=NS(get_input_embeddings=lambda:embed,lm_head=head)
    wanted=canonical_method('Qwen3DFlyDraftModel','_sample_draft_tokens')(ref_model,target,query,block)
    seen=[]
    def lookup(ids):seen.append(ids.clone());return embed(ids)
    owned=NS(model=NS(hidden_correction=correction),embed_input_ids=lookup,compute_logits=head)
    selected=sequential_greedy(owned,query[0,1:8],anchor)
    c.require(torch.equal(selected,wanted),'Actual trained sequential correction selected different IDs')
    c.require(torch.equal(seen[0],anchor) and all(torch.equal(seen[i],selected[:,i-1]) for i in range(1,7)),'Actual predecessor feedback lost')
    c.require(not torch.cuda.is_initialized(),'Actual CPU proof initialized CUDA')
    return {'actual_trained_extra_tensor_count':6,'original_FC_executed':True,
        'actual_live_fusion_dtype':str(fusion.dtype),'actual_live_fusion_probabilities':probs(ref).float().tolist(),
        'per_layer':details,'correction_bitwise_equal':True,'correction_nonidentity':True,
        'all_five_correction_parameters_live':participation,'canonical_sequential_suffix_IDs_equal':True,
        'actual_predecessor_feedback_verified':True,'selected_fixture_tokens':selected.tolist(),
        'canonical_source_methods_executed':['_project_base_context','fusion_probabilities','_build_layer_contexts','_context_noise_kv','HiddenStatesCorrection.forward','_sample_draft_tokens'],
        'input_kind':'synthetic BF16 raw/query/embedding/head fixture with actual saved BF16 draft tensors',
        'actual_original_target_embedding_or_head_executed':False,'full_backbone_forward_executed':False,'cuda_initialized':False}

class ActualWeightsTests(unittest.TestCase):
    def test_actual42_export_reopen_is_raw_byte_exact(self):
        self.assertTrue((HERE/'export_checkpoint.py').is_file(),'missing strict actual checkpoint exporter')
        from . import contracts as c
        from .export_checkpoint import verify_tensor_values
        from safetensors.torch import load_file
        self.assertTrue((c.EXPORT/'model.safetensors').is_file(),'missing actual42 export')
        payload=torch.load(c.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
        restored=load_file(str(c.EXPORT/'model.safetensors'),device='cpu')
        c.validate_state(payload['model']);c.validate_state(restored)
        self.assertEqual(len(verify_tensor_values(payload['model'],restored)),42)

    def test_actual_FC_fusion_shared_KV_and_five_correction_weights_execute(self):
        self.assertTrue((HERE/'hf_step75000_v1/model.safetensors').is_file(),'missing actual42 trained modules')
        proof=actual_weight_functional_proof();self.assertEqual(proof['actual_trained_extra_tensor_count'],6)
        self.assertTrue(all(proof['all_five_correction_parameters_live'].values()))

    def test_export_comparison_rejects_byte_dtype_shape_and_key_changes(self):
        self.assertTrue((HERE/'export_checkpoint.py').is_file(),'missing exact-byte verifier')
        from .export_checkpoint import verify_tensor_values
        source={'x':torch.tensor([1,2],dtype=torch.bfloat16)}
        self.assertEqual(len(verify_tensor_values(source,{'x':source['x'].clone()})),1)
        for bad in ({'x':torch.tensor([1,3],dtype=torch.bfloat16)},{'x':source['x'].float()},{'x':source['x'][None]},{}):
            with self.assertRaises(ValueError):verify_tensor_values(source,bad)

if __name__=='__main__':unittest.main(verbosity=2)
