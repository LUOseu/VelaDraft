"""Layer-specific DFlare/DFly context on native V2 buffers and CUDA graphs.

The persistent context buffer carries all three raw target layers (6144 wide).
Projection/fusion happens inside the native captured context-KV preparation,
so replay reads current buffer values instead of a stale Python tensor pointer.
All model math uses the existing trained reference and public Torch/vLLM ops.
"""
from functools import wraps
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Model,DFlashQwen3ForCausalLM,DFlashQwen3DecoderLayer)

REFERENCE_DIR=Path(__file__).resolve().parents[1]/'qwen36_native_vllm'
sys.path.insert(0,str(REFERENCE_DIR))
import dflare_adapter as flare_reference
import dfly_adapter as fly_reference
assert Path(flare_reference.__file__).resolve()==REFERENCE_DIR/'dflare_adapter.py'
assert Path(fly_reference.__file__).resolve()==REFERENCE_DIR/'dfly_adapter.py'
sys.path.remove(str(REFERENCE_DIR))
_installed=False


def build_contexts(method,raw,fusion,hidden_norm,fc=None):
    if method=='dflare':
        return flare_reference.build_layer_contexts(raw,fusion,hidden_norm)
    if method=='dfly':
        return fly_reference.build_layer_contexts(raw,fc(raw),fusion,hidden_norm)
    raise ValueError(method)


class DedicatedContextLayer(DFlashQwen3DecoderLayer):
    def __init__(self,vllm_config,*,config,layer_idx,cache_config=None,quant_config=None,prefix=''):
        super().__init__(vllm_config,config=config,layer_idx=layer_idx,
                         cache_config=cache_config,quant_config=quant_config,prefix=prefix)
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        for kind in ('k','v'):
            setattr(self.self_attn,kind+'_proj_target',ColumnParallelLinear(
                2048,512,bias=False,gather_output=False,params_dtype=vllm_config.model_config.dtype,
                quant_config=None,prefix=f'{prefix}.self_attn.{kind}_proj_target',return_bias=False))


class ContextBackbone(DFlashQwen3Model):
    context_method=None
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        super().__init__(vllm_config=vllm_config,start_layer_id=start_layer_id,prefix=prefix)
        method=self.context_method;cfg=self.config.dflash_config;dtype=vllm_config.model_config.dtype
        assert cfg['context_layer_ids']==[1,19,37] and cfg['fusion_trainable'] is True
        assert cfg['separate_context_kv']==(method=='dflare')
        self.layer_fusion_weights=nn.Parameter(torch.empty(3,3,dtype=dtype),requires_grad=False)
        if method=='dflare':
            del self.fc
        else:
            assert method=='dfly' and cfg['enable_hidden_correction'] is True
            assert cfg['hidden_correction_intermediate_size'] is None
            self.hidden_correction=fly_reference.HiddenStatesCorrection(
                2048,2048,dtype=dtype,rms_norm_eps=self.config.rms_norm_eps,
                prefix=f'{prefix}.hidden_correction')

    def _build_fused_kv_buffers(self):
        # The stock shared-context packing is not used by per-layer contexts.
        return None

    def load_weights(self,weights):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        state=dict(weights);params=dict(self.named_parameters())
        extra=flare_reference.DEDICATED if self.context_method=='dflare' else fly_reference.CORRECTION_KEYS
        for name in extra:
            param=params[name];value=state.pop(name)
            assert param.shape==value.shape and param.dtype==value.dtype
            getattr(param,'weight_loader',default_weight_loader)(param,value)
        return super().load_weights(state.items()) | set(extra)

    def precompute_and_store_context_kv(self,raw,context_positions,context_slot_mapping=None):
        assert raw.ndim==2 and raw.shape[1]==6144 and context_positions.ndim==1
        n=raw.shape[0]
        contexts=build_contexts(self.context_method,raw,self.layer_fusion_weights,
                                self.hidden_norm,getattr(self,'fc',None))
        keys=[];values=[]
        for layer,context in zip(self.layers,contexts):
            attn=layer.self_attn
            if self.context_method=='dflare':
                key=F.linear(context,attn.k_proj_target.weight)
                value=F.linear(context,attn.v_proj_target.weight)
            else:
                kv=F.linear(context,attn.qkv_proj.weight[attn.q_size:])
                key,value=kv.chunk(2,dim=-1)
            keys.append(attn.k_norm(key.reshape(n,attn.num_kv_heads,attn.head_dim)))
            values.append(value.reshape(n,attn.num_kv_heads,attn.head_dim))
        first=self.layers[0].self_attn
        all_k=torch.stack(keys).contiguous();all_v=torch.stack(values).contiguous()
        offset=context_positions.storage_offset()
        positions=self._q36_context_axes[:,offset:offset+n]
        assert positions.shape==(3,n)
        flat=all_k.view(3*n,first.kv_size)
        dummy=torch.zeros((3*n,first.rotary_emb.head_size),dtype=flat.dtype,device=flat.device)
        rotated,_=first.rotary_emb(positions.repeat(1,3),flat,dummy)
        all_k=rotated.view(3,n,first.num_kv_heads,first.head_dim)
        if context_slot_mapping is None:return
        per_layer=isinstance(context_slot_mapping,(list,tuple))
        for i,layer in enumerate(self.layers):
            slots=context_slot_mapping[i] if per_layer else context_slot_mapping
            attn=layer.self_attn.attn
            if slots is not None:attn.impl.do_kv_cache_update(attn,all_k[i],all_v[i],attn.kv_cache,slots)


class DFlareBackbone(ContextBackbone):
    context_method='dflare'
    decoder_layer_cls=DedicatedContextLayer


class DFlyBackbone(ContextBackbone):
    context_method='dfly'


class RawContextModel(DFlashQwen3ForCausalLM):
    def combine_hidden_states(self,hidden_states):
        assert hidden_states.ndim==2 and hidden_states.shape[1]==6144
        return hidden_states


class DFlashVelaDFlareModel(RawContextModel):
    model_cls=DFlareBackbone


class DFlashVelaDFlyModel(RawContextModel):
    model_cls=DFlyBackbone


def install():
    global _installed
    if _installed:return
    import vllm
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from research.qwen36_v032.spatial_adapter import install as spatial_install
    assert vllm.__version__=='0.32.0'
    spatial_install()
    for cls in (DFlashVelaDFlareModel,DFlashVelaDFlyModel):
        ModelRegistry.register_model(cls.__name__,f'{__name__}:{cls.__name__}')
    original_init=DFlashSpeculator.__init__;original_generate=DFlashSpeculator._generate_draft
    @wraps(original_init)
    def init(self,vllm_config,device):
        original_init(self,vllm_config,device)
        method=getattr(self.draft_model_config.hf_config,'veladraft_custom_method',None)
        if method in ('dflare','dfly'):
            assert self.max_num_reqs==1 and self.draft_logits is None and not self.sample_from_anchor
            self.hidden_states=torch.zeros(self.max_num_tokens,6144,dtype=self.dtype,device=device)
            assert fly_reference.contract.SPECULATIVE_TOKENS==self.num_speculative_steps
    @wraps(original_generate)
    def generate(self,num_reqs,num_tokens_padded,attn_metadata,slot_mappings,
                 num_tokens_across_dp,cudagraph_runtime_mode=CUDAGraphMode.NONE):
        method=getattr(self.draft_model_config.hf_config,'veladraft_custom_method',None)
        if method!='dfly':
            return original_generate(self,num_reqs,num_tokens_padded,attn_metadata,slot_mappings,
                                     num_tokens_across_dp,cudagraph_runtime_mode)
        assert num_reqs==1 and num_tokens_padded==self.num_query_per_req
        hidden=self._run_model(num_tokens_padded,attn_metadata,slot_mappings,
                               num_tokens_across_dp,cudagraph_runtime_mode)
        tokens=fly_reference.sequential_greedy(self.model,hidden[1:self.num_query_per_req],
                                               self.input_buffers.input_ids[:1])
        self.draft_tokens[:1]=tokens.view(1,-1)
    DFlashSpeculator.__init__=init;DFlashSpeculator._generate_draft=generate
    _installed=True
