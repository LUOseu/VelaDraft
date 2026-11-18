"""Owned DFly compile metadata; official query decoder and exact original weights."""
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config
from vllm.model_executor.models import qwen3_dflash as official
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from . import contracts as c
from .correction import HiddenStatesCorrection
from .spatial import precompute

CORRECTION_KEYS=frozenset('hidden_correction.'+name+'.weight' for name in
    ('hidden_norm','embed_norm','gate_proj','up_proj','down_proj'))

@support_torch_compile(dynamic_arg_dims={'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}})
class DFlyOwnedBackbone(nn.Module):
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        c.runtime_receipt();c.validate_launch_config(vllm_config)
        super().__init__()
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size=self.config.vocab_size;self.quant_config=None;self.use_aux_hidden_state=True
        dtype=vllm_config.model_config.dtype;current=get_current_vllm_config()
        self.embed_tokens=official.VocabParallelEmbedding(self.config.vocab_size,4096,prefix=maybe_prefix(prefix,'embed_tokens'))
        self.layers=nn.ModuleList([official.DFlashQwen3DecoderLayer(current,config=self.config,
            cache_config=current.cache_config,quant_config=None,prefix=maybe_prefix(prefix,f'layers.{i+start_layer_id}')) for i in range(3)])
        self.fc=official.ReplicatedLinear(12288,4096,bias=False,params_dtype=dtype,quant_config=None,
            prefix=maybe_prefix(prefix,'fc'),return_bias=False,disable_tp=True)
        self.hidden_norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)
        self.norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)
        self.layer_fusion_weights=nn.Parameter(torch.empty(3,3,dtype=dtype),requires_grad=False)
        self.hidden_correction=HiddenStatesCorrection(4096,4096,dtype=dtype,
            rms_norm_eps=self.config.rms_norm_eps,prefix=maybe_prefix(prefix,'hidden_correction'))
        self._dfly_weights_loaded=False

    embed_input_ids=official.DFlashQwen3Model.embed_input_ids
    forward=official.DFlashQwen3Model.forward
    precompute_and_store_context_kv=precompute

    def load_weights(self,weights):
        state={}
        for name,value in weights:
            c.require(name not in state,'Duplicate DFly tensor: '+name);state[name]=value
        c.validate_state(state)
        params=dict(self.named_parameters())
        # Canonical hidden correction has separate gate/up parameters. Stock's
        # substring packer would otherwise send them into a nonexistent gate_up.
        for name in CORRECTION_KEYS:default_weight_loader(params[name],state[name])
        ordinary={name:value for name,value in state.items() if name not in CORRECTION_KEYS}
        loaded=official.DFlashQwen3Model.load_weights(self,ordinary.items())
        self._dfly_weights_loaded=True
        return loaded|set(CORRECTION_KEYS)
