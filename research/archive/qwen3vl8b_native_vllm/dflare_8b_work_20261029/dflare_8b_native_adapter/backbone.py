"""Independent DFlare backbone/compile metadata with official query sublayers."""
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config
from vllm.model_executor.models import qwen3_dflash as official
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from . import contracts as c
from .spatial import precompute

DEDICATED=frozenset(f'layers.{i}.self_attn.{kind}_proj_target.weight' for i in range(3) for kind in ('k','v'))

class DFlareOwnedDecoderLayer(official.DFlashQwen3DecoderLayer):
    def __init__(self,vllm_config,*,config,cache_config=None,quant_config=None,prefix=''):
        super().__init__(vllm_config,config=config,cache_config=cache_config,quant_config=quant_config,prefix=prefix)
        for kind in ('k','v'):
            setattr(self.self_attn,kind+'_proj_target',nn.Linear(4096,1024,bias=False,dtype=vllm_config.model_config.dtype))

@support_torch_compile(dynamic_arg_dims={'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}})
class DFlareOwnedBackbone(nn.Module):
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        receipt=c.runtime_receipt();c.validate_launch_config(vllm_config,receipt)
        super().__init__()
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size=self.config.vocab_size;self.quant_config=None;self.use_aux_hidden_state=True
        current=get_current_vllm_config()
        self.embed_tokens=official.VocabParallelEmbedding(self.config.vocab_size,4096,prefix=maybe_prefix(prefix,'embed_tokens'))
        self.layers=nn.ModuleList([DFlareOwnedDecoderLayer(current,config=self.config,cache_config=current.cache_config,
            quant_config=None,prefix=maybe_prefix(prefix,f'layers.{i+start_layer_id}')) for i in range(3)])
        self.hidden_norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)
        self.norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)
        self.layer_fusion_weights=nn.Parameter(torch.empty(3,3,dtype=vllm_config.model_config.dtype),requires_grad=False)
        self._dflare_weights_loaded=False

    embed_input_ids=official.DFlashQwen3Model.embed_input_ids
    forward=official.DFlashQwen3Model.forward
    precompute_and_store_context_kv=precompute

    def load_weights(self,weights):
        state={}
        for name,value in weights:
            c.require(name not in state,'Duplicate DFlare tensor: '+name);state[name]=value
        c.validate_state(state)
        params=dict(self.named_parameters())
        # Prevent stock substring .k_proj/.v_proj packing from corrupting target names.
        for name in DEDICATED:default_weight_loader(params[name],state[name])
        ordinary={name:value for name,value in state.items() if name not in DEDICATED}
        loaded=official.DFlashQwen3Model.load_weights(self,ordinary.items())
        self._dflare_weights_loaded=True
        return loaded|set(DEDICATED)
