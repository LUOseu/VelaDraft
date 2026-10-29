"""DFlash2-owned compile metadata/backbone using official vLLM sublayers."""
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config
from vllm.model_executor.models import qwen3_dflash as official
from vllm.model_executor.models.utils import maybe_prefix

from . import contracts as c
from .conv import GroupedDynamicCausalConv
from .layer_logic import decoder_forward
from .spatial import precompute


class DFlash2OwnedDecoderLayer(official.DFlashQwen3DecoderLayer):
    def __init__(self,vllm_config,*,config,cache_config=None,quant_config=None,prefix=''):
        super().__init__(vllm_config,config=config,cache_config=cache_config,quant_config=quant_config,prefix=prefix)
        options=config.dflash_config['dflash2']
        args=dict(block_size=config.block_size,kernel_size=options['conv_kernel_size'],group_size=options['conv_group_size'])
        self.attention_conv=GroupedDynamicCausalConv(config.hidden_size,**args)
        self.mlp_conv=GroupedDynamicCausalConv(config.hidden_size,**args)
    forward=decoder_forward


@support_torch_compile(dynamic_arg_dims={'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}})
class DFlash2OwnedBackbone(nn.Module):
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        approved=c.runtime_receipt();c.validate_launch_config(vllm_config,approved)
        super().__init__()
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size=self.config.vocab_size
        self.quant_config=official.get_draft_quant_config(vllm_config)
        c.require(self.quant_config is None,'DFlash2 checkpoint requires unquantized BF16')
        self.use_aux_hidden_state=True
        current=get_current_vllm_config()
        self.embed_tokens=official.VocabParallelEmbedding(self.config.vocab_size,self.config.hidden_size,prefix=maybe_prefix(prefix,'embed_tokens'))
        self.layers=nn.ModuleList([DFlash2OwnedDecoderLayer(current,config=self.config,cache_config=current.cache_config,
            quant_config=None,prefix=maybe_prefix(prefix,f'layers.{i+start_layer_id}')) for i in range(self.config.num_hidden_layers)])
        self.fc=official.ReplicatedLinear(input_size=12288,output_size=4096,bias=False,params_dtype=vllm_config.model_config.dtype,
            quant_config=None,prefix=maybe_prefix(prefix,'fc'),return_bias=False)
        self.hidden_norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)
        self.norm=official.RMSNorm(4096,eps=self.config.rms_norm_eps)

    embed_input_ids=official.DFlashQwen3Model.embed_input_ids
    _build_fused_kv_buffers=official.DFlashQwen3Model._build_fused_kv_buffers
    forward=official.DFlashQwen3Model.forward
    load_weights=official.DFlashQwen3Model.load_weights
    precompute_and_store_context_kv=precompute
