"""Distinct40-weight Domino model, stock backbone/operators and own metadata."""
from __future__ import annotations
import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models import qwen3_dflash as official
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.utils import maybe_prefix
from contracts import validate_config,validate_state,require_runtime_binding
from spatial_math import precompute_and_store_context_kv

@support_torch_compile(dynamic_arg_dims={'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}})
class DominoBackbone(nn.Module):
    # Independent decorator dictionary; ordinary DFlash metadata is never mutated.
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        require_runtime_binding(vllm_config)
        super().__init__()
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        validate_config(self.config)
        self.vocab_size=self.config.vocab_size
        self.quant_config=official.get_draft_quant_config(vllm_config)
        self.use_aux_hidden_state=True
        current=get_current_vllm_config()
        self.embed_tokens=official.VocabParallelEmbedding(self.config.vocab_size,self.config.hidden_size,prefix=maybe_prefix(prefix,'embed_tokens'))
        self.layers=nn.ModuleList([official.DFlashQwen3DecoderLayer(current,config=self.config,cache_config=current.cache_config,quant_config=self.quant_config,prefix=maybe_prefix(prefix,f'layers.{layer_idx+start_layer_id}')) for layer_idx in range(self.config.num_hidden_layers)])
        self.fc=official.ReplicatedLinear(input_size=self.config.target_hidden_size*len(self.config.dflash_config['target_layer_ids']),output_size=self.config.hidden_size,bias=False,params_dtype=vllm_config.model_config.dtype,quant_config=self.quant_config,prefix=maybe_prefix(prefix,'fc'),return_bias=False)
        self.hidden_norm=official.RMSNorm(self.config.hidden_size,eps=self.config.rms_norm_eps)
        self.norm=official.RMSNorm(self.config.hidden_size,eps=self.config.rms_norm_eps)
        dtype=vllm_config.model_config.dtype
        self.prefix_gru=nn.GRU(input_size=4096,hidden_size=1024,num_layers=1,batch_first=True,bias=False,dtype=dtype)
        self.embed_proj=nn.Sequential(nn.Linear(5120,256,bias=False,dtype=dtype),nn.SiLU(),nn.Linear(256,151936,bias=False,dtype=dtype))

    embed_input_ids=official.DFlashQwen3Model.embed_input_ids
    _build_fused_kv_buffers=official.DFlashQwen3Model._build_fused_kv_buffers
    forward=official.DFlashQwen3Model.forward
    load_weights=official.DFlashQwen3Model.load_weights
    precompute_and_store_context_kv=precompute_and_store_context_kv

class DominoForCausalLM(official.DFlashQwen3ForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        require_runtime_binding(vllm_config)
        nn.Module.__init__(self)
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        validate_config(self.config)
        if vllm_config.model_config.dtype!=torch.bfloat16:raise ValueError('Frozen Domino weights require BF16')
        if getattr(self.config,'draft_vocab_size',None) is None:self.config.draft_vocab_size=self.config.vocab_size
        target_layer_num=vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.config.target_layer_count=target_layer_num
        self.model=DominoBackbone(vllm_config=vllm_config,prefix=maybe_prefix(prefix,'model'),start_layer_id=target_layer_num)
        self.lm_head=official.ParallelLMHead(self.config.draft_vocab_size,self.config.hidden_size,prefix=maybe_prefix(prefix,'lm_head'))
        self.logits_processor=official.LogitsProcessor(self.config.draft_vocab_size,scale=1.0)
        self.draft_id_to_target_id=None

    def load_weights(self,weights):
        pairs=list(weights)
        if len(pairs)!=len(dict(pairs)):raise ValueError('Duplicate checkpoint keys')
        validate_state(dict(pairs))
        return super().load_weights(pairs)
