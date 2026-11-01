"""Three-axis position adapter for upstream vLLM 0.32 DFlash Model Runner V2.

Native input expansion, verification, sampling, graph capture and cache-slot
management are retained. Only the position channels supplied to draft RoPE
and its context K/V projection change. Scope: the frozen Qwen3.6 DFlash, TP1,
batch size one, BF16. This module has no CUDA/Triton kernels.
"""
import hashlib
import inspect
from pathlib import Path

_installed=False


def query_axes(target_axes,query_start_loc,num_rejected,num_query):
    import torch
    last=query_start_loc[1:].long()-num_rejected.long()-1
    anchor=target_axes.index_select(1,last)
    offset=torch.arange(1,num_query+1,device=target_axes.device,dtype=target_axes.dtype)
    return (anchor[:,:,None]+offset).flatten(1)


def install():
    global _installed
    if _installed:return
    import torch
    import vllm
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model as Model
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator as Speculator
    if vllm.__version__!='0.32.0':raise ValueError('This adapter pins vLLM 0.32.0')
    original_init=Speculator.__init__;original_load=Speculator.load_draft_model
    original_propose=Speculator.propose;original_forward=Model.forward
    original_context=Model.precompute_and_store_context_kv
    signature=inspect.signature(original_propose)
    expected=['self','input_batch','attn_metadata','slot_mappings','last_hidden_states','aux_hidden_states','num_sampled','num_rejected']
    if list(signature.parameters)[:len(expected)]!=expected:raise ValueError('Upstream proposer argument contract changed')

    def init(self,vllm_config,device):
        if vllm_config.parallel_config.tensor_parallel_size!=1 or vllm_config.scheduler_config.max_num_seqs!=1:
            raise ValueError('Initial spatial adapter validation is scoped to TP1 and batch1')
        original_init(self,vllm_config,device)
        cfg=self.draft_model_config.hf_config
        if (cfg.hidden_size!=2048 or cfg.num_hidden_layers!=3 or cfg.vocab_size!=248320
                or not cfg.dflash_config.get('spatial_rope_enabled')):
            raise ValueError('Unexpected draft architecture or disabled spatial RoPE')
        self._q36_query_axes=torch.zeros((3,self.max_num_tokens+1),dtype=torch.int64,device=device)
        self._q36_context_axes=torch.zeros((3,self.max_num_tokens+1),dtype=torch.int64,device=device)

    def load(self,target_model,target_attn_layer_names):
        model=original_load(self,target_model,target_attn_layer_names)
        model.model._q36_query_axes=self._q36_query_axes
        model.model._q36_context_axes=self._q36_context_axes
        if not all(isinstance(layer.self_attn.rotary_emb,MRotaryEmbedding) for layer in model.model.layers):
            raise ValueError('Draft failed to select the official multidimensional RoPE operator')
        target=target_model.get_language_model()
        from vllm.v1.worker.gpu.spec_decode.eagle.utils import get_target_lm_head
        target_head=get_target_lm_head(target_model,target)
        target_inner=getattr(target,'model',target)
        if model.lm_head.weight.data_ptr()!=target_head.weight.data_ptr():
            raise ValueError('Target LM head was not shared')
        if model.model.embed_tokens.weight.data_ptr()!=target_inner.embed_tokens.weight.data_ptr():
            raise ValueError('Target embedding was not shared')
        return model

    def propose(self,*args,**kwargs):
        batch=args[0] if args else kwargs['input_batch']
        rejected=kwargs['num_rejected'] if 'num_rejected' in kwargs else args[6]
        dummy=kwargs.get('dummy_run',False)
        if dummy:
            self._q36_query_axes.zero_();self._q36_context_axes.zero_()
        else:
            rope=self.model_state.rope_state
            if rope is None or rope.num_dims!=3:raise ValueError('Target 3-axis positions unavailable')
            n=batch.num_tokens;r=batch.num_reqs
            axes=rope.get_positions(n)
            self._q36_context_axes[:,:n].copy_(axes)
            queries=query_axes(axes,batch.query_start_loc[:r+1],rejected[:r],self.num_query_per_req)
            self._q36_query_axes[:,:queries.shape[1]].copy_(queries)
        return original_propose(self,*args,**kwargs)

    def forward(self,input_ids,positions,input_embeds=None):
        if not hasattr(self,'_q36_query_axes'):
            raise ValueError('Spatial buffers must be attached before draft execution')
        return original_forward(self,input_ids,self._q36_query_axes[:,:positions.shape[0]],input_embeds)

    def context(self,context_states,context_positions,context_slot_mapping=None):
        if not hasattr(self,'_q36_context_axes'):
            # Model loading can initialize buffers before the speculator attaches them.
            if context_slot_mapping is not None:raise ValueError('Spatial context axes unavailable for real cache writes')
            return original_context(self,context_states,context_positions,context_slot_mapping)
        if context_positions.ndim!=1:raise ValueError('Upstream physical positions must stay one-dimensional')
        n=context_states.shape[0];offset=context_positions.storage_offset()
        positions=self._q36_context_axes[:,offset:offset+n]
        if positions.shape!=(3,n):raise ValueError('Context position slice outside stable buffer')
        if not hasattr(self,'_num_attn_layers'):self._build_fused_kv_buffers()
        L=self._num_attn_layers;kv=self._kv_size;hd=self._head_dim;nkv=self._num_kv_heads
        all_k,all_v=self._project_context_kv(context_states,n,L,nkv,hd)
        keys=self._normalize_context_k(all_k).view(L*n,kv)
        rotary=self.layers[0].self_attn.rotary_emb
        dummy_key=torch.zeros((L*n,rotary.head_size),dtype=keys.dtype,device=keys.device)
        keys,_=rotary(positions.repeat(1,L),keys,dummy_key)
        if context_slot_mapping is None:return
        if getattr(self.layers[0].self_attn,'v_scale',None) is not None:
            raise ValueError('Frozen checkpoint does not use attention-value scaling')
        keys=keys.view(L,n,nkv,hd)
        per_layer=isinstance(context_slot_mapping,(list,tuple))
        for i,attn in enumerate(self._attn_layers):
            slots=context_slot_mapping[i] if per_layer else context_slot_mapping
            if slots is not None:attn.impl.do_kv_cache_update(attn,keys[i],all_v[i],attn.kv_cache,slots)

    Speculator.__init__=init;Speculator.load_draft_model=load;Speculator.propose=propose
    Model.forward=forward;Model.precompute_and_store_context_kv=context
    _installed=True


def source_hashes():
    import vllm.model_executor.models.qwen3_dflash as model
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as speculator
    import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as graph
    return {str(Path(module.__file__)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (model,speculator,graph)}
