"""Owned DFly spatial bridge. Math follows the frozen spatial adapter; all operators are official vLLM."""
from contextvars import ContextVar
import inspect
from typing import Any
import torch
from vllm.model_executor.layers.rotary_embedding import mrope as mrope_module
from . import contracts as c
DUMMY_SCOPE=ContextVar('dfly_dummy_scope',default=False)

def absolute_context_positions(seq_lens: torch.Tensor, query_start_loc: torch.Tensor,
                               num_context: int) -> torch.Tensor:
    """Absolute token indices, independent of compressed MRoPE coordinates.

    seq_lens includes padded/rejected positions, as does query_start_loc.
    The official expansion kernel applies rejection offsets to query slots.
    Explicit output_size prevents a repeat_interleave GPU-to-host size sync.
    """
    batch_size = query_start_loc.numel() - 1
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    request = torch.repeat_interleave(
        torch.arange(batch_size, device=query_start_loc.device), lengths,
        output_size=num_context)
    offsets = torch.arange(num_context, device=query_start_loc.device)
    return (offsets - query_start_loc[:-1].index_select(0, request)
            + seq_lens.index_select(0, request) - lengths.index_select(0, request)).long()


def spatial_query_positions(target_axes: torch.Tensor, query_start_loc: torch.Tensor,
                            num_rejected: torch.Tensor | None,
                            num_query_per_req: int) -> torch.Tensor:
    """Keep T/H/W independently and continue from each last accepted token."""
    if target_axes.ndim != 2 or target_axes.shape[0] != 3:
        raise ValueError("Spatial RoPE requires three axes with shape [3, context_tokens]")
    valid_end = query_start_loc[1:].long()
    if num_rejected is not None:
        valid_end = valid_end - num_rejected.long()
    last_valid = target_axes.index_select(1, valid_end - 1)
    offsets = torch.arange(1, num_query_per_req + 1, device=target_axes.device,
                           dtype=target_axes.dtype)
    return (last_valid[:, :, None] + offsets).flatten(1)


def rotate_context_keys(rotary_emb: Any, context_axes: torch.Tensor,
                        flat_keys: torch.Tensor, num_layers: int,
                        *, native: bool = False) -> torch.Tensor:
    """Use the official MRoPE operator for the existing fused layer-major K.

    MRotaryEmbedding requires both Q and K. Treat context K as Q and supply a
    separate one-head zero tensor as the unused K. This avoids pointer aliasing
    and uses the official operator without modifying its kernel/API.
    """
    if context_axes.ndim == 1:
        # Only profiling/dummy_run uses the upstream all-zero 1D buffer.
        context_axes = context_axes.unsqueeze(0).expand(3, -1)
    if context_axes.ndim != 2 or context_axes.shape[0] != 3:
        raise ValueError("Context RoPE requires three axes")
    positions = context_axes.repeat(1, num_layers)
    dummy_key = torch.zeros((flat_keys.shape[0], rotary_emb.head_size),
                           dtype=flat_keys.dtype, device=flat_keys.device)
    operator = rotary_emb.forward_native if native else rotary_emb
    rotated, _ = operator(positions, flat_keys, dummy_key)
    return rotated


def validate_query_batch(num_tokens,num_padded,mode_name,*,dummy,keys_initialized):
    if dummy:
        c.require(1<=num_tokens<=num_padded<=8 and mode_name in ('NONE','PIECEWISE'),'Unsafe DFly dummy shape/mode.')
    else:
        c.require(num_tokens==num_padded==8 and mode_name=='PIECEWISE' and keys_initialized,
            'Real DFly queries require exactly8 initialized PIECEWISE tokens, no padding.')


def context_query_axes(target_axes,seq_lens,query_start_loc,rejected):
    n=target_axes.shape[1]
    c.require(target_axes.ndim==2 and target_axes.shape[0]==3 and query_start_loc.numel()==2 and seq_lens.numel()==1,
        'DFly requires one request and all3 actual spatial axes.')
    return absolute_context_positions(seq_lens,query_start_loc,n),spatial_query_positions(target_axes,query_start_loc,rejected,8)


def original_metadata():return {'input_ids':0,'positions':0,'input_embeds':0}


def corrected_metadata():return {'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}}


def compile_mapping(model_class):
    init=inspect.getclosurevars(model_class.__init__).nonlocals
    call=inspect.getclosurevars(model_class.__call__).nonlocals
    mapping=init['dynamic_arg_dims']
    mark=inspect.getclosurevars(call['_mark_dynamic_inputs']).nonlocals['dynamic_arg_dims']
    c.require(mapping is mark,'Initializer/marker compile mapping identity changed.')
    return mapping


def validate_compile_ownership(owned_class):
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    stock=compile_mapping(DFlashQwen3Model);owned=compile_mapping(owned_class)
    c.require(stock==original_metadata() and owned==corrected_metadata() and stock is not owned,
        'DFly must own its compile dictionary; stock DFlash metadata cannot change.')
    c.require(owned_class.__init__ is not DFlashQwen3Model.__init__ and owned_class.__call__ is not DFlashQwen3Model.__call__,
        'DFly must own initializer/call wrappers.')
    return {'stock_mapping_unchanged':True,'owned_mapping':owned,'shared_with_stock':False}



def precompute(self,raw_context,context_positions,context_slot_mapping=None):
    """Explicit raw12288, original FC-plus-fusion and original shared context KV."""
    from .context import build_layer_contexts,project_shared_context
    c.require(c.enabled(self.config),'Owned DFly context requires its truthful marker')
    c.require(getattr(self,'_dfly_weights_loaded',False),'All42 real tensors must load before context computation')
    c.require(raw_context.ndim==2 and raw_context.shape[1]==12288,'DFly raw context must be [C,12288]')
    n=raw_context.shape[0]
    c.require(context_positions.shape==(3,n),'DFly context needs all3 original axes')
    if context_slot_mapping is not None:
        c.require(context_slot_mapping.shape==(n,),'Physical mapping must cover every context row')
    contexts=build_layer_contexts(raw_context,self.layer_fusion_weights,self.fc,self.hidden_norm)
    keys=[];values=[]
    for layer,context in zip(self.layers,contexts):
        attention=layer.self_attn
        key,value=project_shared_context(context,attention)
        keys.append(attention.k_norm(key));values.append(value)
    all_k=torch.stack(keys).contiguous();all_v=torch.stack(values).contiguous()
    rotary=self.layers[0].self_attn.rotary_emb
    c.require(isinstance(rotary,mrope_module.MRotaryEmbedding),'DFly requires official MRotaryEmbedding')
    all_k=rotate_context_keys(rotary,context_positions,all_k.view(3*n,1024),3).view(3,n,8,128)
    if context_slot_mapping is None:return
    for i,layer in enumerate(self.layers):
        attn=layer.self_attn.attn
        attn.impl.do_kv_cache_update(attn,all_k[i],all_v[i],attn.kv_cache,context_slot_mapping)
