"""New Domino context hook; audited spatial helpers are read-only dependencies."""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'spatial_adapter/official_spatial_adapter.py'
expected=json.loads((ROOT/'source_bindings.json').read_text())['frozen']['spatial_adapter/official_spatial_adapter.py']
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen spatial math changed')
spec=importlib.util.spec_from_file_location('_domino_frozen_spatial_math',path)
math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
absolute_context_positions=math.absolute_context_positions
spatial_query_positions=math.spatial_query_positions
rotate_context_keys=math.rotate_context_keys

def precompute_and_store_context_kv(self,context_states,context_positions,context_slot_mapping=None):
    if context_slot_mapping is not None and (context_positions.ndim!=2 or context_positions.shape[0]!=3):raise ValueError('Real context writes require all three spatial axes')
    if not hasattr(self,'_num_attn_layers'):self._build_fused_kv_buffers()
    num_ctx,L,kv,hd,nkv=(context_states.shape[0],self._num_attn_layers,self._kv_size,self._head_dim,self._num_kv_heads)
    normed=torch.empty_like(context_states)
    ops.rms_norm(normed,context_states,self._hidden_norm_weight,self._rms_norm_eps)
    all_kv_flat=F.linear(normed,self._fused_kv_weight,self._fused_kv_bias)
    all_kv=all_kv_flat.view(num_ctx,L,2,nkv,hd).permute(2,1,0,3,4).contiguous()
    all_k,all_v=all_kv[0],all_kv[1]
    all_k_normed=torch.empty_like(all_k)
    for i in range(L):ops.rms_norm(all_k_normed[i],all_k[i],self._k_norm_weights[i],self._rms_norm_eps)
    rotary=self.layers[0].self_attn.rotary_emb
    if not isinstance(rotary,MRotaryEmbedding):raise RuntimeError('Official MRotaryEmbedding required')
    all_k_flat=rotate_context_keys(rotary,context_positions,all_k_normed.view(L*num_ctx,kv),L)
    if context_slot_mapping is None:return
    all_k_final=all_k_flat.view(L,num_ctx,nkv,hd)
    for i in range(L):
        attn=self._attn_layers[i]
        attn.impl.do_kv_cache_update(attn,all_k_final[i],all_v[i],attn.kv_cache,context_slot_mapping)
