"""Original two-norm, three-matrix DFly SwiGLU residual using official layers."""
import torch
from torch import nn
import torch.nn.functional as F
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

class HiddenStatesCorrection(nn.Module):
    def __init__(self,hidden_size,intermediate_size,*,dtype,rms_norm_eps=1e-6,prefix=''):
        super().__init__()
        self.hidden_norm=RMSNorm(hidden_size,eps=rms_norm_eps,dtype=dtype)
        self.embed_norm=RMSNorm(hidden_size,eps=rms_norm_eps,dtype=dtype)
        for name in ('gate_proj','up_proj','down_proj'):
            in_size=intermediate_size if name=='down_proj' else 2*hidden_size
            out_size=hidden_size if name=='down_proj' else intermediate_size
            setattr(self,name,ReplicatedLinear(in_size,out_size,bias=False,params_dtype=dtype,
                quant_config=None,prefix=f'{prefix}.{name}',return_bias=False,disable_tp=True))

    def forward(self,hidden_states,prev_token_embeds):
        hidden_norm=self.hidden_norm(hidden_states)
        embed_norm=self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        joined=torch.cat([hidden_norm,embed_norm],dim=-1)
        delta=self.down_proj(F.silu(self.gate_proj(joined))*self.up_proj(joined))
        return hidden_states+delta.to(hidden_states.dtype)
