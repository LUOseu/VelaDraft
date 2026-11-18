"""DFly FC-plus-fusion context and the original shared context/query KV weights."""
import torch
import torch.nn.functional as F

def build_layer_contexts(raw_features,fusion_logits,fc,hidden_norm):
    if raw_features.ndim!=2 or raw_features.shape[1]!=12288 or tuple(fusion_logits.shape)!=(3,3):
        raise ValueError('DFly requires explicit ordered [C,12288] and live [3,3] fusion logits')
    stacked=raw_features.reshape(1,raw_features.shape[0],3,4096)
    # Preserve canonical cast and addition order, including live BF16 softmax.
    base=fc(raw_features.to(fc.weight.dtype))
    probabilities=F.softmax(fusion_logits,dim=-1).to(base.dtype)
    return tuple(hidden_norm(base+torch.einsum('t,bstd->bsd',probabilities[i],stacked.to(base.dtype)).squeeze(0))
        for i in range(3))

def project_shared_context(context,attention):
    # Same packed parameter objects as the unmodified official query forward.
    # Official attention uses F.linear on this exact QKV parameter as well.
    weight=attention.qkv_proj.weight[attention.q_size:]
    bias=attention.qkv_proj.bias
    bias=None if bias is None else bias[attention.q_size:]
    key,value=F.linear(context,weight,bias).split([attention.kv_size,attention.kv_size],dim=-1)
    return key.reshape(context.shape[0],8,128),value.reshape(context.shape[0],8,128)
