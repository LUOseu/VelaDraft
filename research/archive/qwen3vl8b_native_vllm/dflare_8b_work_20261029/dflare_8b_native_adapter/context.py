"""Canonical live-fusion and dual-source K/V algebra using public Torch ops."""
import torch
import torch.nn.functional as F


def build_layer_contexts(raw_features,fusion_logits,hidden_norm):
    if raw_features.ndim!=2 or raw_features.shape[1]!=12288 or tuple(fusion_logits.shape)!=(3,3):
        raise ValueError('DFlare requires explicit ordered [C,12288] and live [3,3] fusion logits')
    stacked=raw_features.reshape(1,raw_features.shape[0],3,4096)
    probabilities=F.softmax(fusion_logits,dim=-1).to(raw_features.dtype)
    return tuple(hidden_norm(torch.einsum('t,bstd->bsd',probabilities[layer],stacked)).squeeze(0)
        for layer in range(3))


def project_context(context,key_projection,value_projection):
    key=key_projection(context);value=value_projection(context)
    return key.reshape(context.shape[0],8,128),value.reshape(context.shape[0],8,128)
