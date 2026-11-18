"""Bias-free local Domino equations using public Torch operations."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def domino_gru_step(token_embedding,hidden_state,weight_ih,weight_hh):
    token_embedding=token_embedding.squeeze(1)
    if hidden_state is None:
        hidden_state=token_embedding.new_zeros((token_embedding.shape[0],weight_hh.shape[1]))
    input_reset,input_update,input_new=F.linear(token_embedding,weight_ih).chunk(3,dim=-1)
    hidden_reset,hidden_update,hidden_new=F.linear(hidden_state,weight_hh).chunk(3,dim=-1)
    reset_gate=torch.sigmoid(input_reset+hidden_reset)
    update_gate=torch.sigmoid(input_update+hidden_update)
    new_gate=torch.tanh(input_new+reset_gate*hidden_new)
    hidden_state=new_gate+update_gate*(hidden_state-new_gate)
    return hidden_state.unsqueeze(1),hidden_state

@torch.inference_mode()
def sample_domino(hidden_states,block_input_ids,*,compute_logits,embed_tokens,weight_ih,weight_hh,projector_in,projector_out):
    if hidden_states.ndim!=2 or hidden_states.shape[0]!=8 or tuple(block_input_ids.shape)!=(8,):
        raise ValueError('All8 current hidden rows and8 block IDs are required')
    current_hidden=hidden_states.unsqueeze(0)
    completed_ids=block_input_ids.to(dtype=torch.long).reshape(1,8).clone()
    logits=compute_logits(hidden_states)
    if logits is None or logits.ndim!=2 or logits.shape[0]!=8:
        raise ValueError('All8 full-vocabulary logits rows are required')
    base_logits=logits.unsqueeze(0)
    gru_state=None
    for token_position in range(1,8):
        previous_ids=completed_ids[:,token_position-1:token_position]
        prefix_state,gru_state=domino_gru_step(embed_tokens(previous_ids),gru_state,weight_ih,weight_hh)
        head_position=token_position-1
        next_logits=base_logits[:,head_position:head_position+1,:]
        if head_position>=1:
            features=torch.cat((current_hidden[:,head_position:head_position+1,:],prefix_state),dim=-1)
            correction=F.linear(F.silu(F.linear(features,projector_in)),projector_out)
            next_logits=next_logits+correction
        completed_ids[:,token_position]=next_logits.argmax(-1).squeeze(1)
    return completed_ids[:,1:]
