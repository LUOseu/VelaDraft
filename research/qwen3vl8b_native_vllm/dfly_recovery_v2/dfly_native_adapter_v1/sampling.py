"""Sequential greedy suffix with real selected-token feedback; target is unchanged."""
import torch

def sequential_greedy(model,suffix_hidden,bonus_anchor):
    if suffix_hidden.ndim!=2 or suffix_hidden.shape[0]!=7 or bonus_anchor.shape!=(1,):
        raise ValueError('DFly requires seven suffix hidden rows and one actual target bonus anchor')
    previous=bonus_anchor
    selected=[]
    for i in range(7):
        embeds=model.embed_input_ids(previous).to(suffix_hidden.dtype)
        corrected=model.model.hidden_correction(suffix_hidden[i:i+1],embeds)
        logits=model.compute_logits(corrected)
        if logits is None:raise ValueError('DFly requires the original target output head')
        previous=logits.argmax(dim=-1)
        selected.append(previous)
    return torch.stack(selected,dim=1)
