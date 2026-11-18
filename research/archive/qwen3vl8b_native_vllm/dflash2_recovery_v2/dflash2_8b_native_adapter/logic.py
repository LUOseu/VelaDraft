"""Canonical sequential selector walk over one parallel draft forward."""
import torch


def selector_walk(selector, unary_logits, hidden_states, anchor_ids):
    if unary_logits.ndim != 3 or hidden_states.ndim != 3:
        raise ValueError('Selector requires [batch, slots, vocabulary/hidden] tensors')
    if unary_logits.shape[:2] != hidden_states.shape[:2] or anchor_ids.shape != unary_logits.shape[:1]:
        raise ValueError('Selector batch/slot/actual-anchor shapes differ')
    predecessor = anchor_ids[:, None]
    chosen = []
    for slot in range(unary_logits.shape[1]):
        ids, scores = selector.select(unary_logits[:, slot:slot+1], hidden_states[:, slot:slot+1], predecessor)
        predecessor = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        chosen.append(predecessor)
    return torch.cat(chosen, dim=1)
