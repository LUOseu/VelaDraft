"""Reproduce the drafter's compilable flex-attention call on this env.

Mirrors the failing shapes from the S1a log: [4, 2, 686, 128] bf16 logits
(remember: micro-batch 4), training=True path that transformers compiles with
mode='max-autotune-no-cudagraphs' on torch 2.6.0.
"""
import sys

import torch
from torch.nn.attention.flex_attention import flex_attention

print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
q = torch.randn(4, 2, 686, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(4, 2, 686, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(4, 2, 686, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)


def score_mod(s, b, h, q_idx, kv_idx):
    return s * 0.08838834764831845


mode = sys.argv[1] if len(sys.argv) > 1 else "none"
if mode == "maxautotune":
    compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
else:
    compiled = torch.compile(flex_attention, dynamic=False)
out = compiled(q, k, v, score_mod=score_mod)
(out.float().sum()).backward()
torch.cuda.synchronize()
print("OK mode=", mode, "out", tuple(out.shape), flush=True)
