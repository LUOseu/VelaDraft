"""Repro for the S1a flex-attention Inductor NoValidChoicesError (c1vllm env).

Mirrors the drafter block attention captured in logs/s1a_ab1.log:
  q   [B=4, H=16, Q=80,  D=128] bf16  (10 anchors x block_size 8)
  k/v [B=4, H=2,  KV=686, D=128] bf16 (ctx 606 + 80 draft, GQA 16:2)
  BlockMask over [ctx | draft-block] visibility, plain torch.compile like
  transformers' WrappedFlexAttention (torch 2.11 -> plain compile).

Hypothesis (torch/_inductor/kernel/flex/flex_decoding.py):
  decode path chosen when q_len < 128 and GQA with ratio a power of two;
  BLOCK_M = next_pow2(q_len * G) = next_pow2(80*8) = 1024;
  config filter `SPARSE_Q_BLOCK_SIZE (128) % BLOCK_M != 0` then drops every
  candidate -> NoValidChoicesError. FORCE_USE_FLEX_ATTENTION=True skips the
  decode path. smallq (q_len=16 -> BLOCK_M=128) should compile the decode
  kernel and confirm the arithmetic.

Variants (argv): base force_main tritonb static smallq eager
  base        plain compile, eager create_block_mask (expect FAIL on 2.11)
  force_main  plain compile + kernel_options={'FORCE_USE_FLEX_ATTENTION': True}
  tritonb     plain compile + kernel_options={'BACKEND':'TRITON'}
  static      torch.compile(flex_attention, dynamic=False)
  smallq      q_len=16 probe (BLOCK_M=128 == mask block size)
  eager       no compile
"""
import sys
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

B, HQ, HKV, QLEN, KVLEN, D = 4, 16, 2, 80, 686, 128
CTX = KVLEN - QLEN
BLOCK = 8


def mask_mod(b, h, q_idx, kv_idx):
    return (kv_idx < CTX) | ((kv_idx >= CTX) & (q_idx // BLOCK == (kv_idx - CTX) // BLOCK))


def build_inputs(qlen):
    torch.manual_seed(0)
    q = torch.randn(B, HQ, qlen, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, HKV, KVLEN, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, HKV, KVLEN, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    return q, k, v


def build_mask(qlen):
    # The drafter's BlockMask is head-agnostic (kv_indices [B,1,...] in the
    # production trace); H=1 here is what routes flex_attention into the
    # decoding kernel path (_use_flex_decoding: valid_block_mask_num_heads).
    return create_block_mask(mask_mod, B=B, H=1, Q_LEN=qlen, KV_LEN=KVLEN, device="cuda")


def run(variant):
    qlen = 16 if variant == "smallq" else QLEN
    q, k, v = build_inputs(qlen)
    kwargs = {"block_mask": build_mask(qlen), "enable_gqa": True}
    if variant == "force_main":
        kwargs["kernel_options"] = {"FORCE_USE_FLEX_ATTENTION": True}
    if variant == "tritonb":
        kwargs["kernel_options"] = {"BACKEND": "TRITON"}
    if variant == "static":
        fn = torch.compile(flex_attention, dynamic=False)
    elif variant == "eager":
        fn = flex_attention
    else:
        fn = torch.compile(flex_attention)
    t0 = time.time()
    out = fn(q, k, v, **kwargs)
    out.float().sum().backward()
    torch.cuda.synchronize()
    print(f"[{variant}] OK in {time.time()-t0:.1f}s out={tuple(out.shape)}", flush=True)


def main():
    variants = sys.argv[1:] or ["base"]
    print(f"torch {torch.__version__} cuda {torch.version.cuda}", flush=True)
    for variant in variants:
        try:
            run(variant)
        except Exception as exc:  # noqa: BLE001 - repro script
            head = str(exc).splitlines()[0] if str(exc) else ""
            print(f"[{variant}] FAIL {type(exc).__name__}: {head[:220]}", flush=True)


if __name__ == "__main__":
    main()
