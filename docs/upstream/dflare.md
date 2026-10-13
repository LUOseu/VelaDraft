# Upstream audit — DFlare

Status: **audited; implementation landed**. The audit was completed by reading
the pinned sources line by line before the port was written; the local
implementation mirrors the upstream module structure and keeps every changed
behaviour documented below.

```text
Paper / design:            DFlare: Scaling Up Draft Capacity for Block Diffusion
                           Speculative Decoding — arXiv:2606.02091
Official repo:             Tencent/AngelSpec @ d3412bed231025ea85d35dbf5e0af44f1ac62a5b
Reference implementation:  angelspec/models/draft/dflare.py
                             (DFlareAttention, DFlareDecoderLayer, DFlareDraftModel)
                           angelspec/models/draft/dflash.py (shared DFlashModel /
                             DFlashTrainer stack, reused unchanged upstream)
                           angelspec/config/dflare_draft_config.json   (draft architecture)
                           configs/vllm_qwen3_8b_dflare.yaml           (training recipe)
License:                   MIT (AngelSpec repo LICENSE + file headers)
```

## Semantics recorded from the sources

```text
Two structural changes over DFlash; the training stack (DFlashModel wrapper,
DFlashTrainer, anchor sampling, block-causal mask, decayed block CE) is reused
unchanged upstream:

1. Dual-source KV attention with SEPARATE context/draft projections
   (DFlareAttention): q from draft; draft k/v from k_proj/v_proj; target-context
   k/v from the DFlare-specific k_proj_target/v_proj_target; concat
   [context | draft] -> k_norm -> RoPE (context positions first) -> flex
   attention under the shared block mask. DFlash instead uses one k_proj/v_proj
   for both sources.

2. Per-draft-layer fusion replaces DFlash's single shared context
   (DFlareDraftModel):
     context_feature = stack(captured target layers)        # [B, S, T, D]
     fusion_probs    = softmax(layer_fusion_weights, -1)    # [D_layers, T]
     fused_context_i = einsum("t,bstd->bsd", probs_i, context_feature)
     context_i       = context_norm(fused_context_i)        # RMSNorm
   Draft layer i attends to its own softmax-weighted mixture of the T captured
   target layers. DFlash instead computes one context_proj (Linear(T*D, D))
   shared by all draft layers.

Fusion-logit initialization (DFlareDraftModel._init_fusion_weights):
    logits[d, t] = -alpha * |t - t*_d|,   alpha = 2.0,
    t*_d = round(d / max(D-1, 1) * (T-1))
  i.e. a soft relative-depth prior (draft layer d peaks at the target layer at
  the same relative depth); deliberately NOT zeros/uniform, which would be
  average pooling.

Training objective:        unchanged DFlash (sampled anchors + decayed block
                           CE); the DFlare trainer only swaps the draft model.
Recipe (vllm_qwen3_8b_dflare.yaml):  lr 6e-4, warmup_ratio 0.04, max_grad_norm
                           1.0, block_size 16, num_anchors 512, decay gamma 7.0,
                           num_target_layers 9, max_seq 2048, 2 GPUs, FSDP
                           FULL_SHARD, ttt_length 7.
Draft config (dflare_draft_config.json): hidden 4096, 7 draft layers,
                           target_layer_ids [1, 5, 9, ..., 33] (9 layers of the
                           36-layer target = build_target_layer_ids(36, 9)),
                           target_hidden_size 4096, mask_token_id 151669,
                           tie_word_embeddings false.
Selection:                 same DFlashConfig class, chosen via
                           model_arch="dflare" inside DFlashTrainer.init_model.
Inference:                 same block-parallel decode; the fused context for a
                           block is built from the captured target layers, so
                           serving must extract exactly the captured layer list.
```

## What we port / change / why

```text
What we port:  the two structural changes (separate context K/V + per-draft-layer
               fusion), the fusion normalization placement (RMSNorm after the
               weighted mix, before attention), the distance-prior initialization
               including its constants (alpha=2.0, round-to-preferred-depth), and
               the rest of the DFlash stack unchanged (same anchors, mask, decayed
               block CE, slot conventions).
What we change:
  (1) Feature plumbing. Upstream feeds the trainer a stacked [B, S, T, D] tensor
      produced live by the target forward. This repo's feature stores concatenate
      the captured layers to [B, S, T*H]; consumers declare an explicit subset
      (algorithm.dflare.context_layer_ids) of an explicitly captured superset
      (drafter.capture_layer_ids) and the trainer slices by layer id before the
      draft forward. The values for the consumed layers are identical to what
      upstream would feed; the layout is recorded in the manifest/contract.
  (2) Vision encoder / MRoPE plumbing is inherited from the frozen DFlash VLM
      port (upstream DFlare is demonstrated on a text-only Qwen3-8B recipe);
      nothing in the fusion changes for image tokens.
  (3) Algorithm selection is a registered algorithm + model class
      (Qwen3DFlareDraftModel) instead of the model_arch string switch; the
      dflash_config field names match the upstream config JSON.
  (4) Ablation-only switches (not upstream): separate_context_kv=false keeps
      DFlash's shared K/V projections (isolates change 1), fusion_trainable=false
      freezes the fusion at its prior (isolates change 2), fusion_init="uniform"
      uses zero logits (average pooling) instead of the distance prior.
  (5) Training runs in the repo's staged convention (offline feature stores or
      online teacher-forced capture) instead of the upstream online FSDP recipe;
      optimizer/lr/anchor/block settings are config-level and follow the upstream
      values in the canonical configs where affordable.
Why we change: the repo's single contract (registered algorithms, explicit
               capture lists, unified checkpoints, staged/online feature sources)
               must hold for every algorithm; none of this alters the training math
               of the ported modules.
Parity level:  architecture/semantics level (ported module-by-module against the
               pinned commit; not verified against upstream checkpoints
               token-for-token).
```

Local code: `vlm_dflash/dflare/model.py` (attention + fusion model),
`vlm_dflash/dflare/factory.py` (config plumbing), `vlm_dflash/dflare/trainer.py`
(fusion metrics), `vlm_dflash/algorithms/dflare.py` (registry plugin).
Tests: `tests/test_dflare.py`. User doc: `docs/algorithms/dflare.md`.
