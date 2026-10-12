# Upstream audit — DFlash2 (Experimental port)

Status: **audit complete; port implemented as an experimental reconstruction**
(Stage 5 of the GOAL execution order — the most complex variant, and the only
one whose training objective is *not* published upstream).

## Pinned sources

```text
Paper:                     none published (no arXiv/venue paper found)
Blog:                      Z Lab, "DFlash2" — https://inco.ai/blog/dflash2/
                           (architecture + checkpoint contract + selector math)
Official repo:             z-lab/dflash
                             `dflash/model.py` @ 07ebd93db9f472af339b644bb70221ad8428328a
                             (the revision speculators pins as the adaptation source;
                              MIT-licensed model code — header carried into our port)
Maintainer implementation: vllm-project/speculators
                             `src/speculators/models/dflash2/{config,core,metrics,model_definitions}.py`
                             `docs/user_guide/algorithms/dflash2.md`
                             `tests/unit/models/test_dflash2_model_definitions.py`
Commit inspected:          vllm-project/speculators @ 55ebbd5b9b49e9bd223f47949c986aa4abfffbc3
                           (2026-10-08; the revision our previous audit pinned)
NeMo AutoModel:            NOT present. `NVIDIA-NeMo/Automodel @ bb5b34ce` contains
                           `components/speculative/dflash/{core,domino_core,jetspec_core,draft_qwen3}.py`
                           only; `grep -rl dflash2` over that tree returns nothing
                           (re-checked 2026-10-12). The earlier audit note that pinned
                           `dflash2_core.py` / `draft_qwen3_dflash2.py` there was wrong.
License:                   speculators Apache-2.0; the DFlash2 model code inside it
                           carries an MIT header from Z Lab (Copyright (c) 2026 Z Lab)
                           pointing at the z-lab/dflash revision above.
```

## What DFlash2 is

DFlash2 = the DFlash block-parallel draft backbone **+ two added modules**:

```text
1. GroupedDynamicCausalConv ("local dynamic convolution")
   Each draft layer wraps both its attention sublayer and its MLP sublayer in a
   grouped, token-conditioned, *causal-within-block* convolution. A linear
   projection of the current hidden state emits two dynamic kernels (input-side
   and output-side); a learned base kernel is shared across positions and
   channels inside a group share the dynamic component. The convolution never
   crosses an anchor-block boundary (training flattens many anchored blocks into
   one token dimension, so the taps must reset every `block_size` tokens).

2. CandidateSelector ("candidate reranking")
   A low-rank bilinear edge scorer over the unary logits' top-K candidates:
       score(candidate) = unary(candidate)
                        + dot(predecessor_codebook(prev) * hidden_projection(h),
                              successor_codebook(candidate))
   Inference keeps one parallel draft forward pass and lets only the small
   selector walk the candidate path: it starts at the anchor token and conditions
   each position on the token selected at the preceding position.
```

Config defaults (from `config.py` + the speculators user guide):
`conv_kernel_size=2`, `conv_group_size=16`, `selector_rank=256`,
`selector_top_k=16`, `selector_loss_alpha=1.0`, `sliding_window_non_causal=True`,
5 draft layers, block size 8, `sample_from_anchor=False`, fixed exponential
position weighting, KL divergence unary loss. **DFlash2 requires the full
verifier vocabulary** — pruned draft vocabularies are rejected
(`DFlash2DraftModel.__init__` raises when `draft_vocab_size != target_vocab_size`)
because serving selects candidates *before* any draft→target vocab mapping.

## Training objective (the honest part)

```text
what is official:      the inference architecture, the module shapes, the
                       parameterization (base_kernel + kernel_projection,
                       predecessor/successor codebooks + hidden projection), the
                       candidate-set semantics (rerank the unary top-K only), and
                       the checkpoint key names.
what is inferred:      the selector's *serving* walk (documented in the guide,
                       not implemented in speculators itself — no vLLM runtime
                       code is in the pinned revision).
what is experimental:  the TRAINING objective. The public DFlash2 sources
                       "specify inference but do not publish its training
                       objective"; speculators trains an experimental split
                       objective:
                         (a) unary logits ← the configured DFlash-family compound
                             loss (default: forward KL to the verifier logits,
                             weight 1.0) with the configured position decay;
                         (b) selector    ← hard K-way CE over the unary top-K,
                             teacher-forcing the previous sequence token; if the
                             target id is absent from the top-K, the weakest
                             candidate (rank K-1) is replaced by the target so the
                             CE is always defined — validation uses the same
                             replacement, strictly-scored serving metrics do not.
                       alpha = selector_loss_alpha weights (b). Both terms use the
                       same fixed-exp decay and divide by the *masked position
                       count* (not by the decay-weight sum).
                       The guide explicitly warns that the free codebook
                       parameterization is an *expanded* parameterization and that
                       this prototype must not be read as reproducing the
                       parameter count of the DFlash2 blog.
```

Our port keeps that split objective exactly and therefore ships as
**Experimental** in `ALGORITHM_MATRIX.md` / `SUPPORT_MATRIX.md`.

## What we port

```text
- GroupedDynamicCausalConv + grouped_dynamic_conv   (math, init, block-local taps)
- CandidateSelector                                  (codebooks, projection, scoring)
- DFlash2 decoder layer                              (conv wraps attention + MLP)
- selector_training_candidates + selector K-way CE   (target injection at rank K-1)
- unary forward-KL with fixed-exp decay              (default objective, gamma 4.0)
- validation metrics: top-K candidate recall, target mass, teacher-forced
  selector accuracy, realized selector accepted length (eal)
```

## What we change, and why

```text
1. Teacher signal. Upstream consumes full-vocab verifier logits captured online
   ([B, S, V]). This staged repo's capture store keeps the target's *final hidden
   state* (capture v2) and projects it through the frozen target lm_head at
   training time, chunk-wise under no_grad — an exact computation of the same
   logits (verified separately: max |lm_head(h_last) - target logits| = 0.0).
   The selector's hard target id comes from the same projection's argmax.
2. Serving. The pinned speculators revision has no runtime; upstream serving is
   "vLLM with a speculators config". We implement the documented selector walk
   inside this repo's shared block-parallel decoder (one parallel draft forward,
   then a sequential top-K walk over candidate scores), and export the selector
   weights under the canonical key names so a future vLLM/speculators runtime can
   load them. Marked Experimental for exactly this reason.
3. Target. Ported on Qwen2.5-VL-3B in the multimodal pipeline: the convolution
   and selector are modality-agnostic (they act on the draft block), so the
   multimodal plumbing (MRoPE, image spans, capture) is unchanged.
4. Namespace. Upstream flattening of many anchors into one token dimension is
   preserved by applying the conv per block (`offset % block_size`), the same
   convention the reference uses.
```

## Parity level

```text
Architecture:               maintainer implementation (speculators @ 55ebbd5b) —
                            ported math + state-dict key names, unit-tested
                            against a loop reference.
Training objective:         experimental reconstruction (upstream undisclosed).
Inference procedure:        inferred from the user guide + `CandidateSelector.select`
                            (no runtime in the pinned revision).
Checkpoints:                our unified container (`vlm-dflash-ckpt-v1`) with a
                            `dflash2` metadata block; canonical selector/conv key
                            names inside `model.safetensors` for future interop.
```
