# Upstream audit — DSpark

```text
Paper:                     "DSpark: Confidence-Scheduled Speculative Decoding with
                           Semi-Autoregressive Generation", arXiv:2607.05147
                           (DeepSeek; cited by both upstream implementations)
Official repo:             none published by the authors (paper-only);
                           maintained implementations are in the general speculator stacks
Reference implementations: vllm-project/speculators
                             `src/speculators/models/dspark/{core,config,metrics,model_definitions}.py`
                             `docs/user_guide/algorithms/dspark.md`
                             `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh`
                           NVIDIA-NeMo/Automodel
                             `nemo_automodel/components/speculative/dspark/{markov_head,loss,core,config,draft_qwen3,common,registry,target,target_utils}.py`
                             `docs/guides/speculative/dspark.mdx`
                             `examples/speculative/dspark/qwen3_0.6b_dspark.yaml`
Commit inspected:          speculators @ 55ebbd5b9b49e9bd223f47949c986aa4abfffbc3
                           NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
License:                   Apache-2.0 (both upstreams)

Backbone:                  identical to DFlash — speculators' `DSparkDraftModel`
                           subclasses `DFlashDraftModel` and only adds heads;
                           NeMo's dspark package reuses the dflash core the same way
Extra modules:             (1) low-rank Markov logit-bias head (`MarkovHead` /
                           `VanillaMarkov`): W1 = Embedding(vocab, markov_rank),
                           W2 = Linear(markov_rank, vocab, bias=False);
                           bias = W2(W1(prev_token)) added to the DFlash logits.
                           Variants: `vanilla` / `gated` (bias gated by the block
                           hidden state) / `rnn` (GRU-like state across the block).
                           (2) confidence head: linear(hidden [+ markov_rank], 1),
                           predicts per-slot acceptance probability
Training objective:        DFlash compound loss + confidence BCE
                           speculators: default loss_config = {kl_div: 1.0} (distil against
                           verifier logits) + alpha * BCE(confidence, accept_rate), where
                           accept_rate = 1 - TV(p_draft, p_target) = sum_v min(p, q)
                           (computed on verifier logits during online distillation);
                           confidence_head_alpha default 1.0; decay applies to the
                           confidence BCE via dflash_loss_decay (speculators metrics.py)
                           NeMo: CE + L1-probability term vs aligned verifier logits +
                           confidence BCE with the same accept-rate target
Inference procedure:       block-parallel backbone pass, then *semi-autoregressive*
                           in-block sampling: slot k's logits = base logits_k +
                           W2(W1(token sampled at slot k-1)); slot 0 conditions on
                           the anchor token. DSpark's default `sample_from_anchor=True`
                           drafts the anchor slot too (`block_size` proposals/round).
                           The confidence head is consumed by the sampling policy
                           (dynamic draft length), not by the verifier
Checkpoint format:         speculators HF-style speculator configs (speculators_model_type
                           "dspark", markov_rank, markov_head_type, enable_confidence_head,
                           confidence_head_with_markov, sample_from_anchor) — see
                           `DSparkSpeculatorConfig`
```

## What we port

- The Markov head factorization and its three variants, teacher-forced at
  training (ground-truth previous token per slot) and chained at inference
  (previously *sampled* token) — matching `sample_block_tokens` in both upstreams.
- The confidence head and its BCE term with `confidence_head_alpha`; the head
  input concatenates the previous-token embedding when
  `confidence_head_with_markov` (both upstream defaults are `True`).
- `sample_from_anchor=True` as the DSpark default (upstream parity), including
  the shifted supervision window and the widened verification round.
- W1 init: normal(std=0.01), as in speculators (`NN init normal_`); NeMo leaves
  the default embedding init. We follow speculators (the implementation with
  published checkpoints and vLLM runtime support).

## What we change (and why)

- **Confidence target (documented adaptation).** Upstream trains the confidence
  head against the analytical acceptance rate `1 - TV(p_draft, p_target)` from
  verifier logits (online distillation). Our pipeline is staged and offline: the
  feature shards store target *hidden states* only, not verifier distributions
  (full-vocab logits per position are not storable). The confidence target here
  is the **greedy per-slot acceptance indicator** (draft argmax == target token)
  — the exact event our temperature-0 verification gate measures, and the signal
  a dynamic-draft-length policy would consume. This is the one deliberate
  objective deviation; everything else (heads, bias, chaining, slot alignment)
  follows upstream.
- **Distribution term dropped.** The KL/L1 term against verifier logits is not
  computable from staged features; our base loss stays the frozen decayed block
  CE on hard regen targets. Consequences are tracked as a parity gap, not hidden.
- **Verification extension.** `sample_from_anchor=True` needs a round window of
  `block_size + 1` positions (anchor + `block_size` proposals) in
  `spec_generate`; the draft's own input block stays `block_size`.

## Parity level

**Maintainer-implementation-informed reconstruction** — architecture, head
semantics, sampling chain and slot alignment are ported from the two upstream
implementations at the commits above; the training objective deviates in the
two documented points (confidence target, dropped verifier-distribution term),
because the staged offline pipeline cannot reproduce online logit distillation.
Parity evidence to be recorded after the canonical benchmark: confidence
signal non-constant + calibrated (AUROC/ECE), and acceptance comparison vs
DFlash on identical features.
