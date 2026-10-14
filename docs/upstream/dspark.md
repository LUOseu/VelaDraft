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

- **Verifier-distribution objective (recipe `upstream_dspark`).** The upstream
  default loss is `loss_fn="kl_div"` — per position forward
  `KL(p_verifier || p_draft)` against the verifier logits, decay-weighted, with
  the confidence BCE against `accept_rate = 1 - TV(p_draft, p_target)` — see
  speculators `models/dspark/metrics.py::compute_metrics`,
  `losses/utils.py::compound_loss/loss_function`, `losses/eager.py::kl_div_loss`
  and `tv_loss`, default resolved in `train/config/schema.py`
  (`_DEFAULT_LOSS_FN` falls back to `"kl_div"`; `dflash_decay_gamma=4.0`).
  We reproduce this objective without storing verifier logits: the stored
  target **final hidden states** are projected through the frozen output head
  on demand (`TeacherLogitsStream`, position `label - 1`, bounded fp32 chunks,
  activation-checkpointed). The KL term and the analytic confidence target
  share one chunk stream per forward; the target is computed from the live
  drafter distribution each step (detached), never cached. Keyed by
  `algorithm.train_recipe: upstream_dspark`
  (`configs/dspark/qwen2_5_vl_3b_2k_recipe.yaml`); requires
  `capture.store_final_hidden=true` and a store with `stores_final_hidden`.
- **Plain runs (documented adaptation).** Without the recipe, the loss stays
  the frozen decayed block CE on hard regen targets and the confidence target
  is the **greedy per-slot acceptance indicator** (draft argmax == target
  token) — the exact event our temperature-0 verification gate measures. This
  is the original port behavior, kept as the default so recorded runs and
  their plan digests stay byte-identical.
- **Normalization convention.** Our decay-weighted terms divide by
  `sum(mask * decay)` (the repo-wide `DFlashDecayLoss` convention); the
  speculators `loss_function` divides by the plain mask count. Gradient
  direction is identical; the effective scale differs by the batch's mean
  decay factor. Also: our recipe uses the run's `loss_decay_gamma` (dev configs
  use 7.0; set 4.0 to match the upstream default) and trains over the full
  target vocabulary (upstream's example pipeline can map to a 32k draft vocab).
- **Verification extension.** `sample_from_anchor=True` needs a round window of
  `block_size + 1` positions (anchor + `block_size` proposals) in
  `spec_generate`; the draft's own input block stays `block_size`.
- **Head ownership (implementation detail, no semantic change).** The heads are
  constructed once by `Qwen3DSparkDraftModel` (from the `algorithm.dspark`
  knobs baked into its `dflash_config`); `DSparkTrainerModule` reuses those
  instances instead of building its own copies. This keeps the training loop
  coherent by construction: `module.draft_model.to(dtype)` casts the heads with
  the backbone, and the optimizer's master parameters plus the saved
  `state_dict` come from `draft_model.*`, so head weights are actually updated,
  checkpointed and reloaded. Trainer-side `markov_rank` / `markov_head_type` /
  `enable_confidence_head` / `confidence_head_with_markov` arguments are
  validation-only overrides of the model config.

## Parity level

**Maintainer-implementation-informed reconstruction** — architecture, head
semantics, sampling chain and slot alignment are ported from the two upstream
implementations at the commits above. The training objective is selectable:
`train_recipe: upstream_dspark` reproduces the speculators default objective
(KL distillation + analytic confidence target) from stored final hidden states,
with the three documented convention differences above (normalization divisor,
decay-gamma default, draft vocab); plain runs keep the original
decayed-CE + greedy-confidence port. Objective-level parity evidence (loss and
gradients vs a direct full-vocab reference, chunk invariance, detached analytic
target, gradient coverage of both heads) is recorded in the CPU tests
(`tests/test_objectives.py`, `tests/test_dspark.py`); a real-target GPU
comparison is pending (see the GPU_PENDING list in
`docs/plan/REVIEW_REMEDIATION.md`).
