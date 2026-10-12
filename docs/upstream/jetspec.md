# Upstream audit — JetSpec

Status: **implemented; validated on Qwen2.5-VL-3B** (Stage 4 of the GOAL
execution order). The audit below was completed first; implementation landed in
`vlm_dflash/jetspec/` + `vlm_dflash/algorithms/jetspec.py` on top of the
capture-v2 extension (see "Implementation record" at the bottom).

```text
Paper:                     JetSpec — "Breaking the Scaling Ceiling of Speculative
                           Decoding with Parallel Tree Drafting", arXiv:2606.18394
                           (cited in the maintainer implementation docstring)
Official repo:             NVIDIA-NeMo/Automodel @ bb5b34cee5f629aac00d2c61eaefe8e0ee94dace
Reference implementation:  nemo_automodel/components/speculative/dflash/jetspec_core.py
                             (JetSpecTrainerModule, _gather_teacher_logits)
                           nemo_automodel/components/loss/kd_loss.py (KDLoss)
                           nemo_automodel/recipes/llm/train_jetspec.py
                           examples/speculative/jetspec/qwen3_jetspec.yaml
                           tests/unit_tests/speculative/test_jetspec_core.py
Read-only local mirror:    jacky566/Automodel_lao @ d21fb5aa (carries the same
                           maintainer files; inspected for this audit)
License:                   Apache-2.0 (NVIDIA copyright header)
```

## Semantics recorded from the sources

```text
Backbone:                  unmodified Qwen3DFlashDraftModel; JetSpec only
                           changes the *attention mask* and the *loss*
Extra modules:             none (no new parameters at all)
Attention:                 block-causal in-block mask (query at within-block
                           offset i attends to offsets j <= i), i.e. the
                           causal=True path of create_dflash_block_mask;
                           DFlash's bidirectional in-block default must not change
Training objective:        forward KL(P_target || Q_draft) with temperature
                           scaling, uniform weighting over active draft
                           positions (no depth decay), slots k = 1..bs-1;
                           hard-label CE is NOT used (do not degenerate to CE)
Teacher signal:            full-vocab target logits [B, S, V] captured online by
                           the target wrapper (capture_logits=True); the teacher
                           distribution for the token at position p is the
                           target's logits at p-1, gathered at
                           label_indices[..., :-1]  (anchor .. anchor+bs-2)
Metrics:                   draft-vs-TEACHER greedy agreement (target argmax,
                           not the ground-truth token) + expected accepted-prefix
                           length per block (accept_len via the shared helper)
Inference procedure:       parallel tree drafting with branch conditioning
                           (runtime system; not part of the trainer release)
Checkpoint format:         upstream NeMo; ours uses the unified container
```

## What we port / change / why

```text
What we port:   (1) the causal in-block attention mask (our shared mask builder
                already exposes causal=True; the trainer passes it for JetSpec
                only); (2) the forward-KL objective with temperature scaling and
                uniform position weighting; (3) the teacher gathering rule
                (teacher logits at anchor+k-1 predict the token at anchor+k);
                (4) the draft-vs-teacher greedy metric + acceptance-length metric.
What we change: (1) teacher logits are NOT stored as [B, S, V] (~622 MB/sample
                bf16 - infeasible for a staged corpus). Capture v2 stores the
                target's FINAL hidden state per position ([S, H] fp16,
                ~8 MB/sample); training computes exact full-vocab teacher logits
                on the gathered positions only:
                teacher = lm_head(final_hidden[anchor+k-1]) - same matmul cost
                as the draft's own output head, no top-K approximation.
                (2) the KD loss is chunked over positions to bound memory;
                (3) step/curriculum plumbing stays shared.
Why we change:  the staged-pipeline contract (target off the training GPU,
                frozen features) must hold; (1) preserves the exact objective,
                (2)-(3) are memory-layout choices with identical math.
Parity level:   Maintainer implementation (training semantics + mask + metrics
                ported; staged-pipeline adaptation as above).
Runtime:        NOT ported. Production parallel-tree drafting is unsupported;
                the shared block-parallel decoder can run a JetSpec checkpoint
                with the causal in-block mask as a *proposal reference* only
                (single forward + greedy per-slot sampling, no tree
                conditioning), and must be labeled as such in eval results.
```

## Implementation checklist (Stage 4)

```text
1. capture v2: store final_hidden per sample (features format addition,
   backward compatible; JetSpec training fails loud on old features).
2. vlm_dflash/jetspec/: kd_loss.py (chunked forward-KL, T-scaling),
   trainer.py (JetSpecTrainerModule: causal mask + KD + teacher metrics),
   factory.py, algorithms/jetspec.py plugin.
3. tests: KD math (KL vs reference, temperature scaling, ignore-index mask),
   causal-mask leakage test (position i cannot see j > i), overfit sanity,
   teacher-gather alignment test (teacher position == label-1).
4. smoke + 2K runs; eval through the shared decoder with the causal mask
   (flagged reference-only) or a documented decision to defer eval numbers.
```

## Implementation record (2026-10-12)

```text
Capture v2:  ``capture.store_final_hidden=true`` additionally stores the target's
             final hidden state (post-final-norm, [S, H] fp16/bf16, ~+25% over
             the fused context) per sample; the manifest records
             ``stores_final_hidden``. Old stores load fine for every other
             algorithm; JetSpec training raises a capture-v2 error on them.
Teacher:     verified exact - on a real smoke sample, ``lm_head(hidden_states[-1])``
             equals the target's own output logits bit-for-bit (max|diff| = 0.0,
             argmax agreement 1.0). The KD teacher is the true target
             distribution, not an approximation.
Trainer:     ``JetSpecTrainerModule`` sets ``causal_in_block=True`` (the shared
             mask builder's causal path) and overrides the scoring seam:
             teacher hidden gathered at ``label_indices[..., :-1]`` and projected
             chunk-wise through the frozen lm_head; temperature-scaled forward KL
             (ported ``kd_loss.py``; chunked trainer path verified equal to the
             logits-in reference within 1e-5); metrics are draft-vs-teacher
             agreement and expected acceptance length.
Unit tests:  17 new tests (KD math/temperature/ignore-index/chunking, causal
             mask leak test, teacher alignment, fail-loud contracts, chunked-vs-
             reference loss, overfit, feature-store plumbing, registry/config).
Smoke:       COCO 64-row slice, shared corpus with the other algorithms'
             smoke runs; overfit sanity passed, gradient coverage 36/36,
             drafter visual check passed; eval token-exact (AL 1.123 vs init
             1.000).
Runtime:     the shared block-parallel decoder serves JetSpec checkpoints with
             the training causal in-block mask (``inference_in_block_causal``
             flag; additive mask hiding the in-call future block columns only).
             Mask-mismatch diagnostic: smoke ckpt max |dlogit| 0.070 / 0
             greedy-argmax flips; trained 2K ckpt max 9.0 / 7.6% flips - i.e.
             the trained model relies on the causal factorization, so eval must
             serve the mask. The paper's parallel-tree runtime (top-k branch
             conditioning) is still NOT implemented: proposals are per-slot
             greedy along a single path, and eval results must say so.
```
