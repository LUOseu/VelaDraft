# Upstream audit — JetSpec

Status: **audited; implementation planned** (Stage 4 of the GOAL execution
order). The audit is complete; implementation is scheduled behind the capture
extension described below.

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
