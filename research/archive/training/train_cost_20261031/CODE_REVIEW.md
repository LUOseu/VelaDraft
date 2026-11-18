# Independent implementation review

Date: 2026-10-31. Base and HEAD: `9404b934838483000ae01714b5a2a8cfd130915f`; review covers uncommitted changes. Reviewed `objectives.py`, trainer/factory/config wiring, `tests/test_teacher_projection_dedup.py`, `scripts/benchmark_teacher_projection.py`, and `docs/superpowers/plans/2026-10-31-training-cost.md`. Follow-up scope includes `dflash/chunked_ce.py`, the `ce_chunk_size` route and algorithm capability gates, `tests/test_chunked_ce_training.py`, and the benchmark's CE mode. Reviewer used the `requesting-code-review` skill, made no production edits, and ran no GPU job.

The scope expanded from teacher-projection deduplication to a separate controlled-CE memory optimization after the coordinator reported no stable full-update gain from the initial dedup probe. This review did not independently run or certify those GPU measurements. The CE path is assessed as an engineering memory/time tradeoff, not as a new objective or research novelty.

A final bounded extension adds `ce_backward='precompute'`, retaining checkpoint as the default. The coordinator reported a checkpoint-CE memory improvement with a runtime regression and requested review of this alternative; those GPU numbers are outside this independent CPU review. The new branch caches the scalar CE's hidden-state VJP in forward and applies the upstream scalar in backward.

## Assessment

**Ready within the tested scope; no remaining blocking finding.** No blocking objective/indexing/gradient defect was found for the intended factory-created, frozen, unsharded `torch.nn.Linear` head in either optimization. The benchmark gradient-coverage defect was fixed and its negative test passed. Remaining findings are contract/documentation and regression-coverage improvements. CUDA numerical behavior, performance, TP execution and convergence are not approved by this CPU review.

## Strengths

- Teacher identity is the flattened `(sample, predecessor position)` index, and inverse gather restores the original request order. It neither conflates different samples nor merges anchor-dependent student states.
- Reuse stays inside each existing loss chunk; no cache survives a call. Empty/tail slices, compacted block sample maps, bias and live teacher-value changes preserve the original behavior in the exercised cases.
- Both loss stages and DSpark confidence training are exercised against the unchanged objective and every observed trainable gradient. The execution flag defaults off. The runtime config and constructor reject CE-only/no-teacher configurations.
- Workload counters include activation-checkpoint replay and distinguish requested, projected and reused rows. The counters were independently checked with forced multiple chunks.
- The benchmark uses fixed anchors, alternating arm order, real trainer backward/optimizer work, repeatability observations, raw timings and hashes. Its scope explicitly excludes convergence and complete-pipeline claims.
- Chunked CE reconstructs the original ordered scalar-NLL layout and uses `DFlashDecayLoss._reduce`, preserving the existing weighting, denominator and dtype policy. Nonzero supervised slots are selected after packed-document masks, sample weights and DFly's hidden correction are applied. Prediction IDs preserve accuracy and acceptance metrics without retaining full logits.
- `ce_chunk_size=0` keeps dense scoring. Runtime config and trainer capabilities reject DSpark, DFlash2, JetSpec, Domino, teacher-distribution recipes and variable-prefix objectives. CE execution metadata records the size while model state keys and training-plan digests remain unchanged.

## Findings

### Resolved important finding: benchmark could pass after losing a parameter gradient

**Original location:** `scripts/benchmark_teacher_projection.py:110-128`; resolution at `gradient_drift` and the parity gate, currently lines 45-60 and 140-144.

`drift(grads)` iterates only over the candidate dictionary, whose entries exclude parameters with `grad is None`. It does not compare candidate/reference key sets. If one parameter loses its gradient in the reuse arm while the remaining gradients match, the computed relative L2 can remain zero and `parity.passed` can be true. The recorded parameter count does not participate in the gate. This is a validation gap, not an observed defect in the current dedup arithmetic; the CPU trainer test already checks identical key sets.

**Verified resolution:** `gradient_drift` now records missing/extra names and shape mismatches, checks finite values, and reports `complete_and_finite`; the final gate requires that result for both repeat and optimized arms. `test_benchmark_gradient_gate_rejects_missing_and_nonfinite_gradients` passed, as did the CE script smoke run with complete 14-parameter coverage.

### Minor: exact Linear type is not a complete custom-behavior guard

**Location:** `vlm_dflash/objectives.py:860-864`.

An exact frozen `torch.nn.Linear` can have a forward hook or instance-patched forward that changes row semantics. These pass the new type/parameter check. A deterministic hook subtracting the output's batch mean produced a maximum baseline/reuse logit difference of `0.11294071170662218` in the reviewer's CPU reproduction. The shipped factory's plain Linear is unaffected.

**Recommendation:** explicitly document that only unmodified row-local Linear execution is supported, or fail closed on semantics-changing hooks/instance forward overrides if the public contract promises to reject all custom heads. Do not describe the current structural test as proof of row locality. Read-only instrumentation hooks can be distinguished in documentation rather than assumed safe automatically.

### Resolved important robustness finding: noncontiguous precompute inputs failed

**Location:** `vlm_dflash/dflash/chunked_ce.py:73` and the indexed gradient write in `_FrozenHeadCE.forward`.

The first implementation used `zeros_like(hidden)`, which preserved a dense transposed layout, followed by `grad_hidden.view_as(flat)`. A valid hidden tensor created with `randn(2,12,21).transpose(1,2)` raised a view/stride error. Merely changing `view` to `reshape` would be unsafe because indexed writes could land in a temporary copy.

**Verified resolution:** gradient storage is now explicitly contiguous. The new noncontiguous-input FP64 regression compares the loss and scaled hidden gradient with dense CE, and passed in the final 22-case CE suite. No remaining precompute blocker was found in the tested first-order BF16/FP32 training scope.

### Minor: persist the multiple-chunk replay counter regression

**Location:** `tests/test_teacher_projection_dedup.py:69-89`.

The committed-shape objective tests have 56 rows / 8 blocks, below the 1024-row / 128-block consumer limits. They exercise checkpointing but do not cross a loss chunk boundary or assert replay call counts. The separate direct stream slicing test covers gathers, not chunked loss replay.

**Recommendation:** retain a small test with monkeypatched chunk limits and compare loss, gradients and exact requested/projected/call totals. Independent ad hoc verification passed; this is a future-regression coverage suggestion, not a discovered implementation failure.

## Verification performed

Python: `/Users/Admin/grounding-parallel/veladraft-npu/.venv/bin/python`; CPU only.

1. `pytest -q tests/test_teacher_projection_dedup.py tests/test_objectives.py`: passed, 62 cases at the initial review snapshot.
2. `pytest -q tests/test_recipe_row_compaction.py tests/test_recipe_weight_reporting.py tests/test_trainer_smoke.py tests/test_dfly.py tests/test_dspark.py tests/test_attention_run_reporting.py`: passed, 62 cases.
3. The subsequently added runtime-config rejection cases: `pytest -q -o addopts='' tests/test_teacher_projection_dedup.py -k config_rejects`: **6 passed**.
4. Independent forced chunk sizes `LK_CHUNK_ROWS=11`, `E2E_CHUNK_BLOCKS=3`: KL, hybrid, TV and end-to-end TV losses and student gradients all matched. The 56-request forward plus backward reported 112 requested rows; LK-family reuse projected 78 rows in 12 calls, and E2E projected 64 rows in 6 calls.
5. CPU script smoke run with float32, hidden 64, vocabulary 128, four anchors, both recipe stages and strides 1/8: all four numerical gates passed with zero observed gradient drift. Overlap cases projected 20 of 56 replay-inclusive requests; non-overlap cases projected all 56. Raw review-local output: `/tmp/teacher_projection_review_probe.json`. Its tiny CPU timings are not performance evidence.
6. `git diff --check`: clean at the reviewed snapshot.

### Follow-up CE verification

- `pytest -q -o addopts='' tests/test_chunked_ce_training.py tests/test_training_plan_fusion.py tests/test_regroup.py tests/test_recipe_row_compaction.py tests/test_teacher_projection_dedup.py`: **86 passed**. This suite overlaps the initial test list; it is not 86 additional unique cases. It covers the completed config and benchmark-gate edits. A run while those edits were still in flight had five expected missing-implementation failures; the completed rerun above passed.
- Independent direct-loss checks: **24 combinations** spanning float64/float32/BF16, decay disabled/enabled, `weight_dtype=auto/fp32`, and `normalize=mean/tokens`, with fractional sparse masks and an all-zero-weight sample. Loss, hidden gradients and per-position counts matched within dtype-appropriate tolerances. These BF16 CPU checks are not CUDA BF16 evidence.
- Independent full-trainer checks: DFlash and DFly with two packed documents per sample, reset positions, document boundaries, fractional sample weights and a zero-weight sample preserved loss, weight/count/acceptance metrics, gradient-key coverage and every trainable gradient.
- Explicit controlled-CE configurations for DFlash/DFlare/DFly accept `ce_chunk_size=5`; toggling it leaves `training_plan_digest` unchanged. Dense/chunked modules have identical model-state key sets; `execution_switches` records the chosen chunk size.
- CE benchmark smoke: CPU float32, hidden 64, vocabulary 128, four anchors, chunk size 5, strides 1/8, stage 0. Both numerical gates passed; maximum global gradient relative L2 was `6.52e-8`, every one of 14 parameter gradients was present and finite, and metrics matched. Raw review-local output: `/tmp/chunked_ce_review_probe.json`. Small CPU timings show checkpoint overhead and are not a GPU performance claim.

### Final precompute verification

- Final `pytest -q -o addopts='' tests/test_chunked_ce_training.py`: **22 passed**, including both backward strategies, retained-graph scaling, trainable-head rejection, the benchmark coverage gate and the noncontiguous-input fix.
- Independent **32 direct FP64/FP32 cases** span mean/raw-sum reductions, decay disabled/enabled, original/FP32 reduction dtype, fractional masks and all-zero masks. Dense and precomputed CE losses, scaled hidden gradients (`0.37`) and position counts match within the exercised precision tolerances.
- A full trainer with all sample weights zero matches dense loss, all count/acceptance metrics, gradient-key coverage and every zero gradient; it executes no head projection.
- Inspecting the custom autograd output before backward showed exactly one saved tensor, shape `[2,21,12]`, with no `grad_fn`: the cached hidden-sized VJP. The inner chunk graphs are consumed with `autograd.grad`; no vocabulary-shaped tensor is retained by the outer Function.
- CPU BF16 autocast produces FP32 CE loss on both routes. For an upstream scale of `0.37`, observed hidden-gradient relative L2 was `0.00343`; moving the scalar past BF16 GEMM/casts is numerically different, as documented. This is not a bitwise or convergence-equivalence check.
- CPU CLI smoke with `--optimization ce --ce-backward precompute --ce-chunk-size 5`, float32, hidden 64, vocabulary 128 and four anchors passed: 14 finite parameter gradients, exact metric counts, global gradient relative L2 `2.75e-7`. Raw review-local artifact: `/tmp/precomputed_ce_review_probe.json`.
- The training loop already restricts live draft dtype to BF16/FP32. FP16/GradScaler, higher-order gradients, distributed precompute and CUDA autotuning were not added to this bounded review. `once_differentiable` intentionally excludes higher-order differentiation; fixed supervision weights and frozen unsharded Linear heads are explicit precompute requirements.

## Evidence boundaries for reporting

- `teacher_head_rows` reductions are projection-work reductions, not complete-update speedups. Reuse still expands `[R,V]` logits for the consumer and may simultaneously retain `[U,V]` producer logits, increasing transient memory.
- A stride-1 synthetic workload establishes an overlap mechanism, not its prevalence in real captured batches. The low-overlap control is necessary and must retain regressions.
- The revised script exposes `--compile-tv-kl`; the reviewer did not execute a compiled CUDA baseline. Paper-facing superiority over optimized execution additionally requires compatible compiled/fused baselines and representative fixed-batch replay, as the implementation plan/research review already require.
- Existing `DFlashDecayLoss.forward_fused` (`decay_loss.py:577`) already implements checkpointed head+CE, with FP32-upcast semantics. The new route adds trainer integration, original-dtype scoring, mask compaction and prediction-ID metrics. Describe that engineering difference precisely; checkpointed CE itself is established.
- In `--optimization ce` mode, `--stages 0 1` currently repeats the same controlled-CE objective under two stage labels. Use `--stages 0`, or normalize/report the stage as inapplicable; these are not two distinct CE objectives or independent workload shapes.
- Checkpoint CE replays the selected student projection during backward; precompute projects once and computes its hidden VJP in forward. Thus `student_head_rows` has different replay-inclusive totals between strategies and is not a complete FLOP/runtime measurement. Report time, work and memory separately. Checkpoint CE handles DTensor outputs in code, while precompute rejects sharded heads; distributed execution was not exercised here.
- The 2% BF16 global gradient tolerance is a pilot gate. Recording a repeatability floor does not by itself justify that tolerance or establish per-parameter/long-run convergence agreement.
- DTensor parameter rejection was inspected, not executed under a distributed group. No TP, CUDA, NPU, real-checkpoint quality, or matched-quality cost claim follows from this review.
