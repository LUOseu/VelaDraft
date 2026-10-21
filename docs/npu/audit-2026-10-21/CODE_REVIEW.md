# Independent code review — 2026-10-21

A separate Codex agent reviewed the production diff and audit harnesses. This is an independent code review within the same agent team, not an external cross-model scientific review. No remaining blocking correctness issue was identified in the reviewed paths after the fixes below.

## Scope

- Attention runtime accounting, mixed fused/fallback reporting, and measurement-window semantics.
- EOS-aware proposal/acceptance counters and controlled acceptance/KV-commit fixtures.
- Recipe weight reporting, compaction parity, input binding, paired optimizer/RNG state, and measured-work accounting.
- FP32-master attention-capacity experiments and their natural-data versus shape-only limitations.
- Weight-only continuation, export, and shared-feature forward capture of raw draft hidden states, corrected student hidden states, and actual student logits.

## Findings and resolution

1. **Continuation could export a nonfinite final update while reporting success.** The observer checked forward loss and selected head gradients, but the harness did not validate the optimizer's returned gradient norm or every final model tensor. Two injected-failure tests reproduced successful export with a NaN norm or a NaN final parameter. The harness now rejects either condition before checkpoint/export creation. Both tests pass, together with the existing successful continuation/export test.

   This hardening was made **after the measured continuation**, so it is not presented as a gate that the original run executed. A separate CPU, memory-mapped inspection verified all ten persisted loss/gradient-norm records and all floating/complex tensors in the original final checkpoint. Its semantic weight digest matches the original summary. The supplemental check passed; see [supplemental_integrity.json](raw/continuation/supplemental_integrity.json). Original run artifacts and their source hashes were not modified, and the ten training updates were not rerun.

2. **Directory-form compaction inputs were insufficiently bound to the parity gate.** Hashing only the feature manifest allowed feature shards or teacher-head bytes to change without invalidating an existing PASS. The harness now incrementally hashes the contract and every feature-store file, including head, index, manifest, and shards. Tests verify that changing shard/head bytes with an unchanged manifest changes the binding. File-form bundles retain their existing full-file SHA256 behavior.

   This generic harness fix was also made **after the measured A experiment**. That experiment used a file-form bundle already bound by its full-file SHA256, so its input-binding behavior is unaffected. Its original source hashes and evidence remain preserved; future runs use the hardened harness.

3. **Earlier parity/state verification gaps were closed.** Review identified stale parity-file reuse, absent explicit RNG/state identity evidence, and scalar-only weight checks. The current harness binds parity to checkpoint, input, config, anchor plan, and source; restores RNG between parity legs; checks paired model/master/optimizer/scheduler/RNG digests; and observes actual teacher row mappings, masks, and position-weight tensors. The label for strict numerical comparison no longer claims that the NPU execution was a CPU run.

4. **Recipe reporting scalar differed with zero-padded reduction layout.** The production fix reduces the same ordered nonzero weight terms for active recipes supporting compaction. It preserves signed nonzero terms and does not replace these legacy reporting weights with D-PACE weights. Inspection confirmed that the recipe objective does not normalize itself with this reporting scalar. Tests verify zero-padding invariance and unchanged loss/gradients before versus after the fix within each layout. The original failed A parity evidence is retained; the same gate was rerun under the changed source. This is a reporting bug fix, not a relaxed tolerance.

5. **Initial B optimizer measurements were not production FP32-master measurements.** Those direct-BF16-AdamW pilots remain explicitly archived as preflight, excluded from the final table. The final 18-cell matrix uses the production FP32-master optimizer helper. Tests verify master and moment dtypes, actual gradient transfer, and parameter copyback.

## Nonblocking semantics and claim limits

The subsequent evidence-generator review found fail-open reporting checks,
without finding a discrepancy in the actual stored measurements. Mutation
tests reproduced acceptance of a wrong attention geometry, dense execution
reported as chunked, missing pilot updates, and an incorrect compaction switch.
The generator now checks actual execution counters, geometry, dtype, paired
inputs/source and ordered measurement windows. It also checks DSpark's recorded
geometry and coefficients, and DFly's 42 checkpoint tensor shapes and declared
and observed stage coefficients. The CUDA status is declared in the matrix
legend. The reviewer ran 41 generator/matrix tests successfully; the measured
time/HBM values and original raw artifacts were unchanged.

- `completed_attention_calls` counts completed attention-core execution. Recording occurs before the output projection, so a later output-projection exception can leave that core call counted. It must not be described as a count of successfully completed whole attention modules, model forwards, optimizer updates, or backward kernels. No further production change was made during this review, preserving the source used by the CUDA regression.
- Shared-forward capture's first-head-call rule is valid for the reviewed trainer: the corrected student projection precedes teacher-distribution projections. The hook captures actual corrected student inputs and logits rather than only the uncorrected draft output. Cross-device numerical conclusions still require matching input/anchor/weight identities and the separate comparison artifacts.
- B dense baselines are explicitly described as harness full-query SDPA comparators; they are not stock HF dense-mask-once throughput. All tested shapes fit. The evidence supports a memory/time tradeoff, not an observed OOM capacity boundary or speedup.
- Natural B samples have at most 825 real context tokens and 249 available anchors in the selected sample. Padding a context bucket to 4096 does not establish natural 4096-token training. Fully allocated shape-only tests are separate.
- Continuation is pretrained-weight continuation with a fresh optimizer and fixed stage2, not from-scratch training or cross-backend optimizer/RNG resume. Pending downstream evidence must remain pending until its own checks finish.

## Validation performed by this reviewer

The following final targeted suite passed: **56 tests passed** (collection independently checked: 28 acceptance fixtures, 5 attention-accounting, 5 weight-reporting, 6 compaction, 3 continuation, and 9 CUDA-harness tests).

```sh
.venv/bin/python -m pytest \
  tests/test_acceptance_path_fixtures.py \
  tests/test_attention_run_reporting.py \
  tests/test_recipe_weight_reporting.py \
  tests/test_audit_compaction.py \
  tests/test_audit_continue.py \
  tests/test_audit_cuda.py -q
```

The focused compaction/continuation suite also passed: **9 tests passed**. The new continuation and binding tests were first observed failing before their corresponding fixes. This review does not substitute CPU tests for hardware evidence; A/B/C hardware artifacts retain their own source identities and validation scope.
