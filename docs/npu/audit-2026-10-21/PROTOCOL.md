# PR30 bounded follow-up protocol (2026-10-21)

Baseline: main `8cacc9d`; audited NPU head `4625ef6`. Keep prior smoke evidence unchanged. New tests and results are a separate dated extension of the existing support matrix.

## Fixed claims and stop conditions

1. DSpark `upstream_dspark`, DFly `upstream_two_stage` cold-start and e2e-TV stages must execute the declared losses, consume real teacher distributions and update specific heads. Missing/unsupported paths fail; never replace them with default CE.
2. DFly stage2 row compaction is compared on identical natural records, anchors, snapshot, fresh FP32-master AdamW state, RNG, horizon and attention. Negative performance is a valid result.
3. Dense SDPA, checkpointed dense SDPA, checkpointed query-chunked SDPA differ only in attention execution. Measure full updates, actual memory and OOM; analytical mask bytes stay separate.
4. Actual acceptance paths and EOS/budget accounting are tested with controlled proposals. Real exported weights require nonzero acceptance on a fixed 64-prompt natural held-out set.
5. CUDA regressions use identical base/head inputs and existing software. Busy devices are not preempted. Hardware gaps remain unverified.

Compute cap: 4 NPU-device-hours and 1 GPU-device-hour for this extension, counting pilots and failed runs. Memory lane <=1.5 NPU-h; root recipe/compaction/continuation <=2.5 NPU-h. Per-cell five-update pilot precedes timing legs; no post-hoc shape search after negative results. Unavailable CUDA time is not counted as compute, but is reported.

## Frozen data

64 training rows selected by existing-regeneration token-length quantiles before timing; 64 held-out unique images excluded against every prepared training image ID (24,000 IDs) in the original data. Selection manifest carries record/image hashes. Reuse generated teacher answers; no new large regeneration. Natural input lengths are 297–825 tokens; do not represent these as natural 4096-token examples. Capture a superset of layers [1,9,17,25,33] and final hidden with the same Qwen2.5-VL-3B identity. Shared features originate from NPU capture and are reused on CUDA.

## A: formal matched DFly stage2

Use repository `configs/dfly/qwen2_5_vl_3b_2k_recipe_matched.yaml`: 3 layers, hidden 2048, FFN 11008, 16 Q / 2 KV heads, head dimension128, block8, requested anchors256, microbatch8, BF16 forward/FP32 masters. This is the documented formal **matched three-context-layer** recipe, not the separate five-context variant. Initialize from the existing trained R3 weights when geometry matches; fresh optimizer state is equal for every leg and is explicitly not an optimizer-state continuation of that training run.

Freeze 8 microbatches using masks/lengths: homogeneous valid-anchor groups and alternating low/high valid-anchor groups. Preserve all original supervision. Store batch IDs and CPU-generated anchor tables before evaluating speed. At least one batch must have padding blocks and one low-padding batch; otherwise mark the selection gate failed. Stage rule cold_start_fraction=.5, horizon100, logical stage progress75..89 (always stage2); the optimizer scheduler uses horizon100 starting at local step0 in every15-update leg (distinct from logical recipe stage progress). Attention sdpa_chunked128 and compile-TV/KL off in both arms. Three pairs, seeds42/43/44, alternating OFF/ON and ON/OFF order; each leg5 warmup+10 measured updates.

Correctness on all8 batches compares loss, weight, valid supervision, teacher rows and parameter gradients before timing. Preserve strict existing CPU row-compaction checks (loss rtol1e-9/atol1e-12 and exact gradients). The existing H1 BF16 numerical envelope rtol=.02/atol=.02 is separately reported for hardware comparisons; strict failures are retained even if this looser diagnostic passes. Teacher-row mapping retains existing FP32 rtol1e-5/atol1e-6 reference checks. No tolerance changes after outcomes. If the numerical gate fails, report the failure; performance is not a correctness claim.

## B: capacity

DFlash formal geometry above, microbatch1; (context bucket, requested anchors) = (1024,64),(4096,256),(4096,512). Natural records remain complete and unchanged; separately report effective length, padded allocation and achieved anchors. Full allocated shape-only tensors are explicitly synthetic diagnostics. Compare identical attention-only checkpoint scope (mask+SDPA, not FFN/layer checkpoint). Five pilot updates then three measured updates per fixed cell. Preserve OOM/errors. No fusion extension required.

## CUDA and C

Use existing 3090 environment if an idle device is available; otherwise a genuinely available A6000 slot, never a queued-job gap. Base/head DFlash FlexAttention, formal DSpark objective, DFly stage2 and Domino each use the same short diagnostic microbatch1 override and FP32-master update. No upgrading active environments. C starts from existing R3 weights, performs NPU weight-only continuation with fresh optimizer, then exports. It is not from-scratch NPU training or cross-backend bitwise resume. Shared-feature forward compares the same final weights and CPU anchor tables across devices. On CUDA, native/export formats use the same verifier on64held-out prompts, greedy64newtokens, sequential verification. Report accepted tokens, verification rounds, tau (progress/round), A (accepted drafts/round), proposal denominator, EOS and output equality.

## Delivery

Add objective/profile/stage/backend/mode records to the existing matrix with dtype, geometry, effective rows/chunks, data kind, counts, source revision and artifact digests. Provide a fail-closed validator and recomputable tables. Add cumulative attention execution counters and requested/effective compile gating. Keep implementation, numerical evidence and performance evidence separate.
