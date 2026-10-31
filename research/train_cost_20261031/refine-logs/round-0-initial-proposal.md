# VelaDraft: exploit repeated supervision in block-drafter training

## Problem Anchor
- Bottom-line problem: reduce the total cost of training diffusion block drafters within VelaDraft while preserving the specified learning objective and validating attained quality.
- Must-solve bottleneck: anchor expansion repeats frozen teacher/output-head work and materializes vocabulary-sized intermediates; existing feature caching does not remove this scoring cost.
- Non-goals: change serving semantics, claim a new diffusion architecture, or equate microbenchmark speed with end-to-end savings.
- Constraints: existing main 9404b93; shared four-A6000 host; preserve and restore user tasks if paused; opt-in behavior; CPU and CUDA correctness before long runs.
- Success condition: reproducible objective/gradient checks and lower measured training resource cost, followed by matched-quality multi-seed evaluation before paper claims.

## Candidate routes
A. Exact execution of repeated supervision: map each expanded (sample, anchor, slot) teacher request to (sample, predecessor position), project each distinct teacher row once per bounded chunk, then restore original row order. Keep the student computation and complete vocabulary/objective unchanged. This directly exploits overlapping prediction windows; it does not assume student states are equal across anchors.
B. Frozen-head streaming CE: compact supervised slots, chunk the head and loss, retain hidden-sized first-order gradients. This is useful engineering but overlaps fused linear CE / Cut Cross-Entropy / Liger. Treat it as a baseline/supporting primitive, not standalone novelty.

## Dominant hypothesis
An explicit distinction between reusable teacher state and anchor-dependent student state exposes a workload-dependent redundancy that can be removed without changing the objective. The systems contribution must be a demonstrated execution policy and crossover model, not the claim that memoization or chunking is new. Per-chunk reuse is the smallest first implementation; across-chunk scheduling or caching is conditional on measured residual waste.

## Mechanism and invariants
For request r=(b,a,k), teacher identity u=(b,a+shift+k-1). The full teacher distribution is softmax(H[u] W^T); student state remains indexed by r. Build unique indices and inverse mapping inside the consumer's existing bounded chunk. Project H[unique] and gather by inverse. Never cache across optimizer steps, records, model identities or stale captures. Padded and packed indexing remain governed by the existing label table. Chunk boundaries, dtype rounding and downstream gradient accumulation can cause numerical differences: test them and avoid bitwise claims.

No new trainable components. Use explicit execution switches and counters (requested/projected rows, calls, peak memory, full update time). Unsupported heads or objectives fail clearly. Existing defaults remain the baseline.

## Three validation blocks
1. Cheap falsification: CPU fp64/fp32 loss + every parameter gradient; adversarial repeated positions from different samples, packed/compacted rows, empty chunks; two-stage KL/TV and confidence paths. CUDA BF16 repeatability floor and paired numerical tolerance.
2. Mechanism: replay identical feature batches with fixed anchors; no-reuse vs per-chunk reuse; vary anchor density, block length, vocabulary, chunk size and answer length. Record projection compression, total update time, peak allocated/reserved memory. Low-overlap control must expose the overhead/crossover. Compare existing compiled recipe loss and DFlash2 switches rather than an intentionally weak baseline.
3. Paper gate: same target, data, seeds, sample order, optimizer/update count and quality evaluation; report GPU-hours to a preregistered validation acceptance threshold, complete pipeline stage costs and amortization across methods. At least two targets and three relevant methods, multiple seeds; locked held-out quality and native serving evaluation. Not launched until 1/2 justify spending GPU time.

## Claim ledger at design time
- Repeated teacher rows exist in the implementation: code-observed, not yet workload-quantified.
- Dedup preserves the real-arithmetic objective: derivation, numerical tests pending.
- Training is faster or cheaper: unmeasured.
- Paper-level novelty: unverified; independent review and current primary-source search required.
- Mainstream platform / submission ready: not claimed.
