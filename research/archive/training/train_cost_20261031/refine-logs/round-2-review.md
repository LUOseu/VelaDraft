# Independent research review — round 2

Date: 2026-10-31 (Asia/Shanghai). Reviewed `FINAL_PROPOSAL.md`, `REPORT.md`, generated `summary.md`, and `CODE_REVIEW.md`. This is a focused claim/design review, not an independent reproduction or source-code audit. No production edits or GPU jobs were performed. Same independent Codex reviewer route as round 1; the skill's requested GPT-5.5 was unavailable, so no cross-model claim is made.

**Verdict: REVISE. The evidence supports a bounded engineering memory result; it does not yet support a research contribution or training-cost claim.** The revised proposal correctly treats the execution policy and consumer-side reuse as future hypotheses. Reporting negative results improves evidential reliability, not the novelty of the mechanism.

## What is supported

The reported synthetic padded BF16 cell supports the statement that the combined compacted/precomputed CE arm lowered this process's measured peak allocated memory from 3.058 to 1.769 GiB, approximately 42%. The statement must retain its geometry, one-layer synthetic-backbone scope and combined-arm attribution. It does not establish a 42% reduction in a real Qwen3-VL training run, total device occupancy, total pipeline cost, or the isolated benefit of VJP precomputation.

The local teacher-reuse probe does not establish stable full-update acceleration. The checkpoint-CE probe exposes a memory/time tradeoff rather than a universal improvement. The padded cell's time ratio of 1.526 with interval [0.327, 1.658] cannot support a speed claim. Ten resampled timing pairs on a shared GPU do not remove interference bias or establish transfer to exclusive execution. The report says this clearly.

Fixed-input loss, metrics and gradient checks are evidence for first-order implementation consistency within the tested tolerances. The approximately 0.00514 global relative gradient L2 drift is explicitly incompatible with bitwise equality; global agreement also does not bound each small parameter's relative drift. With `lr=0`, there is no learning-trajectory or quality evidence. The report correctly refrains from making either claim.

## One wording correction

`REPORT.md` says: “Inference and export weights are unchanged by these training-only settings.” This can be read as asserting identical learned weights. Given the documented BF16 gradient differences, that interpretation is not established.

Use: **“Inference architecture, export format and loading path are unchanged; learned weights can differ through numerical training effects.”** If the intended claim is only that changing the switch does not directly mutate an existing checkpoint, say that explicitly. This is a wording correction, not a new implementation defect.

Also keep “preserves the real-arithmetic objective” separate from “tested finite-precision gradients are within a pilot tolerance.” A passing 2% aggregate BF16 gate is not itself a scientific justification of convergence equivalence.

## Next mechanism priority

**First, determine what caused the observed memory gain on real workloads.** Run dense, row-compaction-only, compacted checkpoint CE, compacted precomputed CE, and compatible Liger/CCE configurations on the same saved feature batches and anchor tables. This is the cheapest decisive comparison. It separates generic padding removal from backward-state liveness and from known optimized kernels. Measure real valid-slot distributions; do not choose only the most padded batch.

**Second, establish the actual resource constraint and crossover.** For each representative geometry, measure complete update time and both allocated/reserved peak memory under exclusive execution. Include optimizer state, the real model depth and data transfer. Evaluate equal feasible resources as well as equal fixed batch: memory savings matter operationally if they avoid OOM, enable a useful batch/context, or reduce device requirements. The current process-local 3 GiB synthetic result does not yet demonstrate such an operating point.

**Third, select one prospective novel mechanism from the remaining cost.** Consumer-side unique teacher rows are worth implementing only if full-distribution TV/hybrid/confidence consumers remain a material bottleneck after those baselines. The specific candidate is to consume shared teacher rows without expanding `[R,V]`, and to share normalization/projection lifetime across their consumers while retaining anchor-dependent student rows. Merely adding another `unique`/gather or a cache does not answer the novelty objection.

A simple execution policy is appropriate only after there is a reproducible frontier to select from. Routing among existing kernels is not automatically a novel systems contribution. It needs a structural predictor that generalizes to held-out targets/shapes and a demonstrable advantage over a static strong configuration; calibration/tuning cost and failure cases must be charged. If row compaction plus an existing fused loss explains nearly all gains, adopt it as a platform improvement and reconsider the paper's central mechanism rather than expanding the diagram.

## Updated assessment

| Dimension | Score | Change since round 1 |
|---|---:|---|
| Problem Fidelity | 9 | Still aligned with reducing training cost without silently changing objectives. |
| Method Specificity | 9 | Delivered primitives and support limits are now concrete; the proposed policy remains unimplemented. |
| Contribution Quality | 4 | Unchanged. The implemented ingredients are established techniques, and combined-arm benefits are not isolated. |
| Frontier Leverage | 7 | The many-to-one supervision representation remains plausible but has not produced a distinct validated mechanism. |
| Feasibility | 9 | Implementation and pilot evidence improve confidence in the bounded engineering work. |
| Validation Focus | 9 | Negative cells, uncertainty and missing baselines are clearly retained; real-workload attribution is the correct next gate. |
| Venue Readiness | 4 | Unchanged. No strong-baseline advantage, representative convergence, policy transfer or matched-quality cost evidence. |

Weighted score: **7.2/10** (7.15 before rounding), using the same prescribed weights. The small increase reflects implementation specificity and validation discipline; it is **not** an increase in novelty or publication readiness.

## Final claim boundary

Suitable present-tense description: “VelaDraft now provides opt-in bounded scoring paths and a reproducible diagnostic harness. A synthetic padded BF16 probe showed lower peak allocated memory; acceleration, production-workload savings, convergence and the proposed execution policy remain unverified.”

Unsuitable present-tense descriptions: “42% lower diffusion training cost,” “1.53× faster training,” “lossless training,” “objective-aware optimizer/platform established,” or “submission-ready.” The revised proposal avoids these claims and should continue to do so.

**Recommendation:** retain the tested options as engineering primitives; prioritize real-workload attribution and strong fused-loss baselines. Proceed to consumer-side teacher reuse only if profiling identifies it as the remaining dominant opportunity. No additional modules or automatic scheduler are justified by the current pilot alone.
