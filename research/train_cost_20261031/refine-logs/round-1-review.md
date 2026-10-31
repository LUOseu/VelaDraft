# Independent research review — round 1

Review date: 2026-10-31 (Asia/Shanghai). Reviewed proposal: `round-0-initial-proposal.md`; repository base: `9404b93`. Reviewer route: independent Codex sub-agent with inherited model. The skill's requested `gpt-5.5` model was unavailable, so this is **not** a cross-model GPT-5.5 review. No production files were edited and no GPU experiment was run by this reviewer.

**Verdict: REVISE. Weighted score: 6.9/10.** Proceed with the small implementation as a falsification experiment and platform improvement. Per-chunk teacher-row deduplication alone is not a convincing primary contribution for MLSys/OSDI/SOSP. A unified framework can still become a strong artifact, but packaging existing optimizations is not sufficient evidence of research novelty.

## Scores

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 9 | The work directly targets training cost while preserving the chosen learning objective. |
| Method Specificity | 8 | The index map and location in `TeacherLogitsStream` are precise. The execution policy, resource envelope, and unsupported-head contract remain unspecified. |
| Contribution Quality | 4 | Unique-plus-gather is straightforward common-subexpression elimination; streaming CE and hidden-gradient precomputation overlap established work. |
| Frontier Leverage | 7 | The method correctly separates frozen teacher state from anchor-dependent student state. It does not yet exploit this distinction beyond a local optimization. |
| Feasibility | 9 | A bounded, opt-in version is implementable and cheap to falsify. Large paper claims are not justified by the current evidence. |
| Validation Focus | 8 | Fixed-anchor parity and low-overlap controls are well chosen. Baselines need to include current external systems and objective-compatible fused losses. |
| Venue Readiness | 4 | No measured bottleneck prevalence, crossover, full-update gains or convergence result yet; novelty remains thin. |

Weighted score uses the skill's prescribed weights: 15%, 25%, 25%, 15%, 10%, 5%, 5%.

## Central objection a reviewer will raise

“The proposed optimization projects identical frozen hidden states only once. This is ordinary deduplication, while the paper's other components—offline feature caching, chunked output projection and fused CE—are already established. Why is this a new systems result rather than a useful patch? Why is a full training framework needed to realize it, and how much does it save beyond optimized current baselines?”

The right response is evidence and a more precise systems abstraction, not more components or a stronger name.

## Code-grounded findings

1. `TeacherLogitsStream.logits` currently maps expanded block rows to flattened `(sample, predecessor_position)` indices and projects all of them. Overlapping anchor windows can repeat these exact indices. Deduplication is valid for a deterministic row-local frozen head, including bias, provided inverse gather preserves consumer row order.
2. The student rows are not reusable merely because they predict the same token. Their conditioning anchor, mask and corrupted state differ. Merging student states or counting one student prediction per unique target changes the objective.
3. Default controlled CE does not use this teacher stream. A teacher-reuse result cannot be presented as improving every DFlash training mode. The coverage is teacher-distribution recipes, until other seams are separately integrated and tested.
4. The current LK and end-to-end TV losses call the stream from checkpointed chunk functions. A local dedup optimization will repeat during backward recomputation and cannot eliminate repeated work across chunk boundaries or across the forward/backward boundary. Counter semantics must include these replayed calls.
5. Deduplication alone does not eliminate vocabulary-sized consumer output. The implementation usually holds unique logits `[U,V]` and gathered logits `[R,V]` simultaneously. Relative to the existing `[R,V]` producer output, peak live memory can grow by approximately `U*V*bytes_per_entry`, plus index buffers. A memory reduction claim is especially unsafe.
6. The mapping must include sample identity. Same token id or same position in two different samples is not sufficient. Packed and compacted inputs must reuse the exact existing sample/position map; clamp behavior and invalid slots must remain unchanged.
7. Smaller GEMMs can have lower hardware utilization and different floating-point rounding. `R/U` is a projection-row ratio, not an expected wall-time speedup or a bit-exactness claim.

## Closest prior work and overlap

| Primary source | What already exists | Consequence for VelaDraft |
|---|---|---|
| [DFlash, 2026](https://arxiv.org/abs/2602.06036), [official project](https://z-lab.ai/projects/dflash/) | Block drafting conditioned on target features; training design already exposes clean context and masked blocks. | Anchor-expanded training and offline target-feature reuse are background, not new contributions. |
| [SpecForge, 2026](https://arxiv.org/abs/2603.18567), [current official training documentation](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md) | Target/draft decoupling, efficient training infrastructure and memory-conscious gradient kernels; current documentation includes diffusion drafter families. | Required external systems baseline. Compare pinned implementations with matched objective/configuration; paper-v1 EAGLE results alone do not describe today's software. |
| [Block Parallelism for Efficient Distributed Long-Context Diffusion Language Model Training, Sep 2026](https://arxiv.org/abs/2609.19242) | Separates shared clean sequence state from block-local corrupted computation, with a semantics-preserving distributed schedule; includes DFlash2 training. | No “first exploit diffusion block independence/shared context” claim. Differentiate scoring reuse at practical multimodal contexts from distributed long-context attention/communication. |
| [Cut Your Losses / Cut Cross-Entropy, ICLR 2025](https://openreview.net/pdf?id=E4Fk3YuG56), [official implementation](https://github.com/apple/ml-cross-entropy) | Avoids materializing full logits in large-vocabulary CE. | A direct CE baseline; plain chunking is not novelty. Report exact versus approximate/filtering configuration carefully. |
| [Liger Kernel, 2024](https://arxiv.org/abs/2410.10989), [official implementation](https://github.com/linkedin/Liger-Kernel) | Fused linear CE, chunked input and early gradient computation. | Frozen-head streaming CE is an engineering component/baseline, not a primary new mechanism. |
| [Efficient Knowledge Distillation for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss, Aug 2026](https://arxiv.org/abs/2608.03796) | Offline sparse teacher targets and fused chunked vocabulary-loss evaluation. | “Cache teacher and chunk loss” is occupied. Distinguish full-distribution semantics, non-KL objectives and repeated anchor supervision. Top-K truncation is not an exact baseline for full-vocabulary KL/TV. |
| [Sparse Logit Sampling, ACL 2025](https://aclanthology.org/2025.acl-long.885/) | Examines teacher-logit caching and sparse probability estimation, including shortcomings of naive top-K approximations. | Include sparsification only as a separate quality/cost frontier, not as objective-equivalent execution. |
| [KL Cache Trick for Full-Vocab Distillation, author's technical note, May 2026](https://jonathanc.net/blog/kl-cache-trick) | Derives hidden-sized cached `teacher_probs @ frozen_student_head` for exact forward-KL hidden gradients. This is a technical note, not a peer-reviewed paper. | The frozen-head sufficient statistic itself is not an original VelaDraft claim. It is a strong compatible baseline for pure KL, and does not directly handle TV/hybrid/confidence terms. |

Search included exact and related formulations for teacher-logit deduplication, repeated anchor supervision, diffusion-drafter training cost, block diffusion training memory, SpecForge and frozen-head losses. No exact prior implementation of this repository's proposed per-consumer-chunk index dedup was found in this bounded search. **Failure to find a match does not establish novelty**, and ordinary deduplication remains conceptually unsurprising even without a paper naming this exact patch.

## Strongest defensible direction

Recommended thesis, still a hypothesis:

> Diffusion block-drafter training has a many-to-one supervision graph: anchor-dependent student states consume repeated frozen target state. An objective-aware execution plan can remove redundant full-distribution scoring while respecting memory limits and preserving the original stochastic training workload.

The compelling contrast is not “diffusion versus AR” in general. It is the mismatch between the expanded student layout and reusable teacher identities, including the different lifetime requirements of KL, TV, hybrid losses, confidence targets and selector metrics. Pure KL has a known compressed sufficient statistic; TV-like objectives still depend on the full distribution and student state. A useful planner should exploit that distinction rather than force all objectives through one expensive representation.

For an MLSys paper, the minimal addition beyond the local patch is a **consumer-aware, bounded execution policy with a demonstrated crossover**. It need not be a general compiler or learned scheduler. Start with three possible routes: the current dense request order; local exact reuse; and an objective-compatible known sufficient-statistic/fused-loss route. Only add grouped or tiled full-distribution consumption if measurements show that gather traffic and duplicated backward projection remain material. Never add a cache hierarchy merely to make the diagram larger.

A stronger future mechanism would let the loss consume unique teacher rows plus inverse indices directly, avoiding expansion of the teacher slab and sharing projection/normalization across its consumers. This is harder than unique-plus-gather and has a more defensible mechanism-level distinction, especially for TV/hybrid objectives. Its novelty still requires direct implementation comparison and current-source review.

## Minimal cost model and falsification

Let `R` be expanded requests in a consumer chunk, `U` their unique teacher identities, `V` vocabulary size, `H` head width and `s` bytes per logit. Dense projection performs approximately `2*R*H*V` FLOPs; dedup performs `2*U*H*V` plus unique-index work and an `R*V*s` gather/write. Let `P(n)` be measured projection latency for `n` rows, not a constant-throughput assumption. The reuse path is worthwhile only when:

`P(R) - P(U) > T_unique(R) + T_gather(R,V) + additional_allocation_cost`.

Peak memory must include both producer and consumer tensors. If the teacher path occupies fraction `f` of full update time, even an infinitely fast teacher path has ideal speedup at most `1/(1-f)`. This is an Amdahl bound, not a performance prediction.

Cheap kill criteria:

- If representative production traces have `U≈R`, report that and do not prioritize this as the main paper contribution.
- If the saved projection work is below gather/index overhead, keep it opt-in or route away from it; report the negative regime.
- If full-update gains disappear after current compiled/fused baselines, the local patch is not the main thesis.
- If increased peak memory reduces feasible batch size, evaluate at equal resource constraints; a fixed small-batch speedup alone is misleading.

## Required changes and experiments, in priority order

**CRITICAL — empirical bottleneck evidence.** Before a long training run, replay representative captured batches for each relevant recipe. Record anchor density, `R`, `U`, teacher-forward/recompute counts, time share and complete update time. Include a CE-only control to make the applicability boundary visible.

**CRITICAL — strong baselines and exact semantics.** Compare to current VelaDraft main including its existing compaction/compile switches, a pinned SpecForge configuration with matching objective where available, and known fused/compact KL or CE methods in their compatible regimes. Keep an objective table: CE, forward-KL, TV/hybrid and confidence are not interchangeable.

**IMPORTANT — numerical contract.** Test FP64/FP32 objective and every trainable gradient against the identical anchor table. For BF16, measure a same-code repeatability/shape floor before choosing tolerances. Check checkpoint replay, non-power-of-two accumulation scaling, empty/tail chunks, cross-sample collisions and compacted/packed maps. Long-run quality is needed before a convergence-equivalence claim.

**IMPORTANT — policy transfer.** Calibrate any routing threshold on held-out workload shapes, freeze it, then evaluate other densities, vocabulary/head sizes and hardware. Compare against a per-cell oracle to expose missed savings and regressions. Per-cell hand-tuning cannot establish an automatic policy.

**IMPORTANT — matched-quality cost.** Use repeated seeds and preregistered validation acceptance/quality thresholds. Report time-to-threshold and censored failures, plus complete pipeline stage costs and cache-build amortization. Do not replace native serving speed with training acceptance length.

## Simplification and drift

Simplification opportunities: (1) ship local dedup as one optional primitive, (2) reuse established fused-loss kernels rather than inventing a competing CE implementation immediately, (3) derive a simple measured policy before considering a general compiler or multi-tier cache.

Modernization opportunities: use objective-compatible sufficient statistics/fused loss as baseline primitives; a learned scheduler or new diffusion model is unnecessary.

Drift warning: presenting a single-head microbenchmark as total training-cost reduction would drift from the Problem Anchor. Claiming a new diffusion training method would also drift: this proposal preserves the method and changes execution.

**Overall novelty assessment: LOW for the local patch; unresolved for the proposed objective-aware execution system. Recommendation: PROCEED WITH CAUTION.** Build the small experiment, retain negative results, and decide the paper's main mechanism only after measuring the dominant cost and crossover.
