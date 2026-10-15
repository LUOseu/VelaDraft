# Benchmarks

All acceptance numbers below use **sequential verification** (the only
token-exact path), greedy decoding, and the target's own greedy output as the
reference. Do not mix experimental parallel-runtime numbers into these tables.

## Acceptance (trained DFlash vs random-init drafter)

| Target | Corpus | Train rows | Prompts | AL (trained) | AL (random init) | Token-exact | Per-slot acceptance (trained) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Qwen2.5-VL-3B | 512 | 278 kept† | 64 | 1.198 | 1.003 | 64/64 | see eval_result.json |
| Qwen2.5-VL-3B | 2K | 1050 kept | 128 | 1.630 | 1.003 | 128/128 | .413, .149, .048, .015, .004, .001, 0 |
| Qwen2.5-VL-3B (dspark) | 2K | 1050 kept | 128 | 1.735 | 1.003 | 128/128 | .432, .173, .075, .034, .015, .007, .003 |
| Qwen2.5-VL-3B (domino) | 2K | 1050 kept | 128 | 1.764 | 1.003 | 128/128 | .443, .180, .078, .036, .016, .009, .005 |
| Qwen3-VL-8B | 512 | 274 kept† | 32† | 1.322 | 1.003 | 32/32 | see eval_result.json |
| Qwen3-VL-8B | 2K | 1996 kept | 32 | 1.665 | 1.000 | 32/32 | .413, .160, .063, .027, .010, .002, .0004 |
| Qwen2.5-VL-3B | 5K | 2629 kept | 128 | 1.817 | 1.003 | 128/128 | .482, .210, .080, .030, .011, .005, .001 |
| Qwen2.5-VL-3B (dflash2, experimental) | 2K | 1050 kept | 128 | 1.954 | 1.003 | 128/128 | .495, .243, .118, .060, .027, .013, .006 |

Notes
- † **Number-trace pending (2026-10-15).** The kept/prompt cells of the two
  512 rows disagree with the stage records in `docs/measured-runs.md`
  (3B/512: 274 kept / 64-prompt eval; 8B/512: 512/512 kept / 64 prompts).
  The underlying run artifacts live on the reference host, so both records
  stay visible until they are re-checked; the marked cells must not be used
  for capacity planning. AL / token-exact are quoted the same in both records.
- The 512-corpus 3B row is the 2026-10-11 rerun (original v0.1: AL 1.220);
  bf16 runs reproduce the protocol, not the bits (< ~2% AL noise).
- Slot-0/1 acceptance dominates; 2K is the first point with stable non-zero
  acceptance through slot 5, 5K extends it through slot 6.
- 5K training: block CE 6.84 -> 2.58 (first/last-10% means), 1974 steps,
  peak 30.0 GB, ~40 min on one A6000; overfit sanity passed.
- 8B / 2K: regen kept 1996/2000 rows, capture 1996 samples (visual-path check
  passed), training 1500 steps (resumed from 375), final loss 2.69, slot-0
  teacher-forced accuracy 0.73; eval 32 prompts / 128 tokens, speedup 0.81x.
- DSpark / 2K: same corpus, features, block size, protocol and eval prompts as
  the 3B DFlash row (directly comparable). AL 1.735 vs DFlash 1.630; every
  slot improves and the gain grows with block depth (slot 2: .048 -> .075,
  slot 3: .015 -> .034, slot 4: .004 -> .015, slot 5: .001 -> .007,
  slot 6: 0 -> .003) - the semi-autoregressive Markov bias relieves exactly
  the suffix-acceptance decay the block-parallel drafter suffers. 792 steps,
  final loss 3.53, overfit sanity + gradient coverage + visual check passed;
  confidence head calibration BCE 0.590 / ECE 0.048 / AUROC 0.745 (not
  constant). Speedup vs target-only 0.86x (sequential verification path).
- DSpark COCO smoke (64 rows, pipeline check only): AL 1.123 vs init 1.000,
  token-exact 8/8 after the verify-parity fix.
- Domino / 2K: same corpus, features, block size and protocol as the rows
  above. AL 1.764 (best of the three at this point), token-exact 128/128;
  slot gains over DFlash again grow with depth (slot 3: .048 -> .078,
  slot 4: .015 -> .036, slot 5: .004 -> .016, slot 6: .001 -> .009,
  slot 7: 0 -> .005). 792 steps; the dual-logit log shows final_loss below
  base_loss by the end of the curriculum; overfit sanity, gradient coverage
  (audited at 10%), and the drafter visual check (logits_diff 7.46) passed.
  Speedup vs target-only 0.84x (sequential verification path).

### Algorithm-zoo scoreboard (3B / 2K corpus / 128 prompts / 128 tokens)

All rows use the same data, regen, captured features, block size 8, seed
protocol, sequential (token-exact) verification and eval prompts. Ablations
are the same run recipe with one algorithm knob flipped.

| Run | AL | slot 1 | slot 2 | slot 3 | slot 4 | slot 5 | slot 6 | slot 7 | Token-exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DFlash (frozen baseline) | 1.630 | .413 | .149 | .048 | .015 | .004 | .001 | 0 | 128/128 |
| DSpark (anchor mode, upstream default) | 1.735 | .432 | .173 | .075 | .034 | .015 | .007 | .003 | 128/128 |
| DSpark `sample_from_anchor: false` (DFlash-width ablation) | 1.719 | .418 | .170 | .075 | .033 | .016 | .006 | .003 | 128/128 |
| Domino (correction enabled) | 1.764 | .443 | .180 | .078 | .036 | .016 | .009 | .005 | 128/128 |
| Domino `enable_correction_head: false` (ablation) | 1.662 | .457 | .148 | .042 | .013 | .002 | .001 | 0 | 128/128 |
| JetSpec (causal-mask serving, faithful) | 1.599 | .408 | .137 | .040 | .011 | .003 | .001 | 0 | 128/128 |
| JetSpec (shared runtime, no in-block mask) | 1.338 | .242 | .069 | .020 | .005 | .001 | .0004 | .0002 | 128/128 |
| DFlash2 (conv + selector, experimental objective) | 1.954 | .495 | .243 | .118 | .060 | .027 | .013 | .006 | 128/128 |
| DFlash2 `conv_kernel_size: 1` + `selector_loss_alpha: 0` (ablation) | 1.798 | .482 | .203 | .077 | .027 | .009 | .003 | .001 | 128/128 |
| DFlare (fusion + separate context K/V, matched `[1,17,33]`) | 1.637 | .421 | .154 | .049 | .011 | .003 | .001 | 0 | 128/128 |
| DFly (fusion residual + hidden correction, matched `[1,17,33]`) | 1.682 | .384 | .160 | .073 | .037 | .019 | .008 | .004 | 128/128 |
| DFlare (5-layer recipe `[1,9,17,25,33]`) | 1.670 | .435 | .162 | .055 | .016 | .004 | .001 | .0001 | 128/128 |
| DFly (5-layer recipe `[1,9,17,25,33]`) | 1.709 | .394 | .169 | .079 | .037 | .019 | .010 | .005 | 128/128 |
| DFly `enable_hidden_correction: false` (ablation) | 1.569 | .391 | .126 | .038 | .011 | .004 | .001 | .0004 | 128/128 |
| DFlare `fusion_trainable: false` (fixed prior, ablation) | 1.637 | .422 | .154 | .049 | .010 | .002 | .001 | .0001 | 128/128 |
| DFlare `separate_context_kv: false` (shared K/V, ablation) | 1.579 | .401 | .129 | .037 | .010 | .002 | .001 | .0002 | 128/128 |

- DSpark decomposition: at the frozen DFlash round width the Markov head alone
  lifts AL 1.630 -> 1.719 (+0.089, deep slots roughly doubled); drafting the
  anchor slot as well adds +0.016 on top (mostly slot 1). Both correction
  families relieve the same suffix-acceptance decay the block-parallel drafter
  suffers, and DSpark/Domino per-slot profiles differ (Domino keeps more of
  slot-1 acceptance while gaining deeper), so the two heads are not redundant.
- JetSpec decomposition: the trained checkpoint relies on its causal in-block
  factorization. Serving it under the shared runtime's unmasked block forward
  (the default for every other algorithm) loses **0.26 AL** (1.599 -> 1.338);
  a diagnostic on one captured sample shows the runtime forward flips 7.6% of
  the checkpoint's per-slot greedy argmaxes relative to the training mask
  (max |dlogit| 9.0). Serving with the mask (now the default for JetSpec
  checkpoints, `inference_in_block_causal`) restores per-slot acceptance to
  essentially the DFlash profile at this scale. Under the faithful mask JetSpec
  (1.599) sits just below the frozen DFlash CE baseline (1.630): with uniform
  soft-distribution weighting and this tiny corpus/step budget, KD does not beat
  hard-label CE here; the paper's scaling claims are not testable at 2K.
- Domino ablation: the causal GRU correction is worth +0.102 AL at the same
  window; the disabled run reproduces roughly the DFlash objective on the
  shifted window (its 1.662 vs 1.630 difference reflects the shifted-label
  supervision and ~2% protocol noise, not the head).
- DFlash2 / 2K (**experimental objective**): same corpus, features, block
  size, seed protocol and eval prompts as the rows above. AL 1.954,
  token-exact 128/128; per-slot `[.495, .243, .118, .060, .027, .013, .006]`
  — every slot ahead of DFlash (its frozen baseline) and of DSpark/Domino at
  this scale. 792 steps, unary KL 3.41 -> 1.20 and selector CE 2.20 -> 0.28
  (first/last-10 means), train candidate recall@16 0.78, overfit sanity
  0.995/1.0, gradient coverage 51/51 (conv + selector tensors included),
  drafter visual check logits_diff 7.92; peak 31.1 GB, 22.6 min. Speedup vs
  target-only 0.82x (sequential parity path).
- DFlash2 selector contribution (serving-time decomposition, 16 prompts, same
  checkpoint, `scripts/diag_dflash2_selector.py` ->
  `eval/selector_ablation.json`): reranking the unary top-16 with the trained
  transition scores gives AL 2.060 vs **1.965** when the *same* candidate set is
  ranked by unary logits alone. The selector changes the pick at ~60% of
  proposal positions, and its gain is concentrated in the deep slots
  (slot 3 .046 -> .083, slot 4 .025 -> .039, slot 6 .005 -> .011), while
  unary-only ranking keeps more of slot 1 (.548 vs .511). On the ablated
  checkpoint the same diagnostic reports `selector_override_rate: 0.0`
  (selector and unary rankings identical, both AL 1.897 at 16 prompts),
  confirming the knob really turns the trained reranking off.
- DFlash2 caveat (read before quoting +0.32): the objective is **not
  published**; the number above is the maintainer implementation's
  experimental split objective, so it must not be cited as "DFlash2's training
  method" (see `docs/upstream/dflash2.md`). It also differs from the frozen
  DFlash baseline on two axes at once — forward-KL unary instead of decayed
  hard-label CE, plus conv + selector — so the scoreboard delta cannot be
  attributed to a single module. The completed local-mixing + selector
  ablation (`algorithm.dflash2.conv_kernel_size: 1`,
  `selector_loss_alpha: 0.0`, same recipe/features/seed, run
  `qwen25vl3b_dflash2_2k_ablate_nomix`) bounds the joint conv+selector share:
  **1.954 -> 1.798 (-0.156 AL)**, all else equal. The ablated checkpoint
  still beats DSpark (1.735) and Domino (1.764), so the remaining +0.17 vs
  the frozen baseline goes with the objective axis (hard-label CE ->
  forward-KL), which is **not isolated yet** — a `unary_loss: ce` control
  would be needed before quoting that residual as the objective's value.
  (Ablation run: 792 steps, first/last-10 loss 3.31 -> 1.01, overfit sanity
  0.995/0.98 passed, visual check logits_diff 9.51, peak 30.1 GB, 19.5 min,
  token-exact 128/128.)
- `AL = new_tokens / target_verification_steps` (bonus-token-inclusive);
  random-init drafter AL ~1.003 means proposals are almost never accepted.
- DFlare (matched `[1,17,33]`, AngelSpec port): parity with the
  frozen DFlash baseline at this budget — AL 1.637 vs 1.630 (+0.007, inside
  the ~2% run-to-run noise), slots within ±0.004 of the baseline profile. The
  fusion logits train (fp32-master drift from the distance prior: prob L1
  0.011 at step 594; per-layer entropy [0.496, 0.671, 0.396] vs prior
  [0.441, 0.668, 0.439]), but the prior still dominates the mixture at 2K —
  the run tests the architecture + prior, not a freely re-shaped fusion.
  792 steps, first/last-10 loss 7.39 -> 3.73 (final 3.64), overfit sanity
  passed (CE drop 0.9997, slot0 1.0), visual check logits_diff 10.59, peak
  30.2 GB, 24 min; sequential speedup 0.85x. Both runs token-exact 128/128;
  the frozen DFlash 2K baseline re-checked on this branch reproduces
  1.6298365840801265 bit-identically (`tests/reference_dflash/manifest.json`).
- DFly (matched `[1,17,33]`, AngelSpec port): AL 1.682
  (+0.052 over the frozen baseline) with the deep slots carrying the gain
  (slot 3: .049 -> .073, slot 5: .003 -> .019, slot 6: .001 -> .004) while
  slot 1 gives some back (.421 -> .384) — the same suffix-decay relief shape
  as DSpark/Domino. The hidden-states correction trains from its zero init
  (correction |delta| 0 -> 1.81 by step 792; `down_proj` |w| 0.0065) and the
  fusion stays near its prior (drift 0.0039). 792 steps, loss 6.29 -> 1.60
  (final 1.68), visual check logits_diff 5.63, peak 31.6 GB, 16 min;
  sequential speedup 0.775x (the per-slot corrected head walk costs wall time
  — see `docs/algorithms/dfly.md`). Token-exact 128/128.
- Layer-rich runs (the upstream-style wider context on the same store, same
  recipe/seed): DFlare `[1,9,17,25,33]` takes AL 1.637 -> **1.670** (+0.033;
  every slot at or above the matched run, but wall speedup drops to 0.781x —
  twice the context K/V projections to run) and its fusion moves further from
  the prior (fp32-master drift 0.0094; per-layer entropy [0.523, 0.830, 0.421]
  vs prior [0.458, 0.794, 0.458]). DFly `[1,9,17,25,33]` takes 1.682 ->
  **1.709** (+0.027; slots
  `[.394,.169,.079,.037,.019,.010,.005]`) with the correction still active
  (`down_proj` |w| 0.0066) and a near-prior fusion (drift 0.0014). Both
  792 steps, visual check passed, token-exact 128/128.
- DFly no-correction ablation: the zero-init hidden-states correction is worth
  **+0.113 AL** (1.682 -> 1.569) and its gain is concentrated in the deep
  slots (slot 2 .160 -> .126, slot 3 .073 -> .038, slot 5 .019 -> .004 while
  slot 1 barely moves) — i.e. it relieves exactly the suffix-acceptance decay
  of the block-parallel drafter, not the first proposal. Training-side the
  ablated run also fits worse (final loss 3.98 vs 1.68) and it is *faster* at
  serving (0.851x vs 0.775x vs target-only) because the per-slot corrected
  head walk disappears. The ablated checkpoint is below its frozen DFlash
  baseline (1.569 < 1.630): the DFlash backbone plus a near-prior fusion and
  the layer slicing does not beat plain DFlash by itself; the correction is
  what carries DFly's gain at this scale.
- DFlare fixed-prior ablation: freezing the fusion at the distance prior
  changes AL by **0.0005** (1.6374 trained -> 1.6369 fixed; per-slot within
  ±0.002) — at this corpus/step budget the trained fusion buys nothing the
  prior does not already provide, consistent with the small fp32-master drift
  measured on the trained run. The knob still moves the mechanism (the
  gradient audit covers `layer_fusion_weights` and the layer-rich run drifts
  further, 0.0094), so quote DFlare's 2K numbers as "architecture + distance
  prior", not "learned fusion".
- DFlare separate-context-KV ablation (`separate_context_kv: false`, same
  recipe/seed/store, fusion knob unchanged): AL **1.579** vs 1.637 matched
  and 1.630 frozen baseline — the dedicated context K/V projections are the
  component with a measurable effect at this scale (**+0.058** over the
  shared-KV variant), while the fusion stays prior-dominated (fixed-prior
  AL Δ0.0005; the shared-KV run's fusion drifts a bit further, prob L1 0.0133
  vs 0.0110, without a gain). Reading the ablations together: DFlare's 2K
  numbers are "dedicated context K/V + distance prior"; a learned-fusion gain
  is not established at this budget. The ablation trains cleanly (792 steps,
  first/last-10 loss 7.40 -> 3.68, final 3.53, overfit sanity passed, visual
  check logits_diff 8.5, coverage 36/36, token-exact 128/128 vs init 1.003;
  sequential speedup 0.79x — the lower AL costs target rounds, peak
  30.2 GB).

- **DFly upstream two-stage recipe (canonical, 792 steps):** AL **1.839** vs
  1.682 for the controlled_ce record (+0.157, +9.3%) — the highest DFly AL
  recorded, with every slot above the shared-CE run
  (`[.440,.207,.103,.051,.024,.014,.007]` vs
  `[.384,.160,.073,.037,.019,.008,.004]`) and token-exact 128/128 on the
  same store/seed/step budget; the only knob is `train_recipe:
  upstream_two_stage` (cold-start fraction 0.5, stage boundary at step 396).
  Stage evidence: LK/D-PACE loss 8.31 -> 1.25 over steps 1-396; CE 2.00 ->
  0.98 with e2e TV 0.79 and slot-0 accuracy 0.47 -> 0.67 afterwards; final
  loss 1.977; sanity passed (drop 0.926, slot0 0.9986). Drafting is slightly
  cheaper (0.86 s vs 0.94 s per prompt), eval speedup 0.80x. Run details and
  costs: `docs/plan/RUN_LOG.md` R-20261014-11, `docs/training-costs.md`.

### AngelSpec ports — Qwen3-VL-8B reduced validation (256-row slice, 192 steps, 16-prompt eval)

Same reduced protocol as the zoo release runs (shared capture-v2 store, 3
layers `[1,17,33]`, token-exact gate).

| Run | final loss (first-10% -> last-10%) | AL (init 1.000) | token-exact | peak VRAM |
| --- | --- | ---: | --- | ---: |
| DFly reduced | 4.16 (20.26 -> 4.13) | 1.2256 | 16/16 | 22.6 GB |
| DFlare reduced | 6.22 (12.83 -> 6.02) | 1.1852 | 16/16 | 20.4 GB |

Both reduced runs were exported with the final algorithm metadata; their
step-144 fusion masters sit close to the prior (drift 0.0014 / 0.0003) and
DFly's correction `down_proj` is still small (|w| 0.0025) — at 192 steps the
reduced runs validate the pipeline on Qwen3-VL-8B, not converged quality.

## Runtime probe (32 prompts, bf16, parity backend, same card)

| Run | Mode | Token-exact | Wall s/sample | vs target-only | Note |
| --- | --- | ---: | ---: | ---: | --- |
| 3B / 2K (AL 1.68) | sequential | 32/32 | 5.25 (8B run) / - | 0.86x | correctness path |
| 3B / 2K | parallel | 8/32 | - | 1.25x | experimental, not lossless |
| 3B / 2K | strict τ=0.25 | 7/32 | - | 1.03x | diagnostic |
| 8B / 512 (AL 1.33) | sequential | 32/32 | 5.25 | 0.84x | correctness path |
| 8B / 2K (AL 1.67) | sequential | 32/32 | 5.60 | 0.81x | correctness path |
| 8B / 512 | parallel | 6/32 | 4.31 | 1.02x | experimental, not lossless |
| 8B / 512 | strict τ=2.0 | 10/32 | 6.61 | 0.67x | diagnostic |

Mechanism (documented in `docs/REFERENCE.md`): block-forward vs step-forward
logits differ by ~0.07 mean |Δ| at the same position/input (bf16); accepted
tokens' KV cache entries are computed by the block forward, so the drift
propagates and per-round guards cannot restore exactness. Parallel mode must be
reported as experimental throughput only.

## Public workload: TextVQA (Qwen2.5-VL-3B zoo, pinned export)

TextVQA validation split, 128-prompt seeded subset (seed 42; subset definition
and per-row question/image hashes pinned in
`benchmarks/prompts/textvqa_v1.jsonl`, protocol in
`benchmarks/manifests/02_textvqa.json`); greedy, `max_new_tokens=32`,
`verification: sequential` (token-exact gate); same 3B/2K checkpoints as the
scoreboard above.

| Algorithm | AL | acceptance rate | token-exact | per-slot acceptance (1..7) |
| --- | ---: | ---: | --- | --- |
| DFlash | 1.263 | 3.15% | 128/128 | [.178,.031,.004,.001,0,0,0] |
| DSpark | 1.253 | 2.64% | 128/128 | [.162,.035,.005,.002,0,0,0] |
| Domino | 1.268 | 3.21% | 128/128 | [.174,.037,.005,.002,0,0,0] |

Short grounded answers (32-token budget) accept less than the long-caption
val split (~1.63); the ordering across algorithms compresses to within 0.02 AL.
Wall-clock speedup vs target-only is 0.94-0.98x on this card - the sequential
path is the correctness reference, not a speed claim. Reproduce with
`python scripts/prepare_benchmark.py --workload textvqa --source <download>`
then `python scripts/benchmark.py --run-dir runs/qwen25vl3b_2k --workload
textvqa` (see `docs/benchmark.md`).

## Public workloads: DocVQA and ChartQA (Qwen2.5-VL-3B zoo, pinned exports)

DocVQA (`lmms-lab/DocVQA` DocVQA/validation, 5349 rows) and ChartQA
(`lmms-lab/ChartQA` test, 2500 rows), 128-prompt seeded subsets (seed 42;
subset definitions and per-row question/image hashes pinned in
`benchmarks/prompts/docvqa_v1.jsonl` / `benchmarks/prompts/chartqa_v1.jsonl`,
protocols in `benchmarks/manifests/03_docvqa.json` / `04_chartqa.json`);
greedy, `max_new_tokens=64`, `verification: sequential` (token-exact gate);
frozen 3B/2K DFlash checkpoint (`runs/qwen25vl3b_2k`).

| Workload | AL | acceptance rate | token-exact | per-slot acceptance (1..7) | wall/s | target-only/s |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| DocVQA | 1.201 | 1.99% | 128/128 | [.125,.012,.002,0,0,0,0] | 1.02 | 1.06 |
| ChartQA | 1.221 | 2.36% | 128/128 | [.144,.015,.002,.001,0,0,0] | 0.75 | 0.66 |

Dense-document and chart questions accept less than long-caption datasets
(short factual answers at a 64-token budget); the two workloads land within
0.02 AL of each other. Wall-clock speedup vs target-only is 0.89-1.04x on this
card - the sequential path is the correctness reference, not a speed claim.
E3 evidence: run `R-20261014-10`, report
`benchmarks/reports/e3_dflash_docvqa_chartqa.{json,md}`. Reproduce with
`python scripts/prepare_benchmark.py --workload docvqa --source <download>`
(then `chartqa`) and `python scripts/benchmark.py --run-dir runs/qwen25vl3b_2k
--workload docvqa --workload chartqa` (see `docs/benchmark.md`).

## Reproducibility

- Post-zoo regression (2026-10-12): re-evaluating the frozen 3B-2K DFlash
  checkpoint with the full five-algorithm code reproduces AL **1.6298**
  bit-identically at the canonical 128-prompt / 128-token protocol (both
  checkpoints token-exact 128/128); the 8-prompt registry path reproduces
  1.6096. Frozen baselines, semantics and re-measurements:
  `tests/reference_dflash/manifest.json`.
- Release-commit re-eval (2026-10-13): the frozen baseline reproduces
  1.6298365840801265 bit-identically again; 16-prompt release regressions are
  token-exact for DSpark (1.827), Domino (1.854), JetSpec (1.661) and DFlash2
  (2.047) on the same checkpoints (sequential, random-init 1.002; subset AL
  reads higher than the 128-prompt rows above).
- Qwen3-VL reduced validation (2026-10-13, 256-row slice, 192 steps, 16
  prompts): DSpark 1.221 / Domino 1.228 / JetSpec 1.119 / DFlash2 1.210,
  token-exact 16/16 each — compatibility evidence for the Qwen3-VL column of
  `SUPPORT_MATRIX.md`, not a scale comparison (full table in
  `RELEASE_VALIDATION.md`).
- Same seed + same data: `rows_sha256` and regen keeps reproduce; captured
  shard hashes and bf16 training losses do not (kernel nondeterminism).
- Treat AL differences < ~2% as noise unless the protocol pins more seeds.
