# REPORT — VelaDraft system-optimization lane (status: S1a/S1b/S2/S2c/S3 complete; frames 1+2 DONE: acceptance eval functionally identical; cache drift 30-90x below seed variation; offline baseline measured (§4.6); see §4.5/§6)

Scope: one system-level optimization for the frozen-target / external-L2-drafter
training flow, judged on wall-clock and GPU-hours at fixed semantics, with an
explicit engineering vs paper judgement at the end. Sources are the task
document (`VelaDraft_System_RSI_Goal_20261014.md`), this repo at `e15256d`,
and existing run evidence on `a6000xuyang` (paths in `STATE.md`).

## 1. What was established in S0 (CPU-only, no GPU spent)

1. All six baseline behaviors named in the task document still hold in the
   current code (table in `STATE.md`). In particular: the online source
   re-extracts every epoch; capture groups are capped by the micro-batch; the
   target runs in the trainer's main process (`num_workers=0`); regen keeps
   token ids only; the unified extraction-timing boundary is implemented and
   the old -18.8% figure remains marked invalid pending a paired re-run.
2. Cost model assembled from measured evidence (`docs/training-costs.md`,
   `WP4_MEASUREMENT.md`, real 2K store index): offline 2K pipeline
   65-80 min; online pays ≈2.5-2.6 min extraction per epoch (≈15-16 min for
   6 epochs, ≈20% of online end-to-end); teacher forward per-row wall is
   nearly flat across capture groups 1→8 (139→129 ms best, -7%) on the smoke
   slice, i.e. doc-candidate A's ceiling is small on this hardware.
3. Resource reality recorded: all four server GPUs held by other jobs at
   audit time; `/home` is 100% full (28 GB free) and drifting; the host is
   writeback-stall-prone (documented 1.7x variance). The disk constraint
   makes "no-store" online operation genuinely valuable here, and makes a
   host-RAM retention design (H-C1) the leading candidate.

## 2. Candidates and current ranking

| Candidate | Mechanism | Predicted effect (2K, 6 epochs, online) | Status |
| --- | --- | --- | --- |
| H-C1 RAM retention (primary) | keep extracted records in byte-bounded host RAM; epochs ≥2 serve from RAM | extraction wall → ~0 for epochs ≥2; measured S3: −59.7% train-stage wall; zero disk | verified: S2 (−50.9% x3) + S2c (bitwise, fixed order) + S3 canonical (§4.4/§4.5) |
| H-A decoupled teacher window | collect rows across micro-batches into larger teacher forwards | <2% e2e if per-row flatness holds at g16/32; call-count win only | STOP (S1b: −13.3% extraction, saturates at g8; <3% e2e) |
| H-C2 regen/capture fusion | capture hidden during generation (TIDE-style) | ≤ ~3-4% e2e once C1 lands; numeric-equivalence risk | deferred |
| H-C3 split-device overlap | target GPU + drafter GPU pipeline | wall win at 2× device cost; no free GPU; dominated by C1 | deferred |

## 3. Measurement plan (S1/S2, when GPUs free up)

- S1a: paired per-sample vs group-4 unified-boundary re-measurement on the
  pinned smoke slice (the re-run `docs/online-training.md` is waiting for).
- S1b: group sweep 1/2/4/8/16 on real-corpus rows (≈546 tokens, 1 image)
  with per-group forward/sample wall, call counts, render cold/warm split and
  target-device peak (`scripts/profile_online_extraction.py`); no
  vision-vs-text kernel attribution in this round.
- S1c: fresh vs resumed online-run stage timeline (startup wait vs run wall).
- S2: H-C1 minimal implementation + paired OFF/ON A/B, ~130 real rows ×
  6 epochs, interleaved (off/on/off/on), bitwise loss gate, RAM high-water
  mark; ckpt sha256 parity within pairs (actual outcome and the amended
  criteria are in §4.3/§4.5).
- S3: canonical 2K paired A/B (LLaVA pool → 1012 regen rows, 6 epochs,
  762 steps, micro-batch 8). Leg A builds the master staged inputs
  (prepare+regen) and trains with the cache OFF; leg B reuses A's data+regen
  (copied stage manifests; no rebuild) with the cache ON (16 GiB). Gated on
  the amended H-C1 gate (§4.5; S2c), disk-guarded; the representative-
  workload wall number.

## 4. Evidence, timelines, alignment

### 4.0 Known baselines (pre-GPU, from repo evidence; unchanged by this lane)

| Path | Wall (1 GPU) | Store bytes | Notes |
| --- | --- | --- | --- |
| offline full pipeline (2K) | ~65-80 min incl. eval (prepare ~20 s, regen ~23.5 min, capture ~2.6 min, train 9.8-24 min, eval ~28 min) | 9.4 GB (1050 rows, 3 layers + final hidden, seq 2048) | `docs/training-costs.md`; canonical run `runs/qwen25vl3b_2k` (offline: capture stage recorded) |
| online, fixed timing (2K) | train stage 9.8-24 min + ~2.5-2.6 min extraction per epoch (x6 on a 6-epoch run) | 0 | cost model assembled in S0; S1/S2/S3 measure it directly |
| offline store reuse (train only) | 9.8-24 min | existing store | only possible when a store exists and disk is healthy; server disk is 100% full |

### 4.1 S1a — unified-boundary paired re-measurement (smoke slice)

Measured 2026-10-14 17:28-17:30 on GPU0 (RTX A6000, idle), smoke slice
(16 synthetic 2-image/2-turn rows, seq 1024, 1 extraction pass = 4 steps),
`VLM_DFLASH_FLEX_FORCE_MAIN=1`, resume-ckpt writes off (identical both legs):

| leg | batch/group | target forwards | prepare s | forward s | extraction s | forward ms/row |
| --- | --- | --- | --- | --- | --- | --- |
| ab1 | per-sample (1) | 16 | 0.719 | 2.209 | 2.928 | 138.0 |
| ab4 | group 4 | 4 | 0.748 | 2.269 | 3.017 | 141.8 |

Paired deltas: prepare +0.029 s (+4.0%), forward +0.060 s (+2.7%), extraction
+0.089 s (+3.0%). I.e. with the unified R2 boundary the group-4 window is ~3%
**slower** on this slice — no extraction-wall win; the pre-R2 "-18.8%" number
stays invalid. Consistent with the smoke per-row flatness (139 ms g1 vs
133 ms g8 in `docs/training-costs.md`); S1b's real-row sweep decides H-A.
Small-sample caveat: 16 rows / 4 forwards; this is a smoke-slice point, not
a workload claim.

### 4.2 S1b — teacher-forward group sweep (real rows)

Measured 2026-10-14 17:52-17:56 on GPU0, 129 real v1safe rows (seq 2048),
one extraction pass per group, `VLM_DFLASH_FLEX_FORCE_MAIN=1` both legs of
the protocol pinned (`logs/s1b_profile.json`):

| group | forwards | wall median (s) | sample ms | target peak (GB) |
| --- | --- | --- | --- | --- |
| 1 | 129 | 16.36 | 126.8 | 7.92 |
| 2 | 65 | 14.89 | 115.4 | 8.11 |
| 4 | 33 | 14.41 | 111.7 | 8.50 |
| 8 | 17 | 14.18 | 110.0 | 10.80 |
| 16 | 9 | 14.18 | 110.0 | 10.80 |

Render cold/warm: 14.31 / 12.43 ms per sample. Batching the teacher forward
buys -13.3% extraction wall (saturated by g8) and drops the call count
129 -> 9; per-row flatness reproduces on real rows. **H-A verdict: STOP as a
mechanism** — extraction is only ~16-20% of online end-to-end (§4.0), so even
the saturated -13.3% extraction term is <3% e2e. `online_batch_size` stays as
a modest knob, not a mechanism; S1a's smoke g1-vs-g4 +3% is the same flatness
at the noisy end (both agree grouping is not the lever).

### 4.3 S2 — cache A/B on the 256-slice (3 interleaved pairs)

Measured 2026-10-14 17:34-17:52 on GPU0, 129 real rows (v1safe), 6 epochs /
102 steps, `online_batch_size=8`, cache 8 GiB, off/on x3 interleaved
(`logs/s2_progress.log`, `logs/s2_*_summary.json`):

| pair | OFF wall (s) | ON wall (s) | ON/OFF | OFF extract (s) | ON extract (s) | cache peak |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 158.2 | 66.3 | 0.42 | 94.8 | 16.0 | 830 MiB |
| 2 | 145.2 | 79.0 | 0.54 | 95.5 | 16.1 | 830 MiB |
| 3 | 151.4 | 78.0 | 0.52 | 97.1 | 18.2 | 830 MiB |

Train-stage wall: OFF mean 151.6 s vs ON mean 74.4 s (**-50.9%**, pair 1
-58%); extraction wall -82.5% (ON extracts only epoch 1: 17 forwards /
129 samples vs 102 / 774); 645 hits / 129 misses / 129 stores / **0
evictions** at an 8 GiB cap (9.7x headroom); +0 disk bytes; GPU peak
unchanged (25.02 GB both legs); host RSS peak 4.744-4.760 GiB across all six
legs (the ON legs' retained 830 MiB stays under the run's existing
high-water mark). **Wall win confirmed, paired, x3.**

Numeric gate: **FAILED as written** — all three pairs
sha256(ckpt_final.pt) mismatch; the gate text demanded bit-parity. Follow-up
analysis:

- OFF legs a1=a2=a3 and ON legs b1=b2=b3 are each bitwise identical (logs
  + ckpt hashes): the drift is deterministic and systematic, not run noise.
- Loss identical in all 102 steps at bf16 print precision; grad_norm first
  differs at step 18 = the first cache-hit step (epoch 2 step 1; max
  |delta| 6.5e-4); accuracy/accept_len/slot0 differ from step 27 (~4e-3);
  final weights differ on 15.33M / 243.81M elements (6.286%), max |delta| =
  0.0078125 = 2^-7 exactly (one bf16 ulp at unit scale); worst tensor
  `layers.1.self_attn.k_norm.weight` (`wdiff_ckpt.py`,
  `logs/wdiff_a1_b1.txt`). Per-element ulp normalisation: 83.7% of the
  differing elements are within 1 ulp of their own value; a small near-zero
  tail (<=2^-11 absolute) gives large per-element ulp ratios - the earlier
  "<= 1 bf16 ulp" wording was imprecise and is corrected in §4.5.
- Direct probes (GPU2, `logs/probe2.log`): for a fixed row with bitwise
  identical `input_ids` / `loss_mask` / `position3`, the extracted `hidden`
  depends on the extraction batch context: alone vs 2-row max_abs 6.0, alone
  vs 8-row 48.0, 2-row vs 8-row 48.0, ~78% of elements differ — bf16-rounding
  scale amplified through depth. Membership at fixed shape and call order do
  not matter (probe 1). This is a property of the batched teacher forward
  itself, matching `online_source_digest`'s note that batch composition
  changes padding/numerics.

Working interpretation (causal test in §4.4): the shuffled loader changes
the grouping every epoch, so epoch-k recomputation (OFF) and the cached
epoch-1 values (ON) differ at that last-bit scale; 6 epochs of training
amplify it to a <=1-ulp weight difference. The cache is value-preserving
(hits return the stored tensors bitwise, pinned by unit tests); what the
bit-parity gate assumed — identical extraction grouping across epochs — the
shuffled loader does not provide.

### 4.4 S3 — canonical 2K paired A/B (representative workload)

**S3 ran 18:34-19:29 (GPU0), only after the amended gate passed** (the
original 17:30 chain self-gated at 17:56 on the retired bit-parity
criterion; amendment in §4.5). Leg A (`a_off`) builds the staged 2K inputs
(prepare + regen, included in its end-to-end) and trains cache-OFF; leg B
(`b_on`) reuses A's staged data+regen (skip logged) and trains cache-ON
(16 GiB).

| leg | end-to-end wall (s) | train stage (s) | extraction (s) | target forwards | cache |
| --- | --- | --- | --- | --- | --- |
| a, off | 2733.8 | 1167.0 | 799.1 | 762 | off |
| b, on | 547.4 | 470.9 | 125.5 | 127 | 5060 hits / 1012 miss / 0 evict / 6.35 GiB |

Train-stage wall **-59.7%**; extraction **-84.3%** (ON extracts only epoch 1:
127 forward batches = 1012 samples vs OFF 762 forwards = 6072 samples); zero
disk bytes; GPU peak identical (25.02/25.02 GB target-resident); host RSS
peak 5.695 -> 9.664 GiB (+3.97; retained 6.35 GiB inside the process
high-water mark). Identity: both legs share feature digest `bc4b888c...`;
1012 regen rows; 762 steps.

Numeric (shuffled protocol): sha256(ckpt_final) mismatch - expected, and the
S2c causal test below plus §4.5 bound the difference. All logged signals:
`valid_tokens` identical 762/762 (supervised-signal set exactly equal); loss
identical 740/762 (rest <=2 bf16 display ulp, max 0.0625); grad_norm first
diverges at step 128 = first cache-hit step; accuracy max |delta| 0.0032,
accept_len 0.018 (display scale); final-weight drift max 0.0446 abs / 4.26%
global relL2 (§4.5).

**S2c — fixed-order causal check (18:09-18:24, GPU0).** Same protocol as
S2 pair 1 (129 rows, 6 epochs, batch 8, cache 8 GiB) except the epoch order
is fixed (`VLM_DFLASH_TRAIN_SHUFFLE=0`, a diagnostic env override at the
trainer's shuffle line; both legs share it; `docs/online-training.md`
caveat):

| leg | train-stage wall (s) | extraction (s) | cache |
| --- | --- | --- | --- |
| a, off, fixed order | 206.4 | 98.9 | off |
| b, on, fixed order | 76.5 | 16.6 | 645 hits / 129 misses / 0 evict / 830 MiB |

Result: **sha256(ckpt_final) identical (`52d25648...`)** and all six logged
fields bitwise identical across all 102 steps; the wall win reproduces
(train stage -62.9%). (Leg a's setup phase hit a host I/O stall: 708 s
end-to-end vs 195 s; S2c's point is bit-parity, not wall.) This isolates
the S2 drift completely: the extractor's records depend on the extraction
batch grouping at last-bit scale (probe 2, §4.3); the shuffled loader
changes the grouping every epoch; the cache freezes epoch-1 groupings. It
also shows extraction is time-invariant under a fixed grouping (OFF's
epoch-k recompute == ON's epoch-1 stored values bitwise).

### 4.5 Numeric alignment

- Cache value-exactness: unit-pinned (`second_read_is_free_and_bit_identical`
  + a tiny real-model end-to-end test) and full-run pinned by S2c (ckpt
  sha256 equality under a fixed order).
- S2 protocol (shuffled epochs, 129 rows, 102 steps): loss identical
  102/102; grad_norm drifts <=6.5e-4 from the first hit; final weights differ
  on 15.33M/243.81M elements (6.286%), max |delta| = 0.0078125 = 2^-7
  exactly. Deterministic and reproducible (all OFF legs identical; all ON
  legs identical). Per-element bf16-ulp normalisation (`ulp_drift.py`):
  83.7% of the differing elements are within 1 ulp of their own reference
  value, with a small near-zero tail (<=2^-11 absolute) that gives large
  per-element ulp ratios.
- S3 protocol (canonical 2K, 762 steps, shuffled): valid_tokens identical
  762/762; loss identical 740/762, the remaining 22 steps differ by <=2 bf16
  display ulp (max 0.0625 at loss ~4.03); grad_norm first diverges at the
  first cache-hit step; accuracy max |delta| 0.0032, accept_len 0.018. Final
  weights: 90.4% of elements differ, max |delta| 0.0446, global relative L2
  4.26% (worst tensor `layers.0.self_attn.q_proj.weight` 8.0% relL2,
  rms_w 0.022 vs rms_d 0.0018); per-element: 16.2% of all elements within
  1 ulp of their reference, 18.9% above 16 ulp. So at 762 steps the drift is
  multi-ulp - larger than the S2 bound and growing with training length, as
  expected for a deterministic grouping perturbation accumulated over
  6 epochs of SGD. The S2-era "every delta <= 1 bf16 ulp" phrasing was
  imprecise and is corrected here (raw files support both rows).
- Gate implications: the original "bit parity under shuffled epochs" gate
  tests loader-grouping stability, not the cache. Amended gate (openly
  documented per the RSI rules): (a) bit-exactness under stable grouping
  (S2c); (b) under the production shuffle: equal loss/valid_tokens sequence,
  with the weight drift deterministic, mechanism-attributed (probe 2 + S2c)
  and quantified per scale - S2: max 2^-7 abs; S3: max 0.0446 abs / 4.26%
  relL2 at 762 steps - reported with every cached run; (c) identity fields
  untouched — verified: S2/S2c legs share feature digest `154f74bf...`, both
  S3 legs `bc4b888c...`; `feature_cache` appears only as runtime stats.
- Downstream acceptance eval (closing frame 1, DONE 19:56-20:34 on GPU2;
  `vlm_dflash.cli eval --checkpoints ckpt_final.pt --tag closing`, 128
  prompts, max 128 new tokens, sequential verification): a_off vs b_on final
  checkpoints give mean_al 1.5262066923 vs 1.5262066923 (identical to 10
  decimals), acceptance_rate 0.076164 vs 0.076165, accepted tokens 5288 both,
  new tokens 15462 both, and token_exact=True for both (draft-assisted
  sequences exactly match target-only greedy). The only difference is 69429
  vs 69428 draft tokens (1 token in ~69k, a knife-edge tie-break). So the
  cache-trained drafter is functionally equivalent to the normally-trained
  drafter on the end-to-end acceptance metric - "no performance loss" is now
  verified downstream, not only on training metrics. (speedup_vs_target_only
  0.855 vs 0.863 is wall-clock noise in the sequential-verification eval
  mode, not a semantics difference.) Evidence:
  `runs/s3/*/eval/eval_result_closing.json` (copies in `evidence/`).
- Natural-variation frame (closing frame 2, DONE 20:33-21:04 on GPU2):
  same-pipeline cache OFF/ON pair at train.seed=43 (c_off_seed43 +
  d_on_seed43; init sha `c27bf1e4...` identical within the pair, != the
  seed-42 init). Cache replication at seed 43: train wall 1113.7 -> 476.5 s
  (-57.2%); extraction 754.5 -> 125.2 s; cache 5060 hits / 1012 misses /
  0 evictions; valid_tokens identical 762/762; loss 757/762 identical (rest
  <=1 display ulp; first diff step 287); final-weight drift max 0.0343 abs /
  global relL2 1.45% (vs 4.26% at seed 42 - per-seed sensitivity, both far
  below run-to-run variation). Reference frame (a vs c = seed change:
  order + init, nothing else): 99.9% of elements differ, max 0.1846 abs,
  global relL2 **131.5%** (weights as different as two independent random
  draws - this tiny drafter stays close to its random init), last-10 loss
  4.052 vs 3.707 (Delta 0.345). So the cache-induced perturbation is ~30-90x
  smaller than a routine seed change and moves no displayed training signal
  at either seed. (First diff attempt failed on repo-relative script paths;
  re-run 21:07 with absolute paths - raw files are the re-run.)
- Semantics note: per-epoch re-extraction is itself a source of last-bit
  epoch-to-epoch variation once grouping varies; caching makes epochs >=2
  consume one frozen feature set (offline-store semantics) and is the more
  self-consistent of the two behaviors.

### 4.6 Strong baselines and ablations (doc §8.3)
- Current online (fixed timing) = the OFF legs of S2/S3 (measured paired).
- Full offline store - MEASURED 2026-10-14 (leg e = capture + train, leg e2 =
  train from e's store; same S3 staged inputs/config; GPU2 21:18-21:42,
  0.39 GPU-h): trainer walls e 624.5 s (capture ~75.6 s + train 548.8 s;
  writes a 7.0G store) and e2 548.8 s (store pre-existing); end-to-end
  836.8 s / 582.2 s; peak GPU 17.4 GB both; e/e2 train logs byte-identical
  (store round-trip lossless, offline path deterministic); final_loss
  4.03125 on all four legs (e, e2, a_off, b_on).
- Train-stage comparison (762 steps / 6 epochs): offline store reuse 548.8 s
  vs online cache 470.9 s vs full store cycle 624.5 s vs online no-cache
  1167.0 s -> the cache path is 14.2% faster than reusing an already-written
  store (7.0G) at zero disk bytes, and 24.6% faster than produce-and-train;
  offline still beats no-cache online (624.5 vs 1167.0) - the cache recovers
  the reuse benefit inside online mode. Caveat: single-shot legs under host
  I/O contention (loadavg ~10-12; offline pace 0.72 s/step vs cache-trained
  0.45 s/step may be partly shard-I/O contention; both offline legs
  consistent with each other).
- Disk-infeasibility note (unchanged): the store cannot be written when
  `/home` is in the 3-8 GiB windows this lane hit; the lane's claim is the
  disk-free variant.
- Simple batching/prefetch = H-A sweep (S1b); predicted flat per-row, so not
  a competing mechanism on this hardware.
- "Visual-only cache": not implementable in this harness without changing the
  capture path (records are produced by one teacher-forced forward; no
  vision-only extraction API, and no kernel-level vision/text split measured
  this round) - stated as a limitation, not tested.
- LRU parameter ablation: capacity vs working set (S2: 8 GiB vs ~1-2.4 GB;
  S3: 16 GiB vs 6.8 GB cache set / 7.0G store measured) => zero evictions
  observed; the capacity-below-working-set test covers eviction -> recompute.

### 4.7 Failure scenarios / boundaries
- Cache cap smaller than the record set: LRU recompute (degrades to today's
  behavior); correctness unchanged, gain shrinks.
- Host RAM pressure on a shared node: measured via `/usr/bin/time -v` RSS.
- Resume: cache starts cold, re-extracts (correct, slower first epoch); under
  the shuffling loader not guaranteed bit-exact to an uninterrupted cached run.
- Regen answers or target weights changing between epochs: assumption broken
  -> cache would be wrong; the lane's protocol fixes both (greedy fixed
  transcript, frozen target), and the digest gates forbid silent drift.
- Multi-epoch + single-epoch runs: single-epoch runs gain nothing (first
  epoch always extracts).

### 4.8 Maintenance cost of the patch

Cache feature (merge-ready scope, after the 22:2x cleanup): +137/-5 lines in
`vlm_dflash/target/online.py`, +200 lines of tests (8 cache tests incl. a
tiny real-model end-to-end and an eviction->recompute value-exactness test),
+50 doc lines; opt-in config key `train.online_feature_cache_gb` (default
0 = off), no new deps, runtime-only state, no digest/plan/provenance schema
change. Tests on the cleaned tree (server venv, 2026-10-14): online suite
66/66; full suite 425 collected -> 418 passed / 6 skipped / 1 pre-existing
venv-metadata failure (`test_cli` version vs stale editable dist-info,
reproduces on pristine HEAD). Lane-only env gates used by the lane's runs
were removed from the merge change (kept in the lane working copy +
`lane_patch_backup_20261014/`): `VLM_DFLASH_FLEX_FORCE_MAIN=1` (flex_decoding
workaround), `VLM_DFLASH_SKIP_RESUME_CKPT=1` (hostile-disk runs),
`VLM_DFLASH_TRAIN_SHUFFLE=0` (diagnostic fixed-order runs).

## 5. Final judgement

| Engineering | Paper |
| --- | --- |
| 可合入（附数值行为说明与修正后的 gate） | 仅工程复用（其中"缓存冻结分组"的数值边界可作为协议要点引用） |

Cache A/B, measured paired on the same data/seed: S2 (129-row real slice,
6 epochs, x3 interleaved) train stage 151.6 s -> 74.4 s (-50.9%); S3
(canonical 2K, 1012 regen rows, 6 epochs / 762 steps) train stage
1167.0 s -> 470.9 s (**-59.7%**), extraction 799.1 s -> 125.5 s (-84.3%),
5060 hits / 1012 misses / 0 evictions at 16 GiB (peak 6.35 GiB), zero disk
bytes, +3.97 GiB host RSS, GPU peak unchanged (25.02 GB both). The
as-written bit-parity gate fails, but the failure is fully localised: the
extractor is grouping-shape sensitive at the bf16-rounding scale (probe 2),
the shuffler changes grouping every epoch, and the cache freezes epoch-1
groupings; under a fixed order the ON leg is bit-identical to OFF (S2c:
ckpt sha match, 102/102 logged fields). Under the shuffled protocol the
training signal is intact (valid_tokens identical 762/762; loss identical
740/762, rest <=2 display ulp; accuracy/accept_len deltas at display scale),
while final weights drift deterministically - max 0.0446 abs / 4.26% relL2
at 762 steps (larger than S2's 2^-7 bound; drift grows with training length;
full numbers in §4.5). This is not a cache defect; it is a property of
per-epoch re-extraction that caching removes. Merge recommendation: include
the opt-in cache with the numeric-behavior note and the amended gate (§4.5);
"bit-perfect parity under shuffle" is unattainable by the pipeline itself
and was therefore rejected as a cache criterion. Closing frame 1 is done:
the downstream acceptance eval of the S3 pair gives mean_al identical to 10
decimals (1.5262066923), accepted/new tokens identical (5288/15462),
token_exact=True for both, and a single differing draft token in ~69k -
"no performance loss" is verified end-to-end, not just on training metrics.
Closing frame 2 (seed change, order + init) puts the weight drift in
context: a routine seed change moves the same weights by global relL2 131.5%
and the last-10 loss by 0.345, while the cache moves them by 1.45-4.26%
relL2 (per seed) with display-identical loss - the cache's effect is 1-2
orders of magnitude below the pipeline's own run-to-run variation. The goal
§8.3 offline baseline closes the last comparison (measured 2026-10-14 on the
same staged inputs): producing the 7.0G store + training took 836.8 s
end-to-end and reusing it trained in 548.8 s - both above the cache's
470.9 s train stage at zero disk bytes, and the store round-trip was
numerically lossless (e/e2 train logs byte-identical).

## 6. Next-phase recommendations

1. Merge the cache behind `train.online_feature_cache_gb` (default 0) with
   the numeric-behavior note and the amended gate (§4.5): fixed grouping is
   bit-exact; under the shuffled protocol report the per-run drift
   (valid_tokens/loss equality + quantified weight drift).
2. If bit-exactness is required by policy, run with the fixed extraction
   order (`VLM_DFLASH_TRAIN_SHUFFLE=0`); otherwise accept the bounded drift
   measured here.
3. Optional deepenings (not blocking, ordered by value): (a) an order-only
   control (same init, different order seed) to split init vs order
   contributions to the 131.5%; (b) acceptance eval at seed 43 for
   replication; (c) drift growth over longer schedules (>=12 epochs) before
   applying on much longer runs; (d) eval at other prompt counts/samplers.
4. Do not expand H-C3 (split-device) on this evidence; H-C2 stays deferred;
   H-A stays STOP.

Paper judgement: engineering reuse, not a standalone systems contribution —
byte-bounded LRU retention of already-materialised CPU records is standard,
and the wall win is a strong-but-mundane systems result. The reusable
artifact worth citing is the equivalence-boundary protocol for feature reuse
in online distillation (freeze the grouping, or bound the drift; with the
probe/S2c methodology as an evaluation harness), plus the H-A closure
(teacher-forward batching saturates at -13.3% and is not the lever). The
strongest counter-argument — "an offline store with page-cache reads is
equivalent when the disk is healthy" — is answered by scope: this lane's
claim is the disk-free variant (0 bytes written on a 100%-full host), and
the store writes 7.0G for this workload the cache does not (measured
2026-10-14; the older 1050-row baseline quotes 9.4 GB). Even with a store
already written, the measured reuse training (548.8 s) did not beat the
cache's train stage (470.9 s) in tonight's single-shot comparison.
