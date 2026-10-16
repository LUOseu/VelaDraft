# Parallel verification: measured speedup, cost decomposition and mismatch diagnosis

Status: **E2/E3 complete for R3 (`ckpt_final.pt`); E4 (full 256) not run.** No retraining was done,
the R3 checkpoint and all previous benchmark artifacts are untouched; every number here comes from
new tags (`profile_r3_*_32p*`, `diag_first_divergence_*`) written next to the old ones.

Harness: `scripts/scale/profile_parallel.py` (new) and `scripts/scale/speed_paired.py` (extended with
the dual AL metrics and the phase breakdown). Same 32-prompt prefix of the fixed dev split, greedy,
`max_new_tokens=128`, bf16, target = Qwen2.5-VL-3B (frozen), drafter = R3 `ckpt_final.pt`,
GPU = RTX A6000, one card per run, 3 warmup prompts excluded from the measured window.

## 1. Headline table (E3: 32 prompts, 128 output tokens)

| path | wall speedup vs target-only | `A` (AL_progress) | AL_draft | `r` | `d` | `S_model = A/(r+d)` | target fwd / token | token-exact |
|---|---|---|---|---|---|---|---|---|
| target-only | 1.00x (reference) | 1.0 | - | 1.0 | 0 | - | 1.00 | reference |
| **sequential** (lossless) | **0.853x** | 2.804 | 1.798 | 2.799 | 0.436 | 0.867 | 1.02 | **32/32** |
| **parallel** (current fast path) | **1.855x** | 2.874 | 1.868 | **1.000** | 0.447 | **1.987** | 0.36 | **18/32** |
| strict_parallel (margin 2.0, sweep run) | 0.678x | 2.858 | 1.853 | 3.625 | 0.427 | 0.687 | 1.32 | 19/32 |
| optimized parallel (BlockVerifier fast path) | not built - the measured addressable overhead is 0.8% of wall (§2, decision in §4) | - | - | - | - | - | - | - |

- `A`, `r`, `d` are `AL_progress`, `block-verify-s / single-token-decode-s` and
  `draft-s / single-token-decode-s`; the single-token decode time comes from a two-point reference
  probe (`generate(max_new_tokens=1)` vs `generate(max_new_tokens=L)` including a correction for
  early EOS stops) and is ~29 ms/token on this card/context.
- `S_model` (1.99) sits ~7% above `S_wall` (1.86) for the parallel path; that residual is the target
  prefill, which `A/(r+d)` does not model (3.35 s of 38.1 s of the speculative wall here, partly
  offset by the same prefill inside the target-only baseline).
- **`r = 1.000` means the block verification is already at its floor**: one forward per round costs
  exactly one single-token decode. There is nothing left to win by making verification cheaper.
- **The two live levers are `A` and `d`**: at `r = 1`, `S = A/(1+d)`, so a free drafter would give
  `S = 2.87` (+55%) and `A = 4` at the current `d` would give `S = 2.76` (+49%).
- The strict guard is **worse than useless in this configuration**: it buys one extra exact prompt
  (19/32 vs 18/32) and costs more than the lossless sequential path (0.678x vs 0.853x). The full
  sweep in §3.6 shows the only exact setting (fallback every round, 32/32) lands at 0.655x, i.e.
  dominated by plain sequential verification.

## 2. Where the wall clock actually goes (E2, parallel path, 32 prompts)

| component | ms per round | share of speculative wall |
|---|---|---|
| target block verify (1 forward) | 29.77 | 62.5% |
| drafter (1 block forward) | 13.00 | 27.3% |
| target prefill (amortised) | 4.19 | 8.8% |
| cache commit (crop + writes) | 0.22 | 0.5% |
| verify prep (mask/position/inputs) | 0.05 | 0.1% |
| round prep | 0.08 | 0.2% |
| misc (python glue, sampling, stop scans) | 0.35 | 0.7% |

**This is the direct answer to "where is the block verifier's wall-clock overhead":** it is not in
KV management, not in mask/position construction, and not in the commit path - those are 0.5% and
0.1% respectively. It is the verify forward itself, which is already one forward per round, plus the
drafter at 27%. See `docs/vispec_runtime_comparison.md` §4: porting ViSpec's preallocated-KV/gather
design would optimise the 0.5%, so it is **not** justified for our linear-block shape.

The task doc's results table asks for an "optimized parallel" column; it is intentionally absent
because no optimised path was built - §4 gives the measured reason (the addressable overhead is the
0.8% of wall that the rewrite would touch, against a verify forward that is already at its `r = 1`
floor, and the task doc forbids unattributable rewrites). The "Delta" column is therefore empty by
decision, not by omission.

## 3. Mismatch diagnosis (P0)

Protocol: `scripts/scale/diag_parallel_first_divergence.py` runs the same prompt through four paths
(target-only reference, sequential, parallel, parallel under the narrowed/parity SDPA backend, plus
strict) and, for the first divergent position, compares the decision trace captured inside
`spec_generate` (`capture_decision_trace=True`): absolute position, forward shape, KV length at the
forward, argmax, top-1..top-5 logits and the top-1/top-2 margin.

### 3.1 The one prompt that flips at 32 tokens (dev row `000000115506`, first divergence at output index 14)

| path / forward shape | KV len at forward | argmax | top-5 ids | top-5 values |
|---|---|---|---|---|
| sequential (1 query row) | 441 | **264** (=reference) | 264, 23880, 50911, 19653, 16158 | 15.0, **15.0**, 14.875, 14.5625, 14.1875 |
| parallel block row that committed it | 447 | **23880** | 23880, 264, 50911, 19653, 16158 | 15.0, **14.9375**, 14.875, 14.5625, 14.1875 |

- The **same candidate set** comes out of both shapes; only the top-2 ordering moves, by **one bf16
  ULP (0.0625 at magnitude ~15)**. In the sequential shape the two candidates are *exactly tied*
  (both 15.0), and the tie resolves to the lower id.
- The narrowed/parity SDPA backend reproduces the flip identically, so it is **not** an SDPA-backend
  selection artifact: it is the block-shape vs single-token-shape numerical difference.
- `--target-only-repeats 2` shows the reference path is bit-stable on reruns, so this is not the
  "baseline is unstable" case either.
- Rows computed inside a block *beyond* the accepted prefix condition on draft tokens that are then
  discarded: those rows (visible in the trace with different top-5 sets) are not evidence of a bug
  and must not be compared against the reference.

**Classification: Case B - numerical-path divergence at a near-tie, reproducible, no logic/offset
bug found.** A wrong decision needs a margin below ~1-2 ULP, which is why the strict guard can catch
a specific instance but cannot make the path exact in general.

### 3.2 Characterisation over all 14 divergences (E3, 32 prompts x 128 tokens, R3)

The 14 prompts that came out non-identical in the official parallel run were re-run through all four
paths (`diag_first_divergence_r3_mismatch14.json`):

| path | prompts with a first divergence |
|---|---|
| sequential (lossless reference) | **0 / 14** |
| parallel (current fast path) | **14 / 14** |
| parallel, narrowed/parity SDPA backend | **14 / 14**, at the *same* first-divergence index as the plain parallel path |
| strict_parallel (margin 2.0) | 11 / 14 (3 fixed outright; on others it only moves the divergence later, e.g. index 34 -> 109 and 14 -> 78) |
| target-only reference, 2 reruns | bit-stable on 14 / 14 |

At the deciding position of each divergence:

```
median top-1/top-2 margin   sequential : 0.0000
                            parallel   : 0.0625
max |logit delta| between the two forward shapes, for the same token : 0.125
```

0.0625 is one bf16 ULP at magnitude 14-16, 0.125 is two; the margins at these positions are
literally 0.0 or 1 ULP. In other words **every divergence in the sample happens at an exact or
one-ULP tie**, and the two shapes agree on the candidate set to within that.

### 3.3 Classification: numerical-path divergence (Case B), with the Case-A checklist performed

The task doc's decision tree assigns "parallel performance and parallel parity both wrong at the same
position" to Case A (look for logic bugs). We ran that checklist explicitly before accepting the
numerical explanation:

| Case-A item | check | result |
|---|---|---|
| logits offset | trace keys are absolute positions; the deciding parallel row and the sequential row carry the *same* absolute position | no offset found |
| first-mismatch / correction-token handling | emitted token equals the argmax of the very row that committed it; the token before the divergence matches the reference | consistent |
| KV commit / crop | committed KV length at the deciding forward equals the sequential path's (441 vs 441, 526 vs 526, ...); only the block row's own `kv_len` is larger, as expected | consistent |
| `cache_position` / position ids / mRoPE | decision traces line up position-for-position across shapes for the whole prefix | consistent |
| attention mask / image-text positions | prompt-side prefill is shared by both paths; divergence positions are far into the text tail | not implicated |
| EOS / bonus token | divergences occur mid-sequence, not at stops | not implicated |
| magnitude of the difference | same candidate set, top-2 swapped by 1 ULP | numerical, not structural |

Two further observations support Case B over Case A: the narrowed SDPA backend changes nothing (so
the cause is the *forward shape*, i.e. GEMM/kernel shapes in the block forward, not the SDPA backend
selection), and the margin guard does not remove the phenomenon, it only relocates it (a deterministic
logic error would not be "fixed" by falling back at one position and reappear at another).

**Caveat on the doc's terminology:** our `eval.verify_sdpa_backend=parity|reference` switch only
narrows the SDPA backend; it does **not** provide a second shape-independent implementation of the
block forward. So the branch that the decision tree calls "parity agrees with the reference" cannot be
produced by that switch, and the classification above rests on the direct logit evidence instead.

### 3.4 What "exact" can honestly mean here

- Under a fixed deterministic math path (single-token sequential verification) the acceptance logic
  is exact: 32/32 prompts in E3, 256/256 rows in the eval round.
- Under the block-shape path, exactness holds only up to bf16 decision ties: identical candidate
  sets, identical conditioning, a last-ULP argmax flip at positions the model itself treats as ties.
  The honest label is **"numerically equivalent up to last-ULP argmax flips (18/32 bit-identical,
  14/14 divergences at margins <= 1 ULP)"**, never "lossless".
- Making it literally lossless needs the *decision* to reproduce, e.g. an fp32 verify head, a
  re-check of only the near-tie positions in the sequential shape, or a documented epsilon-exactness
  contract. None of those is implemented yet, and the cost of the re-check variant is exactly what the
  strict experiment measured indirectly (it is expensive when the threshold is 2.0).

### 3.5 E4 status

The full-256 run was **not** executed: the task doc gates E4 on "no correctness regression, wall
speedup > current parallel, and the gain not being a timing artifact". The current parallel path
fails the first condition (18/32 bit-identical) and no optimized BlockVerifier exists yet that passes
it - see §4.

### 3.6 Strict-margin guard sweep: the exactness/speed frontier (R3, 32 prompts)

The `strict_parallel` guard re-verifies a round sequentially whenever the smallest top-1/top-2 raw
logit gap over the round's used decisions (accepted positions + bonus token) falls below a threshold.
Sweeping that threshold traces the exactness-vs-cost frontier
(`eval/profile_r3_strict_m{0.25,1.0,2.0,1e9}_32p.json`; same 32 prompts, same GPU, same protocol as
§1-§2, sweep run 12:27-12:41 with the card otherwise idle; `m=1e9` is the control in which every
round falls back):

| margin | token-exact | wall speedup | `r` | `d` | `A` | fallback rounds | target fwd / token |
|---|---|---|---|---|---|---|---|
| none (plain parallel) | 18/32 | 1.855x | 1.000 | 0.447 | 2.874 | 0% | 0.36 |
| 0.25 | 16/32 | 1.208x | 1.815 | 0.441 | 2.838 | 25.9% | 0.66 |
| 1.0 | 19/32 | 0.746x | 3.243 | 0.435 | 2.805 | 77.5% | 1.18 |
| 2.0 | 19/32 | 0.678x | 3.625 | 0.427 | 2.858 | 92.2% | 1.32 |
| 1e9 (control: every round falls back) | **32/32** | 0.655x | 3.828 | 0.444 | 2.804 | 100% | 1.37 |

What the curve says:

1. **No setting is both exact and fast.** Exactness is reachable only at `m=1e9`, where every round
   pays one wasted block verify *plus* a sequential re-verify (`r = 3.83 ~= 1 + 2.80`), landing at
   0.655x - *below* the plain lossless sequential path (0.853x), because on top of identical
   re-verification work it still pays one discarded block forward per round. Run uniformly, the
   guard's fallback *is* the sequential verification that §1 measured at 32/32.
2. **The frontier is not monotone at the bottom.** `m=0.25` makes exactness *worse* than no guard
   (16/32 vs 18/32) while already giving up a third of the speed. A partially-firing guard creates
   mixed numeric histories - fallback rounds commit token-shaped KV, non-firing rounds commit
   block-shaped KV - and the mixed stream mutates at new positions that no per-round margin check
   inspects. Only uniform 100% fallback restores the path to a single, matched numeric shape.
3. **The threshold semantics explain the saturation.** The guard compares an absolute raw-logit
   gap, while the flips it targets scale with one ULP of the local logit magnitude (~0.06-0.13 at
   logit magnitude 15-30). Rising m from 1.0 to 2.0 sweeps that whole band and buys zero net
   exactness (19/32 at both) while still raising cost (`r` 3.24 -> 3.63).

Consequence: the strict guard is reported as a *measured frontier*, not as a fix. It is dominated
everywhere - its exact point is slower than just running sequential verification, and its fast
points are not exact. This closes the "can a cheap guard buy faithfulness back?" question from
§3.4 with a measured no; the residual 14/32 last-ULP flips are not separable by any absolute
margin threshold at acceptable cost.

## 4. On the BlockVerifier fast path: measured decision, not implemented

The task doc's P1 item was a `Qwen25VLBlockVerifier` (prefill / verify_block / commit) modelled on
ViSpec's target-side runtime. **It was consciously not implemented in this round**, and the reason is
the measurement in §2:

- the target-side KV/position/commit machinery it would rewrite accounts for **0.8%** of the
  speculative wall (commit 0.22 ms + verify prep 0.05 ms + round prep 0.08 ms per round, against
  29.8 ms of verify and 13.0 ms of draft);
- the verify forward it would replace is already **one forward per round** and already costs
  `r = 1.000` single-token decodes - there is no scheduling win left in the block verify itself;
- therefore a rewrite would optimise <1% of the wall while touching the path that produces the
  round's correctness evidence, which the task doc explicitly forbids
  ("不要同时重构 verifier + drafter + dataset，导致无法 attribution").

What the numbers license instead: `S = A/(1+d)` with `A = 2.87-2.87` and `d = 0.44-0.45`, so the two
levers that matter are `A` and `d`, and the fidelity question in §3 is the gate on quoting any of it.

## 5. Official harness cross-check (`scripts/scale/speed_paired.py`)

The same measurement taken by the pre-existing harness (the one the training round used), which is
the number quoted in `reports/REPORT_R3R5_NEXT_ROUND.md` §7.3:

| | R3 | R5 |
|---|---|---|
| wall speedup (ratio of means) | 1.902x | 1.775x |
| paired mean / median of per-sample ratios | 2.117x / 2.043x | 1.958x / 1.865x |
| `token_exact` | 18/32 | 18/32 |
| AL_progress / AL_draft | 2.874 / 1.868 | 2.856 / 1.848 |
| target forwards per round | 1.040 | 1.040 |
| draft / verify / prefill share | 28.0% / 60.9% / 9.6% | 32.2% / 57.9% / 8.5% |

R3's 1.902x (harness) and 1.855x (instrumented profile, separate run) agree within run-to-run noise;
R5 is the same story with a 3-4% larger drafter share, i.e. **the deeper drafter is worse on AL and on
cost at the same time, in the parallel path as well as in the sequential one.**

## 6. Answers to the task doc's eight questions

1. **Is VelaDraft's AL the same quantity as ViSpec's `tau`?** No - `tau` is draft-only, ours was
   progress-inclusive; the aligned numbers are `al_draft` 1.85 (dev) / 1.80 (test) for R3 vs their
   2.19, still not comparable across candidate budgets. `docs/benchmark_metric_alignment.md`.
2. **What is ViSpec's target-verification runtime?** Preallocated KV + CPU-side length pointer,
   commit by `index_select`+`copy_`, one tree-wide forward per round with the tree mask injected into
   the 4D causal mask, explicit mRoPE positions rebuilt from cached `rope_deltas`.
   `docs/vispec_runtime_comparison.md` - which also holds the same-machine micro-reproduction (§7
   there): official checkpoint on SQA/T=0 gives `tau = 1.98` and 2.04x on this GPU, in the same
   league as our parallel path, and not token-exact vs its own baseline either.
3. **Where is our block verifier's wall-clock overhead?** 62.5% target verify forward (already one
   forward per round, `r = 1.00`), 27.3% drafter, 8.8% prefill amortisation, 0.8% all KV/position
   plumbing combined, 0.7% glue. §2.
4. **Which ViSpec techniques are worth porting?** The discipline (verifier owns positions/mask/length,
   commit = pointer update) and their `rope_deltas` reuse if prep ever grows; **not** the tree
   gather/runtime, and not the 2508-line model fork. §2 + `docs/vispec_runtime_comparison.md` §4-§5.
5. **Before/after `A, r, d`?** No code change was made to the verify path, so there is no before/after
   yet: `A = 2.874`, `r = 1.000`, `d = 0.447`; the sequential reference is `A = 2.804, r = 2.799,
   d = 0.436`. §1.
6. **Before/after end-to-end speedup?** Before = sequential 0.853x (lossless) / parallel 1.855-1.902x
   (18/32 exact). No "after" exists because the fast-path rewrite was rejected on evidence (see §4).
7. **Is the mismatch a logical bug or numerical-path divergence?** Numerical: 14/14 divergences at
   margins of 0-1 ULP, same candidate sets, parity backend irrelevant, Case-A checklist clean. §3.3.
8. **Can the parallel path be called exact/lossless?** No. Accurate formulation: "numerically
   equivalent up to last-ULP argmax flips; 18/32 prompts bit-identical on the official 32-prompt
   set; the lossless path is sequential verification at 0.85x". A strict-margin guard does not buy
   exactness back - its only exact setting is the all-fallback control at 0.655x (§3.6). §3.4.
