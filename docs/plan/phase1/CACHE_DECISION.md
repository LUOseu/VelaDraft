# Phase-1 decision: all-online vs all-offline vs a fixed-space cache

Question (route document §15.6 / §4.3): compare **all-online**, **all-offline**
and a **simple fixed-space cache**, and implement the mixed materialization
planner only if those simple plans are shown insufficient.

What the repository supports today (verified on this worktree):

| Plan | Status | Entry point |
| --- | --- | --- |
| All-offline | shipped, default | `capture` stage -> feature store; `train.feature_mode=offline` |
| All-online | shipped | `train.feature_mode=online` (`--mode online`); live frozen target |
| Mixed (some units cached, others live) | **not implemented** | - (a run is entirely one source; `--features` is mutually exclusive with online) |

## The cost model, instantiated with recorded numbers

Route §4.3's serial approximation: keep feature unit `u` iff
`n_u * c_u > c_u + w_u + n_u * r_u`, i.e. saving per kept unit
`S(u) = (n_u - 1) * c_u - w_u - n_u * r_u`, under the disk budget `B`.

Recorded instantiations ([M] measured / [D] derived, sources in
`PROFILING.md`):

| Symbol | Meaning | Value used |
| --- | --- | --- |
| `c_u` | extract one unit (render + teacher forward) | ~0.156 s/row at ~600 tokens, 2 images [M components: render 17 ms + forward 139 ms] |
| `w_u` | extra write cost of the store | ~0.01 s/row [D: capture 0.149 s/row amortised minus the forward share] |
| `r_u` | read cost during training | ~0.01 s/row [D estimate from shard-local reads; no dedicated IO measurement - `PROFILING.md` §7] |
| `s_u` | stored bytes | 16 KB/token (C=3) -> 8.4 MB at L=512, 33.6 MB at L=2048, 67.1 MB at L=4096 [D from the store layout] |
| `n_u` | expected reads (epochs x algorithms that can consume the unit) | recorded program: 6 epochs x up to 7 algorithms sharing one layer set |

Per-unit saving at `n_u = 6`: `S = 5 x 0.156 - 0.01 - 6 x 0.01 ~= 0.71 s/row`
[D]. For the recorded 2K corpus (1050 rows) that is **~12.4 min saved per
algorithm-run** versus all-online, at 9.4-12.4 GB of disk; shared across the
seven algorithms the same store saves ~7x that.

Even at `n_u = 2` (two epochs, one algorithm: `S = 0.15 s/row`) keeping the
unit is positive. Offline is the right default for any unit read more than
once - which is what the repository already does.

## Plan comparison on the recorded geometry

| | All-online (per algorithm-run) | All-offline (per store) | Fixed-space simple cache |
| --- | --- | --- | --- |
| Target compute in training | 6 epochs x 0.156 s x 1050 rows ~= 16.4 min [D] | 0 (store read only) | `epochs x (1 - stored_frac) x c` |
| One-time extraction | 0 (folded above) | 2.6 min [M] | capture of the stored units |
| Disk | 0 | 9.4 GB / 12.4 GB [M] | `B` (cap) |
| Target resident RAM/VRAM | 7.5 GB target + 13.7 GB train peak [M] | not loaded during training | target still resident for the live share |
| Cross-algorithm reuse | none | one store serves every algorithm whose layers are a subset [M, zoo program] | only the stored fraction |
| Resume / determinism | live target forward, fast-forward accounting [M] | store digest pinned; resume gate refuses changed stores [M] | needs a per-row source switch (**not implemented**) |

## Does the mixed planner earn its implementation?

1. **On the audited host's budget the question is nearly moot.** The audited
   host had 61 GB free and the full 2K store is 9.4-12.4 GB: all-offline fits
   with room to spare, and it is the plan with the best measured behaviour
   (one store, seven algorithms). There is no budget pressure to solve at the
   recorded scale.
2. **When the budget does bite, the "planner" collapses to a length sort.**
   Under the recorded cost structure `c_u ~= a_vision(pixels) + b * L` (forward
   is prefill-bound) while `s_u = 16 KB * L`, so the saved teacher-time per
   stored byte `c_u / s_u` is **largest for short/medium units** (the visual
   cost is amortised over fewer bytes). A deterministic "shortest-first up to
   the budget" prefix policy captures that ranking; the rest of §4.3's cost
   model (per-unit `n_u`, differing signals) adds nothing measurable until
   units differ in *more* than length. Per the route document's own rule
   ("implement only if the simple plans are insufficient"), the planner is
   **not justified** at this stage; a plan that ranks by length is a policy
   variable inside the existing offline path, not a new subsystem.
3. **At document scale, neither plan fits; the fix is not a planner either.**
   Formula: 100k rows at L=4096 -> 6.7 TB (C=3) / 10.1 TB (C=5)
   (`PROFILING.md` §3). With 61 GB free, a fixed-space cache stores
   `61e9 / 67.1e6 ~= 0.9%` of the units - a saving of `0.009 x 0.71 s/row`
   [D], i.e. invisible. At that scale the honest options are: keep all-online
   (target resident), reduce captured layers, or capture from a serving engine
   at inference time (route §4.5's "target available window" constraint), not
   a smarter selection.

## Decision (recorded)

- **Keep the two shipped plans as the first version.** All-offline stays the
  default for corpora that fit the budget (the recorded behavior); all-online
  exists for single-target convenience and for corpora whose stores do not
  fit (document scale).
- **Do not implement the mixed materialization planner now.** The simple
  baselines cover the recorded geometry; the planner's marginal gain over a
  deterministic shortest-first prefix is a bounded constant on teacher time,
  and that time is *not* on the training path in offline mode (the dominant
  costs are regen 33% + eval 40%, which no cache changes - `PROFILING.md` §1).
- **Trigger conditions for revisiting** (write these into the ledger):
  (a) a corpus whose store exceeds the budget by >2x *and* whose units have
  heterogeneous per-byte extraction costs beyond length (e.g. pixel-budgeted
  images, per-recipe teacher-signal needs); or (b) the online extraction
  share of step time exceeds ~20% at the target scale (currently a projection,
  re-measurement is GPU_PENDING #2 in `REVIEW_REMEDIATION.md`); or (c) a
  serving-engine capture path exists whose `c_u` differs from the training
  host's (engine-side capture is the competitor-audited fix for exactly this;
  see `COMPETITIVE_AUDIT.md`).
- If the mixed mode is implemented later, the semantic constraint from §4.3
  stays: the batch composition may not change optimizer membership; mixing
  stored/live units must preserve the sample order and any bucketing becomes
  a separately-labeled training strategy.
