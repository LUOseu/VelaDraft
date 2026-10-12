# Verification modes and the runtime probe

`eval.verification` selects how the target verifies drafted blocks (mechanism
in `docs/REFERENCE.md`):

- `sequential` (default) - generate()-equivalent numerics; the **token-exact
  correctness gate**. All correctness/acceptance numbers use this mode.
- `parallel` - one target forward per drafted block; an **experimental
  throughput path**, not bitwise-exact in bf16.
- `strict_parallel` - parallel path plus a near-tie guard with sequential
  fallback (`eval.strict_parallel_margin` sets the top-1/top-2 logit gap
  threshold); still not bitwise-exact.

Parallel-mode numbers must always be reported together with their agreement
rate and never mixed into the correctness tables.

## Probe: Qwen3-VL-8B, 512-sample drafter (`runs/qwen3vl8b_dflash`, AL ~1.33)

| mode | token-exact vs target-only greedy | wall / sample | speed vs target-only |
| --- | --- | --- | --- |
| `sequential` (default; the gate) | **32/32 prompts** | 5.25 s | 0.84x |
| `parallel` (one forward per block) | 6/32 | 4.31 s | 1.02x |
| `strict_parallel`, margin 0.25 | 11/32 (4.6% of rounds re-verified) | 4.71 s | 0.97x |
| `strict_parallel`, margin 2.0 | 10/32 (45.8% re-verified) | 6.61 s | 0.67x |
| `strict_parallel`, margin 1e9 (control) | **32/32** | 8.66 s | 0.51x |

## Probe: Qwen2.5-VL-3B, 2K-scale drafter (`runs/qwen25vl3b_2k`, AL ~1.68)

| mode | token-exact | AL | wall / sample | speed vs target-only |
| --- | --- | --- | --- | --- |
| `sequential` (gate) | **32/32** | 1.692 | 4.35 s | 0.86x |
| `parallel` | 8/32 | 1.677 | 2.97 s | **1.25x** |
| `strict_parallel`, margin 0.25 | 7/32 (14.0% re-verified) | 1.690 | 3.66 s | 1.03x |

Protocol: `ckpt_final.pt`, 32 held-out prompts, greedy, bf16, one A6000,
`verify_sdpa_backend: parity`; raw numbers in `eval/eval_result_*.json`, drift
probe in `scripts/diag_verify_drift.py`.

## Reading

- Block-parallel verification is not bitwise-equivalent to step mode in bf16:
  the same position with identical inputs moves by up to ~0.31 logits (prefill
  deltas are 0). Accepted tokens carry that drift into the KV cache, so a
  per-round near-tie guard reduces but cannot eliminate divergence - the
  control run re-verifies every round and lands exactly on 32/32 with the same
  AL, which pins the residual divergence on numerics rather than bookkeeping.
- Speed scales with drafter quality, exactness does not follow: the parallel
  path measured 1.02x at AL 1.33 (512-sample 8B drafter) and 1.25x at AL 1.68
  (2K-scale 3B drafter), while the sequential gate - the only bitwise-exact
  mode - was 0.84x and 0.86x on the same settings.
- `sequential` therefore remains the correctness gate; parallel modes are
  throughput experiments whose agreement rate is reported per run.
