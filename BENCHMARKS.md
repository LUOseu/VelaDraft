# Benchmarks

All acceptance numbers below use **sequential verification** (the only
token-exact path), greedy decoding, and the target's own greedy output as the
reference. Do not mix experimental parallel-runtime numbers into these tables.

## Acceptance (trained DFlash vs random-init drafter)

| Target | Corpus | Train rows | Prompts | AL (trained) | AL (random init) | Token-exact | Per-slot acceptance (trained) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Qwen2.5-VL-3B | 512 | 278 kept | 64 | 1.198 | 1.003 | 64/64 | see eval_result.json |
| Qwen2.5-VL-3B | 2K | 1050 kept | 128 | 1.630 | 1.003 | 128/128 | .413, .149, .048, .015, .004, .001, 0 |
| Qwen3-VL-8B | 512 | 274 kept | 32 | 1.322 | 1.003 | 32/32 | see eval_result.json |
| Qwen3-VL-8B | 2K | 1996 kept | 32 | 1.665 | 1.000 | 32/32 | .413, .160, .063, .027, .010, .002, .0004 |
| Qwen2.5-VL-3B | 5K | 2629 kept | 128 | 1.817 | 1.003 | 128/128 | .482, .210, .080, .030, .011, .005, .001 |

Notes
- The 512-corpus 3B row is the 2026-10-11 rerun (original v0.1: AL 1.220);
  bf16 runs reproduce the protocol, not the bits (< ~2% AL noise).
- Slot-0/1 acceptance dominates; 2K is the first point with stable non-zero
  acceptance through slot 5, 5K extends it through slot 6.
- 5K training: block CE 6.84 -> 2.58 (first/last-10% means), 1974 steps,
  peak 30.0 GB, ~40 min on one A6000; overfit sanity passed.
- 8B / 2K: regen kept 1996/2000 rows, capture 1996 samples (visual-path check
  passed), training 1500 steps (resumed from 375), final loss 2.69, slot-0
  teacher-forced accuracy 0.73; eval 32 prompts / 128 tokens, speedup 0.81x.
- `AL = new_tokens / target_verification_steps` (bonus-token-inclusive);
  random-init drafter AL ~1.003 means proposals are almost never accepted.

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

## Reproducibility

- Same seed + same data: `rows_sha256` and regen keeps reproduce; captured
  shard hashes and bf16 training losses do not (kernel nondeterminism).
- Treat AL differences < ~2% as noise unless the protocol pins more seeds.
