# Qwen3.6 target verification cost, draft lengths 1–7

Server root: `/home/lyh/qwen36_verifylen_20261101`.
Active attempt: `active/` -> `attempt2_g12/`, physical A6000 GPUs 1/2, TP2. This pair crosses NUMA sockets; its topology is recorded in `TOPOLOGY.json`. The first attempt on GPUs 0/1 was aborted when another user's process entered GPU0 and is excluded. Do not mix timings from these layouts.
Target: Qwen3.6-35B-A3B BF16. Drafter: final 75000-step DFlash, unchanged B8 weights. Runtime query width is K+1; K is the number of proposed draft tokens, not KV-cache page size.

Eight fresh engines run sequentially: native AR, then actual DFlash speculative decoding at K=1,2,3,4,5,6,7. Target FULL_AND_PIECEWISE and draft PIECEWISE graph modes are retained; capture width is matched to K+1. All parameters except K and the necessary graph width remain fixed within SD.

Three initial contexts are shared across arms: first general-VQA session, and first/last turns of the first multi-turn session from the frozen 12-session diagnostic subset. The untimed AR history is reused from the verified completed reference under `qwen36_k_sweep_20261101/dfly_k1_r1_a2.evidence/reference.json`. Prompt token IDs are checked. Each context is warmed, then profiled three times with up to 128 generated tokens. Continuations can diverge; this is an actual-runtime profile, not an exact fixed-prefix kernel microbenchmark.

Metrics:
- `target_forward`: target model forward stream time and host-call time; excludes proposal and separate sampling.
- `verification_execute_plus_sampling`: disjoint target execute and target sampling/rejection spans, per rank before taking the slower rank. Includes forward/metadata/head and sampling work, excludes draft and time gaps between calls. No nested forward is added twice.
- Per-context count, mean, median and P90. AR uses actual scheduled width 1, SD uses full-width verify steps with actual scheduled width K+1. Prefill, fallback and short final steps remain in raw profiles but are excluded from the main width curve.
- Proposal components are separate; inclusive parent spans overlap children and must not be summed.

Outputs: `STATUS.json`, `queue.log`, `stages/<arm>/profile.json`, `logs/<arm>.log`, `verification_latency.csv`, `component_latency.csv`, `SUMMARY.json`, `RESULTS.md`, `resources.jsonl`.

Timing is observer-affected CPU/CUDA-event profiling. CUDA-stream intervals include relevant waits and gaps, not only kernel execution. No profiled wall time is reported as end-to-end speedup. Accepted counters and actual output counts are separate; actual per-spec-round committed counts after EOS remain unavailable, never guessed from AL+1.

Instrumentation files are `research/qwen36_k_sweep/profile_worker.py` and `profile_stage.py` in this isolated checkout; orchestration and aggregation live here. The original 8B and Qwen3.6 speed result files are preserved. GPU PID ownership is checked throughout; foreign compute processes invalidate the profile.

Before launch, 14 CPU tests passed: 7 short-budget/config/math tests, 3 timing-wrapper tests, and 4 aggregation tests. GPU correctness/support and timings must be established by the actual runs; no values are prefilled.
