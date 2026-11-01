# Shared DFlash path diagnosis

Requested 2026-11-01 after the K-sweep launch. DFlash now precedes DFlash2 on the (2,3) TP2 lane; DFly continues on (0,1). DFlash's first valid K1 E2E result triggers a separate AR / SD K1 / SD K7 profile before its remaining uninstrumented K trials.

The profile reuses DFlash K1's real canonical AR histories: one general-VQA context and the first/last turns of the first multi-turn diagnostic session. All arms use identical initial prompt token IDs. Warmup precedes hook installation. Each measured profile request generates at most 128 tokens. Subsequent generated prefixes may diverge, so this is not an exact fixed-prefix kernel benchmark.

Record per rank and scheduled step: host span, CUDA-stream elapsed span, actual scheduled token count, scheduled draft count, first input tensor shape, target execute/forward, sampling+proposal, proposal total, feature fusion, input/position construction, attention metadata, input construction/context K/V, backbone, sampling and LM head. Inclusive spans overlap their children and must not be summed. Stream elapsed time is not pure GPU compute time; communication and gaps may be included.

Use official accepted/proposed/round counters and actual output token count separately. Actual per-spec-round committed output after EOS truncation is unavailable and stays null. Do not label `1+accepted/rounds` as measured actual progress.

## Instrumentation changelog

| File | Change | Scope |
|---|---|---|
| profile_worker.py | Created | Opt-in worker RPC and CPU/CUDA-event wrappers; never imported by main E2E runs |
| profile_stage.py | Created | Warmup, fixed initial contexts, per-rank spans and counters |
| profile_driver.py | Created | Fresh AR/SD engines under a TP2 resource lease |
| test_profile_worker.py | Created | Return/call-count preservation and correct current-step attribution |
| queue.py | Modified | Run separate profile after valid DFlash K1, then continue E2E sweep even if profile fails |

Results will be under `/home/lyh/qwen36_k_sweep_20261101/profile_dflash_k1_k7/`. No profile numbers exist until these GPU runs finish. Main K results remain in `RESULTS.md`/`RESULTS.json` and never use the instrumented wall times.
