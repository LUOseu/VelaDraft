# Fixed-K backend experiments

The completed upgrade and reproduction preset are documented in
[`../qwen36_v032/README.md`](../qwen36_v032/README.md). The four candidates
below are retained as screened alternatives, not the adopted configuration.

This experiment isolates backend changes from the earlier K=7 to K=2 change.
The control is the already tuned K=2 DFlash configuration on A800 TP1, BF16.
The target, draft, prompts, native verification and timing boundaries are fixed.
No custom CUDA/Triton kernels or weight changes are introduced.

## Candidates

1. Tune the official vLLM MoE tile configuration at M=3 using real layer-0
   expert weights and shared/disjoint/random routing probes. Preserve every
   other entry, especially M=1, in the previous tuning table.
2. Fuse the shared-expert final gate with official `torch.compile`. Preserve
   BF16 rounding points with `emulate_precision_casts`. Use only unquantized
   TP1, hidden size 2048 and query sizes 1..8; all other paths use the original.
   This is a common backend improvement and must also be offered to AR.

The real-weight probes use synthetic inputs; their speedups are not E2E
speedups or guarantees of numerical equivalence. Both candidates first enter
separate 4-condition comparisons: baseline AR/SD and candidate AR/SD, with
reversed process order for the gate experiment. Three fixed contexts, three
warm-engine repetitions and max 128 output tokens are screening evidence only.
All measured calls include the existing native request preparation and
generation boundary. Model loading and compilation are outside steady-state
request latency in both arms. No profiler is active during these comparisons.

## Acceptance

- Compare SD latency before/after at fixed K, as well as paired AR/SD speedup.
- Report generated token counts and paired output equality separately.
- Never select a candidate because it slows AR. Keep the fastest validated AR
  control visible when interpreting net gain.
- Require immutable source hashes, runtime snapshots and GPU resource records.
- A larger 12-session confirmation follows only promising screening results.
  These sessions are diagnostic data, not an untouched paper benchmark.

## Files

`shared_gate_probe.py`, `moe_fixed_search.py`: isolated CUDA-graph probes.
`shared_gate.py`, `worker.py`: reversible opt-in gate backend and worker audit.
`e2e_stage.py`, `e2e_compare.py`: native model request comparison and controller.
`run_probe.sh`, `run_compare.sh`: server launchers, no model downloads to client.

Server artifacts: `/root/autodl-tmp/qwen36_backend_20261101`.
Baseline artifacts: `/root/autodl-tmp/qwen36_rootcause_20261101`.
