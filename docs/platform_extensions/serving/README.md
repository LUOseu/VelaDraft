# Serving extension implementation guide

Scope: `project_scope=platform_extension`, `include_in_paper=false`.
This directory contains the independent Agent 03 serving integration for
vLLM 0.22.1 and Qwen2.5-VL-3B.

This guide covers the implementation and CPU-side review path. Executed GPU
correctness and lifecycle evidence, completion levels and measured limitations
are delivered by the stacked evidence follow-up PR; this PR does not claim
those results.

## Entry points

```console
python -m pytest -q tests/serving
python scripts/run_serving_gate.py fixture-check \
  --manifest tests/serving/fixtures/manifest.json
python scripts/run_serving_gate.py audit \
  --model /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --out configs/platform_extensions/serving/capabilities.json \
  --source-commit 0decac0d96c42b49572498019f0a0e3600f50398
```

The remaining entry points are documented by `--help`:

- `serve`: launch a pinned config while holding the GPU UUID flock.
- `offline`: run AR or native N-Gram `LLM.chat` correctness.
- `compare`: compute AR/AR and SD/AR token-exactness.
- `measure`: run SSE, non-stream, cancel, C=1/2/4 and same-text/different-image
  checks against a live OpenAI-compatible server.
- `summary`: merge correctness and lifecycle documents into completion levels.

## Contract

`vlm_dflash/serving/contracts.py` implements the `veladraft-platform-v1`
RunRecord wire format. Every record carries target, draft, backend and protocol
identity; AR records explain the null draft identity in `missing_reasons`.
Unknown counters remain null with a reason. `round_trace` is not fabricated
because vLLM 0.22.1 exposes only aggregate speculative counters.

Session/Proposal/Commit types keep `based_on_prefix_version` explicit. A Commit
is valid only after a verified proposal; the live engine does not expose this
internal event stream as a public hook.

## Safety

The launcher refuses a GPU with compute processes or insufficient free memory,
then takes `/tmp/veladraft_gpu_locks/<UUID>.lock` with `flock`. It never stops
or cleans another process. GPU-hours in measurement documents start at the
captured allocation epoch, so model load, compilation, warmup and failed
windows are included.

Do not promote a run to `verified_greedy_exact` based on HTTP status or
`SD_EXECUTED` alone. The correctness gate and the lifecycle gate are separate.
