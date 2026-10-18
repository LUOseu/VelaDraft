# VelaDraft serving extension - vLLM 0.22.1 + Qwen2.5-VL-3B

Status: **AR validated; native N-Gram executed but not AR-equivalent; full serving lifecycle pending**
Branch: `platform/serving-vllm` (baseline `7711eda504f4f28653a11cbb164b2ea673b4d682`)
Scope: `project_scope=platform_extension`, `include_in_paper=false`

This is the independent Agent 03 delivery. It records only work executed on
the pinned host and installed engine. It does not change the paper manifests,
the global CLI, `pyproject.toml`, the training environment or any existing
checkpoint.

## Completion levels

| Level | Result | Evidence |
| --- | --- | --- |
| `AR_API_PASS` | **true** | AR stream/non-stream, boundary and C=1/2/4 record checks |
| `AR_OFFLINE_PASS` | **true** | 120/120 AR/AR token-exact; zero request errors |
| `SD_EXECUTED` | **true** | 1,038 N-Gram draft rounds and 4,146 proposed tokens in the offline gate |
| `SD_CORRECTNESS_PASS` | **false** | only 81/120 N-Gram outputs match AR token-for-token (67.5%) |
| `SERVING_LIFECYCLE_PASS` | **false** | cancel probes were idle-baseline only; request-correlated abort and output attribution are unavailable |
| `ar_status` | `verified_greedy_exact` | three independent AR repeats and SD/AR comparison completed |
| `sd_status` | `experimental_numeric_divergence` | deterministic N-Gram repeat, but not the same transcript as AR |

The first-round external claim is **not** granted: the task requires both
`SD_CORRECTNESS_PASS` and `SERVING_LIFECYCLE_PASS`, and both are false. The
corrected replay treats the API smoke checks separately from the full
lifecycle. The N-Gram mode is real and observable; it is not a lossless
replacement for AR on this fixture set.

## What is delivered

- An optional vLLM launcher and OpenAI-compatible client for Qwen2.5-VL-3B.
- A real native **N-Gram** speculative-decoding mode (`method=ngram`,
  `num_speculative_tokens=4`, `prompt_lookup_min=2`, `prompt_lookup_max=2`).
- A frozen 32-request natural image-text suite (8 OCR, 8 chart/doc, 8 VQA,
  8 caption), 8 boundary requests and 8 repetition diagnostics.
- A lightweight `veladraft-platform-v1` RunRecord serializer/validator and
  Session/Proposal/Commit contract, plus the explicit
  `veladraft-serving-wire` downstream profile.
- CPU/mock tests and three reproducible entry points: capability audit,
  AR/SD correctness gate, and stream/cancel/concurrency measurement.

Not claimed:

- No fixed speedup. The measured N-Gram/AR paired total is below 1 at every
  tested concurrency (see below).
- No lossless claim. `SD_CORRECTNESS_PASS=false` is the binding result.
- No complete lifecycle or concurrent-cancellation claim. The recorded cancel
  probes ran after each window drained and have no request-correlated engine
  abort evidence. Same-text/different-image output attribution is unverified.
- N-Gram is not evidence that this project's trained DFlash/DFly drafter is
  served natively. DFlash remains experimental/blocked.
- No claim about arbitrary sampling, TP>1, multi-node, Kubernetes or another
  engine.

## Repository layout

| Path | Purpose |
| --- | --- |
| `vlm_dflash/serving/contracts.py` | `veladraft-platform-v1` RunRecord validation, Session/Proposal/Commit types and capability validation |
| `vlm_dflash/serving/records.py` | target/draft/backend identity, protocol hash and resource accounting |
| `vlm_dflash/serving/fixtures.py` | frozen manifest loading, image data URIs and OpenAI request bodies |
| `vlm_dflash/serving/openai_api.py` | dependency-free SSE/non-SSE client and token-id collection |
| `vlm_dflash/serving/metrics.py` | Prometheus parsing, active gauges and speculative-counter diffs |
| `vlm_dflash/serving/launch.py` | pinned server command, busy-GPU refusal and per-UUID flock |
| `vlm_dflash/serving/offline.py` | AR/N-Gram `LLM.chat` correctness runs |
| `vlm_dflash/serving/compare.py` | AR/AR and SD/AR token comparison and completion levels |
| `vlm_dflash/serving/measure.py` | SSE, cancel, C=1/2/4, non-stream and same-text/different-image probes |
| `vlm_dflash/serving/audit.py` | installed-source capability audit and maturity labels |
| `vlm_dflash/serving/profile.py` | downstream wire profile, counter scope and normalizer |
| `vlm_dflash/serving/summary.py` | strict common-protocol, pairing and lifecycle aggregation |
| `scripts/run_serving_gate.py` | `serve`, `audit`, `fixture-check`, `offline`, `compare`, `measure`, `summary` |
| `configs/platform_extensions/serving/` | launch configs, environment lock and generated capabilities |
| `tests/serving/` | contract, fixture, client, compare, launch, metrics, measure, profile, summary and gate tests |

The contract keeps `project_scope=platform_extension` and
`include_in_paper=false` on every record. A Session has an isolated request
prefix; a Proposal carries `based_on_prefix_version`; only a validated Commit
may advance the committed output/cache. vLLM 0.22.1 does not expose a public
per-round propose/commit hook, so `round_trace` remains `unsupported` and no
internal event stream is fabricated. Unknown counters are null with a reason,
never silently converted to zero.

## Wire profile

The downstream profile is `veladraft-serving-wire` version `1.0`, wrapped in
schema `veladraft-platform-v1`. Request, window and global counters keep their
recorded scope: request counters require proven request attribution, window
counters are not request counters, and global counters are never allocated to
individual requests. Unknown values remain null with a reason.

The profile source is `vlm_dflash/serving/profile.py`. Its real-shaped CPU
fixture is `tests/serving/fixtures/wire_profile_record.json`; the
`describe_profile()` output is embedded in
`configs/platform_extensions/serving/capabilities.json`. This profile is not
field-compatible with the advisor validator despite the shared schema name. A
downstream adapter must map it explicitly; the current records are not claimed
to import without that layer.

## Locked environment

See `configs/platform_extensions/serving/environment.lock.json` and
`configs/platform_extensions/serving/capabilities.json`.

| Item | Value |
| --- | --- |
| Host | `a6000xuyang`, 4x RTX A6000 48 GB |
| GPU used | index 3, UUID `GPU-06d42c3b-89d2-20d5-8189-9b4564fec135` |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| Python | 3.12.13 (`/home/lyh/envs/pace`, venv over `c1vllm`) |
| vLLM | 0.22.1 |
| vLLM source commit | `0decac0d96c42b49572498019f0a0e3600f50398` (upstream `v0.22.1` tag) |
| Torch | 2.11.0+cu130 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| Target | `/home/lyh/models/Qwen2.5-VL-3B-Instruct` |
| Model config SHA-256 | `7ed3eed5be6924cc800e8a5e53fc405c1aab1aaf36bad65c33403b36c56827f5` |
| Tokenizer SHA-256 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| Processor SHA-256 | `f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0` |
| Fixture manifest SHA-256 | `0830ede336fbd5da7d61a9b6d74ee2460e6c3012beff379053dff64b792a6ca1` |

Both configs use BF16, TP=1, greedy decoding, `max_model_len=8192`, prefix
caching disabled, `gpu_memory_utilization=0.75`, `max_num_seqs=8`, pixel bounds
`200704..1003520`, and `async_scheduling=false`.

## Capability audit

Generated from the installed source tree and the final evidence:

| Capability | Status | Maturity | Key limit |
| --- | --- | --- | --- |
| AR image | supported | **validated** | one A6000, TP=1, BF16, greedy, prefix caching off |
| N-Gram image | supported | **gpu_smoke** | executed, but correctness gate failed; CPU token-id suffixes only |
| DFlash image | experimental | **blocked** | WP5 AL gap and position mismatch unresolved |
| Stream | supported | **gpu_smoke** | SSE timing is a client-side boundary; full lifecycle is pending |
| Cancel | supported | **gpu_smoke** | idle-baseline socket close and global gauge reclaim only; no request-correlated abort log |
| Round trace | unsupported | implemented | only aggregate draft/accept counters are public |
| `return_token_ids` | supported | **gpu_smoke** | vLLM extension field; standard clients ignore it |
| Metrics | supported | **gpu_smoke** | global counters, no request tag |
| Wire profile | supported | **cpu_tested** | `veladraft-serving-wire` 1.0; explicit downstream mapping required |

Installed-source evidence is recorded with exact symbol names and line numbers
in `configs/platform_extensions/serving/capabilities.json`:

- `vllm/config/speculative.py`: `SpeculativeConfig` and native `"ngram"`.
- `vllm/v1/spec_decode/ngram_proposer.py`: `NgramProposer` and suffix lookup;
  the draft path has no model forward.
- `vllm/v1/worker/gpu_model_runner.py`: engine construction of `NgramProposer`.
- `vllm/model_executor/models/qwen2_5_vl.py`:
  `Qwen2_5_VLForConditionalGeneration`, `get_mrope_input_positions`,
  `recompute_mrope_positions`.
- `vllm/entrypoints/openai/chat_completion/protocol.py`: `return_token_ids`.
- `vllm/v1/spec_decode/metrics.py`: aggregate draft, proposed and accepted
  token counters.

The startup log also confirms `Async scheduling is disabled`, the FLASH_ATTN
backend, FlashAttention 2, the fast `Qwen2VLImageProcessor` and the native
`speculative_config=SpeculativeConfig(method='ngram', model=None,
num_spec_tokens=4)`.

## Fixtures

`tests/serving/fixtures/manifest.json` is frozen with per-image and per-prompt
SHA-256 hashes. `fixture-check` exited 0 with `natural=32`, `boundary=8`,
`diagnostic=8`.

| Category | Count | Source |
| --- | --- | --- |
| OCR | 8 | TextVQA regen set, rows 0-7 |
| chart/doc | 8 | ScienceQA test rows 9,15,21,67,109,128,180,214 |
| VQA | 8 | LLaVA/COCO val rows 0-7 |
| caption | 8 | LLaVA/COCO val rows 8-15, fixed caption prompt |
| diagnostic | 8 | repetition prompts with TextVQA images; excluded from natural speedup stats |
| boundary | 8 | two-turn, different resolution, empty text, no image, invalid media, long prompt, bad structure, two images |

The 32 natural samples are 3.9 MB total; no sample is downloaded at run time.
The manifest records source row ids and content hashes for partner re-runs.

## Protocols

- AR: no `speculative_config`.
- SD: native N-Gram, `num_speculative_tokens=4`, fixed lookup `2/2`.
- Offline correctness: `LLM.chat`, one request at a time, 32 natural plus 8
  diagnostic fixtures repeated three times; token IDs come from vLLM, not from
  re-tokenizing text.
- HTTP measurement: `/v1/chat/completions` with `stream=true`,
  `return_token_ids=true`, `stream_options.include_usage=true`; TTFT is the
  first non-empty content/token chunk and includes queue, media transfer and
  preprocessing.
- A stream succeeds only after a success `finish_reason` and `[DONE]`. Partial
  EOF, SSE error events, malformed JSON, duplicate finish and missing finish are
  protocol failures, not successful completions. Intentional cancellation is a
  separate terminal state.
- Exact `tpot_s` is null because the SSE API exposes chunk timestamps rather
  than per-token timestamps. `client_visible_token_tpot_proxy_s` is a clearly
  labelled visible-token proxy, and `client_chunk_gap_mean_s` is a separate
  chunk-gap mean that is never called ITL. The legacy pre-rework `tpot_s`
  values are not reused as exact measurements.
- Spec counters are attributed per request only at C=1. At C>1 they are
  reported once per session/window because vLLM's counters have no request tag.
- `--max-new-tokens` is copied into the actual request body and recorded with
  `request_control.source=actual_request_body`. The dedicated cancel probe is
  the explicit exception: it sets `max_tokens=512` and `ignore_eos=true`, and
  its counters are not mixed into natural latency. The legacy raw records did
  not contain `request_control`; the corrected summary therefore refuses to
  promote their speed ratios to a claim.

## Correctness results

### AR/AR consistency

| Metric | Result |
| --- | --- |
| Requests | 120 (40 fixtures x 3 repeats) |
| Errors | 0 |
| Output tokens | 7,965 |
| AR/AR token-exact | **120/120** |
| Load / allocation wall | 609.36 s / 810.32 s |
| Peak GPU memory | 41,927 MiB |
| Allocation GPU-hours | 0.225088 |

Evidence: `evidence/ar_full.json`, `evidence/ar_full.log`.

### N-Gram execution and repeat consistency

| Metric | Result |
| --- | --- |
| Requests | 120 |
| Errors | 0 |
| Output tokens | 8,148 |
| Draft/verify rounds | 1,038 |
| Proposed / accepted tokens | 4,146 / 1,953 |
| Acceptance length | 2.88 tokens per round |
| N-Gram/N-Gram token-exact | **120/120** |
| Load / allocation wall | 152.18 s / 289.81 s |
| Peak GPU memory | 36,863 MiB |
| Allocation GPU-hours | 0.080501 |

Evidence: `evidence/ngram_full.json`, `evidence/ngram_full.log`.

The same input is deterministic in N-Gram. The failure is therefore not a
repeat-time sampling or engine-jitter artifact in the N-Gram path.

### AR versus N-Gram

| Metric | Result |
| --- | --- |
| Token-exact pairs | **81/120 (67.5%)** |
| Finish-reason exact | 114/120 (95.0%) |
| OCR | 9/24 |
| chart/doc | 24/24 |
| VQA | 3/24 |
| caption | 21/24 |
| diagnostic | 24/24 |

Diagnostic requests entered verification and produced 396 draft rounds,
1,584 proposed tokens and 1,320 accepted tokens, so `SD_EXECUTED=true`.

Natural requests produced 2,562 proposed tokens over 642 rounds. Importantly,
**27/96** natural requests had zero proposals (all 24 caption repeats and 3
chart/doc repeats); those cases are retained, not filtered out. Natural
acceptance was 633/2,562 proposed tokens. Representative divergences include
AR `mentee` versus N-Gram `student`, and AR `skate park` versus N-Gram
`concrete surface`; OCR differences often appear in the middle or at the tail
of a long response.

Evidence: `evidence/comparison.json`.

### Scheduling control

vLLM 0.22.1 does not support asynchronous scheduling together with N-Gram.
To test whether scheduling alone explained the mismatch, one 8-OCR control
ran both AR and N-Gram with `--no-async-scheduling` (synchronous for both).
Both runs exited 0, and **5/8** OCR outputs still differed. Scheduling mode is
therefore not a sufficient explanation. `evidence/ar_sync8.json` and
`evidence/ngram_sync8.json` retain both runs.

The result is reported as `experimental failure`, not renamed as
`verified_greedy_exact`. Batch invariance is not claimed.

## HTTP lifecycle results

Each mode used three sessions at C=1,2,4. Each timed window contains 32 natural
requests, one diagnostic normal-end request and one same-text/different-image
pair (35 records). A dedicated cancel probe runs after the timed window drains.
Every mode also exercises 8 boundary requests, one stream/non-stream probe and
4 warmup requests.

| Metric | AR | N-Gram |
| --- | --- | --- |
| Windows | 9/9 | 9/9 |
| Window records | 315 | 315 |
| Record errors | 0 | 0 |
| Cancel probes observed / reclaimed | 9/9 | 9/9 |
| Cancel reclaim range | 0.0101-0.2783 s | 0.0115-0.2761 s |
| Cancel probe concurrent with healthy requests | no | no |
| Request-correlated engine abort evidence | unavailable | unavailable |
| Boundary checks | 8/8 | 8/8 |
| Stream/non-stream match | yes | yes |
| Same-text/different-image submitted pairs | 9/9 | 9/9 |
| Same-text/different-image output attribution | unverified | unverified |
| Measurement wall | 222.30 s | 251.16 s |
| Allocation wall (load + warmup + measure) | 661.61 s | 385.72 s |
| Peak GPU memory | 36,295 MiB | 36,291 MiB |
| Allocation GPU-hours | 0.183782 | 0.107145 |

`AR_API_PASS=true` because the recorded windows, records, boundaries and
non-stream probe pass. `SERVING_LIFECYCLE_PASS=false` because a full lifecycle
also requires a concurrent healthy request, request-correlated engine abort/end
evidence and verified output ownership; those are absent. TTFT, E2E and the
non-exact timing proxies over the natural records are in
`evidence/summary.json`. At C=1 the AR TTFT p50/p95 is 0.1108/0.2255 s and the
N-Gram TTFT p50/p95 is 0.1046/0.2071 s; these are client-side boundaries, not
engine token timestamps. Exact TPOT is null.

Cancellation evidence is the client close time, the `vllm:num_requests_running`
and `..._waiting` gauges sampled during and after the request, and the
reclamation interval. The vLLM INFO logs did not echo the request ids, so
server-log line matching was 0/9 in both modes; the logs are still included for
audit. Each probe uses an idle baseline, a 512-token `ignore_eos` request and a
10 s reclamation window. It is therefore a narrow socket-close/global-gauge
smoke, not proof that a request cancelled concurrently with other traffic
stopped decoding.

The same-text/different-image checks verify submitted prompt and image hashes,
distinct request IDs and absence of request errors. The corrected gate does not
interpret those inputs as output attribution, and it makes no no-cross-session
claim.

Evidence: `evidence/ar_measure.json`, `evidence/ngram_measure.json`,
`evidence/summary.json` and both `*_lifecycle_server.log` files.

## Measured throughput (diagnostic wall times)

These are paired wall-clock measurements on the same host, firmware, model,
scheduler and fixture set. The corrected summary reports them for diagnosis but
sets `speedup.claim_eligible=false`: the legacy raw records lack the weight
content hash, effective template source, engine-applied backend labels and
actual request-body budget required by the common-protocol gate.

| C | AR total wall | N-Gram total wall | N-Gram/AR ratio (diagnostic) | Raw session ratios |
| --- | --- | --- | --- | --- |
| 1 | 107.23 s | 114.31 s | **0.938x** | 0.827, 0.894, 1.146 |
| 2 | 59.26 s | 67.65 s | **0.876x** | 0.866, 0.884, 0.877 |
| 4 | 38.31 s | 44.64 s | **0.858x** | 0.824, 0.813, 0.941 |

Only one of the nine session pairs is faster than AR (C=1 session 3). The
aggregate result is below 1 at every concurrency. N-Gram was not presented as a
speedup: its proposal lookup is CPU token-id suffix matching, its scheduling
path differs, this delivery did not tune the engine for speed, and the legacy
protocol record does not meet the claim gate.

## Commands and exit codes

CPU tests (local rework checkout, Python 3.9.6, no GPU):

```console
python -m pytest tests/serving -q -o addopts=''
# 97 passed in 5.13s
```

Fixture check:

```console
python scripts/run_serving_gate.py fixture-check --manifest tests/serving/fixtures/manifest.json
# EXIT:0; natural=32 boundary=8 diagnostic=8
```

Offline correctness (one process per mode, GPU UUID flock, `CUDA_VISIBLE_DEVICES=3`):

```console
python scripts/run_serving_gate.py offline --mode ar --repeats 3 --gpu-index 3 \
  --model /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --manifest tests/serving/fixtures/manifest.json \
  --code-commit 0decac0d96c42b49572498019f0a0e3600f50398 \
  --out /tmp/veladraft_serving/evidence/ar_full.json
# EXIT:0

python scripts/run_serving_gate.py offline --mode ngram --repeats 3 --gpu-index 3 \
  --model /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --manifest tests/serving/fixtures/manifest.json \
  --code-commit 0decac0d96c42b49572498019f0a0e3600f50398 \
  --out /tmp/veladraft_serving/evidence/ngram_full.json
# EXIT:0

python scripts/run_serving_gate.py compare \
  --ar /tmp/veladraft_serving/evidence/ar_full.json \
  --ngram /tmp/veladraft_serving/evidence/ngram_full.json \
  --manifest tests/serving/fixtures/manifest.json \
  --out /tmp/veladraft_serving/evidence/comparison.json
# EXIT:2 (expected); SD_EXECUTED=true; SD_CORRECTNESS_PASS=false
```

HTTP lifecycle (server already healthy; no GPU visibility is needed in the
measurement client):

```console
date +%s > /tmp/veladraft_serving/evidence/ar_allocation_start_epoch
CUDA_VISIBLE_DEVICES=3 VLLM_SOURCE_COMMIT=0decac0d96c42b49572498019f0a0e3600f50398 \
  python scripts/run_serving_gate.py serve \
  --config configs/platform_extensions/serving/ar.json --gpu-index 3 --port 8100 \
  > /tmp/veladraft_serving/logs/ar_lifecycle_server.log 2>&1 &

DEVICES='[{"index":3,"name":"NVIDIA RTX A6000","uuid":"GPU-06d42c3b-89d2-20d5-8189-9b4564fec135","compute_capability":"8.6"}]'

CUDA_VISIBLE_DEVICES="" python scripts/run_serving_gate.py measure \
  --mode ar --base-url http://127.0.0.1:8100 \
  --model Qwen2.5-VL-3B-Instruct \
  --model-path /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --manifest tests/serving/fixtures/manifest.json \
  --out /tmp/veladraft_serving/evidence/ar_measure.json \
  --run-id ar-lifecycle --code-commit 0decac0d96c42b49572498019f0a0e3600f50398 \
  --devices-json "$DEVICES" --gpu-index 3 \
  --server-log /tmp/veladraft_serving/logs/ar_lifecycle_server.log \
  --allocation-start-epoch "$(cat /tmp/veladraft_serving/evidence/ar_allocation_start_epoch)" \
  --concurrency 1,2,4 --sessions 3 --no-async-scheduling
# EXIT:0

date +%s > /tmp/veladraft_serving/evidence/ngram_allocation_start_epoch
CUDA_VISIBLE_DEVICES=3 VLLM_SOURCE_COMMIT=0decac0d96c42b49572498019f0a0e3600f50398 \
  python scripts/run_serving_gate.py serve \
  --config configs/platform_extensions/serving/ngram.json --gpu-index 3 --port 8101 \
  > /tmp/veladraft_serving/logs/ngram_lifecycle_server.log 2>&1 &

CUDA_VISIBLE_DEVICES="" python scripts/run_serving_gate.py measure \
  --mode ngram --base-url http://127.0.0.1:8101 \
  --model Qwen2.5-VL-3B-Instruct \
  --model-path /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --manifest tests/serving/fixtures/manifest.json \
  --out /tmp/veladraft_serving/evidence/ngram_measure.json \
  --run-id ngram-lifecycle --code-commit 0decac0d96c42b49572498019f0a0e3600f50398 \
  --devices-json "$DEVICES" --gpu-index 3 \
  --server-log /tmp/veladraft_serving/logs/ngram_lifecycle_server.log \
  --allocation-start-epoch "$(cat /tmp/veladraft_serving/evidence/ngram_allocation_start_epoch)" \
  --concurrency 1,2,4 --sessions 3 --no-async-scheduling
# EXIT:0
```

Merge and audit:

```console
python scripts/run_serving_gate.py summary \
  --correctness /tmp/veladraft_serving/evidence/comparison.json \
  --ar-measure /tmp/veladraft_serving/evidence/ar_measure.json \
  --ngram-measure /tmp/veladraft_serving/evidence/ngram_measure.json \
  --manifest tests/serving/fixtures/manifest.json \
  --out /tmp/veladraft_serving/evidence/summary.json
# EXIT:0; SERVING_LIFECYCLE_PASS=false; speedup.claim_eligible=false

CUDA_VISIBLE_DEVICES="" python scripts/run_serving_gate.py audit \
  --model /home/lyh/models/Qwen2.5-VL-3B-Instruct \
  --out configs/platform_extensions/serving/capabilities.json \
  --source-commit 0decac0d96c42b49572498019f0a0e3600f50398 \
  --gpu-smoke-ar /tmp/veladraft_serving/evidence/ar_full.json \
  --gpu-smoke-ngram /tmp/veladraft_serving/evidence/ngram_full.json \
  --correctness /tmp/veladraft_serving/evidence/comparison.json \
  --lifecycle /tmp/veladraft_serving/evidence/summary.json \
  --dflash-wp5 docs/plan/WP5_REPORT.md
# EXIT:0; AR/image validated; N-Gram/stream/cancel gpu_smoke; wire profile cpu_tested; DFlash/blocked
```

## Rework verification (2026-10-18)

| Review item | Change | CPU regression |
| --- | --- | --- |
| S1 | Expected request x repeat coverage comes from the manifest; errors, invalid output and zero execution block PASS | Diagnostic-only, failed-SD, zero-proposal, duplicate, missing-repeat and AR-error cases reject; complete exact case passes |
| S2 | Explicit content/finish/done/error/cancel states; partial EOF, SSE error, malformed JSON and duplicate finish fail | Parser and record/window/lifecycle regressions pass |
| S3 | Common protocol, per-request pairing, weight/template/backend identity and actual budget checks | Field mutations, duplicate/missing windows, changed weights and unverified backend labels reject; allowed AR/SD differences pass |
| S4 | Lifecycle claim reduced to the observed socket-close/global-gauge smoke | `SERVING_LIFECYCLE_PASS=false`; cancel and cross-session claims are explicitly unvalidated |
| S5 | Exact TPOT is null; visibility and chunk-gap proxies are separate; resource exposure is non-additive; request budget comes from the sent body | TPOT, resource and CLI-budget regressions pass |
| S6 | README path fixed; wire profile and real-shaped CPU fixture published | Link/path/CLI docs checks pass in the local environment |

Error-PASS cases from the review now have direct regressions: diagnostic-only SD
coverage fails, identical SD tokens with generation errors fail, identical SD
tokens with zero proposals fail, partial SSE EOF is truncated, and a mixed
protocol cannot produce a claim-eligible speedup. The corrected raw replay is
in [RESULTS.md](serving/RESULTS.md) and `evidence/summary.json`.

## Failures observed during development

These failures are part of the audit trail, not silent retries:

- The first AR lifecycle attempt exited 1 before measurement because the AR
  window record omitted `missing_reasons['draft_identity']`. The validator was
  correct; the record mapping was fixed and a regression test was added.
- The first N-Gram lifecycle attempt exited 1 because the `measure` parser did
  not expose `--num-speculative-tokens`, `--prompt-lookup-min` or
  `--prompt-lookup-max`. The parser and a test were fixed.
- One N-Gram C=4 cancel probe had already been reclaimed before the first
  post-close gauge scrape. The probe was made stronger with an idle baseline,
  `ignore_eos`, in-flight sampling and a dedicated reclamation window; both
  modes were then rerun end to end.

The failed allocations are included in the GPU-hour accounting below and are
not presented as successful measurements.

## GPU budget

| Work item | GPU-hours |
| --- | --- |
| AR offline full | 0.225088 |
| N-Gram offline full | 0.080501 |
| N-Gram limited smoke | 0.042656 |
| AR synchronous 8-OCR control | 0.135973 |
| N-Gram synchronous 8-OCR control | 0.055412 |
| AR lifecycle attempt A (superseded after probe fix) | 0.161742 |
| N-Gram lifecycle attempt A (superseded after probe fix) | 0.162775 |
| AR lifecycle final | 0.183782 |
| N-Gram lifecycle final | 0.107145 |
| **Total consumed of 8 GPU-h** | **1.155074 (remaining 6.844926)** |

GPU-hours include model load, compilation, warmup, failed attempts and the
measurement window on the single assigned GPU. No other GPU was used by this
task. Jobs holding the GPU 3 flock were waited out, not terminated. The
2026-10-18 rework used **0.000000 new GPU-hours**; all corrected summaries are
offline replays of the original raw documents.

## DFlash status

`DFLASH_BLOCKED` for this delivery. WP5 established a native load path but an
unresolved ~17% AL gap (1.4059 in vLLM vs 1.6866/1.6924 in-repo), a
1-D-RoPE-vs-MRoPE draft-position mismatch, and same-engine non-determinism
(12/32 token-exact reruns). This task does not rename that result. The N-Gram
delivery is independent of the DFlash position path.

## Partner re-run on their own 32 samples

1. Copy `tests/serving/fixtures/` and replace the images/prompts. Keep one
   manifest entry per sample; the validator currently requires 8 entries per
   natural category, 8 boundary and 8 diagnostic entries. If the partner wants
   a different count, change the constants and state the new counts explicitly.
2. Recompute every `image_sha256`, `prompt_sha256` and `prompt_token_count`.
   Run `fixture-check` until it exits 0.
3. Point `configs/platform_extensions/serving/ar.json` and `ngram.json` at the
   partner model paths and GPU index. Keep BF16, TP=1, greedy, prefix caching
   off and the same pixel bounds unless the comparison protocol is redesigned.
4. Start `serve --config .../ar.json --gpu-index <n>` and `serve --config
   .../ngram.json --gpu-index <n>` on the same GPU, one after the other. The
   launcher refuses a busy GPU and holds the per-UUID flock. Never kill another
   job to obtain the GPU.
5. Run `offline --mode ar|ngram --repeats 3`, then `compare`. A result is not a
   pass just because both processes exited 0: inspect `SD_EXECUTED` and
   `SD_CORRECTNESS_PASS` separately.
6. Run `measure` for C=1,2,4 with `--sessions 3`, then `summary`. Use a fresh
   `--allocation-start-epoch` captured before server launch so load and warmup
   are included in GPU-hours.
7. Keep `project_scope=platform_extension` and `include_in_paper=false`, and
   replace `/home/lyh/...` paths and the vLLM pin with the partner's own.

## Known limitations

- `target_forward_calls` is null: vLLM 0.22.1 exposes aggregate draft/accept
  counters, not target-forward calls. It is never replaced by 0.
- N-Gram proposals are CPU token-id suffix matches; they do not observe image
  features directly. The target's MRoPE/image processing is unchanged.
- N-Gram forces synchronous scheduling in this vLLM version. That is a pinned
  engine behavior, not a guarantee of batch invariance.
- The image processor is the fast `Qwen2VLImageProcessor`; the slow/fast choice
  is pinned behavior, not silently normalized.
- SSE chunk timing is not token-level ITL. Exact `tpot_s` is null; the visible
  token proxy and chunk-gap mean are reported under separate names.
- The server INFO logs do not contain request ids. The recorded cancel probes
  establish client close and global active-gauge reclamation only; they do not
  establish request-correlated abort, concurrent cancellation or stopped
  decode.
- Same-text/different-image checks verify submitted inputs and error-free
  completion; they do not verify output ownership, so no cross-session claim is
  made.
- Backend attention/graph labels are accepted as applied evidence only when the
  server startup log echoes the configured values. CLI labels alone remain
  `cli_declared_unverified` and make speed ratios claim-ineligible.
