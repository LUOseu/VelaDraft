# Serving result summary (one page)

Scope: `platform_extension`; `include_in_paper=false`.
Engine: vLLM 0.22.1 at `0decac0d96c42b49572498019f0a0e3600f50398`.
Target: Qwen2.5-VL-3B-Instruct, BF16, one RTX A6000, TP=1, greedy.
GPU: index 3, UUID `GPU-06d42c3b-89d2-20d5-8189-9b4564fec135`.

This page is the corrected replay of the original raw runs. The raw files and
their logs are unchanged. `comparison.json` and `summary.json` were regenerated
with the strict coverage and lifecycle gates; no new GPU run was performed.

| Completion level | Result |
| --- | --- |
| `AR_OFFLINE_PASS` | **true** (120/120 exact) |
| `AR_API_PASS` | **true** (recorded HTTP smoke checks) |
| `SD_EXECUTED` | **true** (1,038 rounds, 4,146 proposals) |
| `SD_CORRECTNESS_PASS` | **false** (81/120 exact, 67.5%) |
| `SERVING_LIFECYCLE_PASS` | **false / pending internal evidence** |

The N-Gram mode is real and executed, but it is not an AR-equivalent or
lossless replacement on this suite. The original lifecycle smoke is retained;
it is not promoted to a complete production-lifecycle validation.

## Correctness

| Metric | Result |
| --- | --- |
| AR/AR token-exact | **120/120** |
| N-Gram/N-Gram repeat consistency | **120/120** |
| N-Gram/AR token-exact | **81/120 (67.5%)** |
| N-Gram/AR finish-reason exact | **114/120 (95.0%)** |
| OCR exact | 9/24 |
| chart/doc exact | 24/24 |
| VQA exact | 3/24 |
| caption exact | 21/24 |
| diagnostic exact | 24/24 |

Strict coverage found no missing, duplicate, extra or identity-mismatched
request x repeat pairs: 40 fixtures x 3 repeats = 120 expected pairs for each
side. The failed IDs are retained in `comparison.json`; no repeat borrows
another repeat's reference.

Natural requests produced 2,562 proposed tokens over 642 rounds; 27/96 natural
requests produced zero proposals and remain in the result. Diagnostic requests
produced 1,584 proposals, 1,320 accepted tokens and 396 verification rounds,
which establishes `SD_EXECUTED=true`.

## Lifecycle

| Check | AR | N-Gram |
| --- | --- | --- |
| Timed windows present | 9/9 | 9/9 |
| Window records / errors | 315 / 0 | 315 / 0 |
| Boundary requests | 8/8 | 8/8 |
| Stream/non-stream text and finish match | yes | yes |
| Cancel probe close observed / gauge reclaimed | 9/9 | 9/9 |
| Cancel probe concurrent with healthy requests | no | no |
| Request-correlated engine abort evidence | unavailable | unavailable |
| Same-text/different-image output attribution | unverified | unverified |

The cancel probes ran only after each timed window drained. The captured
server logs contain no request-correlated abort/finish line. Therefore client
socket close and global active-gauge reclamation are reported as a narrow API
signal, not as a validated concurrent cancellation or full lifecycle pass.

The pair checks prove that the two requests were submitted with the same text
and different image hashes and completed without request errors. They do not
prove that each response belongs to the intended image, so no cross-session
correctness claim is made.

## Paired throughput (diagnostic only)

The ratio is AR total wall / N-Gram total wall. Values below 1 mean N-Gram was
slower. These legacy wall times remain useful for diagnosis, but are **not
claim-eligible** as a validated speedup because the original measure records do
not contain the newly required weight-content hash, effective template source,
engine-applied backend labels, or actual request-body budget.

| C | AR total | N-Gram total | N-Gram/AR | Claim eligible |
| --- | --- | --- | --- | --- |
| 1 | 107.23 s | 114.31 s | **0.938x** | no |
| 2 | 59.26 s | 67.65 s | **0.876x** | no |
| 4 | 38.31 s | 44.64 s | **0.858x** | no |

The corrected summary reports three session pairs per concurrency and explains
the rejection reasons in `protocol.reasons` and `pairing.reasons`. It never
uses the ratios to claim a speedup.

## Timing and resources

- Exact `tpot_s` is null in the corrected contract because SSE chunk
  timestamps are not per-token timestamps. `client_visible_token_tpot_proxy_s`
  and `client_chunk_gap_mean_s` are separate, explicitly named proxies; the
  latter is not ITL.
- Additive GPU-hours come only from the unique allocation ledger:
  0.183782 for AR HTTP and 0.107145 for N-Gram HTTP.
- Legacy per-request GPU-hour sums (0.104351 AR, 0.117666 N-Gram) overlap under
  concurrency and are retained for audit only; they are not additive.
- Peak memory is a process/session sampler value shared across requests, not a
  per-request exclusive maximum.
- `--max-new-tokens` now controls the actual request body and the protocol and
  record labels. The cancel probe is the explicit 512-token exception.

## Wire profile

Serving records expose profile `veladraft-serving-wire`, version `1.0`, under
the shared schema name `veladraft-platform-v1`. Request, window and global
counters keep separate scopes; unknown counters stay null with a reason. The
profile is documented by `vlm_dflash/serving/profile.py` and exercised by the
real-shaped fixture `tests/serving/fixtures/wire_profile_record.json`.

This profile is not directly interchangeable with the advisor validator. A
downstream compatibility layer must map fields explicitly; the current serving
records are not claimed to import without that layer.

## Evidence index

- Correctness replay: `evidence/comparison.json`.
- Lifecycle and speedup replay: `evidence/summary.json`; concise status:
  `evidence/summary.stdout.json`.
- Original raw runs: `evidence/ar_full.json`, `evidence/ngram_full.json`,
  `evidence/ar_measure.json`, `evidence/ngram_measure.json` and their logs.
- Capability manifest and wire profile: `configs/platform_extensions/serving/capabilities.json`.
- Frozen fixtures: `tests/serving/fixtures/manifest.json` (SHA-256 locked).

## Replay commands

```console
python -m pytest tests/serving -q -o addopts=''
# 97 passed

python scripts/run_serving_gate.py compare \
  --ar docs/platform_extensions/serving/evidence/ar_full.json \
  --ngram docs/platform_extensions/serving/evidence/ngram_full.json \
  --manifest tests/serving/fixtures/manifest.json \
  --out docs/platform_extensions/serving/evidence/comparison.json
# EXIT:2 is expected: SD_CORRECTNESS_PASS=false

python scripts/run_serving_gate.py summary \
  --correctness docs/platform_extensions/serving/evidence/comparison.json \
  --ar-measure docs/platform_extensions/serving/evidence/ar_measure.json \
  --ngram-measure docs/platform_extensions/serving/evidence/ngram_measure.json \
  --manifest tests/serving/fixtures/manifest.json \
  --out docs/platform_extensions/serving/evidence/summary.json
# EXIT:0; SERVING_LIFECYCLE_PASS=false; speedup.claim_eligible=false
```

`PYTHONPATH=. python scripts/check_docs.py` passed link, path and CLI-example
checks locally. The generated-doc stage could not run because the local Python
environment has no `torch`; CI remains responsible for the full docs check and
is not reported as locally verified.

## Unresolved and budget

- `SD_CORRECTNESS_PASS=false`; N-Gram cannot be called AR-equivalent here.
- `SERVING_LIFECYCLE_PASS=false`; concurrent request-correlated abort evidence
  and same-text/different-image output attribution are still missing.
- N-Gram is slower than AR in every aggregated concurrency bucket; the ratios
  are diagnostic because the legacy protocol record is incomplete.
- DFlash remains `experimental / blocked` (WP5 AL gap and position mismatch).
- `target_forward_calls` and per-round Proposal/Commit traces are not exposed by
  vLLM 0.22.1.
- No new GPU-hours were consumed by this rework. The original delivery ledger
  remains 1.155074 GPU-h of the 8 GPU-h allocation, leaving 6.844926 GPU-h.
