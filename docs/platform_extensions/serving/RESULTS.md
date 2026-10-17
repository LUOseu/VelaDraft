# Serving result summary (one page)

Scope: `platform_extension`; `include_in_paper=false`.
Engine: vLLM 0.22.1 at `0decac0d96c42b49572498019f0a0e3600f50398`.
Target: Qwen2.5-VL-3B-Instruct, BF16, one RTX A6000, TP=1, greedy.
GPU: index 3, UUID `GPU-06d42c3b-89d2-20d5-8189-9b4564fec135`.

| Completion level | Result |
| --- | --- |
| `AR_OFFLINE_PASS` | **true** (120/120 exact) |
| `AR_API_PASS` | **true** |
| `SD_EXECUTED` | **true** (1,038 rounds, 4,146 proposals) |
| `SD_CORRECTNESS_PASS` | **false** (81/120 exact, 67.5%) |
| `SERVING_LIFECYCLE_PASS` | **true** |

## Headline measurements

| Metric | AR | Native N-Gram |
| --- | --- | --- |
| Offline requests / errors | 120 / 0 | 120 / 0 |
| Offline output tokens | 7,965 | 8,148 |
| Offline load / allocation | 609.36 s / 810.32 s | 152.18 s / 289.81 s |
| Offline GPU-hours | 0.225088 | 0.080501 |
| Peak GPU memory | 41,927 MiB | 36,863 MiB |
| HTTP windows / records | 9 / 315 | 9 / 315 |
| HTTP errors | 0 | 0 |
| Cancel probes reclaimed | 9/9 | 9/9 |
| Boundary / pairs | 8/8, 9/9 | 8/8, 9/9 |
| Lifecycle GPU-hours | 0.183782 | 0.107145 |
| Peak HTTP GPU memory | 36,295 MiB | 36,291 MiB |

## Correctness

- AR/AR: **120/120** token-exact over three repeats.
- N-Gram/N-Gram: **120/120** token-exact over three repeats.
- N-Gram/AR: **81/120** token-exact; finish reason 114/120.
- Category exactness: OCR 9/24, chart/doc 24/24, VQA 3/24, caption 21/24,
  diagnostic 24/24.
- Natural N-Gram proposals: 2,562 over 642 rounds; 27/96 requests had zero
  proposals (all 24 caption repeats and 3 chart/doc repeats), retained in the
  report. Accepted natural tokens: 633.
- A synchronous 8-OCR control still differed on 5/8 samples. Scheduling alone
  does not explain the divergence.

## Paired throughput (same host and fixture set)

The ratio is AR total wall / N-Gram total wall. Values below 1 mean N-Gram was
slower.

| C | AR total | N-Gram total | N-Gram/AR |
| --- | --- | --- | --- |
| 1 | 107.23 s | 114.31 s | **0.938x** |
| 2 | 59.26 s | 67.65 s | **0.876x** |
| 4 | 38.31 s | 44.64 s | **0.858x** |

No fixed speedup is claimed. Eight of nine individual session pairs were slower
than AR.

## Evidence index

- Offline: `ar_full.json`, `ngram_full.json`, `comparison.json` and `.log` files.
- Scheduling control: `ar_sync8.json`, `ngram_sync8.json` and `.log` files.
- Lifecycle: `ar_measure.json`, `ngram_measure.json`, `summary.json` and both
  `*_lifecycle_server.log` files.
- Capabilities: `configs/platform_extensions/serving/capabilities.json`.
- Fixtures: `tests/serving/fixtures/manifest.json` (SHA-256 locked).

## Reproduce

The three entry points are `fixture-check`, `offline` + `compare`, and
`measure` + `summary` in `scripts/run_serving_gate.py`. Exact commands, exit
codes, limitations and the partner re-run procedure are in
`docs/platform_extensions/serving.md`.

## Unresolved

- `SD_CORRECTNESS_PASS=false`; N-Gram cannot be called AR-equivalent on this
  image-text suite.
- N-Gram is slower than AR at every tested concurrency.
- DFlash remains `experimental / blocked` (WP5 AL gap and position mismatch).
- `target_forward_calls` and per-round Proposal/Commit traces are not exposed by
  vLLM 0.22.1.
