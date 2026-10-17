# VelaDraft workload advisory report

> **SYNTHETIC FIXTURE** — these numbers are generated CPU fixtures, not real accelerator measurements.
> They must not be copied into a production or paper results table.

- schema: `veladraft-platform-v1`
- project scope: `platform_extension`; include_in_paper: `false`
- workload: `mini-enterprise-v1`
- target: `Qwen/Qwen2.5-VL-3B-Instruct` (bfloat16)
- measurement scope: `local_client`; concurrency=1
- groups: 24; sessions: 3; tasks: 3
- protocol scope hash(es): `67db79b7bbf99fc614cba05d707b6fa25aedb4c71f497999cfb9b13c44856ba3`
- bootstrap: seed=7, n_boot=2000, paired by group

## Recommendation

- action: **insufficient_evidence**
- method: `None`
- threshold: paired speed-ratio CI lower > 1.05
- coverage: {"buckets_insufficient": {"dflash-exact:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dflash-exact:target_V": ["1024-2048", "256-512", "512-1024"], "dflash-exact:task": ["chart_qa", "general_vqa", "invoice_fields"], "dfly-fast-wrong:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dfly-fast-wrong:target_V": ["1024-2048", "256-512", "512-1024"], "dfly-fast-wrong:task": ["chart_qa", "general_vqa", "invoice_fields"]}, "min_groups": 20, "n_groups": 24, "n_sessions": 3, "n_tasks": 3, "task_group_counts": {"chart_qa": 8, "general_vqa": 8, "invoice_fields": 8}}
- Synthetic fixture evidence: no real recommendation is made.

Excluded candidates:
- `dflash-exact` — synthetic_fixture: synthetic fixtures never enter a real recommendation.
- `dfly-fast-wrong` — correctness_failed: 24 record(s) failed.; quality_failed: task quality is below the frozen threshold.; synthetic_fixture: synthetic fixtures never enter a real recommendation.

## Candidates

| candidate | records | groups | correctness | quality | speed ratio [95% CI] | memory fit | latency mean AR -> candidate (s) |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `dflash-exact` | 72 | 24 | verified_greedy_exact=72 (failures 0/72) | pass | 1.430 [1.421, 1.438] | True | 0.999 -> 0.699 |
| `dfly-fast-wrong` | 72 | 24 | failed=24, verified_greedy_exact=48 (failures 24/72) | fail | 2.084 [2.070, 2.095] | True | 0.999 -> 0.480 |

## Latency and resources

| method | e2e mean | p50 | p95 | n | serial output tok/s | TTFT | ITL | peak allocated bytes | GPU-h (max record) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `ar` | 0.999 | 0.999 | 1.149 | 72 | 60.867 | null | null_no_token_visibility | {"synthetic-GPU-0": 20000000000} | 0.002000 |
| `dflash-exact` | 0.699 | 0.701 | 0.788 | 72 | 87.016 | null | null_no_token_visibility | {"synthetic-GPU-0": 20000000000} | 0.002000 |
| `dfly-fast-wrong` | 0.480 | 0.486 | 0.541 | 72 | 126.837 | null | null_no_token_visibility | {"synthetic-GPU-0": 20000000000} | 0.002000 |

TTFT/ITL stay null when the backend does not expose token visibility; no proxy is substituted.
The throughput column is `serial_output_tokens_per_second`, not a concurrent serving throughput.

## Bucket evidence

Frozen tuning boundaries; buckets with fewer than the required independent groups are `insufficient_evidence`.

### `dflash-exact` by task

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| chart_qa | 8 | insufficient_evidence | point=null; CI=null |
| general_vqa | 8 | insufficient_evidence | point=null; CI=null |
| invoice_fields | 8 | insufficient_evidence | point=null; CI=null |

### `dflash-exact` by target_V

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| 1024-2048 | 4 | insufficient_evidence | point=null; CI=null |
| 256-512 | 6 | insufficient_evidence | point=null; CI=null |
| 512-1024 | 14 | insufficient_evidence | point=null; CI=null |

### `dflash-exact` by ar_R

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| 16-32 | 4 | insufficient_evidence | point=null; CI=null |
| 32-64 | 9 | insufficient_evidence | point=null; CI=null |
| 64-96 | 9 | insufficient_evidence | point=null; CI=null |
| 96-128 | 2 | insufficient_evidence | point=null; CI=null |

### `dfly-fast-wrong` by task

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| chart_qa | 8 | insufficient_evidence | point=null; CI=null |
| general_vqa | 8 | insufficient_evidence | point=null; CI=null |
| invoice_fields | 8 | insufficient_evidence | point=null; CI=null |

### `dfly-fast-wrong` by target_V

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| 1024-2048 | 4 | insufficient_evidence | point=null; CI=null |
| 256-512 | 6 | insufficient_evidence | point=null; CI=null |
| 512-1024 | 14 | insufficient_evidence | point=null; CI=null |

### `dfly-fast-wrong` by ar_R

| bucket | n groups | status | ratio [95% CI] |
| --- | ---: | --- | --- |
| 16-32 | 4 | insufficient_evidence | point=null; CI=null |
| 32-64 | 9 | insufficient_evidence | point=null; CI=null |
| 64-96 | 9 | insufficient_evidence | point=null; CI=null |
| 96-128 | 2 | insufficient_evidence | point=null; CI=null |

## Correctness and quality

- Correctness statuses follow the protocol enum; `verified_greedy_exact` requires per-request token comparison against a protocol-matched AR reference.
- A 0/N failure count retains N; it is not evidence that the failure probability is strictly zero.
- Task quality is reported separately from token exactness and run errors.
- frozen scorers: `{"chart_qa": {"name": "normalized_exact_match", "threshold": 1.0, "version": "1"}, "general_vqa": {"name": "normalized_exact_match", "threshold": 1.0, "version": "1"}, "invoice_fields": {"name": "field_f1", "threshold": 0.8, "version": "1"}}`

## Gaps and next steps

- Replace synthetic fixture timings with a real HF backend run on an allocated GPU (4 GPU-h first-round cap).
- Buckets still below the minimum group count: `{"dflash-exact:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dflash-exact:target_V": ["1024-2048", "256-512", "512-1024"], "dflash-exact:task": ["chart_qa", "general_vqa", "invoice_fields"], "dfly-fast-wrong:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dfly-fast-wrong:target_V": ["1024-2048", "256-512", "512-1024"], "dfly-fast-wrong:task": ["chart_qa", "general_vqa", "invoice_fields"]}`.
- No cross-hardware, cross-engine or cross-customer generalization is claimed.

## Replay

```bash
python scripts/workload_advisor.py recommend --records records.jsonl --out-dir out --bootstrap-seed 7
```
