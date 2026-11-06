# VelaDraft workload advisory report (share package)

- schema: `veladraft-platform-v1`
- project scope: `platform_extension`; include_in_paper: `false`
- This package contains public IDs, hashes, aggregate scores and reason codes only.
- Prompts, media, answers, predictions, raw error text and local paths stay private.

## Recommendation

- action: **insufficient_evidence**
- method: `None`
- threshold: `1.05`
- assignment verified: `True`
- input integrity: `ok`
- coverage: `{"buckets_insufficient": {"dflash-exact:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dflash-exact:target_V": ["1024-2048", "256-512", "512-1024"], "dflash-exact:task": ["chart_qa", "general_vqa", "invoice_fields"], "dfly-fast-wrong:ar_R": ["16-32", "32-64", "64-96", "96-128"], "dfly-fast-wrong:target_V": ["1024-2048", "256-512", "512-1024"], "dfly-fast-wrong:task": ["chart_qa", "general_vqa"]}, "min_groups": 20, "n_expected_requests": 24, "n_groups": 24, "n_sessions": 3, "n_tasks": 3, "task_group_counts": {"chart_qa": 8, "general_vqa": 8, "invoice_fields": 8}}`

## Candidates

| candidate | groups | pairs | correctness | quality | speed ratio [95% CI] |
| --- | ---: | ---: | --- | --- | --- |
| `dflash-exact` | 24 | 72 | `True` | `pass` | 1.430 [1.421, 1.438] |
| `dfly-fast-wrong` | 16 | 48 | `False` | `fail` | 2.098 [2.087, 2.108] |

Excluded candidates:

- `dflash-exact` — synthetic_fixture
- `dfly-fast-wrong` — synthetic_fixture, correctness_failed, run_errors, quality_failed, insufficient_coverage

## Replay

Replay the local records and manifest under the documented protocol; no private sample text is included here.
