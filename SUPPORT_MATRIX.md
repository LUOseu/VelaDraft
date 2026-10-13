# Support matrix

Statuses: `stable` / `validated` / `reduced_validation` / `experimental` /
`architecture-only` / `unsupported`. A target column is `validated` when the
canonical protocol has been recorded for that family, and
`reduced_validation` when the full pipeline was recorded at a reduced
protocol (smaller slice / prompt budget). Numbers behind "validated" are in
`BENCHMARKS.md`; protocol details in the linked docs.

This file is generated: the algorithms table below is rendered from
`vlm_dflash/algorithms/<name>.py::<Algorithm>.metadata` (the same data behind
`vlm-dflash doctor --algorithm <name>` and exported model cards). Regenerate
with `python scripts/render_support_matrix.py`.

## Algorithms

| Algorithm | Qwen2.5-VL | Qwen3-VL           | Image     | Multi-image | Video       | Train        | Eval      | Export    |
| --------- | ---------- | ------------------ | --------- | ----------- | ----------- | ------------ | --------- | --------- |
| dflash    | validated  | validated          | validated | unsupported | unsupported | validated    | validated | validated |
| dflash2   | validated  | reduced_validation | validated | unsupported | unsupported | experimental | validated | validated |
| domino    | validated  | reduced_validation | validated | unsupported | unsupported | validated    | validated | validated |
| dspark    | validated  | reduced_validation | validated | unsupported | unsupported | validated    | validated | validated |
| jetspec   | validated  | reduced_validation | validated | unsupported | unsupported | validated    | validated | validated |

- "Train" is the training maturity: DFlash2 trains with an experimental
  objective reconstruction (upstream objective undisclosed); everything else
  is validated.
- JetSpec eval numbers are validated with its causal in-block mask (its
  parallel-tree production runtime is not ported); every algorithm's runtime
  story is sequential token-exact verification only.
- Eval/export for every algorithm load the exported checkpoint through the
  algorithm's own loader dispatch and produced token-exact output in the
  recorded runs (`RELEASE_VALIDATION.md`).

## Targets

| Target | Adapter | Regen | Capture | Train | Eval | Export |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL-3B-Instruct | qwen25_vl | validated | validated | validated | validated | validated |
| Qwen3-VL-8B-Instruct | qwen3_vl | validated | validated | validated | validated | validated |

Both are validated on single-image + text with greedy decoding. Qwen3-VL was
validated for the four zoo algorithms at reduced scale (256-row slice,
16-prompt sequential eval, export + reload, all token-exact - see
`RELEASE_VALIDATION.md`); DFlash has the full 2K-protocol run there.
Multi-image, video and extra target families are roadmap items, not support
claims.

## Verification modes

| Mode | Exactness | Use |
| --- | --- | --- |
| sequential | token-exact (gate) | all acceptance/correctness numbers |
| strict_parallel | partial fallback; see probe | diagnostics only |
| parallel | not exact (measured) | experimental throughput |

## Benchmark workloads

| Workload | Status | Prompt set |
| --- | --- | --- |
| caption_general | scaffold | run's own val split (provisional) |
| textvqa | validated | pinned export (lmms-lab/textvqa validation, 128-prompt seed-42 subset; `benchmarks/prompts/textvqa_v1.jsonl`) |
| docvqa | planned | protocol pinned, adapter pending |
| chartqa | planned | protocol pinned, adapter pending |
