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
| dflare    | validated  | reduced_validation | validated | validated   | unsupported | validated    | validated | validated |
| dflash    | validated  | validated          | validated | validated   | unsupported | validated    | validated | validated |
| dflash2   | validated  | reduced_validation | validated | validated   | unsupported | experimental | validated | validated |
| dfly      | validated  | reduced_validation | validated | validated   | unsupported | experimental | validated | validated |
| domino    | validated  | reduced_validation | validated | validated   | unsupported | validated    | validated | validated |
| dspark    | validated  | reduced_validation | validated | validated   | unsupported | validated    | validated | validated |
| jetspec   | validated  | reduced_validation | validated | validated   | unsupported | validated    | validated | validated |

- "Train" is the training maturity: DFlash2 trains with an experimental
  objective reconstruction (upstream objective undisclosed) and DFly trains
  the ported architecture with the shared DFlash objective by default —
  upstream's two-stage LK/D-PACE -> e2e-TV recipe is ported and selectable
  (`algorithm.train_recipe: upstream_two_stage`) but has no canonical GPU run
  recorded yet (see `docs/upstream/dfly.md`); everything else is validated.
- Multi-image is the schema-v2 data path (multiple images per row, images on
  any user turn of a multi-turn conversation). It is validated end-to-end for
  DFlash on the recorded E2 smoke protocol (2-image / 2-turn rows, 16 train +
  4 val, token-exact eval; `configs/smoke_multiturn.yaml`) and exercised for
  all algorithms by the shared CPU suite; the other algorithms have not been
  re-run on multi-image data individually. Batched online capture for
  multi-image rows is still a WP4 measurement item.
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
Multi-image / multi-turn (schema v2) was recorded for DFlash on Qwen2.5-VL-3B
via the E2 smoke (see the algorithm footnote); video and extra target families
remain roadmap items, not support claims.

## Verification modes

| Mode | Exactness | Use |
| --- | --- | --- |
| sequential | token-exact (gate) | all acceptance/correctness numbers |
| strict_parallel | partial fallback; see probe | diagnostics only |
| parallel | not exact (measured) | experimental throughput |

## Feature modes

This section is static text in the renderer (feature modes are orthogonal to
the algorithm registry above).

| Feature mode | Status | Evidence |
| --- | --- | --- |
| `offline` (staged capture) | validated | the canonical protocol runs in `BENCHMARKS.md` / `RELEASE_VALIDATION.md` are all offline-path numbers |
| `online` (live target extraction) | experimental | full CPU suite + GPU plumbing smokes on Qwen2.5-VL-3B and Qwen3-VL-8B (`docs/online-training.md`): token-exact eval, provenance, digests, exact-resume gates, per-phase peaks |

Online has no canonical-quality or split-device recording yet; do not relabel
offline benchmark numbers as online (see `docs/online-training.md`).

## Benchmark workloads

| Workload | Status | Prompt set |
| --- | --- | --- |
| caption_general | scaffold | run's own val split (provisional) |
| textvqa | validated | pinned export (lmms-lab/textvqa validation, 128-prompt seed-42 subset; `benchmarks/prompts/textvqa_v1.jsonl`) |
| docvqa | validated | pinned export (lmms-lab/DocVQA DocVQA/validation, 128-prompt seed-42 subset; `benchmarks/prompts/docvqa_v1.jsonl`) |
| chartqa | validated | pinned export (lmms-lab/ChartQA test, 128-prompt seed-42 subset; `benchmarks/prompts/chartqa_v1.jsonl`) |
