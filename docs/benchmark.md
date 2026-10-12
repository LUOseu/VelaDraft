# Benchmark harness

One command measures a trained run against fixed workload manifests and writes
a report with its own protocol attached:

```bash
python scripts/benchmark.py --run-dir runs/<run> --list
python scripts/benchmark.py --run-dir runs/<run> --workload caption_general
python scripts/benchmark.py --run-dir runs/<run>          # all runnable workloads
```

Manifests live in `benchmarks/manifests/` (JSON). Each pins:

- `prompt_source` — `{"type": "run_val"}` (the run's held-out split) or
  `{"type": "sharegpt_jsonl", "path": ..., "meta": ...}` (a fixed external
  prompt file);
- `num_prompts`, `max_new_tokens` — the output budget;
- `status` — `validated` / `scaffold` / `planned` (planned workloads pin the
  protocol but have no adapter yet; the harness lists and skips them).

External prompt paths in manifests are written relative to the repository root
(`load_eval_rows` itself resolves relative paths against the run directory, so
the harness pre-resolves them); set `VLM_DFLASH_BENCH_DATA` to keep exported
benchmark data outside the checkout.

Public workloads with their own download pin the export step. TextVQA
(`lmms-lab/textvqa` validation split, 128-prompt seeded subset, seed 42):

```bash
python scripts/prepare_benchmark.py --workload textvqa \
    --source <path-to-lmms-lab/textvqa-download> --out benchmarks/data/textvqa
```

The adapter writes the images, `textvqa_v1.jsonl` + `meta.json` and a
`manifest.json` with per-shard and per-row question/image hashes, and
refreshes the repo-tracked prompt set definition
`benchmarks/prompts/textvqa_v1.jsonl` (ids + hashes, no image bytes). Re-running
with the same seed over the same shards reproduces the same subset; validate
the export against the pinned shard hashes before comparing numbers.

Fixed protocol for every measured workload:

- greedy decoding (temperature 0);
- target-only greedy reference generated per workload;
- acceptance measured with `--verification sequential` (the only
  bitwise-exact path); `parallel` / `strict_parallel` runs are marked
  `experimental_runtime` in the report and must never be mixed into
  correctness tables;
- token-exact comparison on every measured prompt.

Reports are written to `benchmarks/reports/` (gitignored) as JSON + Markdown.
Promote numbers into `BENCHMARKS.md` only with the same protocol block
(workload, prompts, max_new_tokens, checkpoint, verification).

Grading rules borrowed from the v0.1 acceptance criteria:

- `Tier 0 / smoke` numbers (tiny slices) prove the pipeline runs; they are not
  quality numbers.
- Regression comparisons require the same workload, prompt count, output
  budget and verification mode; document any deviation next to the numbers.
