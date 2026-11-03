# Qwen3.6 seven-method MMSpec runner

This PR implements the MMSpec portion of the broader [benchmark handoff](AGENTS.md).
It runs native vLLM 0.32 AR and seven draft methods at fixed K=3, TP1, BF16 and batch 1.
The full MMSpec test contains 600 sessions and 723 turns; smoke selects 12 sessions.

**A100 hardware acceptance is pending.** CPU tests use fake engines. There are no new
hardware results, quality scores or verified acceleration claims in this change.

## Scope and remaining work

Implemented: pinned checkpoints, CPU export, input hashes, reference histories, natural
max1024 and fixed512 protocols, three alternating AR/SD repeats, per-topic resume,
GPU resource sampling, paired bootstrap summaries and incremental HF publication.

The handoff's ScienceQA, MM-Vet, MME, TextVQA, COCO, VizWiz, GQA, SEED-Bench and DOCCI
suites are not implemented here. Neither are official task scorers, development-set
K=1..7 selection, A100 MoE tile search, temperature-1 evaluation or automatic promotion
from smoke to full evaluation. MMSpec is an additional benchmark, not a replacement
for those suites. The configured AR settings are inherited from earlier development
work; they have not been established as the fastest AR settings on A100.

## Environment and inputs

Use a server environment containing vLLM 0.32.0, its compatible PyTorch/Transformers,
PyYAML and huggingface_hub. FlashInfer JIT needs the matching CUDA toolkit. Installations
and model downloads belong on the GPU server. Set `Q36_PYTHON` to that environment's Python.

Edit [the configuration](../../configs/eval/qwen36_a100_seven.yaml), or set these variables
to directories allowed by your server policy before running:

| Variable | Purpose |
| --- | --- |
| `Q36_A100_ROOT` | Run outputs and manifests |
| `Q36_A100_EXPORTS` | Content-addressed draft exports |
| `Q36_TARGET_PATH` | Complete local target snapshot at the configured revision |
| `Q36_MMSPEC_CLONE` | Dedicated MMSpec clone at the pinned commit |
| `VELADRAFT_GPU_LOCK_DIR` | Shared advisory GPU locks |
| `CUDA_HOME` | Compatible CUDA toolkit |
| `HF_HOME`, `TMPDIR`, `TRITON_CACHE_DIR` | Server cache and scratch locations |

Optional: `Q36_LOCAL_CHECKPOINTS` reuses local checkpoints only when their hashes match
[the pinned manifest](../qwen36_seven_20261102/checkpoint_manifest.json).
`Q36_MOE_TUNED_DIR` selects a measured A100 MoE table. `Q36_PRE_STAGE_HOOK` is an optional
site command receiving the GPU index; hook failure stops that stage. No hook is enabled
by default. All selected GPUs must be free at preflight.

GPU ownership defaults to PID checks. Sites with container PID namespaces can explicitly
set `VELADRAFT_GPU_OWNERSHIP=count`, which checks process-count bounds. Sampling cannot
prove continuous exclusivity; resource evidence records this limitation.

HF authentication uses the existing credential store or `HF_TOKEN`. Preflight checks
identity and token role; fine-grained write permissions are only confirmed by the first
successful upload. Never place tokens in the YAML or command line.

## Run and resume

```bash
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --preflight --publish
```

```bash
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-smoke --publish
```

Inspect smoke results for every method before explicitly starting full evaluation:

```bash
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full --publish
```

```bash
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full --publish --resume
```

`--gpus 0,1` selects lanes. `--methods dflash,dspark` selects methods. `--run-id` selects
an existing run with `--resume`. A run requires a clean checkout and freezes its code
commit and configuration. Resume checks the commit and input hashes, retries failed
jobs, and keeps completed shards. YAML changes do not modify an existing run's settings.
Each job must resume on its original GPU index and physical GPU. Use a new run to change
hardware or settings. A launcher lock prevents concurrent resumes of the same run.

## Measurement and output

Each method/protocol job alternates AR0, SD0, SD1, AR1, AR2, SD2 on one card. Every stage
starts a fresh engine and warms it up outside measurement. Multi-turn input uses frozen
untimed AR history; timed prompt token IDs must match the reference. Request timing
includes image reading, RGB decoding, templating, processing, prefill, generation and
detokenization. Engine startup and compilation are excluded.

The stock-default AR job runs independently and is reported separately without a paired
speedup claim. Natural SD also runs an untimed acceptance-counter pass. Accepted draft
tokens per verification round excludes the bonus token. The additional bonus-inclusive
number is only an upper bound because bonus emission is not measured.

Results become visible only after the stage succeeds and GPU resource checks pass.
A run is complete only when every expected session, turn and repeat, stock AR job and
requested acceptance pass is present. Partial pairs cannot be reported as a complete
benchmark. Quality remains pending because this MMSpec input supplies no reference answers
or task scorer; generated answers are retained for later evaluation.

`summary.json` and `summary.csv` report paired request-time ratios, session-clustered
bootstrap 95% intervals, per-token ratios, token equality, lengths, stop reasons and
latency percentiles. Topic summaries and macro topic speedups are separate from pooled
request totals. A complete timing run does not mean task-quality acceptance is complete.

Run files live under `<root>/runs/<run_id>/`. Raw rows are under `raw/mmspec/`; full local
stage diagnostics are under `stages/`, with resource samples and logs kept separately.
`--publish` mirrors selected evidence fields below the configured HF `eval/` directory.
Site environments, local paths, GPU UUIDs and diagnostic tracebacks stay local. Uploads
use immutable byte snapshots, preserve other runs in `INDEX.json`, retry moved repository
heads, and verify downloaded bytes at the resulting commit. Publication errors leave
local results intact; resume retries the outstanding upload.

## CPU validation

```bash
python -m pytest research/qwen36_a100_seven_eval -q
```

The repository CI runs these tests alongside its existing CPU, packaging and documentation checks.
