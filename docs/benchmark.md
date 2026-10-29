# Evaluation protocols

Performance and checkpoint diagnostics have separate entry points and report formats.

| Entry | Purpose | Runtime and timer | Primary E2E eligible |
| --- | --- | --- | --- |
| `vlm-dflash native-eval` | Paired steady-state request E2E | Native vLLM AR and SD; RGB/template/processor/vision/prefill/decode/output | Yes, after complete scope/source/resource validation; quality remains separate |
| `vlm-dflash eval` | Checkpoint acceptance/token diagnostics, including NPU | Existing HF local verifier, prepared inputs | No |
| `scripts/benchmark.py` | Historical fixed-workload HF diagnostics | Existing HF local verifier, prepared inputs | No |

## Native vLLM E2E

Use a separate CUDA environment with vLLM **0.22.1** and a local target plus an
already prepared, complete native draft export. The command does not convert a
training checkpoint, remove MRoPE, discard learned heads, or silently fall back
to HF. Keep the main training/NPU environment unchanged.

```bash
vlm-dflash native-eval \
  --target /models/Qwen2.5-VL-3B-Instruct \
  --draft /models/prepared-native-draft \
  --source /bench/prompts.jsonl --images-dir /bench/images \
  --num-prompts 120 --max-new-tokens 1024 --speculative-tokens 7 \
  --repeats 3 --gpu 0 --out /results/native.json
```

The source is one-image ShareGPT JSONL with unique IDs and alternating
`human`/`gpt` conversations. All human turns are measured. Dataset assistant
answers are ignored. One untimed native AR creator supplies the actual causal
history for later turns; every measured AR and SD request must reproduce its
full input token IDs. Counts are strict; a short source is an error.

A dry run adds `--dry-run`: it checks paths, declared SHA pins and source scope,
and writes the plan without constructing an LLM or initializing CUDA. CPU
checkpoint-to-export tensor preservation is checked before real execution.

### Existing spatial and method adapters

Our multimodal MRoPE exports and additional learned modules are **not** automatic
stock-vLLM integrations. `--runtime-manifest /path/runtime.json` activates an
explicit, reviewed external adapter in the SD client and spawned workers only.
It is mandatory for method labels other than `dflash`, and for spatial DFlash
exports that the stock loader does not support. Example manifest structure:

```json
{
  "method": "dflash",
  "scope": "local trained DFlash with preserved three-axis spatial RoPE",
  "bootstrap": "adapter/sitecustomize.py",
  "checkpoint": "checkpoints/ckpt_final.pt",
  "environment": {},
  "files": {
    "adapter/sitecustomize.py": "SHA256 of the existing activation source",
    "checkpoints/ckpt_final.pt": "SHA256 of the complete training checkpoint"
  }
}
```

Paths resolve relative to this manifest. Include all adapter source/dependency
pins in `files`, not only the two shown. An activation file that defines an
installer instead of executing it can add `"call": "install"`. Add its existing
method-specific enable/receipt environment values to `environment`; device,
thread, Python path and precision overrides are rejected. SHA256 values are
literal hex digests (`sha256sum <file>`), not the explanatory strings above.
Only trusted activation code may be supplied. The protocol does not audit a
third-party adapter's algorithm for you. It verifies all checkpoint tensor
keys, shapes, dtypes and bytes against `model.safetensors`, including scalar
parameters, and records the implementation scope. This is CPU preservation,
not a claim of GPU equality with a canonical implementation.

All seven labels are accepted for prepared integrations; this PR does not port
seven complete upstream systems. In particular, local shared-block JetSpec
and fixed-block DSpark should retain those scope descriptions when used.

### Qwen3-VL-8B campaign code

The [Qwen3-VL-8B native-vLLM research snapshot](../research/qwen3vl8b_native_vllm/README.md)
contains the method-specific adaptation, quality scoring, queue and audit code
used for this campaign. It records machine-bound paths and is not a turnkey
installation; regenerate its receipts and runtime manifests for a new machine.

An optional `QWEN8B_DIGEST_CACHE=/path/to/digests.sqlite` avoids re-reading
unchanged files by caching SHA-256 against device, inode, size, mtime and ctime.
This is an I/O optimization, not an adversarial integrity boundary. Without the
variable, full SHA-256 is calculated at each observation. The GPU is leased
before expensive plan and preservation checks for non-dry runs.

### Measurement and validity

- AR retains native automatic attention/graph/matrix selection. SD requests
  stock FlashAttention for target and drafter, linear DFlash verification,
  batch1/TP1, BF16, S7/B8 by default, PIECEWISE capture8. Read-only worker RPC
  records actual backend classes, draft causality and numerical flags.
- Native matrix defaults are never replaced by Flex/P2/F2. Parent and worker
  numerical settings must match the untimed native AR reference. The public
  vLLM rejection/commit algorithm remains responsible for verification.
- Default three repeats use **AR0, SD0, SD1, AR1, AR2, SD2**, each a fresh process
  and LLM. Every source turn is warmed before each timed stage. Compile, model
  loading and full warmup are excluded. RGB/image bytes, template, processing,
  vision, prefill, decode, CUDA sync and full public output extraction are in
  the request wall boundary. Prefix/image processor caching is disabled and
  every physical request gets a fresh image UUID.
- Headline `speedup_sum_ar_over_sd = sum(AR wall) / sum(SD wall)`. It is not a
  mean of per-request ratios. The 95% interval resamples whole sessions with
  all their turns and repeats, 2000 draws/seed42. It describes session variation
  for this fixed sequence, not independent seed/card/run variance.
- Raw JSON stages preserve all output IDs, EOS/cap/finish reasons, counts and
  durations. Observed token equality is separate from timing eligibility;
  BF16 numerical differences do not automatically fail a measurement. Input
  mismatches, failed stages or incomplete coverage suppress the headline.
- A UUID advisory lock and sampled NVML descendant ownership cover the full
  comparison. Foreign GPU processes or query errors invalidate the result;
  no foreign task is stopped. This is sampled ownership, not continuous or
  host-wide exclusivity. Model, input and declared adapter bytes are pinned.
- Real quality/accuracy against dataset gold is a **separate required eval**.
  `paper_ready` remains false: neither finite token agreement nor a CI containing
  zero proves mathematical losslessness, quality equivalence or noninferiority.
  One-repeat/one-session runs are labeled smoke, not paper results.
- Public speculative counters, when available, report accepted **draft-only**
  tokens per round and fraction. No synthetic `+1` is called acceptance length.
  Native timing disables logging overhead in both arms; unavailable counters
  stay unavailable rather than being inferred from output lengths.

The compatibility command `python scripts/vllm_bridge_check.py` now invokes this
same protocol. It requires a draft and no longer returns failure solely because
sampled AR/SD token outputs differ. No new GPU performance or quality claim
is made by changing the evaluator. A [functional GPU smoke receipt](validation/native_eval_smoke_20261027.json)
records one session/repeat, five physical calls and the observed tiny-sample
ratio, explicitly outside paper-performance evidence. It ran from source with
vLLM0.22.1/Torch2.11/Transformers5.11; CPU regressions used the training tested
extra (Torch2.10/Transformers5.12.1). This does not validate every dependency combination.

## HF checkpoint diagnostics (legacy workload harness)

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

DocVQA (`lmms-lab/DocVQA` `DocVQA/validation` split) and ChartQA
(`lmms-lab/ChartQA` test split) use the same command with their own
`--workload` / `--source` / `--out`; both pin `max_new_tokens=64`. The adapter
writes the images, `<workload>_v1.jsonl` + `meta.json` and a `manifest.json`
with per-shard and per-row question/image hashes, and refreshes the
repo-tracked prompt set definition `benchmarks/prompts/<workload>_v1.jsonl`
(ids + hashes, no image bytes). Re-running with the same seed over the same
shards reproduces the same subset; validate the export against the pinned
shard hashes before comparing numbers.

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
