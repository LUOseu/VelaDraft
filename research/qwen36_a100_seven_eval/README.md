# Qwen3.6-35B-A3B seven-method MMSpec benchmark (A100)

Runner for the contract in [`AGENTS.md`](AGENTS.md): native vLLM 0.32 AR versus DFlash, DSpark, Domino,
JetSpec, DFlash2, DFlare and DFly at K=3 on the complete MMSpec `test` (600 sessions, 723 turns),
TP1 batch 1, BF16, results mirrored to Hugging Face while it runs.

Status: code and CPU tests are in this branch; **A100 hardware acceptance is pending** until a
real run has loaded the model, passed the per-method checks, read the data, written to HF and
resumed. Nothing here is an A100 result yet.

## Setup (once per machine)

```bash
# 1. Runtime: vLLM 0.32.0 brings torch 2.13 (cu130); a complete CUDA 13.0 toolkit is needed for FlashInfer JIT.
uv venv -p 3.12 .venv-q36 && VIRTUAL_ENV=.venv-q36 uv pip install "vllm==0.32.0" huggingface_hub pyyaml pytest
export Q36_PYTHON=$PWD/.venv-q36/bin/python CUDA_HOME=/usr/local/cuda-13.0

# 2. Target weights at the pinned revision (re-check the revision before a run).
hf download Qwen/Qwen3.6-35B-A3B --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 --local-dir /data/models/Qwen3.6-35B-A3B
export Q36_TARGET_PATH=/data/models/Qwen3.6-35B-A3B

# 3. Work locations (all on the server disk; nothing in a client cache).
export Q36_A100_ROOT=/data/qwen36_a100_seven Q36_MMSPEC_CLONE=/data/MMSpec
export VELADRAFT_GPU_LOCK_DIR=/dev/shm/q36_gpu_locks

# 4. HF write access to the results repository (preflight checks it).
hf auth login
```

Optional variables: `Q36_A100_EXPORTS` (draft exports, default under the root),
`Q36_LOCAL_CHECKPOINTS` (an existing `<root>/<method>/train/ckpt_final.pt` tree; used only when the
SHA-256 matches the pin), `Q36_MOE_TUNED_DIR` (an A100 MoE tile table; empty keeps vLLM defaults),
`Q36_PRE_STAGE_HOOK` (a site command called with the GPU index before each stage, e.g. to release
idle placeholders on that card).

## Run

```bash
bash scripts/run_qwen36_a100_benchmarks.sh --preflight --config configs/eval/qwen36_a100_seven.yaml --publish
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full \
  --methods dflash,dspark,domino,jetspec,dflash2,dflare,dfly --draft-length 3 --publish --resume
```

`--preflight` lists every item that needs a person (GPUs, toolkit, target, data, HF access, disk)
and exits. The first run downloads the seven checkpoints at their pinned revisions
(`research/qwen36_seven_20261102/checkpoint_manifest.json`), exports them for vLLM, clones MMSpec at
`290486861e`, verifies the paper scope (600 / 723 / 100 per topic) and freezes all hashes into
`manifest.json`. `--resume` continues the latest run of the suite after any interruption; finished
shards are never recomputed and a changed code commit is refused. `--suite mmspec-smoke` (12
sessions) is a functional check only and is never a formal table.

## What a run measures

- Jobs: every method under `natural` (greedy, natural stop, max 1024 new tokens) and `fixed512`
  (exactly 512 tokens, EOS ignored, explicit stop ids removed). A job owns one GPU and alternates
  AR and SD stages on it (AR0 SD0 SD1 AR1 AR2 SD2), each a fresh engine with an untimed warmup.
  Stock-default AR (`ar_default`) is timed in its own lane alongside the tuned native AR.
- Multi-turn history is the untimed native AR `reference` (fixed AR history for every arm); every
  timed request must reproduce the reference prompt token ids.
- Timing per request: image read and decode, template, processing, vision, prefill, decode and
  detokenization; engine load, compile and warmup are excluded. GPU processes are sampled every
  2 s; a foreign process inside a timed window withdraws that stage's shards and reruns it.
- `natural` SD jobs end with an untimed stats-enabled pass for native acceptance counters.

`summary.json` / `summary.csv` are recomputed from raw shards: request-level speedup
`sum(AR wall) / sum(SD wall)` with a session-clustered paired bootstrap 95% CI, the per-token time
ratio (a separate metric), acceptance (accepted draft tokens per round; the bonus-inclusive length is
labelled separately), token-exact rate, finish reasons and latency percentiles, per topic, pooled and
as a labelled macro mean. Task quality is `pending`: MMSpec `test` ships no references or scorer;
all generations are kept for later scoring.

## Layout

```
<root>/runs/<run_id>/            mirrored to khazic/veladraft-qwen36-400k-dflash@main:eval/qwen36_a100_seven_v1/runs/<run_id>/
  manifest.json  status.json  summary.json  summary.csv  sessions.json
  raw/mmspec/<method>/<protocol>__<stage>__<topic>.jsonl     immutable once written
  raw/mmspec/_reference/natural__reference__<topic>.jsonl
  stages/<owner>/<protocol>__<stage>.json                    engine args, versions, worker metadata, resources
  resources/  logs/  claims/  done/  failed/  publish_state.json
```

Not implemented yet: the temperature-1 extension (contract step 5, only after the greedy table) and
an A100 MoE tile search (set `Q36_MOE_TUNED_DIR` if one is measured).
