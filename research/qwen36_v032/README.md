# Qwen3.6 DFlash: official vLLM 0.32 backend

This is a measured backend upgrade for the final trained DFlash checkpoint on
Qwen3.6-35B-A3B, A800 80 GB, TP1, BF16, batch1 and K=2. It is a research
runner with a narrowly scoped position adapter, not a general serving plugin.

The engine, target verification, sampler, attention kernels and CUDA graph
manager are upstream vLLM. The adapter supplies the drafter's trained three-axis
positions and uses the official MRoPE operator for context keys. It leaves
physical cache slots and target computation under the native engine. Target
embedding/head sharing is checked, and the exported trained tensors are compared
bitwise with the checkpoint before and after evaluation. No custom CUDA/Triton
kernel or quantization is introduced.

## Validated configurations

| Workload | Configuration | Measured speedup |
|---|---|---:|
| Natural EOS, max 1024 | v0.32 MRV2, DFlash FULL graph, GDN `triton`, K2 | 1.252x / 1.253x on two cards |
| Natural EOS, max 1024 | Earlier optimized v0.22, K2 | 1.186x / 1.188x |
| Fixed 512 output tokens | v0.32 MRV2, GDN `cuda`, K2 | 1.231x / 1.232x on two cards |
| Fixed 512 output tokens | Earlier optimized v0.22, K2 | 1.158x |

Each full condition uses 12 diagnostic sessions, 16 turns and three warm-engine
repeats (48 timed requests per arm). Every AR/SD pair is measured on the same
GPU. Natural-completion repeats use reverse process order on the second card.
The primary ratio is sum(AR request wall time) / sum(SD request wall time).
Reported intervals resample 12 paired session clusters, retaining all turns and
warm repeats together. They do not cover run/card variance or held-out data.

The different GDN choices must remain distinct rows. CUDA fusion improves
fixed-budget throughput here, while its finite-precision generation differs in
length from AR; the Triton setting gives the better natural-EOS result in these
samples. Changing the GDN option left all 48 AR outputs unchanged on each card.
Do not attribute every divergence to floating point without checking it.

Quality scores were not collected in this backend campaign. Token exactness is
reported independently; these results do not establish strict token-exact
losslessness and are not an untouched paper main table.

## Runtime and reproduction

- vLLM 0.32.0; PyTorch 2.13.0+cu130; official package hashes pinned in
  `/root/autodl-tmp/qwen36_backend_20261101/v032_pypi_hashes.txt`.
- Use a complete CUDA **13.0** compiler/header toolkit. The initially resolved
  13.4 NVCC with 13.0 headers failed FlashInfer compilation. Both arms now use
  the same existing 13.0 toolkit; `TOOLCHAIN.json` records its path and digest.
- Target weights stay in `/dev/shm/qwen36_tp1_target`; no client model cache.
- All server evidence lives in `/root/autodl-tmp/qwen36_backend_20261101`.

On the prepared A800 server, choose a new output path and an available GPU:

```bash
bash /root/autodl-tmp/qwen36_backend_20261101/repo/research/qwen36_v032/reproduce_natural.sh \
  /root/autodl-tmp/qwen36_backend_20261101/new_natural_confirmation 0
```

The runner leases an empty GPU, preserves previous evidence and writes raw
requests, runtime metadata, source hashes and sampled GPU resource records.
It uses native FlashAttention for target full-attention layers and the draft;
GDN is the target's separate linear-attention/state-update component.

`research/qwen36_fixedlength` implements the separate fixed-output comparison:
`SamplingParams(temperature=0, max_tokens=512, ignore_eos=True)`, with explicit
stop-token lists removed. It asserts that every timed output has exactly 512
tokens and ends by length. Tokens after EOS, if any, are included. This uses the
same output-length mechanism as the [official latency benchmark](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/benchmarks/latency.py),
with the existing real-image paired workload and request timing boundary.
Natural-output quality must not be scored on those forced-length continuations.

## Evidence and discarded candidates

`build_evidence_report.py` recomputes request sums and session bootstrap CIs
without importing the evaluator's statistics function. `RESULTS.csv`,
`RESULTS.json`, `EVIDENCE_AUDIT.json` and `ARTIFACT_MANIFEST.json` are generated
in the campaign evidence directory and mirrored on the server.

The larger MoE tile search, shared-gate fusion, standalone context graph and
shared-stream threshold changes did not produce a robust end-to-end win on the
previous runtime. Target Triton Attention was also not adopted. Those probe
results are retained separately. The first gate microbenchmark did not retain
all graph input storage; the corrected `shared_gate_probe_v2.py` is the valid
microbenchmark. The model-level gate comparison was independent of that issue.

K1 and K3 fixed-length development pilots do not justify replacing the validated
K2 setting: K1 was slower and K3 was effectively tied with K2. Pilot results
are not included as full benchmark rows.
