# Native evaluation: Qwen3.6-35B-A3B

This runner evaluates DFlash, DFlash2, DSpark, Domino, JetSpec, DFlare, and DFly with native vLLM verification and the shared short-query expert-grouping policy. It uses vLLM **0.32.0**, BF16, TP1, batch size 1, and three proposed tokens. Natural stopping and fixed-512 diagnostics are kept separate.

## Inputs

Use a separate vLLM 0.32.0 environment with the package's evaluation dependencies and a complete CUDA toolkit. Provide a local target-model snapshot and local trained checkpoints; this anonymous source release does not include checkpoint weights or author-hosted download links.

```bash
export Q36_TARGET_PATH=/path/to/Qwen3.6-35B-A3B
export Q36_LOCAL_CHECKPOINTS=/path/to/checkpoints
# Required layout: $Q36_LOCAL_CHECKPOINTS/<method>/train/ckpt_final.pt
bash scripts/run_qwen36_a100_benchmarks.sh \
  --config configs/eval/qwen36_a100_seven.yaml --gpus 0 --preflight
bash scripts/run_qwen36_a100_benchmarks.sh \
  --config configs/eval/qwen36_a100_seven.yaml --gpus 0 --suite mmspec-smoke
```

For a full run, replace `mmspec-smoke` with `mmspec-full`. `--methods dflash,dflash2` selects a subset; `--resume` resumes the same frozen inputs. Results remain under `run.root`. Uploading results is disabled in this release.

The default checkpoint manifest records the expected weight sizes and hashes. To evaluate separately trained checkpoints, set `checkpoints.manifest` to your own JSON mapping each method to its `size` and `sha256`. Exports and run receipts are generated from the files actually used. Do not reuse manifests generated for a different source tree.

The default YAML selects one A100 GPU. For other supported hardware, adjust `run.gpu_name_contains`, memory utilization, and `engine.cuda_home`. The supplied A800 expert-tile table is hardware-specific; use `Q36_MOE_TUNED_DIR` only with the matching GPU. MMSpec is obtained from the third-party repository and commit recorded in the YAML.

## Code map

- `cli.py`, `data.py`, `prepare.py`, `lane.py`, `stage.py`: input freezing and evaluation.
- `summarize.py`: local run summaries, separate from raw requests.
- `../qwen36_seven_20261102/`: method bridges, export preparation, and inference settings.
- `../qwen36_native_vllm/`: shared checkpoint exporter and method-specific reference operations.
- `../qwen36_v032/`: three-axis position adapter.
- `../qwen36_grouped_e2e_20261102/dispatch.py`: query-dependent expert grouping.
- `../qwen36_nextopt_20261102/worker.py`: native worker composition.

The launch protocol is a reproducible runner, not a claim that every hardware/protocol combination reproduces the paper's reported timing. Check the paper's experiment settings before comparing speedups.
