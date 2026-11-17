# Native evaluation: Qwen3-VL-8B

The `native/` adapters support all seven methods: DFlash, DFlash2, DSpark, Domino, JetSpec, DFlare, and DFly. They retain the trained method-specific heads and use vLLM's native linear greedy verifier with spatial MRoPE.

Use a separate **vLLM 0.22.1** environment. This runner expects the continuation checkpoints (125,000 total updates), a local target snapshot, and MMSpec ShareGPT-format requests and images. Weights and image datasets are not bundled.

```bash
export PYTHON=/path/to/vllm0221/bin/python
export Q3VL8B_ROOT=/path/to/evaluation
export TARGET=/path/to/Qwen3-VL-8B-Instruct
export SOURCE=/path/to/mmspec_sharegpt_600.jsonl
export IMAGES=/path/to/mmspec/images
export RESULT_ROOT=/path/to/results
bash research/qwen3vl8b_cont_mmspec/scripts/prepare_exports.sh \
  /path/to/checkpoints "$Q3VL8B_ROOT/exports"
bash research/qwen3vl8b_cont_mmspec/scripts/run_qwen3vl8b_cont_mmspec.sh --preflight
bash research/qwen3vl8b_cont_mmspec/scripts/run_qwen3vl8b_cont_mmspec.sh --gpus 0 --suite smoke
```

Checkpoint layout is `<checkpoints>/<method>/train/ckpt_final.pt`. The export script creates local exports and fresh per-method runtime manifests from the current source and checkpoint bytes. For a complete run, use `--suite all`; `--methods` selects a subset and `--resume` preserves validated completed stages. Results remain local; `--publish` is explicitly disabled.

This supplied continuation protocol uses K=7, BF16, TP1, batch size 1, three repeats, 600 sessions / 723 turns, natural stopping, and at most 1,024 generated tokens. It should not be conflated with a different checkpoint or session-count protocol in the paper. `vlm-dflash native-eval --help` exposes the underlying evaluation interface.

`native/export_checkpoint.py` performs CPU export; `native/make_runtime_manifest.py` records the current source hashes; `native/bootstrap_*.py` installs the selected method; `native/spatial_adapter.py` handles multimodal positions. The two shell scripts orchestrate export and evaluation without embedding machine paths or remote author accounts.
