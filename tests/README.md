# Core regression tests

These tests use small synthetic tensors, temporary feature records and mocked
runtime interfaces. They cover the seven drafter implementations, multimodal
feature reuse, batching and scoring semantics, checkpoint export, and the native
inference protocol. They do not require downloaded checkpoints, private datasets,
historical experiment reports, or a GPU. Native protocol tests use mock engine
responses and do not measure performance.

Run from the repository root after installing the development dependencies:

```bash
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
```

For a shorter check of the training optimizations:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_feature_contract.py tests/test_cache_budget.py \
  tests/test_stage_cache.py tests/test_regroup.py \
  tests/test_recipe_row_compaction.py tests/test_teacher_projection_dedup.py \
  tests/test_chunked_ce_training.py
```

`test_dflash2.py` and `test_jetspec.py` read their packaged smoke configuration
files under `configs/`; run the suite from the repository root. The tests use the
package's declared PyTorch and Transformers versions, including the Qwen model
configuration classes. CPU execution can take several minutes because some tests
compare complete gradients and perform small checkpoint round trips.
