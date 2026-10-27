# Native vLLM evaluation implementation plan

> **For agentic workers:** Execute inline, then request an independent code review before opening the PR.

**Goal:** Provide an executable native vLLM paired E2E protocol; keep HF checkpoint diagnostics explicitly distinct from performance evidence.

**Architecture:** `eval.native` owns a CPU-only plan and complete-scope accounting. `eval.native_runtime` owns fresh-process public vLLM execution and read-only worker metadata. `eval.native_resources` owns GPU leases and sampled ownership. Custom trained methods use explicit pinned external adapter manifests and prepared full-weight exports. No adapter, model weight, numerical precision, or rejection kernel is invented or silently substituted.

**Tech Stack:** Python, vLLM0.22.1 public LLM API, Torch, Transformers AutoProcessor, Pillow; pytest CPU regressions plus a real native DFlash smoke when a GPU is available.

- [x] Add failing tests in `tests/test_native_eval.py`: constructor arguments retain default AR; SD Flash/linear; missing scope/input mismatch suppress speed; ratio of sums; bootstrap by session; numerical token changes remain visible; only complete repeats aggregate; invalid precision overrides rejected.
- [x] Implement `vlm_dflash/eval/native.py`: `engine_args(plan, arm)`, `summarize(plan, reference, reports)`, plan/source/adapter validation and isolated process coordinator; atomic raw reports and CLI.
- [x] Implement `vlm_dflash/eval/native_runtime.py`: public `LLM.generate`, fresh image UUID, RGB/template-to-full-token wall boundary, all-prompt warmup, canonical history, metadata before/after, raw rows and official-counter deltas outside timing.
- [x] Implement `vlm_dflash/eval/native_resources.py`: actual UUID mapping, shared GPU lock, periodic NVML descendant ownership; any contamination invalidates aggregation.
- [x] Route `vlm-dflash native-eval` and `scripts/vllm_bridge_check.py` to the new executable protocol. Preserve explicit old `eval`/`benchmark` commands as HF diagnostic reports and label their preprocessed-prompt timer correctly.
- [x] Update `docs/benchmark.md`, README EN/ZH and CLI help. State prepared exports/adapter requirements and scopes; report no new seven-method GPU or quality claim from CPU tests.
- [ ] Run `python3 -m pytest tests/test_native_eval.py tests/test_eval_checkpoints.py tests/test_benchmark_manifests.py tests/test_cli.py`; execute a minimal GPU integration without modifying existing experiment bundles.
- [ ] Independent code review, fix material issues, commit the bounded change and open PR against latest `main`. Do not merge.
