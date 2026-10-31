# Diffusion training cost implementation plan

**Goal:** remove redundant scoring work on sampled diffusion blocks and expose reproducible evidence for a systems paper.
**Architecture:** retain existing objectives and anchors; add opt-in bounded teacher-row reuse inside TeacherLogitsStream. Add controlled-CE slot compaction with checkpoint and frozen-head precompute backends as engineering components.
**Tech stack:** Python, PyTorch, existing trainer/factory/config/counters, CPU pytest, CUDA paired benchmark.

- [x] Inspect current main, training objectives, existing optimizations, GPU ownership and baseline tests (16 passed).
- [x] Add failing tests in tests/test_teacher_projection_dedup.py comparing projected outputs, loss, gradients and projection row counts; include cross-sample identity and compacted blocks.
- [x] Implement bounded dedup in vlm_dflash/objectives.py, configuration forwarding in dflash/factory.py, validation and execution counters in dflash/trainer_core.py. Preserve default behavior.
- [x] Run teacher-stream and trainer regressions. Add a paired benchmark script that serializes raw timings, numerical drift, workload counters and environment.
- [x] Validate on CUDA using an isolated copy; use spare GPU memory and then a newly idle GPU with occupancy guards. No task was paused; restoration was unnecessary.
- [x] Independent research/code review; record findings and implement necessary fixes.
- [x] Write measured results and an honest paper-facing experiment roadmap; commit the completed change on codex/diffusion-train-cost-20261031.

The implementation also includes `ce_chunk_size` / `ce_backward`, negative controls, reviewer-requested gradient-coverage and noncontiguous-input regressions. Paper-level real-workload and time-to-quality validation remain explicitly queued in FINAL_PROPOSAL.md.
