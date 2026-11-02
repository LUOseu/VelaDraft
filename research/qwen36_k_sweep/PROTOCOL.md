# Qwen3.6 A6000 K-budget diagnostic, 2026-11-01

Base VelaDraft commit: dd3a944c9d868f28df747c28da8d8c6f4935cc7a.
This is a new small diagnostic campaign, not a replacement for the A100 main table.

- Target: existing BF16 Qwen3.6-35B-A3B, two A6000 cards, target/draft TP2.
- Methods: final 75000-update DFlash2 and DFly checkpoints, pinned HF revisions and SHA256 in prepare.py.
- K: 1, 3, 5, then 7 as a same-device control; actual runtime draft query width K+1 including the anchor. Trained block size and all weight tensors stay B8.
- Data: first two MMSpec full-test sessions per topic, restricting the multi-turn topic to sessions with at least two turns. All turns are included. IDs frozen before seeing sweep results. This is test-set diagnosis, not an independent development set.
- One paired repeat per configuration, with untimed native AR history and full warmup, then AR and SD. Twelve sessions; 1024 maximum new tokens; no forced-length generation.
- Preserve original Qwen3.6 non-thinking prompt, image bounds 65536..1048576, context 8192, seed43, no prefix/MM caching, native AR, FlashAttention SD dense attention, native hybrid layers, target FULL_AND_PIECEWISE graph, draft PIECEWISE. GPU memory fraction .85 for both arms.
- Independent isolated environment: torch inherited from frozen 8B runtime, pinned vLLM 0.22.1 source files reused read-only, transformers 5.12.1 installed only in this environment. All model/cache/output data stay on server.
- Main outcome: sum AR request wall / sum SD request wall, with output token totals and output agreement. Startup/compile/warmup excluded. One-repeat small-sample values are preliminary, not publication-level significance.
- Within each K, prompt token IDs are required to match AR/SD. Cross-K histories must be inspected before interpreting cross-K changes in multi-turn workloads.
- Per-device PID ownership checks; any foreign-process sample aborts this campaign rather than sharing a card during timing.
- User authorized stopping their 8B tasks if needed. Preemption is scoped to lyh-owned known 8B evaluation process trees, records PIDs/start-times, and occurs only after a second usable card remains free for 30 seconds. Other users' tasks are never stopped.

CPU evidence: budget/config guards, short convolution boundaries, MRoPE continuation after rejection, sequential predecessor correction, matching K/graph width, and process-tree/PID-reuse safety. GPU support and speed remain unverified until actual runs complete.
