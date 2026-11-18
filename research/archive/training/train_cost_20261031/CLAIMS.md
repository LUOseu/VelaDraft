# Claim ledger

| Claim | Measurement / rejection condition | Status | Evidence |
|---|---|---|---|
| Supported scoring paths preserve first-order loss/gradient/metric semantics within tested precision | Identical input and anchor replay; reject missing/nonfinite gradients, FP32 disagreement or BF16 drift above the probe gate | [measured] | `cpu_tests.xml`, `full_cpu_tests.xml`, `a6000_bf16_ce_padded256.json`, `CODE_REVIEW.md` |
| Chunked CE reduces peak allocated memory in the tested shapes | Full-update allocated peak, including optimizer; reject a memory reduction claim if peak does not fall | [measured] | `a6000_bf16_ce.json`, `a6000_bf16_ce_padded256.json`, `summary.json` |
| Local teacher dedup provides a stable full-update speedup on the first pilot | Paired complete-update measurements; reject a speed claim if interval includes 1 or interference unresolved | [negative] | `a6000_bf16_pilot.json`, `summary.json` |
| Combined CE improves update time on two guarded-exclusive synthetic shapes | Paired full updates, baseline/optimized ratio interval > 1; occupancy checked before/after each arm | [measured] | `a6000_exclusive_ce_padded256.json`, `a6000_exclusive_ce_unpadded256.json`, `summary.json` |
| The combined CE optimization improves arbitrary real training workloads | Actual feature-store shapes, current fused baselines and resource-matched ablations | [to do] | `FINAL_PROPOSAL.md` |
| Lower GPU-hours to equal quality, including pipeline setup/evaluation | Multiple seeds, preregistered quality threshold, stage cost accounting | [to do] | `FINAL_PROPOSAL.md` |
| Local patches alone establish systems-paper novelty | Independent current-prior-art comparison and clear contribution delta | [negative] | `refine-logs/round-1-review.md`, `refine-logs/round-2-review.md` |
