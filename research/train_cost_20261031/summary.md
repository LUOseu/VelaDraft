# Paired scoring probes

Synthetic shape probes; occupancy context is retained in JSON (exclusive cells are named exclusive). Intervals resample timing pairs, not training seeds; they do not remove interference bias.

| Source / case | Baseline ms | Optimized ms | Ratio [95% paired interval] | Peak allocated GiB baseline → optimized | Numerical gate |
|---|---:|---:|---:|---:|---|
| a6000_bf16_pilot.json, stride 1, stage 0 | 68.47 | 75.16 | 0.911 [0.163, 1.014] | 2.461 → 2.461 | True |
| a6000_bf16_pilot.json, stride 1, stage 1 | 60.69 | 63.92 | 0.950 [0.815, 4.428] | 2.007 → 2.007 | True |
| a6000_bf16_pilot.json, stride 8, stage 0 | 67.38 | 77.02 | 0.875 [0.195, 1.012] | 2.463 → 2.463 | True |
| a6000_bf16_pilot.json, stride 8, stage 1 | 64.03 | 60.53 | 1.058 [0.595, 1.226] | 2.007 → 2.007 | True |
| a6000_bf16_ce.json, stride 1, stage 0 | 73.97 | 119.22 | 0.620 [0.524, 0.782] | 2.129 → 1.547 | True |
| a6000_bf16_ce_precompute.json, stride 1, stage 0 | 41.66 | 50.94 | 0.818 [0.551, 4.953] | 2.129 → 1.547 | True |
| a6000_bf16_ce_padded256.json, stride 1, stage 0 | 102.20 | 66.98 | 1.526 [0.327, 1.658] | 3.058 → 1.769 | True |
| a6000_exclusive_ce_padded256.json, stride 1, stage 0 | 54.07 | 38.73 | 1.396 [1.350, 1.411] | 3.058 → 1.769 | True |
| a6000_exclusive_ce_unpadded256.json, stride 1, stage 0 | 54.82 | 51.03 | 1.074 [1.062, 1.097] | 3.058 → 1.769 | True |
