# Experiment B: measured attention capacity

All 18 FP32-master cells passed; none OOM. Each cell has 5 pilot and 3 measured full optimizer updates. Model: Qwen2.5-VL-3B geometry, DFlash 3 layers, hidden 2048, FFN 11008, 16 query / 2 KV heads, head dimension 128, block 8, microbatch 1, BF16 forward/backward with production FP32-master AdamW.

Natural data: select one longest complete sample from 64 fixed natural records. Its length is 825; context buckets 1024/4096 zero-pad the context without changing supervision. Requested 256/512 anchors both cap at 249. Shape-only rows use synthetic inputs/features and the actual frozen target head; these are separate capacity diagnostics, not natural-data training.

Checkpoint scope: mask materialization plus SDPA only; K/V head repetition stays outside. No projection, MLP, whole-layer, or loss checkpoint. Dense comparators use the same lazy-mask and explicit SDPA/GQA dispatch as chunked, materializing a full mask per layer; these are harness baselines, not stock HF dense-mask-once trainer throughput.

| Source | Requested C/A | Actual C/A | Backend | Chunks/call | Q / supervised rows | Mean update ms | Peak allocated HBM GiB | OOM |
|---|---:|---:|---|---:|---:|---:|---:|---|
| natural | 1024/64 | 825/64 | dense | 1 | 512/448 | 54.60 | 5.606 | no |
| natural | 1024/64 | 825/64 | dense_checkpoint | 1 | 512/448 | 68.89 | 5.610 | no |
| natural | 1024/64 | 825/64 | chunked_checkpoint | 4 | 512/448 | 112.05 | 5.610 | no |
| natural | 4096/256 | 825/249 | dense | 1 | 1992/1743 | 152.43 | 8.862 | no |
| natural | 4096/256 | 825/249 | dense_checkpoint | 1 | 1992/1743 | 182.45 | 7.137 | no |
| natural | 4096/256 | 825/249 | chunked_checkpoint | 16 | 1992/1743 | 297.62 | 6.537 | no |
| natural | 4096/512 | 825/249 | dense | 1 | 1992/1743 | 152.43 | 8.862 | no |
| natural | 4096/512 | 825/249 | dense_checkpoint | 1 | 1992/1743 | 183.18 | 7.137 | no |
| natural | 4096/512 | 825/249 | chunked_checkpoint | 16 | 1992/1743 | 306.38 | 6.537 | no |
| shape | 1024/64 | 1024/64 | dense | 1 | 512/448 | 56.42 | 5.606 | no |
| shape | 1024/64 | 1024/64 | dense_checkpoint | 1 | 512/448 | 68.82 | 5.610 | no |
| shape | 1024/64 | 1024/64 | chunked_checkpoint | 4 | 512/448 | 112.23 | 5.610 | no |
| shape | 4096/256 | 4096/256 | dense | 1 | 2048/1792 | 127.58 | 9.016 | no |
| shape | 4096/256 | 4096/256 | dense_checkpoint | 1 | 2048/1792 | 152.13 | 7.238 | no |
| shape | 4096/256 | 4096/256 | chunked_checkpoint | 16 | 2048/1792 | 309.31 | 6.607 | no |
| shape | 4096/512 | 4096/512 | dense | 1 | 4096/3584 | 268.82 | 15.764 | no |
| shape | 4096/512 | 4096/512 | dense_checkpoint | 1 | 4096/3584 | 333.63 | 11.687 | no |
| shape | 4096/512 | 4096/512 | chunked_checkpoint | 32 | 4096/3584 | 550.48 | 9.085 | no |

Small natural and shape-only output/loss/all-parameter-gradient comparisons pass the existing H1 BF16 envelope (atol=rtol=0.02). Dense checkpoint is bitwise equal to dense on every compared tensor. Chunked maximum absolute differences: natural 0.0078125, shape-only 0.000244140625; detailed per-tensor errors are retained in correctness JSON.

The initial correctness artifact copy/save is included in the first pilot update, never the three measured updates. First-five timing estimates are retained per cell. Earlier direct-BF16-optimizer results are archived under `preflight_BF16optimizer_not_final/` and excluded from this table.

Sum of final cell runtimes (setup + updates, excluding Python startup) = 508.1 seconds. All final cell core-source hashes match. Raw JSON and matrix.log/matrix.exit are adjacent; summary.json includes timing ranges and per-cell provenance.

NPU1 was exclusive to B; recipe work on NPU0 shared the host CPU/RAM. This may affect update timings. Results show a memory/time tradeoff, not a speedup or an OOM capacity boundary: even the dense 4096/512 shape fits. Against dense checkpoint at that shape, chunking reduces peak allocated HBM by 22.3% and increases mean update time by 65.0%.
