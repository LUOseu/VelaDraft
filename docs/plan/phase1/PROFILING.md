# Phase-1 limited profiling: teacher compute, feature bytes, I/O, training, checkpoints

Scope: report the cost shares of the recorded pipeline **by realistic visual and
text length**, using only already-recorded runs (no new GPU work; the plan's
budget was not charged for this document). Every number is tagged:

- **[M]** measured, with the run/log it came from;
- **[D]** derived from measured numbers by an explicit formula;
- **[F]** formula projection: the model is exact (bytes) or standard (FLOPs
  shape), the *coefficients are unmeasured* - the doc names the measurement
  that would pin them, and none of these numbers may be quoted as measured.

Reference hardware/protocol for all [M] numbers unless stated otherwise:
Qwen2.5-VL-3B-Instruct target, one RTX A6000 48 GB, bf16, offline feature
store, micro-batch 8, seq cap 2048, the matched 2K program (`docs/training-costs.md`)
and the WP4 profiling slice (`docs/plan/WP4_MEASUREMENT.md`, `R-20261014-12`).

## 1. Stage costs of the canonical 2K pipeline [M]

From `docs/training-costs.md` (base run file timestamps + `provenance.json` +
`train_summary.json`):

| Stage | Wall | Share of 70 min | Dominated by |
| --- | ---: | ---: | --- |
| prepare | ~20 s | 0.5% | dataset pool scan + media copy |
| regen (target generate) | ~23.5 min | 33% | 2000 prompts x <=160 generated tokens, batch 8 |
| capture (target forward + write) | ~2.6 min | 3.7% | 1050 rows teacher-forced forward, seq 2048 cap |
| train (792 steps, 8 samples/step) | 9.8-24 min | 15-33% | drafter forward/backward; shared-host wall variance |
| eval (128 prompts x <=128 tokens) | ~28 min | 40% | sequential verify: target forward per speculative block |
| **end-to-end** | **~65-80 min** | 100% | 1.1-1.3 GPU-h |

Teacher compute (regeneration + capture + eval verification) is **~77% of the
wall** at this protocol; drafter training is the minority stage. A length
argument alone does not move that split, because regen/eval costs scale with
answers, not with context length - the context-length lever acts on capture
bytes and on per-row forward cost (below).

## 2. Per-row unit costs at the measured length (~600 tokens, 2 images) [M]

`R-20261014-12` profiling (median of 4 passes; first-pass warmup dropped),
rows 600-608 tokens, seq 1024, 2 images per row:

| Step | Per row | Note |
| --- | ---: | --- |
| render: PIL decode | 2.0-2.4 ms | I/O + decode |
| render: processor (resize/normalize/tensorize + template) | 10-19 ms | CPU |
| render total (`prepare`) | 16.5-18.6 ms | measured boundary since R2 |
| teacher forward (teacher-forced prefill), group 1 | 139.3 ms | CUDA-synchronised, incl. finalisation |
| teacher forward, group 2 / 4 / 8 | 136.4 / 129.3 / 133.2 ms | batching removes launch/host overhead, not FLOPs |
| capture stage amortised (forward + write + media load) | ~149 ms | [D] 2.6 min / 1050 rows |
| prepared input tensors (for any prepared-input cache) | 4.6 MB | 968 patches x 1176 fp32 |

Length split inside the forward is **not measured** (vision tower vs text
prefill vs finalisation): the profiling script records totals per capture
group only. The bucketed re-profiling item (GPU_PENDING) is what would fit
`t_forward(L) = a_vision(pixels) + b_text * L` coefficients.

## 3. Feature bytes as a function of visual/text length [F for other lengths]

The store layout is one tensor per row with the actual `seq_len` (the index
records it; `capture.seq_length` is a truncation cap, not padding), bf16:

```
bytes(row) = L * (C + 1) * H * 2        # C captured context layers + 1 final hidden
```

with `H = 2048` (Qwen2.5-VL-3B), `C = 3` for the base store, `C = 5` for the
DFlare/DFly store. At the measured aggregate 9.4 GB over 1050 rows this
implies **~546 stored tokens/row** [D] (9.4e9 / (4 x 2048 x 2) = 573.7k
tokens). Per-token constants: 16 KB/token (C=3), 24 KB/token (C=5).

| Per-row length L | C=3 store | C=5 store | C=1 (minimal) |
| ---: | ---: | ---: | ---: |
| 512 tokens | 8.4 MB | 12.6 MB | 2.1 MB |
| 1024 tokens | 16.8 MB | 25.2 MB | 4.2 MB |
| 2048 tokens | 33.6 MB | 50.3 MB | 8.4 MB |
| 4096 tokens (multi-image / long doc) | 67.1 MB | 100.7 MB | 16.8 MB |

Corpus totals [F], same layout: 10k rows at 2048 tokens -> 336 GB (C=3) /
503 GB (C=5); 100k rows at 4096 tokens -> 6.7 TB / 10.1 TB. This is the
recorded store geometry, not the route document's 4096-hidden example; the
formula is the same and the conclusion does not change: full materialization
at "realistic document" scale does not fit a shared host (the audited host had
61 GB free at `RESOURCE_LEDGER.md` measurement time).

## 4. Training cost per algorithm and per sample [M]

`docs/training-costs.md`, matched 792-step / 8-samples-per-step protocol:
0.74-1.82 s/step (DFlash 0.74, DSpark 0.90, Domino 0.98, DFly 1.22, JetSpec
1.56, DFlash2 1.72, DFlare 1.82), i.e. 92-227 ms per sample at micro-batch 8;
peak 29.5-38.4 GB. The DFly recipe adds ~1.5x over its controlled_ce record
(LK cold-start stage + e2e TV forward work): uncontended ~1.9 s/step vs 1.22
(46.5 GB peak is the overfit-sanity batch; main loop 28.6 GB) [M,
`R-20261014-11`].

## 5. I/O shares (disk) [M + D]

| Artifact | Bytes | Share note |
| --- | ---: | --- |
| Feature store (3 layers + final / 5 layers) | 9.4 GB / 12.4 GB | per corpus, shared by every algorithm whose layer set is a subset |
| Run checkpoints: init (fp32 master) | 1.06 GB | written once |
| Run checkpoint: final | 0.53 GB | written once (the ~3.7 GB transient `ckpt_last.pt` is prunable) |
| Run dir (train+eval+export, store excluded) | 2-6 GB | - |

At 6 epochs the store is read ~6 times (per epoch): 56 GB of reads per
algorithm-run (C=3) [D]. The training wall on the shared host was dominated by
compute and by host contention (identical protocols vary up to ~1.7x, see
`training-costs.md`), not by an identified store-read bottleneck; a dedicated
IO-share measurement was **not** taken. Checkpoint bytes are ~17% of the store at
this scale and **independent of data length** (they scale with draft size),
so their relative share only falls as corpora grow.

## 6. Online mode, for contrast [M]

Same slice, live extraction in the trainer process: target resident 7.5 GB
(3B), train-phase target-device peak 13.67 GB, 16 -> 4 target forwards for 16
rows with capture group 4; extraction was ~2% of the training wall on this
tiny slice. Projected to a 2K online run, extraction is ~20% of step time
[the plan's earlier projection, `WP4_MEASUREMENT.md` §Decision]. Under online
mode the checkpoint set is identical; the store does not exist.

## 7. What this profiling does not cover (register, do not assume)

1. **Length-bucketed teacher timing** (vision pixels vs text tokens vs
   finalisation split): the profiler's `--length-buckets` per-row mode now
   provides the per-bucket attribution (CPU-tested, 2026-10-14); the runs at
   2-3 pinned lengths on the real target are GPU_PENDING #6 in
   `REVIEW_REMEDIATION.md`.
2. **Paired online/offline extraction re-timing under the unified R2
   boundary**: registered as GPU_PENDING #2 in `REVIEW_REMEDIATION.md`.
3. **Checkpoint write wall** (not just bytes): no write timing was recorded;
   the checkpoint bytes are [M] but the write share is unknown.
4. **2K-scale online extraction share**: the current 20% is a projection from
   a 16-row slice; do not quote it as measured.

Bottom line for planning: on the recorded protocol the dominant costs are
target generation (33%) and sequential verification (40%), both
*answer-length-driven*; feature storage scales with context length and is
what breaks first on longer visual inputs. Any "system efficiency" paper
claim must therefore separate answer-length effects (regen/eval) from
context-length effects (capture bytes, per-row forward), exactly as this
document does.
