# HYPOTHESES — VelaDraft system-optimization lane

Each candidate: mechanism → prior evidence → competing explanations →
related work → prediction (with falsifier) → budget → decision. Decisions only
change with recorded evidence in `RUNS.jsonl`; status labels are
KEEP / REVISE / PIVOT / STOP.

## H-A — decouple the teacher capture window from the drafter micro-batch

**Mechanism.** Today `OnlineBatchIterator` can only group
`min(train.online_batch_size, micro_batch_size)` rows per teacher forward
(verified in `online.py`). Decoupling: collect rows across several micro-batch
boundaries into one larger teacher forward (bounded window + byte cap), then
serve the records back in the original order; optionally prefetch the next
window while the current one trains. Claim: larger teacher batches raise
per-row teacher throughput (fewer calls, better matmul utilization), so
per-epoch extraction wall drops; optimizer-consumed samples/order/weights stay
identical.

**Prior evidence.** WP4 profiling (16 smoke rows, seq 1024, 3B target, A6000):
per-row forward wall 139.3 / 136.4 / 129.3 / 133.2 ms at groups 1/2/4/8 —
nearly flat (≈7% best gain at g4); per-forward time scales ~linearly with the
group. Implied effective throughput ~35-45 TFLOPS, far below the A6000 bf16
dense peak (~155), and batching does not move MFU in the measured range.
Call-count reduction is real (16→4) but call overhead is not the wall driver.

**Competing explanations for the flatness** (unresolved; cheap to check in S1):
(a) vision-tower cost per row dominates and batches poorly in this HF path;
(b) attention/SDPA kernel limitation at these shapes (bf16, mask layout);
(c) some per-row serialized step in finalization/positions (CPU-side);
(d) measurement artifact of median-of-4 small passes. A group sweep at g16/32
on real corpus rows with kernel-attribution (torch profiler) distinguishes
(a)/(b) from (c)/(d).

**Related work.** Independent batching / bucketing / prefetch are standard;
SpecForge/Speculators/TorchSpec stream features at their own batch sizes
(hard evidence that decoupling is a solved *shape* problem in disaggregated
trainers). For this lane the question is only whether it pays on this
single-process, single-GPU placement.

**Prediction.** On real 2K-corpus rows (~546 tokens, 1 image): larger windows
(g16+) improve per-row forward wall by <10% (probably ~0-5%); extraction wall
−<10%; end-to-end −<2%. Batch-count drop is real and free. **Falsifier:** a
sweep showing ≥15% per-row improvement at g16/g32 on real rows would reopen
this as a real mechanism (then implement the windowed iterator).

**Budget.** ~0.05-0.1 GPU-h (sweep only, reuses `scripts/profile_online_extraction.py`).
**Paper value.** None by itself (standard technique).
**Closure (S1b, 17:52-17:56, 129 real rows, GPU0).** Extraction wall
16.36 -> 14.18 s (-13.3%) g1 -> g16, saturated by g8 (sample ms 126.8 ->
110.0); render cold/warm 14.31/12.43 ms/sample; target peak 7.9 -> 10.8 GB
(`logs/s1b_profile.json`; RUNS `S1b-profile-20261014-1752`). Extraction is
~16-20% of online e2e, so even the saturated term is <3% e2e — the predicted
flatness holds on real rows. **Decision: STOP as a mechanism** (guidance
only: keep `online_batch_size` as a small knob; do not build the windowed
iterator).

## H-C1 — bounded host-RAM retention of extracted records across epochs (primary)

**Mechanism.** In online mode the same frozen teacher produces the same record
for a given row in every epoch (fixed target, fixed regen answers, eval-mode
determinism, identical batch grouping from the fixed seed). Today those CPU
records are discarded after use and re-extracted each epoch.
Change: keep extracted records in a byte-bounded host-RAM store (keyed by
sample index; LRU with recompute-on-miss; runtime-only, never persisted;
size cap configurable, default off). Epoch 1 extracts; epochs ≥2 serve from
RAM. Optimizer-consumed samples, order, grouping and numerics unchanged; the
cache is execution state, not part of the source identity (digest/plan gates
untouched; resume re-fills).

**Prior evidence.** (1) Extraction ≈ 2.5-2.6 min/epoch × 6 epochs ≈ 15-16 min
on the 2K protocol (from capture-stage wall), ≈20% of online end-to-end;
(2) records are already materialized as detached CPU tensors
(`OnlineFeatureDataset._normalize`) — retention is bookkeeping, not new
transport; (3) total record bytes for the 2K corpus ≈ 9.4 GB (same bytes as
the offline store) vs 197 GB free server RAM; (4) disk is 100% full — the
offline store cannot be relied on as the alternative.

**Competing explanations / risks.** (a) Wall gain could be eaten by RAM
traffic/GC on a shared host — measure paired, interleaved. (b) Records could
differ from re-extraction (bf16 nondeterminism via different grouping?) —
records are epoch-invariant only if grouping is identical across epochs
(same seed/sampler ⇒ same groups; verified by the existing DataLoader-order
parity tests); the A/B loss sequence must be bit-identical or the mechanism
is wrong. (c) Bigger corpora / 8B targets exceed a sane RAM cap — LRU degrades
to today's behavior (recompute); cap must be recorded in provenance. (d) It
duplicates an existing option: an offline store with page-cache reads is close
in wall terms *when the disk is healthy and the store exists*. Measured
2026-10-14 (REPORT §4.6): store reuse trains in 548.8 s vs the cache's
470.9 s train stage, and the store cycle costs capture + 7.0G disk — on this
host the disk-free variant is stronger, not merely equal.

**Related work / A+B card (doc §2.2).**
- **A (mature work)**: SpecForge disaggregated training (docs: bounded
  in-flight feature bytes, optimizer-boundary delivery, no per-epoch
  re-extraction; ICML 2026) and Speculators' online feature transport.
- **B (our need)**: single-process online training whose only feature
  "transport" is the trainer's own CPU tensors; re-extraction each epoch is
  pure waste once bytes exist; disk store infeasible (100% full).
- **Match**: A proves bounded feature retention + single-shot delivery is
  production practice; B is the degenerate single-node case where the
  "transport" is a dict. Minimal adaptation: byte-bounded retention window
  inside `OnlineFeatureDataset` with LRU eviction, feature-mode provenance.
- **Failure handling**: if the cap cannot bound tail memory, fall back to
  per-epoch re-extraction (no behavior change). Do not build a scheduler.

**Prediction.** Extraction wall for epochs ≥2 → ~0 (only LRU misses); total
online wall −(E−1)/E of extraction ≈ −13 min on the 6-epoch 2K run (≈ −15%
end-to-end); losses bit-identical; peak host RAM +≈9.4 GB (2K corpus).
**Falsifier:** paired wall gain <5% (noise), or any loss/record mismatch, or
RAM growth unbounded → REVISE/STOP.

**Result (S2, 17:34-17:52, 129 rows x 6 epochs, off/on x3).** Wall win
confirmed: train stage 151.6 s -> 74.4 s mean (-50.9%; pair 1 -58%),
extraction -82.5%, 645 hits / 129 misses / 0 evictions at 8 GiB (+0 disk,
+830 MiB host RAM under the existing process high-water mark). The
as-written bit-parity gate failed (3/3 ckpt sha mismatches), but the drift
is deterministic (all OFF legs identical; all ON legs identical) and bounded:
loss identical 102/102 steps; grad_norm drifts <=6.5e-4 from the first hit;
weights differ on 6.3% of elements, max |delta| = 2^-7 (unit-scale 1 ulp;
per-element ulp mix in REPORT 4.5).

**Causal test (S2c, 18:09-18:24).** With a fixed epoch order the ON leg
reproduces OFF bitwise (sha256(ckpt_final) match `52d25648...`, all six log
fields identical across 102 steps; wall win reproduces, -62.9%). So the
cache is value-exact; the S2 gap comes from the shuffled loader changing the
extraction grouping every epoch (extractor is shape-sensitive at the
bf16-rounding scale, probe 2), which the cache freezes at epoch-1 groupings.
(Correction to risk (b): groups are NOT identical across epochs under the
default shuffler — that premise was wrong; kept visible per the RSI rules.)

**Result (S3, canonical 2K, 18:34-19:29, 1012 regen rows x 6 epochs = 762
steps).** Paired A/B (cache off vs 16 GiB on, same data/seed): train-stage
wall 1167.0 -> 470.9 s (**-59.7%**); extraction 799.1 -> 125.5 s (-84.3%,
762 -> 127 target forwards); cache 5060 hits / 1012 misses / 0 evictions at
16 GiB (peak 6.35 GiB); zero disk bytes; GPU peak unchanged; host RSS +3.97
GiB. Numeric: valid_tokens 762/762 identical; loss identical 740/762 (rest
<=2 display ulp); grad_norm first diverges at the first cache-hit step;
accuracy max |delta| 0.0032, accept_len 0.018; final weights 90.4% elements
differ, max |delta| 0.0446, global relL2 4.26% (multi-ulp at 762 steps;
deterministic; mechanism = frozen epoch-1 grouping). Closing frame 1 DONE
(20:15-20:34): acceptance eval a_off vs b_on - mean_al identical to 10
decimals (1.5262066923), accepted/new tokens identical (5288/15462),
token_exact=True for both, 1 draft token in ~69k differs -> functionally
equivalent downstream. Closing frame 2 (seed change: order + init) DONE:
cache replication at seed 43 (train wall -57.2%, valid_tokens identical,
loss 757/762 identical, drift relL2 1.45% vs 4.26% at seed 42) and the
reference frame a-vs-c: relL2 131.5% / last-10 loss Delta 0.345 -> the
cache moves the run 30-90x less than a routine seed change.

**Budget.** Spent ≈3.2 GPU-h of the lane's 16: S1/S2/S2c/probes 0.86 +
S3 (incl. data build) 0.91 + closing frames 1.06 + offline baseline 0.39.
**Paper value:** engineering adaptation;
no new mechanism (LRU + retention is standard). The reusable artifact is the
equivalence-boundary protocol (freeze grouping or bound the drift) with its
probe/S2c methodology. **Decision: KEEP** — mergeable with the
numeric-behavior note and the amended gate (REPORT §4.5); canonical 2K
number now measured (-59.7% train stage); closing frames: acceptance eval
DONE (functionally equivalent, mean_al identical to 10 decimals);
seed-change control DONE (drift 30-90x below a seed change). Offline
baseline (goal §8.3) DONE 2026-10-14: store 7.0G; e end-to-end 836.8 s;
e2 reuse train 548.8 s vs cache b_on 470.9 s -> the cache path beats store
reuse by 14.2% at zero disk; e/e2 train logs byte-identical (store
round-trip lossless).

## H-C2 — fuse regen with capture (deferred)

**Mechanism.** Regen already runs the target over prompt + generated tokens;
capture later re-runs a teacher-forced prefill. TIDE/SpecForge-style reuse:
capture prompt-prefill and per-decode-step hidden states during generation,
stitch records, no separate capture pass.

**Prior evidence / risks.** Verified: regen writes token ids only; capture
recomputes everything. But (1) decode-with-KV vs prefill kernels are not
bit-equal in bf16 (docs and task doc both warn); the repo's culture gates on
bitwise online↔offline update alignment (GPU smoke 5); (2) missing
final-token/EOS states need patching; (3) regen runs batched greedy generate —
capturing intermediate layers per decode step adds transfer + may disturb
generate; (4) saving bounded to the capture pass cost (~2.6 min) once C1
exists. Multi-image/MRoPE/late-arriving-image visibility all need proof.

**Prediction.** Ceiling ≈ capture-pass cost (~2.6 min one-off, ≈3-4% e2e)
minus added regen overhead; numeric-equivalence argument will eat the effort.
**Decision:** DEFERRED. Reopen only if (i) C1 lands and (ii) the capture pass
remains on the critical path after C1 (e.g., single-epoch runs, or
regen-cost-dominated pipelines). Requires tolerance analysis with a fixed
transcript before any implementation.

## H-C3 — split-device producer/consumer overlap (deferred)

**Mechanism.** Target on a second GPU, drafter on the first (supported:
`target.device=cuda:1`, `train.device=cuda:0`); overlap teacher extraction of
window n+1 with drafter train of window n (bounded double buffer). Today the
two devices alternate idle.

**Prior evidence / risks.** Per-epoch extraction (~2.5 min) vs per-epoch train
(~1.6-4 min) are comparable, so overlap could hide ~min(extract, train) per
epoch; but (1) device-hours rise (2 GPUs busy), and the doc requires comparing
at equal device count + reporting all-device GPU-hours; (2) no second GPU is
free on this server (GPU0-3 all held at 13:20); (3) C1 removes most
re-extraction anyway (overlap only helps epoch 1 + misses). **Prediction:**
wall win, GPU-hour loss; not compelling at equal device-hours once C1 lands.
**Decision:** DEFERRED. Only reopen with an explicitly budgeted 2-GPU slot
and after C1, comparing against the same 2-GPU budget spent on 2 independent
single-GPU runs.

## Cross-candidate measurement design (S1/S2)

- **Paired, interleaved legs** on identical data/regen/seed/config; 3 paired
  steady windows where the run length allows; medians + ranges.
- **Numeric gates (before any wall claim):** identical loss sequences
  (bitwise for C1 legs), identical record bytes (hash a sampled subset),
  unchanged digest/plan/provenance fields.
- **Workloads:** (S1) pinned smoke slice for the R2 unified-boundary
  re-measurement + group sweep; 32-64 real-corpus rows for the realistic
  sweep. (S2) 128-256 real-corpus rows × 6 epochs for the C1 A/B.
- **Reported quantities:** stage walls (prepare/regen/extract-split/train/eval),
  per-epoch extraction wall, count of teacher forwards/samples, peak GPU and
  host RAM, disk bytes written, device + contention state, all-device
  GPU-hours. Wall numbers without device/contention context are not evidence.
