# STATE — VelaDraft system-optimization lane (`research/online-systems`)

Task source: `VelaDraft_System_RSI_Goal_20261014.md` (user-provided, 2026-10-14).
Started: 2026-10-14 (Asia/Shanghai). This file is self-contained; do not assume
any other agent's context.

## Code baseline

- Branch: `research/online-systems`, worktree `/Users/Admin/grounding-parallel/veladraft-systems`
  (off the shared `vlm-dflash` clone; created 2026-10-14 ~13:10).
- Baseline SHA: `e15256d7c4bd10a490a7ac821eb12a0749a96fdd` = `origin/main`
  ("Merge pull request #4 from LUOseu/review-remediation", 2026-10-14 11:18 +0800).
  This equals the baseline named in the task document. Local `main` worktree
  (`vlm-dflash` @ `38c2260`) is an ancestor; do not read it as current.
- Server working copy still to be created (plan: `/home/lyh/VelaDraft-systems`
  on `a6000xuyang`, clone of the same SHA; not yet done as of 14:05).
- Other sessions' worktree (`vlm-dflash-recipe`, branch `review-remediation`)
  had uncommitted `vlm_dflash/objectives.py` edits at 13:00 — not mine, do not
  touch. Its branch has 3 commits not on `origin/main` (f904953, 7e89240,
  c5a32e5); irrelevant to this lane except as coordination context.

## Doc-claim verification (task item 1, 2026-10-14)

All six claimed behaviors verified in code at `e15256d`:

| Claim | Verified where |
| --- | --- |
| online ≠ pipelined 3 stages; regen still pre-generated | `docs/online-training.md`; `cli.py::cmd_train` runs prepare → regen → (capture unless online) → train |
| capture group ≤ min(online_batch_size, micro_batch_size) | `vlm_dflash/target/online.py::OnlineBatchIterator` (`capture_batch` split loop); docstring states `min(...)` |
| GPU target in main process, `num_workers=0` required | `OnlineFeatureDataset.require_main_process`; guard in `__getitem__`; CLI + `dflash/train.py` checks |
| full teacher-forced forward → select layers → CPU | `forward_and_capture_batch`: `output_hidden_states=True`, `logits_to_keep=1`, per-row `cat` of layers then `.to(dtype).cpu()` |
| regen records tokens only, no hidden states | `regen.py` writes `regen/target_data.jsonl` with `full_ids`; no hidden capture anywhere (grep) |
| old -18.8% extraction-wall number invalid; unified boundary implemented, paired re-measurement pending | `docs/online-training.md` R2 note; `online.py` `prepare_seconds`/`forward_seconds` with `_sync_target()` around each stage; `docs/plan/WP4_MEASUREMENT.md` R2 correction |

## Resources & budget (measured 2026-10-14, times Asia/Shanghai)

- GPU server `a6000xuyang` (ssh `oab-remote`, 10.65.14.8, user lyh):
  4× RTX A6000 48 GB.
  - 08:19-13:20+: user's unrelated active job ("gateC", MBPP eval,
    `fpt_work/third_party/RandOpt-main`) holds GPU0/1/2 (~41.7 GB each,
    util 70-100%); GPU3 held by another user (`dongzp`, sglang-spec, ~44 GB,
    ~100% util). **No GPU available for this lane at 13:20.** gateC progress
    1065/5000 parsed at 12:55 (~4/min recent rate); ETA uncertain (hours).
  - Server RAM: 252 GB total, ~197 GB available, 180 GB page cache.
  - Disk `a6000xuyang:/home`: 15 TB, **28 GB free, 100% used** and drifting
    (61 GB at 02:55 → 28-32 GB at 13:00). Wall-clock on this host is
    contaminated by writeback stalls (documented in
    `docs/training-costs.md`: identical protocols vary up to 1.7x; E4 run hit
    a host-wide stall). Treat storage as a first-class constraint of this lane.
- Local Mac (M4, 16 GB): CPU tests/docs only.
- Budget: task doc suggests this lane may take ≤16 of the route's 32
  suggested local GPU-hours, reusing existing authorization, no per-run
  re-confirmation. Lane ledger so far: ~0.2 GPU-h spent (16:47 chain attempt + flex repro; per-run detail in RUNS.jsonl).
  Coordinate devices/times with other agents (channel used by previous
  sessions: `docs/plan/COORDINATION.md` / `RESOURCE_LEDGER.md` — that ledger's
  8 GPU-h cap belongs to the earlier `VelaDraft_01_PLAN.md` program, not this
  lane; do not double-count or silently charge).
- Device policy for this lane: one job at a time, ≤ ~2 h wall each; check
  `nvidia-smi` immediately before launch; leave another user's GPU3 alone;
  prefer a device with 0% util and >40 GB free; record device + contention in
  `RUNS.jsonl`; keep net-new disk bytes small (prune own `ckpt_last.pt` when
  a run's evidence is extracted).

## Pinned workload (small real subset + reliable recipe)

- Recipe: `configs/smoke_multiturn.yaml` (16 synthetic 2-image/2-turn rows,
  seq 1024) — the WP2/WP4 pinned smoke; and the canonical real 2K corpus
  recipe `configs/qwen2_5_vl_3b_2k.yaml` (LLaVA-Instruct pool → 2000 train /
  128 val, regen keeps 1050 rows ≥128 tokens; seq ≤2048; micro-batch 8;
  6 epochs = 792 steps; DFly/DFlash trainer).
- Target: `/home/lyh/vlm-dflash/assets/models/Qwen2.5-VL-3B-Instruct`
  (already on the server; 7.5 GB resident bf16).
- Existing run evidence (read-only for this lane): `/home/lyh/vlm-recipe-smoke/`
  (E1/E2/E4/WP4 runs; `wp4_ab1/ab4` logs; `~/recipe-smoke-config/wp4_*.log`);
  `/home/lyh/vlm-dflash/runs/` (smoke-1..7 run dirs).
- Real 2K feature-store stats (from `qwen25vl3b_dflare_2k/features/index.jsonl`,
  1050 rows): seq_len mean 546 / median 543 / p95 620 / max 727; image tokens
  mean 358 of which ~188 text; answer tokens mean 152. Store (3 layers +
  final hidden) 9.4 GB; 5-layer variant 12.4 GB.

## Cost model (measured evidence, pre-remeasurement)

From `docs/training-costs.md` (2K base run, 2026-10-11) + `WP4_MEASUREMENT.md`
(2026-10-14, pre-R2 timing boundary):

- Offline end-to-end (one A6000): prepare ~20 s; regen ~23.5 min (largest
  target-compute block); capture ~2.6 min (1050 rows); train 9.8-24 min
  (0.74-3.34 s/step, algorithm-dependent); eval ~28 min; total 65-80 min.
- Online (current) pays capture-equivalent extraction **per epoch**:
  ~2.5-2.6 min/epoch × 6 epochs ≈ 15-16 min extra, no 9.4 GB store. Implied
  end-to-end ≈ 78-93 min; extraction ≈ 20% of end-to-end (matches the
  doc's projection).
- Smoke-slice profiling (16 rows, seq 1024, group medians): render
  15-18.6 ms/row (warm 15.1); teacher forward per row 139.3 (g1), 136.4 (g2),
  129.3 (g4), 133.2 (g8) ms — **per-row wall nearly flat across group size**
  (best gain ~7% at g4); per-forward wall scales ~linearly with group size.
  Also measured: ~35-45 TFLOPS effective on the 3B forward (well below the
  A6000 bf16 dense peak ~155), batching does not improve that MFU in the
  measured range → candidate A's ceiling is small **on this hardware**
  (verify at g16/32 on real rows before closing).
- Train wall on the smoke is dominated by resumable checkpoint writes
  (28-49 s each on the 100%-full disk), not by steps or extraction; paired
  A/B walls are noise-dominated → any A/B must mirror the real protocol
  (bigger slice, fewer/skips ckpt writes) to be discriminative.

## Working hypotheses (details + predictions in HYPOTHESES.md)

- H-A (doc candidate A; decoupled teacher window): predicted weak on this
  HW (flat per-row forward); close out via group sweep before any code.
- H-C1 (primary): bounded **host-RAM retention of extracted records across
  epochs** in online mode — skip re-extraction for epochs ≥2 (bounded bytes,
  LRU + recompute on miss, runtime-only). Predicted: −(E−1)/E of extraction
  wall ≈ −13 min on a 6-epoch 2K online run (≈ −15% end-to-end), bit-identical
  losses (records are the same tensors), zero disk footprint. Falsifiers:
  losses differ; wall gain ≤ noise at paired measurement; RAM cap forces
  thrash; page-fault/GC costs eat the gain.
- H-C2 (deferred): fuse regen+capture into one target pass set (TIDE/SpecForge
  style) — numeric-equivalence risk (KV-decode vs prefill kernels), cost
  saving bounded by the capture pass (~2.6 min) once C1 lands.
- H-C3 (deferred): split-device producer/consumer overlap — needs 2 GPUs;
  must be compared at equal device count vs serial placement; on this shared
  server no second GPU is free, so not first-round.

## Decisions log (this lane)

- 2026-10-14 ~13:45 — D1: verified all doc claims; no code behavior changed
  vs the doc. Baseline recorded.
- 2026-10-14 ~13:50 — D2: no GPU available (gateC + dongzp). Do S0/S1-prep on
  CPU: records, cost model, run scripts, sweep design; GPU results stay
  pending. No sleep/mock "speedup" evidence will be produced.
- 2026-10-14 ~13:30 — D4: full `tests/test_online_mode.py` (65 tests incl.
  the 7 new cache tests) passes on the server (exit 0, log
  `online_full_tests_v3.log`; the all-fail `cache_tests.log` entry was a
  pre-fix iteration). 0 GPU-h spent so far.
- 2026-10-14 ~13:40 — D5: run-script corrections before first launch:
  (a) `run_s1a`: `run_leg` no longer deletes the staged `ab4` data copy
  (mirrors the original `wp4_ab.sh` reuse pattern); (b) `run_s2`: legs now
  repeat `--limit 256` because stage digests include `limit` (a mismatch
  would *refuse* training, not silently rebuild); (c) new `run_s3_ab.sh`:
  canonical 2K paired A/B whose master data+regen is built inside leg A and
  reused by leg B through the copied stage manifests; gated on S2
  cache-hits + ckpt parity; 12 GiB disk guard; harvest + ckpt_init prune
  built in; (d) `analyze_ab.py` prints the paired gate report (loss
  sequences, sha256 parity, extraction deltas); (e) S2 runs 3 interleaved
  off/on pairs (doc §9 asks ≥3 paired windows) and S2/S3 legs record host
  RSS with `/usr/bin/time -v`; (f) waiter restarted with a 20 h horizon
  (until ~09:40 on 10-15) and `LANE_STATUS.md` left in the lane root.
- 2026-10-14 ~14:56 — D7 (incident, self-inflicted, 0 GPU-h lost): an
  `ssh`-quoted heredoc mistake at 13:41:11 made the remote shell execute the
  backticked lines of a draft note — it created `STOP_WAITER` and ran
  `screen -S syslane -X quit`, silently killing the 20 h waiter. Detected at
  14:54; removed the spurious file and relaunched the waiter (screen
  `syslane`, 30 h from 14:56, i.e. until ~20:56 on 10-15). No runs existed to
  lose; chain scripts unchanged. Lesson applied: notes/scripts are written
  locally and `scp`-ed, never heredoc-ed through `ssh` quoting.
- 2026-10-14 ~13:45 — D6: `cli train` never runs eval (eval is a separate
  subcommand), so leg walls are prepare/regen + extraction + drafter train
  only. Canonical 2K A/B cost: leg A builds the master (prepare+regen
  ≈ 27 min) and trains with cache OFF; leg B ≈ 10-12 min with cache ON.
  Full chain S1a→S1b→S2→S3 ≈ 1.5-2 h GPU when a device frees up.
- 2026-10-14 ~14:00 — D3: candidate ranking for the first round:
  C1 (RAM retention) primary; A (decoupled batching) closure-only via sweep
  (predicted STOP unless g≥16 shows >10% extraction gain); C2/C3 deferred
  with explicit conditions.

## Next actions (in order)

1. Write HYPOTHESES.md, RUNS.jsonl, REPORT.md skeletons (this session).
2. [DONE 13:12-13:45] Server lane set up; waiter armed (`screen -r syslane`,
   `watch_gpu.sh`, 10 h from 13:12 Asia/Shanghai); chain scripts shipped and
   syntax-checked; S3 gate + harvest tooling in place.
3. S1 (GPU, ~0.2-0.3 GPU-h when available): (a) paired per-sample vs group-4
   unified-boundary re-measurement on the pinned smoke slice (the R2
   re-measurement the repo docs wait for); (b) extended sweep
   groups 1,2,4,8,16 on 32-64 real-corpus rows (~546-token rows) + memory;
   (c) stage timeline of a fresh online run vs resume (startup wait split).
4. S2 (GPU, target ≤1.5 GPU-h): C1 minimal implementation + paired A/B on a
   128-256-row real slice × 6 epochs (cache OFF vs ON), interleaved repeats;
   numeric gate = bit-identical loss sequence and record digests.
5. When the waiter fires: watch `logs/{s1a,s1b,s2,s3}_progress.log`; run
   `python3 /home/lyh/vlm-systems/analyze_ab.py`; paste into REPORT.md
   §4-§5; append RUNS.jsonl entries (device + contention + disk state).
6. Update REPORT.md with results; decide KEEP/REVISE/PIVOT/STOP per candidate;
   engineering judgement + paper judgement.

## Resume commands

```bash
# repo/worktree
cd /Users/Admin/grounding-parallel/veladraft-systems   # branch research/online-systems @ e15256d
# GPU status / contention check
ssh oab-remote 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; df -h /home | tail -1'
# existing evidence (read-only)
ssh oab-remote 'cat ~/recipe-smoke-config/wp4_profile.log; cat ~/vlm-recipe-smoke/runs/wp4/ab1/train/train_summary.json'
# server-side lane root (once created)
ssh oab-remote 'ls /home/lyh/VelaDraft-systems /home/lyh/vlm-systems'
```

Timing discipline for every future run: diagnostic counters only where the
platform already syncs (`prepare_seconds` / `forward_seconds`); whole-run wall
in performance mode; paired legs interleaved; ≥3 steady windows where
feasible; record device, contention and disk state in `RUNS.jsonl`.

## 17:2x — post-chain triage (16:47 chain) and unblocks

The 16:47 S1a->S2->S1b->S3 chain failed within ~5 min (S3 correctly
self-gated); zero usable GPU results from that attempt.

- Blocker A (flex-attention Inductor), root-caused 17:15: torch 2.11 picks the
  flex_decoding kernel for q_len<128 + GQA (ratio 8; BlockMask with H=1). There
  `BLOCK_M = next_pow2(q_len * G) = 1024 > SPARSE_Q_BLOCK_SIZE (128)`, and the
  candidate filter `SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0` rejects every Triton
  config -> `NoValidChoicesError`. Repro matrix (logs/flex_repro_v2.log, GPU0):
  base FAIL, dynamic=False FAIL, `MAX_AUTOTUNE_GEMM_BACKENDS=ATEN` FAIL;
  `FORCE_USE_FLEX_ATTENTION=True` OK, `BACKEND=TRITON` OK, q_len=16 OK.
  Workaround: env-gated patch `VLM_DFLASH_FLEX_FORCE_MAIN=1` in
  `vlm_dflash/dflash/draft_model.py` (passes kernel_options
  `{"FORCE_USE_FLEX_ATTENTION": True}`); exported by all lane scripts; both
  A/B legs share it. It only changes shapes that would otherwise crash
  (q_len>=128 already used the main kernel in normal runs).
- Blocker B (repo/HEAD): LLaVA-Instruct pool fails `check_v1_question_layout`
  at HEAD e15256d — 78752/157712 rows (49.9%) are text-before-image; the
  2026-10-11 canonical run predates the check, so the canonical recipe cannot
  be rebuilt at HEAD without a filtered pool. Resolution:
  `data/llava_v1safe.json` (78960 rows, 100 MB); run_s2/s1b/s3 point
  `VLM_DFLASH_LLAVA_JSON` at it; pool delta documented; no repo code change.
- Script fixes before relaunch: exit-code capture (`ec=$?`; the earlier fix
  still had top-level `local ec=` bugs in S1b/S2 build blocks), build-failure
  guards that abort dependent stages, S1a disk guard (>=4 GiB per leg).
- Disk: /home free has been 3-8 GiB all afternoon (host-wide 100% full); lane
  writes are ckpt-bounded (~1.5 GiB per 2-leg pair, pruned after harvest).

## 17:31 — S1a complete; chain relaunched

- S1a rerun 17:28:40-17:30:30 (GPU0, screen `s1a`): ab1 + ab4 both exit=0.
  R2 paired numbers (16-row smoke, 1 extraction pass): prepare 0.719/0.748 s,
  forward 2.209/2.269 s, extraction 2.928/3.017 s for g1/g4 -> group-4 is
  ~+3.0% (slower), pre-R2 "-18.8%" stays invalid. (First attempt 17:25: ab1
  ok, ab4 skipped by the new 4-GiB guard when the host disk touched 2 GiB;
  guard worked as intended.)
- Lane patches for the hostile disk: `VLM_DFLASH_SKIP_RESUME_CKPT=1`
  (train.py; skips ckpt_last writes, ~3.4 GiB/leg transient removed; identical
  both legs) in addition to `VLM_DFLASH_FLEX_FORCE_MAIN=1` (draft_model.py);
  both exported by all lane scripts. S1a/s2/s3 guards lowered to 4-5 GiB
  (per-pair peak now ~3 GiB). Disk during S1a: 2-7 GiB free (host-wide 100%).
- Chain started 17:30:49 in screen `chain` on GPU0: S2 -> S1b -> S3. S2 build
  (v1safe pool, --limit 256) exit=0 at 17:34:50; s2_a1_off started.

## 18:05 — S2 (cache A/B) + S1b sweep complete; drift investigation

- Chain (screen `chain`, GPU0) finished 17:56: S2 3 pairs -> S1b profile -> S3
  self-gated "SKIPPED: S2 gate failed" (expected on sha mismatch).
- S2 result (129 real rows, 6 epochs / 102 steps, interleaved off/on x3):
  train-stage wall OFF 158.2/145.2/151.4 s vs ON 66.3/79.0/78.0 s
  (mean -50.9%; pair 1 -58%). Extraction wall OFF ~95.8 s mean vs ON ~16.8 s
  (epoch 1 only). Cache: 645 hits / 129 misses / 129 stores / 0 evictions,
  peak 830 MiB vs 8 GiB cap, zero extra disk. H-C1 wall win CONFIRMED
  (paired, 3 repeats, mechanism-level).
- Numeric gate: all 3 pairs ckpt_final sha256 MISMATCH. But a1=a2=a3 and
  b1=b2=b3 bitwise (weights + logs): the drift is deterministic and
  systematic, not run noise. Loss identical in all 102 steps (bf16 print);
  grad_norm first differs step 18 = first cache-hit step (max|delta| 6.5e-4);
  metrics differ from step 27 (~4e-3); final weights differ on 15.33M /
  243.81M elems (6.286%), every delta <= 1 bf16 ulp (max 0.0078125), worst
  tensor layers.1.self_attn.k_norm.weight (`wdiff_ckpt.py`, logs/wdiff_a1_b1.txt).
- Mechanism probes (GPU2 idle; logs/probe2.log): a row's extracted hidden is
  bitwise sensitive to batch path/shape (its input_ids/loss_mask/position3
  equal): alone vs 2-row max_abs 6.0, vs 8-row 48, 2-row vs 8-row 48, ~78% of
  elements differ, at bf16-rounding scale amplified through depth; membership
  at fixed shape and call order do not matter (probe 1). Matches the
  online_source_digest note "batch composition changes the padding/numerics".
- Working root cause of the S2 gap: RandomSampler reshuffles every epoch, so
  epoch-k grouping != epoch-1 grouping; the cache serves epoch-1-computed
  values in epochs >=2 while the OFF leg recomputes under the epoch-k
  grouping -> last-bit drift from the first hit onward. The bit-parity gate as
  written assumed per-epoch grouping stability that the shuffled loader does
  not provide.
- S2c (screen `s2c`, GPU0, started 18:09:11): fixed-order pair
  (VLM_DFLASH_TRAIN_SHUFFLE=0, new env gate at dflash/train.py's main
  shuffle=True) cache off vs on. Prediction: bitwise parity if the grouping
  explanation holds; a persistent mismatch would move the cause to
  process-state effects of skipping the target forward.
- Disk: pruned ckpts to one init + two finals (s2/a1, s2/b1 kept for the
  recorded diff; other hashes already recorded); /home free 7.9 GiB at 18:08.
- H-A closure (S1b profile, 129 real rows): extraction wall 16.36 -> 14.18 s
  (-13.3%) g1 -> g16, sample_ms 126.8 -> 110.0, saturation by g8; render
  cold/warm 14.31/12.43 ms/sample; target peak 7.9 -> 10.8 GB. Verdict:
  STOP-as-mechanism (extraction is only ~16-20% of e2e; a <13% extraction
  term is <3% e2e and saturates); keep online_batch_size as a knob.

## 18:30 — S2c causal check: fixed-order ON == OFF bitwise; decisions

- S2c (18:09-18:24, GPU0): fixed epoch order (VLM_DFLASH_TRAIN_SHUFFLE=0),
  everything else identical to S2 (129 rows, 6 epochs, cache 8 GiB):
  sha256(ckpt_final) MATCH (52d25648d79a4351fc8d387f13b40f22ae033c5509821a4b268be7da01616917)
  and all six train-log fields bitwise identical across all 102 steps
  (evidence/s2c_*, logs/s2c_ckpt_hashes.txt). Train-stage wall 206.4 s (off)
  vs 76.5 s (on), -62.9%; extraction 98.9 vs 16.6 s; cache 645 hits / 129
  misses / 0 evictions / 830 MiB peak. (Leg a's setup hit a host I/O stall:
  708 s end-to-end vs 195 s for leg b; walls above are from run summaries.)
- Mechanism fully localised: the cache is value-exact; S2's drift = shuffled
  loader changes the extraction grouping every epoch x the extractor's
  batch-shape sensitivity (probe 2) => cached epoch-1 values differ from
  epoch-k recomputes at the bf16-rounding scale; 6 epochs amplify to <=1 ulp
  weights. The "bit parity under shuffle" gate was a gate on loader-grouping
  stability, not on the cache.
- Decisions: H-C1 KEEP (wall win paired x3 + bit-exact under stable grouping;
  amended numeric gate in REPORT section 4.5). H-A STOP. H-C2/H-C3 stay
  deferred (conditions unchanged).
- S3 (canonical 2K pair) not run this round: the chain self-gated on the old
  sha criterion before S2c existed, and the host disk was tight (now 32 GiB
  free). Documented next step: run S3 with the amended gate when convenient;
  budget available (~1.2 of 16 GPU-h spent).
- Hygiene: s2c ckpts pruned after hashing (kept s2/a1+b1 finals for the
  recorded wdiff); 65/65 tests pass on the server venv with the patched code;
  py_compile clean; docs/online-training.md caveat added.

## 18:35 — S3 relaunched with the amended gate

- Gate amended (run_s3_ab.sh; documented in the script header and REPORT
  section 4.5): (a) S2 cache run had hits; (b) S2c fixed-order pair is ckpt
  sha256-identical; (c) S2 shuffled pair has identical loss/valid_tokens and
  grad_norm drift <=1e-3 (actual 6.5e-4). The original sha-parity-under-
  shuffle criterion is retired (probe 2 + S2c: it tested loader-grouping
  stability, not the cache).
- S3 started 18:34:27 in screen `s3` on GPU0 (free 38 GiB; leg A builds the
  canonical 2K master then trains cache-off; leg B reuses data+regen,
  cache 16 GiB). ETA ~60-75 min; expect sha mismatch on the shuffled pair
  (by design) -> record loss equality + wdiff ulp bound + walls as the
  amended acceptance evidence; prune after.

## 19:30-20:05 — S3 canonical 2K completed; numeric closing (ulp/relL2)

- S3 DONE 19:29:26 (screen `s3`, GPU0). Legs: a_off (build + cache off;
  end-to-end 45:34; train stage 1167.0 s; extraction 799.1 s / 762 forwards)
  -> b_on (staged inputs reused via digest skip; 9:07 end-to-end; train
  470.9 s; extraction 125.5 s / 127 forwards; cache 5060 hits / 1012 misses
  / 0 evictions / 6.35 GiB of 16). Walls: train -59.7%, extraction -84.3%;
  0 disk bytes; GPU peak 25.02 GB both; host RSS peak 5.695 -> 9.664 GiB;
  feature digest `bc4b888c` identical both legs; init ckpt sha identical
  (pruned). Cost 0.91 GPU-h (booked lane total ~1.8 of 16).
- Numeric A/B (762 steps, shuffled): valid_tokens 762/762 identical; loss
  740/762 identical, 22 differ (1x0.5, 20x1, 1x2 bf16 display ulp; first at
  step 173); grad_norm first diverges step 128 = first cache-hit step (max
  0.041); accuracy max |d| 0.0032; accept_len 0.018; sha256 mismatch
  (expected; both legs individually deterministic).
- Numeric closing - HONEST CORRECTION of the S2-era "<=1 ulp" wording: built
  `server/ulp_drift.py` (per-element bf16-ulp normalisation) + a rel-L2
  pass; ran on both pairs (evidence/ulp_s2.*, evidence/ulp_s3.*):
  * S2 pair: max |d| = 2^-7 exactly; 83.7% of differing elements within
    1 ulp of their own value; near-zero tail <=2^-11 abs -> large per-element
    ratios; the old phrasing was imprecise (REPORT 4.3/4.5 corrected).
  * S3 pair: 90.391% elements differ; max |d| 0.0446; global rel-L2 4.26%
    (worst `layers.0.self_attn.q_proj.weight` 8.0% relL2, rms_w 0.022 vs
    rms_d 0.0018); 16.2% of all elements within 1 ulp, 18.9% above 16 ulp ->
    multi-ulp drift at 762 steps, growing with training length (expected for
    a deterministic grouping perturbation over 6 epochs of SGD).
- Records updated this session: RUNS.jsonl += S3 entries; REPORT
  header/2/3/4.3/4.4/4.5/5 rewritten to the S3 reality; HYPOTHESES H-C1
  updated.
- Closing frames LAUNCHED automatically 19:56:28 on GPU2 (freed card):
  screen `closing` + `watch_gpu_closing.sh` (idle guard: 0 procs / >40 GiB
  free / util<5%, confirmed 60 s apart; 14 h horizon; STOP_WAITER honored;
  flock lane.lock). `run_closing_frames.sh` steps: (1) acceptance eval of
  runs/s3/{a_off,b_on} (`vlm_dflash.cli eval --checkpoints ckpt_final.pt
  --tag closing` -> <run>/eval/eval_result_closing.json; AL / speedup /
  token_exact); (2) seed-change frame: c_off_seed43 + d_on_seed43
  (train.seed=43, staged inputs copied from a_off) + wdiff/ulp for (a,c)
  [order+init variation] and (c,d) [cache effect at seed 43]. Logs:
  closing_waiter.log, closing_progress.log, closing_eval_*.log,
  wdiff_seed43_*.txt, ulp_seed43_*.*; harvest
  logs/s3_{c_off_seed43,d_on_seed43}_summary.json. STOP:
  touch STOP_WAITER or `screen -S closing -X quit`.

## 20:35 — closing frame 1 (acceptance eval) DONE: functionally equivalent

- `vlm_dflash.cli eval --checkpoints ckpt_final.pt --tag closing` on
  runs/s3/{a_off,b_on} (GPU2, 19:56:52-20:33:49; 0.62 GPU-h). a_off vs b_on:
  mean_al 1.5262066923 == 1.5262066923 (**identical to 10 decimals**),
  acceptance_rate 0.076164 vs 0.076165, accepted tokens 5288 both, new
  tokens 15462 both, draft tokens 69429 vs 69428 (one knife-edge token of
  ~69k), token_exact=True for BOTH (draft-assisted == target-only greedy).
  Verdict: the cache-trained drafter is functionally equivalent to the
  normally-trained one on the end-to-end acceptance metric -> "no
  performance loss" is verified downstream, not only on training metrics.
  Evidence: runs/s3/*/eval/eval_result_closing.json ->
  evidence/s3_{a_off,b_on}_eval_closing.json.
- Closing frame 2 (seed change frame) DONE 21:03:40 (+ diff re-run 21:07):
  * c_off_seed43: train wall 1113.7 s; extraction 754.5 s / 762 forwards;
    final loss 3.28125; last10 3.707031.
  * d_on_seed43: train wall 476.5 s (**-57.2%**, replicates S3);
    extraction 125.2 s / 127 forwards; cache 5060 hits / 1012 misses /
    0 evictions / 6823170276 B peak (bit-identical footprint to seed 42).
  * c-vs-d numeric: valid_tokens identical; loss 757/762 identical (rest
    <=1 display ulp; first diff 287); grad_norm first diff 128 (first hit);
    final-weight drift max 0.0343 abs / relL2 **1.45%** (seed-dependent
    vs 4.26% at seed 42; both display-benign).
  * a-vs-c frame (order + init): 99.9% elements differ, max 0.1846 abs,
    relL2 **131.5%**; last10 loss 4.052 vs 3.707 (Delta 0.345). The cache
    perturbation is 30-90x smaller than a routine seed change.
  * Note: first diff attempt failed on repo-relative script paths (cwd is
    the repo, scripts live in $LANE/scripts); plog line 21:03:40 claimed
    "done" wrongly - re-ran 21:07 with absolute paths (see
    diff_seed43_all.sh); raw files are the re-run. Script fixed.
  * Cost: frame 2 = 0.44 GPU-h; eval frame = 0.62 GPU-h; lane total
    ~2.9 of 16 GPU-h.

## 21:35 — push readiness: test env resolved, suites run, offline baseline in flight

- Test-env blocker resolved: `/home/lyh/vlm-dflash/.venv` (Python 3.11.16,
  pytest 9.1.1; the c1vllm env has no pytest). Config `addopts = "-q"`
  suppresses the usual summary line, so counts below are from the raw logs.
- Online suite on the server copy `/home/lyh/VelaDraft-systems`
  (md5-identical to the local worktree at commit time):
  `pytest tests/test_online_mode.py` -> **65/65 pass**, exit 0, incl. the 7
  new cache tests. Raw log: `evidence/pytest_online_20261014.log`.
- Full suite: 424 collected -> **417 passed / 6 skipped / 1 failed**. The
  single failure (`tests/test_cli.py::test_version_flag_and_single_version_source`,
  installed dist 0.1.0 vs source 0.3.1) reproduces identically on a
  pristine-HEAD temp worktree -> stale editable dist-info
  (`__editable__.vlm_dflash-0.1.0`) in that venv, an environment artifact
  unrelated to this lane's diff. Raw log:
  `evidence/pytest_full_20261014.log`.
- Branch `research/online-systems` pushed to `origin` 2026-10-14 21:4x
  (PR-style, main untouched; push commit sha recorded in the follow-up
  records note). The push carries the 5 code/docs files + this records tree
  (docs/online-training.md cites `research/system_optimization/REPORT.md`
  §4.3-§4.5, so the records ship with the code).
- Offline baseline (goal §8.3; waiter screen `offline`, launched 21:17):
  leg e (full offline capture, disk store) exit=0 at 21:32:37, store 7.0G;
  leg e2 (reuse via `--features`) started 21:32:45 on GPU2. REPORT §4.6 and
  RUNS.jsonl get the offline numbers when e2 lands.

## 21:42-22:35 — offline baseline DONE; push; PR #6; merge cleanup

- Offline baseline (goal §8.3; screen `offline`, GPU2, 0.39 GPU-h): e =
  capture+train, end-to-end 836.8 s, trainer wall 624.5 s (capture ~75.6 s +
  train 548.8 s), store 7.0G / 36 files, peak GPU 17.4 GB, host RSS 4.78 GiB;
  e2 = reuse e's store (`--features`), end-to-end 582.2 s, train 548.8 s,
  RSS 3.90 GiB. e/e2 train logs byte-identical (md5 b9340d10..., written
  10 min apart) -> store round-trip numerically lossless, offline path
  deterministic. final_loss 4.03125 on all four legs (e, e2, a_off, b_on).
  Train-stage: a_off 1167.0 / e2 548.8 / e 624.5 / b_on 470.9 s -> the cache
  path beats store reuse by 14.2% at zero disk; caveat: single-shot legs
  under host I/O contention (0.72 vs 0.45 s/step may be partly shard-I/O).
- Push: `ea8381b` -> origin `research/online-systems` (closes the push-sha
  note from the 21:35 section); PR #6 opened (base main, by LUOseu, 21:46).
- Merge cleanup per review brief (`f015f4f`): lane-only env gates removed
  from the merge branch (train.py / draft_model.py now identical to main);
  docs resume semantics (cold start; not bit-exact to an uninterrupted
  cached run under the shuffling loader) + hit/miss-precise claim wording;
  online.py comments likewise + "records are shared, not copied"; +1 test
  (eviction->recompute value-exact, capacity below working set).
- Tests on the cleaned tree (server venv): online suite 66/66; full suite
  425 collected -> 418 passed / 6 skipped / 1 pre-existing venv-metadata
  failure. Logs: `evidence/pytest_*_cleanup_20261014.log`.
- Lane-only patch backup: the server lane copy was tested with the cleaned
  files and then restored to the pre-cleanup snapshot
  (`/home/lyh/vlm-systems/lane_patch_backup_20261014/`, md5s verified).
- Lane budget now ≈3.2 of 16 GPU-h.
