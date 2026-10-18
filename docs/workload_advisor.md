# Enterprise workload advisor (Agent 04)

`project_scope=platform_extension`, `include_in_paper=false`.

This module answers one local question: **does this private image/text workload
benefit from VelaDraft speculative decoding, and is the benefit credible?**
It ships a CPU-first workload contract, a machine-readable RunRecord consumer,
paired group statistics, a conservative decision engine, a local HF backend
adapter and a redacted share package.

The global CLI, README, `pyproject.toml`, paper manifests and training data are
not modified. A public `vlm-dflash workload-advisor` hook is provided as a
separate patch (`docs/workload_advisor_public_hook.patch`); it is not required
to run this module.

## Status at this checkout

- PR #14: `platform/workload-advisor-cpu` (CPU contract, statistics, decisions)
- PR #15: `platform/workload-advisor-backend` (stacked HF backend, lease/ledger)
- baseline: `7711eda504f4f28653a11cbb164b2ea673b4d682`
- CPU contract/report/decision: `cpu_tested` (85 tests on the stacked backend
  branch; 63 tests on the CPU-only branch, see below)
- HF GPU backend: `blocked` on this host (macOS/arm64, no `nvidia-smi`, no
  `torch`, no local target weights). A3-A6 execution-path fixes are exercised
  with fake loaders, clocks and leases only. No real GPU smoke, timing or
  speedup exists. The 4 GPU-h first-round cap is unused; the 8 GPU-h extension
  was not requested.
- synthetic fixtures: `docs/workload_advisor_evidence/cpu_fixture/` (always marked
  `synthetic=true`; never part of real result tables)
- blocker evidence: `docs/workload_advisor_evidence/blocked/` and
  `docs/workload_advisor_evidence/environment.json`
- capabilities: `docs/workload_advisor_evidence/capabilities.json` (`state=blocked`)
- rework closure: A1-A9 are closed only to their stated CPU-testable scope in
  `docs/workload_advisor_delivery.md`; real asset/GPU parity remains unverified.

## Commands

```bash
# 1. Read-only validation: no weights, no GPU, no downloads.
.venv-workload/bin/python scripts/workload_advisor.py validate \
  --manifest examples/workload_advisor/mini/manifest.json \
  --json-out out/validation.json

# 2. Capability matrix, request count, timing scope and budget; no ranking.
.venv-workload/bin/python scripts/workload_advisor.py dry-run \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir out/dry-run --budget-gpu-hours 4

# 3. Real run (requires a ready CUDA/HF backend and local weights).
.venv-workload/bin/python scripts/workload_advisor.py run \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir out/pilot --budget-gpu-hours 4 --sessions 3 --warmups 4

# 4. Offline report / recommendation: no torch or CUDA needed.
.venv-workload/bin/python scripts/workload_advisor.py report \
  --records out/pilot/records.jsonl \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir out/report --share

.venv-workload/bin/python scripts/workload_advisor.py recommend \
  --records out/pilot/records.jsonl \
  --manifest examples/workload_advisor/mini/manifest.json \
  --out-dir out/recommend --bootstrap-seed 7

# Optional legacy paths.
.venv-workload/bin/python scripts/workload_advisor.py convert-sharegpt --input legacy.jsonl --out-dir out/converted
.venv-workload/bin/python scripts/workload_advisor.py import-legacy --report old_benchmark.json --out out/legacy_records.jsonl
```

Exit codes: `0` success; `2` malformed input or a failed schema/correctness
structure; `3` backend not ready (a blocker file is written, no records).
If one optional draft checkpoint is missing while the host/target are ready,
the run continues with the remaining candidates and records
`overall_status=partial` plus `skipped_candidates`.

## Workload contract

A workload directory contains `manifest.json`, prompt-only `rows.jsonl`, an
optional answer sidecar `labels.jsonl`, and local media:

```json
{"request_id":"invoice-001-q1","group_id":"invoice-001","task":"invoice_fields",
 "messages":[{"role":"user","content":"提取发票总额，返回 JSON。"}],
 "media":[{"type":"image","path":"images/invoice-001.png"}],"split":"holdout"}
```

Manifest requirements:

- `schema_version=veladraft-platform-v1`, `kind=workload_manifest`;
- frozen target identity (model id, revision, weights hash, tokenizer/processor,
  chat template, dtype, structure, vision bounds);
- AR plus speculative candidates with checkpoint/algorithm identity;
- frozen decode (`temperature=0`, `max_new_tokens=128`, stop ids, EOS policy),
  backend, seed, resolution, measurement scope, concurrency, bucket rules;
- task scorers/thresholds and tuning/holdout group counts;
- SHA256 for rows and labels. A mismatch is a hard error.

Validation is read-only and per-line:

- malformed JSON, duplicate `request_id`, group spanning tuning and holdout,
  labels without a row, missing/escaping media, bad image magic, hash mismatch
  and unknown schema version are errors;
- video/audio, VLA actions/trajectories, tool calls/roles and multi-turn state
  advancement are returned as `unsupported` rows and never silently flattened
  into single-turn text;
- answers are never inserted into prompts; labels are loaded separately.

Legacy ShareGPT rows can be converted deterministically. Answers are dropped
from the prompt export, and `legacy_last_turn_only=true` is recorded in the
conversion report when the old semantics only used the last turn.

## RunRecord wire format

Every row is JSONL with the required top-level keys:

`schema_version, project_scope, include_in_paper, run_id, request_id, group_id,
target_identity, draft_identity, backend_identity, protocol_hash,
correctness_status, measurement_scope, raw_counts, latency, resource_usage`.

Additional recorded semantics:

- `session_id`, `repeat_index`, `execution_order`, `method`, `candidate_id`,
  requested/effective method, fallback, actual visual `target_V`, output `R`,
  error type/stage, local artifact hashes;
- AR has `draft_identity=null` and `missing_reasons.draft_identity=not_applicable_ar`;
  acceptance counts are `null`, not `0`;
- raw counts keep emitted, proposed, accepted, verify rounds, target/draft
  forward calls and a count-definition version;
  `AL_draft=accepted/rounds`, `AL_progress=emitted/rounds`;
- correctness status is one of `not_checked / verified_greedy_exact /
  verified_semantic_acceptance / experimental_numeric_divergence / failed /
  unsupported`; `acceptance_policy=relaxed` can never be
  `verified_greedy_exact`;
- latency keys use seconds and `*_s`; TTFT/ITL stay `null` with
  `no_streaming` / `no_token_visibility` because the HF eval path exposes no
  per-token stream. No proxy is substituted;
- resource usage records all allocated GPU-hours, measured GPU-hours, devices,
  peak allocated/reserved bytes, allocation intervals and memory fit;
- `protocol_hash` covers actual input identity plus generation/stop rules,
  model/processor identity, precision, backend, vision config and timing
  boundary. AR token references are reusable only when both the request-level
  and workload-level protocol hashes match; AR timing is always re-measured.

## Measurement definitions

- `local_client`: request in memory -> preprocessing/encoding/H2D/generate ->
  committed output. Model load and media disk reads are separate cold-start
  measurements and never mixed into client latency.
- TTFT is the first verified client-visible token; ITL is between committed
  tokens. The current backend has no token callbacks, so both are `null`.
- E2E failures/timeouts/OOM are counted separately; slow requests are not
  silently removed.
- Latency reports mean/p50/p95 with sample counts. Small tails are descriptive,
  not an SLA.
- Serial output throughput is called `serial_output_tokens_per_second`;
  concurrency > 1 is `unsupported` for this first-round backend.
- `all_allocated_gpu_hours` is the sum over allocated devices, including load,
  warmup, failures and idle allocation. `measured_run_gpu_hours` is the
  measured session interval only.

## Statistics and decision

- tuning uses only the frozen tuning groups; holdout is never used to choose a
  checkpoint or threshold;
- records are joined by candidate x request x session/repeat only after shared
  input/reference/backend/device/scope protocol checks; missing, duplicate,
  failed, fallback and mismatched rows stay in explicit exclusion counts and
  never enter the successful denominator. A group bootstrap keeps all valid
  sessions for a sampled group;
- tuning rows are descriptive only. Speed CI, coverage and recommendation gates
  use the frozen group-disjoint holdout, and repeated sessions do not increase
  the number of independent groups;
- the primary speed ratio is `sum(AR e2e) / sum(candidate e2e)` over paired
  groups; mean-of-ratios is reported separately and never mixed to pick the
  larger number;
- paired group bootstrap (seed 7, 2000 draws) reports a 95% CI and the number
  of independent groups; repeated timings do not increase that number;
- buckets are frozen before holdout (`task`, `target_V`, actual AR `R`);
  buckets with `<20` independent holdout groups are `insufficient_evidence`;
- a 0/N failure count keeps N; it is not a proof that failure probability is 0.

Default gates for `accelerate`:

1. per-request token-exact correctness against a protocol-matched AR reference;
2. task quality pass (or an explicit `conditional_accelerate` condition when
   labels/scorers are unavailable);
3. memory fit;
4. real per-request measurement (not `aggregate_only`);
5. paired speed-ratio CI lower bound `> 1.05`.

Otherwise the decision is `retain_ar` (evidence of no benefit / non-compliant
candidate) or `insufficient_evidence` (CI crosses the threshold, coverage or
key metrics missing, protocol mismatch, aggregate-only evidence). The threshold
is a business default for this tool, not a paper result. No LLM is used to rank
methods.

`decision.json` includes schema and protocol hash, action/method, scope, CI,
coverage, reasons, excluded candidates, threshold and bootstrap seed.
`decision_id` is a canonical hash and is byte-stable for identical input.

## GPU protocol

- The HF backend resolves the requested logical device and
  `CUDA_VISIBLE_DEVICES` to the one physical UUID it will actually use. It
  acquires an advisory `flock` under
  `/tmp/veladraft_gpu_locks/<UUID>.lock` for that UUID only; unused visible
  GPUs are not locked or charged.
- If a lock is busy, the CLI returns a `gpu_busy` blocker and does CPU work;
  it never stops or cleans another job. The lock does not replace occupancy
  checks or authorization. External occupancy is checked before and after lease
  acquisition.
- The 4 GPU-h first-round cap includes load, calibration, warmup, timing and
  failures. Zero, negative, NaN or infinite budgets are rejected before any
  lease or load. Remaining budget is checked before load, references, warmup
  and every request. `GpuLedger` records allocation intervals, per-device
  peaks and phase intervals. The optional extension to 8 GPU-h requires
  recorded evidence of progress and is not automatic.
- Unsupported parallel verification, server endpoint, streaming, concurrency,
  EOS/runtime combinations are rejected by readiness before GPU work. Local
  execution is sequential HF verification, not a parallel-serving speed claim.
- Warm sessions interleave method order; each candidate/session gets 4
  unrecorded warmups. Three sessions retain request pairing.
- E2E starts only after media is resident and the device is idle. It includes
  chat template, processor, generation, H2D copy, `torch.cuda.synchronize` on
  the actual device and token `tolist()` commit; generate-only time remains a
  separate field. Completed and failed records are appended and flushed, with
  a ledger checkpoint written throughout the run.
- Streaming, cancellation, per-token timestamps and proposal/commit events are
  declared unsupported rather than fabricated. `sequential` speculative
  verification is used as a correctness reference and is labeled as such, not
  as parallel deployment speed.

## Actual CPU evidence

Commands executed on this checkout (macOS/arm64, Python 3.12.13):

```bash
.venv-workload/bin/python -m unittest discover -s tests -p 'test_workload_advisor*.py'
.venv-workload/bin/python scripts/workload_advisor.py validate --manifest examples/workload_advisor/mini/manifest.json
.venv-workload/bin/python scripts/workload_advisor.py dry-run --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/blocked
.venv-workload/bin/python scripts/workload_advisor.py run --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/blocked
.venv-workload/bin/python scripts/workload_advisor.py report --records examples/workload_advisor/fixtures/records.jsonl --manifest examples/workload_advisor/mini/manifest.json --out-dir docs/workload_advisor_evidence/cpu_fixture --share
```

Observed:

- 85 unit tests pass on the stacked backend branch (63 on CPU-only PR #14);
- `validate`: 30 requests, 30 groups (6 tuning / 24 holdout), 3 tasks, no errors;
- `dry-run`: 216 measured runs, 36 warmups, 4 GPU-h cap, `predicted_speedup=null`,
  `ranking_allowed=false`, backend `blocked`;
- `run`: exit 3, writes `blocker.json`, `validation.json`, `plan.json`; no
  `records.jsonl` and no GPU allocation;
- synthetic report: `synthetic=true`, action `insufficient_evidence`, fast but
  wrong candidate excluded for correctness/quality, exact candidate excluded
  only because the evidence is a synthetic fixture;
- identical input replay produces byte-identical `decision.json` and
  `summary.json`.

The rework document also requests the pytest entry point. It was attempted with
the same interpreter and did not run: `.venv-workload/bin/python` has no
`pytest` module (exit 1). The unittest command above is the executed full-suite
evidence; CI runs `pytest tests/ -q` in its own dependency environment.

Host blockers recorded in `docs/workload_advisor_evidence/blocked/blocker.json`:
`torch_missing`, `no_gpu`, `target_missing`, `checkpoint_missing` (fixture
paths). This is why PR #15 contains a backend implementation and a blocker, not
real GPU timings.

## Deliverables

A real pilot writes: `validation.json`, `plan.json`, `records.jsonl`,
`summary.json`, `decision.json`, `report.md`. The share subdirectory adds
`records.share.jsonl`, `summary.share.json`, `decision.share.json`,
`report.share.md` and the share manifest at
`docs/workload_advisor_evidence/cpu_fixture/share/SHARE_MANIFEST.json`.
Synthetic reports additionally write `records.synthetic.jsonl` and never place
those rows in a real `records.jsonl`.

The share package removes prompts, images, answers, absolute private paths,
model/checkpoint local paths and artifact local paths; it keeps protocol hashes,
identities, aggregates and evidence. The local audit package keeps the row/media
mapping through request ids and hashes. Nothing is uploaded automatically and
no remote media is fetched.

## Known gaps

- no real GPU measurement on this host; PR #15 is `blocked`, not validated.
  Loader revision/processor/template application, checkpoint artifact identity,
  device synchronization and lease/budget behavior are covered only by CPU
  spies/fakes, not by a real HF parity smoke;
- TTFT/ITL/distribution and streaming are unsupported by the current HF eval
  path; a serving backend must expose token/commit callbacks before those
  metrics can be claimed;
- parallel speculative verification remains experimental and is excluded from
  recommendations unless a future protocol validates it;
- first-round input is text + local images only; video/VLA/tool/multi-turn state
  are unsupported by contract;
- `aggregate_only` legacy reports cannot produce CIs or per-request correctness
  and therefore always resolve to `insufficient_evidence` unless paired
  per-request records are supplied.

## References

- VelaDraft source: <https://github.com/LUOseu/VelaDraft>
- PyTorch CUDA synchronization: <https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html>
- PyTorch peak memory: <https://docs.pytorch.org/docs/2.9/generated/torch.cuda.max_memory_allocated.html>
- MLCommons inference rules: <https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>
