# A100 runner smoke — prepare → train → eval → benchmark ×2 → resume → summarize

Plumbing verification for the PR #9 review fixes (run on the A6000 dev server
`oab-remote`, 2026-10-15 17:35–18:10 CST). **Not a quality experiment**: training
runs 2 steps so the whole chain finishes in ~35 min; AL=1.0 / acc=0 are the
expected values for a barely-trained drafter.

## Setup

- Code: server checkout `/home/lyh/veladraft-conv/repo` @ `533bfa8` (detached).
- Python: `/home/lyh/miniforge3/envs/c1vllm/bin/python`; GPUs 0/1 (A6000 48GB).
- Matrix: `matrix.smoke.json` (2 entries: dflash controlled seed 42, regroup
  off + grid2; base config `smoke_base.json` = the frozen W3 recipe with
  `save_every_steps=1`). Data: `train_store=/home/lyh/rsi-d1/runs/d1_w3_probe/features`
  (192 rows), val split staged at `/home/lyh/veladraft-conv-smoke/smoke_val/val.jsonl`.
- Workloads: `smoke_caption` (run_val loader, 6 prompts) and `smoke_val2`
  (external `sharegpt_jsonl`, 4 prompts) — the two prompt-source branches.

## Sequence (server log times)

1. **dry-run** (17:35): 2 entries, pinned effective configs under
   `runs/_configs/*.yaml`, per-workload benchmark names; 0 errors.
2. **run** (17:36–18:00): `prepare` ok for both entries — preflight checks
   `algorithm_name+validate_config, target_contract, store_identity+layers+final_hidden,
   val_split+media`; offline train 2 steps from the W3 store; eval 8 prompts /
   16 new tokens `token_exact=True` (both entries); benchmark both workloads.
3. **interrupt → resume** (18:01–18:10): deleted `train_summary.json`, `eval/`
   and the benchmark reports of `dflash__controlled__s42__rggrid2`, then
   `resume --only rggrid2`: prepare skipped (preflight config hash matches),
   train re-ran with `--resume` ("resumed from step 2"), eval + both benchmarks
   re-ran; all stages `ok`.
4. **summarize**: reads the eval result plus each per-workload report
   (`benchmark_<entry>_<workload>.json`, field `al`); outputs `summary.{json,csv,md}`.

Server state log: `matrix_state.jsonl` (prepare/train/eval/benchmark rows for
both entries, with `git_sha 533bfa8...`).

## Findings

- Summary table (from a 2-step run; plumbing only):

| entry | AL | acc | train wall | peak | token_exact | smoke_caption | smoke_val2 |
|---|---|---|---|---|---|---|---|
| grid2 | 1.000 | 0.0000 | 54 s | 4.0 GiB | True | 1.000 | 1.000 |
| off | 1.000 | 0.0000 | 151 s | 23.4 GiB | True | 1.000 | 1.000 |

- `loss_decreased=false` on the 2-step legs is expected here; the frozen
  analyzer would (correctly) flag such legs as invalid in a real matrix run —
  the smoke does not bypass that.
- Legacy data caveat (found by the smoke): `d1q_val64/images` alone is **not
  self-contained** (48 files); the W3 val rows also reference images that live
  in the probe's `data/images` (208 files). The staging contract therefore is:
  a val split directory must ship `val.jsonl` + `meta.json` (or `images/`) with
  **all** referenced images; the smoke built `smoke_val/` (hardlink union, all
  64 images resolve).
- The runner refused to silently rebind the (changed) val symlink of the
  already-trained run dirs — by design; the operator re-pointed deliberately
  because the prompt set was byte-identical (only the media directory moved).
- `guard.min_disk_gb` default (15 GiB) parks on the ~100%-full shared `/home`;
  the smoke used 8 GiB. On the A100 host, set the threshold to local headroom.

## Files

- `matrix.smoke.json`, `smoke_base.json` — inputs.
- `run.log`, `resume3.log` (eval re-run), `resume4.log` (benchmarks), `resume5.log`
  (interrupt→resume) — runner stdout.
- `matrix_state.jsonl` — stage state (with SHAs).
- `preflight_rggrid2.json` — one entry's preflight (store identity/layers, val rows).
- `train_summary_rggrid2.json`, `eval_result_rggrid2.json` — post-resume artifacts.
- `benchmark_dflash__controlled__s42__rggrid2_smoke_{caption,val2}.json` — per-workload reports.
- `summary.{json,csv,md}` — aggregated output (benchmark `al` fields populated).
