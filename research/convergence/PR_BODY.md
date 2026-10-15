# D1 convergence round: regroup identity bypass fix + eval accounting + benchmark/A100 tooling

Branch `research/rsi-d1-conv` (base `origin/main` @ `6d99f04`). Regroup stays
default-off; nothing here changes shipped training semantics unless
`train.regroup.mode` is set explicitly.

## Code changes

- **`vlm_dflash/dflash/train.py` — identity-layout bypass for `train.regroup`**
  (`f65e67f`). The only remaining identity seam between the regroup machinery and
  the shipped path was the loss reduction (bf16 mean vs fp32 preserved-weight
  sum); the previously reported "2048 vs ~1k forward width" seam does not apply
  (feature batches are length-trimmed and collated to batch max). With
  `train.regroup.identity_bypass=true` (default), identity layouts re-run the
  shipped `module(...)` call verbatim; the legacy machinery path stays available
  as an explicit control (`identity_bypass=false`). Verified bitwise on the
  real training loop (3-step smokes, W3 store: loss/grad_norm bitwise identical
  to regroup-off; the machinery path reproduces the historical ±1 ulp signature).
- **`vlm_dflash/eval/run.py` — per-prompt acceptance accounting** (`5750c23`):
  `eval_samples.jsonl` now carries `verify_steps` / `accepted_tokens` /
  `draft_tokens` per prompt (prerequisite for the preregistered paired
  non-inferiority stats).
- **`scripts/convert_train_state_ckpt.py`** (`bc87e8f`): mid-run `ckpt_last.pt`
  snapshots are exact-resume payloads (`vlm-dflash-train-state-v1`); eval/export
  need `vlm-dflash-ckpt-v1` headers. The converter grafts the train-state
  weights onto `ckpt_init.pt`'s provenance block — stability-curve checkpoints
  become evaluable without re-training.
- Tests: `tests/test_regroup.py` (bypass bitwise on canonical + sorted plans,
  legacy control close, split never bypasses), `tests/test_convert_train_state_ckpt.py`.

## Research tooling (default-off, no product behavior)

- `research/convergence/` — frozen pre-registration, settle-rule analyzer,
  quality gates (hierarchical paired bootstrap), run/chain/stability/harness
  launchers, evidence.
- `research/convergence/a100/` — config-driven experiment matrix for the full
  A100 round: `run_matrix.py` (dry-run / run / resume / summarize; 7 drafters ×
  {controlled, native} × seeds × regroup factor, ~37 GPU-h), `RUNBOOK.md`
  (tracks, isolation, adaptation gates), `regroup_adaptation_smoke.sh`.

## Evidence so far

- Identity bypass: bitwise smokes (off vs canonical/8+bypass) + machinery
  control signature reproduced (see `research/convergence/STATE.md` F1/F2).
- Capacity probe (batch 16, A6000 48GB): canonical micro-16 OOM; canonical
  micro-8 fits at 24.21 GiB peak / 12.6% padding; split sub2 9.37 GiB / 4.9%;
  grid sub2 9.37 GiB / 3.3% with wall −18.8% vs canonical micro-8
  (`research/convergence/evidence/capacity/`).
- Systems A/B series (3 reps × arms canonical/split/token/cost/grid on W3 +
  MIX + W1): <pending — final numbers>
- Confirmatory quality legs (canonical ×4, grid2 ×4 at L*; token2/split2 pilots):
  <pending — verdict per `PREREGISTRATION.md`>
- Second-drafter applicability: dflare smoke <pending>; code-level boundary:
  dspark/domino refused (`sample_from_anchor`), dflash2/jetspec refused
  (auxiliary metrics gate), dflare under test.

## Reproduce

```bash
# local tests
PYTHONPATH=. python -m pytest tests/test_regroup.py tests/test_convert_train_state_ckpt.py -q
# server (4×A6000): stability curve -> L*, confirm legs, systems series, capacity
bash research/convergence/scripts/vconv_stability_leg.sh conv_stab_canon_s42 1 800 200 42
bash research/convergence/scripts/vconv_confirm_series.sh <L*> 1,2,3 pilot
bash research/convergence/scripts/vconv_harness_series.sh series1
bash research/convergence/scripts/vconv_capacity_probe.sh
```

No merge to main; unverified features stay default-off.
