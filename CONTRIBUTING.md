# Contributing

Thanks for helping. v0.3.1 (distribution/first-run fixes) and the
`zoo-v0.3.0/1/2` checkpoint releases are published; `main` keeps moving with
changes that carry recorded evidence.

What is accepted today: bug fixes, documentation, packaging, release tooling,
benchmark adapters, reduced validation, CI, and license/security cleanup — and
new algorithms or targets that meet the acceptance bar below. Multi-image /
multi-turn (schema v2) is implemented with a recorded GPU smoke; video and
serving runtimes remain out of scope until the gates in `docs/serving-bridge.md`
are met. (The earlier "v0.3.1 not yet released, freeze everything else" policy
applied while that release was in flight; it is kept in the CHANGELOG history,
not as current guidance.)

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[tested,dev]"     # exact validated dependency combo
pytest tests/ -q                   # CPU suite; no GPU required
```

Tests that need the real target model are marked `integration` and only run
when `VLM_DFLASH_TEST_MODEL` points at a local checkpoint.

## Add a new drafter algorithm

An algorithm is a plugin behind the registry (`vlm_dflash/algorithms/`); read
`docs/adding_an_algorithm.md` first. The zoo's acceptance bar (unchanged since
v0.3) is:

1. upstream audit doc (`docs/upstream/<name>.md`): paper, official repo,
   inspected commit, license, objective, parity level - conclusions alone
   are not enough;
2. architecture + loss behind the `DrafterAlgorithm` interface (no
   algorithm `if` branches in the shared training loop);
3. unit tests (CPU) + a synthetic overfit test;
4. real VLM smoke on Qwen2.5-VL-3B: prepare -> regen -> capture -> train ->
   visual-path check -> eval (token-exact gate) -> export -> reload;
5. an algorithm user doc (`docs/algorithms/<name>.md`) and matrix entries
   (`ALGORITHM_MATRIX.md`, `SUPPORT_MATRIX.md`).

If upstream does not publish the training objective, the algorithm ships as
**Experimental** with the objective explicitly labelled a reconstruction
(DFlash2 is the example).

## Add a new target

A target needs a `TargetAdapter` + contract resolution:

1. `vlm_dflash/adapters/` entry: processor loading, image token expansion,
   pixel bounds, position ids / MRoPE behavior, special tokens;
2. contract validation in `vlm_dflash/contract.py` (everything runtime-resolved
   from the model config/tokenizer/processor - no hardcoded ids);
3. real smoke: prepare/regen/capture with a visual-path check, one drafter
   train + eval (token-exact), export + reload;
4. docs: target row in `SUPPORT_MATRIX.md` (honest status) and any MRoPE
   notes in `docs/REFERENCE.md`.

## Never do this

- architecture-specific `if` branches inside the shared train loop;
- claiming a support status without a recorded run (`BENCHMARKS.md`);
- committing datasets, weights, `runs/` artifacts or absolute local paths;
- mixing experimental parallel-runtime numbers into correctness tables.

## Pull requests

Keep changes small and focused; one topic per PR. Run `pytest tests/ -q`
before opening. If you change recorded numbers, update `BENCHMARKS.md` (and
the frozen reference manifest if a baseline moved) in the same PR.
