"""``_tv_kl`` torch.compile integration (objective-aware opt-in, fallback, parity)."""
from __future__ import annotations

import warnings

import pytest
import torch

from tests.test_recipe_row_compaction import _batch, _run, _tiny_module
from vlm_dflash import objectives
from vlm_dflash.objectives import resolve_recipe


@pytest.fixture(autouse=True)
def _fresh_compile_cache():
    """Re-arm ``_compiled_tv_kl``'s one-shot cache around every test."""
    snapshot = dict(objectives._COMPILED_TV_KL)
    objectives._COMPILED_TV_KL.clear()
    objectives._COMPILED_TV_KL.update({"impl": None, "attempted": False})
    yield
    objectives._COMPILED_TV_KL.clear()
    objectives._COMPILED_TV_KL.update(snapshot)


def _fake_compile(counter, *, fail=False):
    def fake(fn, **kwargs):
        assert kwargs.get("dynamic") is True
        if fail:
            raise RuntimeError("no toolchain")
        counter["compiles"] += 1

        def compiled(*args, **kw):
            counter["calls"] += 1
            return fn(*args, **kw)

        return compiled

    return fake


def test_compiled_tv_kl_builds_once_and_delegates(monkeypatch):
    counter = {"compiles": 0, "calls": 0}
    monkeypatch.setattr(torch, "compile", _fake_compile(counter))
    impl = objectives._compiled_tv_kl()
    assert impl is not None and callable(impl)
    assert impl is objectives._compiled_tv_kl()  # cached: one build
    torch.manual_seed(3)
    student, teacher = torch.randn(3, 5), torch.randn(3, 5)
    tv, kl = impl(student, teacher)
    tv_ref, kl_ref = objectives._tv_kl(student, teacher)
    assert counter == {"compiles": 1, "calls": 1}
    assert torch.allclose(tv, tv_ref) and torch.allclose(kl, kl_ref)
    assert impl.__name__ == "_tv_kl"


def test_compiled_tv_kl_falls_back_when_compile_raises(monkeypatch):
    calls = {"n": 0}

    def broken(fn, **kwargs):
        calls["n"] += 1
        raise RuntimeError("no toolchain")

    monkeypatch.setattr(torch, "compile", broken)
    assert objectives._compiled_tv_kl() is objectives._tv_kl
    assert calls["n"] == 1
    # The decision is cached: no retry of the broken toolchain.
    assert objectives._compiled_tv_kl() is objectives._tv_kl
    assert calls["n"] == 1


def test_compiled_tv_kl_runtime_failure_falls_back_once(monkeypatch):
    counter = {"compiles": 0, "calls": 0}
    monkeypatch.setattr(torch, "compile", _fake_compile(counter))
    impl = objectives._compiled_tv_kl()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    impl._compiled = boom
    student, teacher = torch.randn(2, 4), torch.randn(2, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tv, kl = impl(student, teacher)
    assert sum(issubclass(w.category, RuntimeWarning) for w in caught) == 1
    tv_ref, kl_ref = objectives._tv_kl(student, teacher)
    assert torch.allclose(tv, tv_ref) and torch.allclose(kl, kl_ref)
    # Permanent fallback: the dead compiled path is never re-entered, and no
    # second warning is emitted.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        impl(student, teacher)
    assert not caught


def _stage_inputs(vocab=8, b=2, n=2, s=2, seed=0):
    torch.manual_seed(seed)
    student = torch.randn(b, n, s, vocab)
    mask = torch.ones(b, n, s)
    mask[0, 0, 0] = 0.0
    targets = torch.randint(0, vocab, (b, n, s))
    teacher = torch.randn(b, n, s, vocab)
    return student, targets, mask, teacher


def _run_recipe_loss(stage, *, compile_tv_kl, seed=3):
    student, targets, mask, teacher = _stage_inputs(seed=seed)
    total, components = objectives.recipe_loss(
        stage,
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
        compile_tv_kl=compile_tv_kl,
    )
    return float(total), components


def _two_stage():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly")
    lk_stage = next(stage for stage in recipe.stages if stage.lk_coef > 0.0)
    e2e_stage = next(stage for stage in recipe.stages if stage.lk_coef == 0.0)
    return lk_stage, e2e_stage


def test_recipe_loss_compiles_only_for_stages_that_use_tv_kl(monkeypatch):
    lk_stage, e2e_stage = _two_stage()
    calls = {"n": 0}

    def spy():
        calls["n"] += 1
        return objectives._tv_kl  # a valid impl; only the gate is under test

    monkeypatch.setattr(objectives, "_compiled_tv_kl", spy)
    _run_recipe_loss(lk_stage, compile_tv_kl=True)
    assert calls["n"] == 1
    calls["n"] = 0
    _run_recipe_loss(lk_stage, compile_tv_kl=False)
    assert calls["n"] == 0  # opt-in: default-off never compiles
    calls["n"] = 0
    _run_recipe_loss(e2e_stage, compile_tv_kl=True)
    assert calls["n"] == 0  # objective-aware: e2e_tv is exempt by construction


def test_trainer_threads_the_flag_and_the_e2e_stage_stays_eager(monkeypatch):
    calls = {"n": 0}

    def spy():
        calls["n"] += 1
        return objectives._tv_kl

    monkeypatch.setattr(objectives, "_compiled_tv_kl", spy)
    batch = _batch()
    _run(_tiny_module(stage=0, recipe_compile_tv_kl=True), batch)
    assert calls["n"] == 1
    calls["n"] = 0
    _run(_tiny_module(stage=1, recipe_compile_tv_kl=True), batch)
    assert calls["n"] == 0
    calls["n"] = 0
    _run(_tiny_module(stage=0, recipe_compile_tv_kl=False), batch)
    assert calls["n"] == 0


def test_compiled_tv_kl_matches_eager():
    impl = objectives._compiled_tv_kl()
    if impl is objectives._tv_kl:  # pragma: no cover - toolchain dependent
        pytest.skip("torch.compile/inductor unavailable in this environment")
    torch.manual_seed(7)
    student, teacher = torch.randn(6, 11), torch.randn(6, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tv_c, kl_c = impl(student, teacher)
    tv_e, kl_e = objectives._tv_kl(student, teacher)
    assert torch.allclose(tv_c, tv_e, rtol=1e-5, atol=1e-6)
    assert torch.allclose(kl_c, kl_e, rtol=1e-5, atol=1e-6)
