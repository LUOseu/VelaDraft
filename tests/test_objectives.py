"""Recipe objectives: resolution schema, exact formulas, trainer integration.

The upstream DFly recipe (LK/D-PACE cold start -> e2e TV) is ported as a
config-selectable objective block (``algorithm.train_recipe``) on the shared
block-CE trainer core. These tests pin the contract:

* resolution is a bounded schema (unknown names/keys/algorithms fail loud);
* the plan digest stays byte-identical without a recipe and is sensitive to
  the recipe name / cold-start fraction;
* every loss term matches a direct reference implementation and is invariant
  to the internal chunk sizes;
* a recipe run needs the teacher signal at forward time and at resume time;
* the trainer switches stages via the training-loop progress hook.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash import objectives as obj
from vlm_dflash.objectives import (
    RECIPE_SCHEME_VERSION,
    TeacherLogitsStream,
    assert_recipe_capture,
    dpace_position_weights,
    e2e_tv_term,
    gather_teacher_logits,
    lk_term,
    per_slot_nll,
    plan_recipe_knobs,
    recipe_loss,
    resolve_recipe,
)
from vlm_dflash.training_plan import (
    FUSION_PLAN_VERSION,
    training_plan_digest,
)

# --------------------------------------------------------------------------- #
# resolution schema
# --------------------------------------------------------------------------- #


def test_recipe_unset_resolves_to_none():
    assert resolve_recipe({}, "dflash") is None
    assert resolve_recipe({"algorithm": {"name": "dflash"}}, "dflash") is None
    assert plan_recipe_knobs({"algorithm": {"name": "dflash"}}, "dflash") == {}


def test_recipe_name_and_mapping_forms():
    from_name = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly")
    assert from_name is not None
    assert from_name.name == "upstream_two_stage"
    assert from_name.cold_start_fraction == 0.5
    assert from_name.needs_teacher is True
    assert [stage.name for stage in from_name.stages] == ["cold_start_lk_dpace", "e2e_tv"]

    from_mapping = resolve_recipe(
        {"algorithm": {"train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.25}}},
        "dflash",
    )
    assert from_mapping is not None and from_mapping.cold_start_fraction == 0.25

    controlled = resolve_recipe({"algorithm": {"train_recipe": "controlled_ce"}}, "dflare")
    assert controlled is not None
    assert controlled.name == "controlled_ce" and controlled.needs_teacher is False


@pytest.mark.parametrize(
    "block,match",
    [
        ({"name": "not_a_recipe"}, "unknown recipe name"),
        ({"name": "upstream_two_stage", "typo_key": 1}, "unsupported keys"),
        ({"name": "upstream_two_stage", "cold_start_fraction": 1.5}, "cold_start_fraction"),
        ({"name": "upstream_two_stage", "cold_start_fraction": -0.1}, "cold_start_fraction"),
        (42, "recipe name or a mapping"),
        ("upstream_two_stage ", "unknown recipe name"),
    ],
)
def test_recipe_schema_is_bounded(block, match):
    with pytest.raises(ValueError, match=match):
        resolve_recipe({"algorithm": {"train_recipe": block}}, "dfly")


def test_recipe_rejected_for_other_algorithms():
    with pytest.raises(ValueError, match="not supported for algorithm"):
        resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dspark")


def test_stage_split_and_index():
    recipe = resolve_recipe(
        {"algorithm": {"train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.5}}},
        "dfly",
    )
    assert recipe is not None
    assert recipe.split_step(10) == 5
    assert recipe.stage_index(0, 10) == 0
    assert recipe.stage_index(4, 10) == 0
    assert recipe.stage_index(5, 10) == 1
    assert recipe.stage_index(10, 10) == 1
    # Fraction bounds degrade to "skip a stage" instead of crashing.
    skip_first = resolve_recipe(
        {"algorithm": {"train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.0}}},
        "dfly",
    )
    assert skip_first is not None
    assert skip_first.stage_index(0, 10) == 1


def test_assert_recipe_capture_requires_the_teacher_signal():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
    assert recipe is not None
    assert_recipe_capture(None, {}, algorithm_name="dflash")  # no recipe, no requirement
    with pytest.raises(ValueError, match="store_final_hidden=true"):
        assert_recipe_capture(recipe, {"capture": {}}, algorithm_name="dflash")
    assert_recipe_capture(recipe, {"capture": {"store_final_hidden": True}}, algorithm_name="dflash")


# --------------------------------------------------------------------------- #
# plan digest identity
# --------------------------------------------------------------------------- #


def _plan_cfg(recipe=None) -> dict:
    cfg = {
        "drafter": {
            "mask_token_id": 255,
            "block_size": 8,
            "num_anchors": 256,
            "num_hidden_layers": 3,
            "loss_decay_gamma": 7.0,
        },
        "train": {"seed": 42, "micro_batch_size": 2},
    }
    if recipe is not None:
        cfg["algorithm"] = {"train_recipe": recipe}
    return cfg


def test_plan_digest_recipe_sensitivity():
    base = training_plan_digest(_plan_cfg(), "dflash")
    # The legacy pinned digest (see test_training_plan_fusion): an unset recipe
    # must not perturb pre-recipe plans at all.
    assert base == "b1a468221c98d42428f57f529f2b04079e0b23afe9504459928f8cd8e5ed535b"
    by_name = training_plan_digest(_plan_cfg("upstream_two_stage"), "dflash")
    explicit = training_plan_digest(
        _plan_cfg({"name": "upstream_two_stage", "cold_start_fraction": 0.5}), "dflash"
    )
    shorter = training_plan_digest(
        _plan_cfg({"name": "upstream_two_stage", "cold_start_fraction": 0.25}), "dflash"
    )
    controlled = training_plan_digest(_plan_cfg("controlled_ce"), "dflash")
    assert by_name == explicit           # default fraction == restated 0.5
    assert by_name != base               # selecting a recipe changes the plan
    assert shorter != by_name            # the stage boundary is part of the plan
    assert controlled != base            # explicit controlled_ce is still a record
    knobs = plan_recipe_knobs(_plan_cfg("upstream_two_stage"), "dflash")
    assert knobs["train_recipe"]["scheme"] == RECIPE_SCHEME_VERSION
    assert [stage["name"] for stage in knobs["train_recipe"]["stages"]] == [
        "cold_start_lk_dpace",
        "e2e_tv",
    ]


# --------------------------------------------------------------------------- #
# formula parity
# --------------------------------------------------------------------------- #


def _ref_dpace(confidences: torch.Tensor, alpha: float) -> torch.Tensor:
    smoothed = (1.0 - alpha) * confidences.double() + alpha
    n = smoothed.numel()
    weights = torch.zeros(n, dtype=torch.float64)
    for j in range(n):
        for t in range(j, n):
            prod = 1.0
            for s in range(t + 1):
                prod *= float(smoothed[s])
            weights[j] += prod
    return weights


def test_dpace_position_weights_match_the_suffix_product_reference():
    torch.manual_seed(0)
    confidences = torch.rand(9)
    for alpha in (0.0, 0.5, 1.0):
        got = dpace_position_weights(confidences, alpha)
        ref = _ref_dpace(confidences, alpha)
        assert torch.allclose(got.double(), ref, atol=1e-5), alpha
    # alpha=1 degenerates to the suffix sum of ones (weights n, n-1, ..., 1).
    ones = dpace_position_weights(torch.full((4,), 0.3), 1.0)
    assert torch.allclose(ones, torch.tensor([4.0, 3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="dpace_alpha"):
        dpace_position_weights(torch.rand(4), 1.2)


def _ref_lk(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor, loss_type: str, eta: float) -> float:
    p = torch.softmax(teacher.double(), dim=-1)
    log_q = torch.log_softmax(student.double(), dim=-1)
    q = log_q.exp()
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)
    if loss_type == "hybrid":
        lam = torch.exp(-eta * (1.0 - tv))
        ell = lam * kl + (1.0 - lam) * tv
    elif loss_type == "kl":
        ell = kl
    elif loss_type == "tv":
        ell = tv
    else:
        ell = -torch.log((1.0 - tv).clamp_min(1e-10))
    w = weights.double()
    return float((ell * w).sum() / w.sum().clamp_min(1e-6))


@pytest.mark.parametrize("loss_type", ["hybrid", "alpha", "kl", "tv"])
def test_lk_term_matches_direct_reference(loss_type):
    torch.manual_seed(1)
    rows, vocab = 37, 11
    student = torch.randn(rows, vocab)
    teacher = torch.randn(rows, vocab)
    weights = torch.rand(rows) * (torch.arange(rows) % 3 == 0)   # includes zeros
    got = float(lk_term(student, teacher, weights, loss_type=loss_type, eta=3.0))
    ref = _ref_lk(student, teacher, weights, loss_type=loss_type, eta=3.0)
    assert got == pytest.approx(ref, rel=1e-4, abs=1e-5)


def test_lk_term_rejects_unknown_loss_type():
    with pytest.raises(ValueError, match="unknown lk_loss_type"):
        lk_term(torch.randn(3, 5), torch.randn(3, 5), torch.ones(3), loss_type="nope", eta=3.0)


def test_lk_hybrid_eta_zero_is_pure_tv():
    torch.manual_seed(2)
    student, teacher = torch.randn(5, 7), torch.randn(5, 7)
    weights = torch.ones(5)
    got = float(lk_term(student, teacher, weights, loss_type="hybrid", eta=0.0))
    ref = _ref_lk(student, teacher, weights, loss_type="hybrid", eta=0.0)
    assert got == pytest.approx(ref, rel=1e-4)


def test_lk_kl_and_tv_types_and_the_accept_collector():
    torch.manual_seed(3)
    rows, vocab = 23, 9
    student = torch.randn(rows, vocab)
    teacher = torch.randn(rows, vocab)
    weights = torch.ones(rows)
    p = torch.softmax(teacher.double(), dim=-1)
    q = torch.softmax(student.double(), dim=-1)
    accept_ref = torch.minimum(p, q).sum(dim=-1)

    kl = float(lk_term(student, teacher, weights, loss_type="kl", eta=3.0))
    tv = float(lk_term(student, teacher, weights, loss_type="tv", eta=3.0))
    assert kl == pytest.approx(_ref_lk(student, teacher, weights, "kl", 3.0), rel=1e-4)
    assert tv == pytest.approx(_ref_lk(student, teacher, weights, "tv", 3.0), rel=1e-4)
    # The NeMo DSpark probability term is the L1 geometry: |p - q|_1 == 2 * TV,
    # so no separate branch is needed to reproduce it from the same chunks.
    l1_ref = float((p - q).abs().sum(dim=-1).mean())
    assert 2.0 * tv == pytest.approx(l1_ref, rel=1e-4)

    collected: dict = {}
    total = float(
        lk_term(student, teacher, weights, loss_type="kl", eta=3.0, accept_out=collected)
    )
    assert total == pytest.approx(kl, rel=1e-6, abs=1e-7)
    accept = torch.cat([collected[start] for start in sorted(collected)], dim=0)
    assert accept.shape == (rows,)
    assert not accept.requires_grad
    assert torch.allclose(accept.double(), accept_ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# upstream_dspark recipe (DSpark faithful objective)
# --------------------------------------------------------------------------- #


def test_upstream_dspark_recipe_resolves_for_dspark_only():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_dspark"}}, "dspark")
    assert recipe is not None
    stage = recipe.stages[0]
    assert recipe.needs_teacher
    assert (stage.ce_coef, stage.lk_coef, stage.e2e_tv_coef) == (0.0, 1.0, 0.0)
    assert stage.lk_loss_type == "kl"
    assert stage.weighting == "decay"
    assert stage.confidence_coef == 1.0

    tuned = resolve_recipe(
        {"algorithm": {"train_recipe": {"name": "upstream_dspark", "confidence_coef": 0.25}}},
        "dspark",
    )
    assert tuned is not None
    assert tuned.stages[0].confidence_coef == 0.25

    with pytest.raises(ValueError, match="not supported for algorithm"):
        resolve_recipe({"algorithm": {"train_recipe": "upstream_dspark"}}, "dflash")
    with pytest.raises(ValueError, match="unsupported keys"):
        resolve_recipe(
            {"algorithm": {"train_recipe": {"name": "upstream_dspark", "cold_start_fraction": 0.5}}},
            "dspark",
        )
    with pytest.raises(ValueError, match="confidence_coef must be >= 0"):
        resolve_recipe(
            {"algorithm": {"train_recipe": {"name": "upstream_dspark", "confidence_coef": -0.1}}},
            "dspark",
        )

    assert_recipe_capture(
        recipe, {"capture": {"store_final_hidden": True}}, algorithm_name="dspark"
    )
    with pytest.raises(ValueError, match="store_final_hidden=true"):
        assert_recipe_capture(recipe, {"capture": {}}, algorithm_name="dspark")

    # The confidence coefficient enters the plan digest. Recipes that never
    # select it keep byte-identical stage dicts (their checkpoints stay
    # exactly resumable).
    knobs = plan_recipe_knobs({"algorithm": {"train_recipe": "upstream_dspark"}}, "dspark")
    assert knobs["train_recipe"]["stages"][0]["confidence_coef"] == 1.0
    dfly = plan_recipe_knobs({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly")
    assert "confidence_coef" not in dfly["train_recipe"]["stages"][0]


def test_plan_digest_records_the_dspark_recipe():
    base = training_plan_digest(_plan_cfg(), "dspark")
    by_name = training_plan_digest(_plan_cfg("upstream_dspark"), "dspark")
    tuned = training_plan_digest(
        _plan_cfg({"name": "upstream_dspark", "confidence_coef": 0.5}), "dspark"
    )
    assert by_name != base
    assert tuned != by_name
    knobs = plan_recipe_knobs(_plan_cfg("upstream_dspark"), "dspark")
    assert knobs["train_recipe"]["stages"][0]["confidence_coef"] == 1.0


def _dspark_recipe_fixture():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_dspark"}}, "dspark")
    assert recipe is not None
    torch.manual_seed(21)
    student = torch.randn(2, 3, 5, 13, requires_grad=True)
    teacher = torch.randn(2, 3, 5, 13)
    targets = torch.randint(0, 13, (2, 3, 5))
    mask = torch.ones(2, 3, 5)
    mask[0, 1, 3:] = 0
    conf = torch.randn(2, 3, 5, requires_grad=True)
    return recipe, student, teacher, targets, mask, conf


def test_upstream_dspark_loss_matches_direct_full_vocab_reference():
    recipe, student, teacher, targets, mask, conf = _dspark_recipe_fixture()
    total, components = recipe_loss(
        recipe.stages[0],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
        confidence_logits=conf,
    )
    # Direct double-precision reference over the full vocab (no truncation).
    p = torch.softmax(teacher.double(), dim=-1)
    log_q = torch.log_softmax(student.detach().double(), dim=-1)
    q = log_q.exp()
    kl = (p * (p.clamp_min(1e-9).log() - log_q)).sum(dim=-1)  # [B, n, S]
    accept = torch.minimum(p, q).sum(dim=-1)
    depth = torch.exp(-torch.arange(5).double() / 7.0).view(1, 1, 5)
    w = mask.double() * depth
    kl_ref = float((kl * w).sum() / w.sum().clamp_min(1e-6))
    prob = torch.sigmoid(conf.detach().double())
    bce = -(accept * torch.log(prob) + (1 - accept) * torch.log(1 - prob))
    bce_ref = float((bce * w).sum() / w.sum().clamp_min(1e-6))

    assert components["lk_loss"] == pytest.approx(kl_ref, rel=1e-4, abs=1e-5)
    assert components["confidence_bce"] == pytest.approx(bce_ref, rel=1e-4, abs=1e-5)
    assert components["confidence_target_mean"] == pytest.approx(
        float((accept * mask.double()).sum() / mask.double().sum()), rel=1e-5, abs=1e-6
    )
    assert float(total.detach()) == pytest.approx(kl_ref + bce_ref, rel=1e-4, abs=1e-5)

    # The analytic target is detached: the BCE trains only the confidence head,
    # so the student logits receive exactly the KL gradient;
    # the confidence logits receive their own BCE gradient.
    student_ref = student.detach().clone().requires_grad_(True)
    kl_only, _ = recipe_loss(
        dataclasses.replace(recipe.stages[0], confidence_coef=0.0),
        student_logits=student_ref,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
    )
    kl_only.backward()
    total.backward()
    assert student.grad is not None
    assert torch.allclose(student.grad, student_ref.grad, atol=1e-6)
    assert conf.grad is not None
    assert float(conf.grad.abs().sum()) > 0.0


def test_upstream_dspark_confidence_requires_logits_and_distribution_term():
    recipe, student, teacher, targets, mask, conf = _dspark_recipe_fixture()
    with pytest.raises(ValueError, match="confidence_logits"):
        recipe_loss(
            recipe.stages[0],
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=teacher,
            loss_decay_gamma=7.0,
        )
    with pytest.raises(ValueError, match=r"same \[B, n, S\] shape"):
        recipe_loss(
            recipe.stages[0],
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=teacher,
            loss_decay_gamma=7.0,
            confidence_logits=conf[:, :, :2],
        )
    with pytest.raises(ValueError, match="no distribution term"):
        recipe_loss(
            dataclasses.replace(recipe.stages[0], lk_coef=0.0),
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=teacher,
            loss_decay_gamma=7.0,
            confidence_logits=conf,
        )


def test_upstream_dspark_loss_is_chunk_invariant(monkeypatch):
    recipe, student, teacher, targets, mask, conf = _dspark_recipe_fixture()
    full_total, full = recipe_loss(
        recipe.stages[0],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
        confidence_logits=conf,
    )
    monkeypatch.setattr(obj, "LK_CHUNK_ROWS", 4)
    chunk_total, chunk = recipe_loss(
        recipe.stages[0],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
        confidence_logits=conf,
    )
    assert torch.allclose(full_total, chunk_total, atol=1e-6)
    for key in ("lk_loss", "confidence_bce", "confidence_target_mean", "confidence_pred_mean"):
        assert chunk[key] == pytest.approx(full[key], rel=1e-5, abs=1e-6)


def _ref_e2e(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor):
    p = torch.softmax(teacher.double(), dim=-1)
    q = torch.softmax(student.double(), dim=-1)
    alpha = torch.minimum(p, q).sum(dim=-1)          # [B, n, S]
    m = mask.double()
    alpha_eff = alpha * m + (1.0 - m)
    prefix = torch.cumprod(alpha_eff, dim=-1)
    gamma = m.sum(dim=-1).clamp(min=1.0)
    accept = (prefix * m).sum(dim=-1)                # [B, n]
    valid = (m.sum(dim=-1) > 0).double()
    per_block = 1.0 - accept / gamma
    loss = (per_block * valid).sum() / valid.sum().clamp(min=1.0)
    accept_len = (accept * valid).sum() / valid.sum().clamp(min=1.0)
    return float(loss), float(accept_len)


def test_e2e_tv_matches_direct_reference_with_masked_slots():
    torch.manual_seed(3)
    student = torch.randn(2, 3, 5, 9)
    teacher = torch.randn(2, 3, 5, 9)
    mask = torch.ones(2, 3, 5)
    mask[0, 1] = 0          # fully masked block -> excluded from the mean
    mask[0, 0, 2:] = 0      # partial mask: masked slots act as identity
    mask[1, 2, 3] = 0
    loss, accept_len = e2e_tv_term(student, teacher, mask)
    ref_loss, ref_accept = _ref_e2e(student, teacher, mask)
    assert float(loss) == pytest.approx(ref_loss, rel=1e-4, abs=1e-6)
    assert float(accept_len) == pytest.approx(ref_accept, rel=1e-4, abs=1e-6)


def test_loss_arithmetic_is_chunk_invariant(monkeypatch):
    torch.manual_seed(4)
    student = torch.randn(50, 13)
    teacher = torch.randn(50, 13)
    weights = torch.rand(50)
    full = float(lk_term(student, teacher, weights, loss_type="hybrid", eta=3.0))
    student4 = student.view(2, 5, 5, 13)
    teacher4 = teacher.view(2, 5, 5, 13)
    mask = torch.ones(2, 5, 5)
    mask[1, 3, 4] = 0
    e2e_full = e2e_tv_term(student4, teacher4, mask)

    monkeypatch.setattr(obj, "LK_CHUNK_ROWS", 7)
    monkeypatch.setattr(obj, "E2E_CHUNK_BLOCKS", 3)
    assert float(lk_term(student, teacher, weights, loss_type="hybrid", eta=3.0)) == full
    e2e_chunked = e2e_tv_term(student4, teacher4, mask)
    assert torch.allclose(e2e_chunked[0], e2e_full[0], atol=1e-6)
    assert torch.allclose(e2e_chunked[1], e2e_full[1], atol=1e-6)


def test_per_slot_nll_matches_cross_entropy():
    torch.manual_seed(5)
    logits = torch.randn(4, 3, 6, 17)
    targets = torch.randint(0, 17, (4, 3, 6))
    got = per_slot_nll(logits, targets)
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 17).double(), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    assert got.shape == targets.shape
    assert torch.allclose(got.double(), ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# recipe_loss stage behavior
# --------------------------------------------------------------------------- #


def _stage_fixture():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dfly")
    assert recipe is not None
    torch.manual_seed(6)
    student = torch.randn(2, 4, 6, 10, requires_grad=True)
    teacher = torch.randn(2, 4, 6, 10)
    targets = torch.randint(0, 10, (2, 4, 6))
    mask = torch.ones(2, 4, 6)
    mask[0, 2, 4:] = 0
    return recipe, student, teacher, targets, mask


def test_recipe_stage_one_is_lk_only_and_needs_the_teacher():
    recipe, student, teacher, targets, mask = _stage_fixture()
    with pytest.raises(ValueError, match="teacher"):
        recipe_loss(
            recipe.stages[0],
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=None,
            loss_decay_gamma=7.0,
        )
    total, components = recipe_loss(
        recipe.stages[0],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
    )
    assert set(components) == {"lk_loss"}
    assert float(total.detach()) == pytest.approx(components["lk_loss"], rel=1e-5)
    total.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_recipe_stage_two_mixes_dpace_ce_and_e2e_tv():
    recipe, student, teacher, targets, mask = _stage_fixture()
    total, components = recipe_loss(
        recipe.stages[1],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=teacher,
        loss_decay_gamma=7.0,
    )
    assert set(components) == {"ce_loss", "e2e_tv_loss", "e2e_accept_len"}
    assert float(total.detach()) == pytest.approx(
        components["ce_loss"] + components["e2e_tv_loss"], rel=1e-5, abs=1e-6
    )
    assert 0.0 <= components["e2e_accept_len"] <= 6.0
    total.backward()
    assert student.grad is not None


def test_recipe_controlled_ce_stage_needs_no_teacher():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "controlled_ce"}}, "dflash")
    assert recipe is not None
    torch.manual_seed(7)
    student = torch.randn(2, 3, 4, 8, requires_grad=True)
    targets = torch.randint(0, 8, (2, 3, 4))
    mask = torch.ones(2, 3, 4)
    total, components = recipe_loss(
        recipe.stages[0],
        student_logits=student,
        target_ids=targets,
        block_mask=mask,
        teacher_logits=None,
        loss_decay_gamma=None,
    )
    assert set(components) == {"ce_loss"}
    assert torch.isfinite(total)
    total.backward()
    assert student.grad is not None


# --------------------------------------------------------------------------- #
# teacher gather
# --------------------------------------------------------------------------- #


def test_gather_teacher_logits_aligns_slots_and_is_detached():
    torch.manual_seed(8)
    lm_head = torch.nn.Linear(6, 13, bias=False)
    teacher_hidden = torch.randn(2, 9, 6)
    label_indices = torch.tensor(
        [
            [[1, 2, 3, 4], [0, 5, 6, 7]],
            [[2, 3, 4, 5], [1, 2, 8, 3]],
        ]
    )  # [B=2, n=2, bs=4]
    out = gather_teacher_logits(lm_head, teacher_hidden, label_indices, first_slot=1)
    assert out.shape == (2, 2, 3, 13)
    assert not out.requires_grad
    for b in range(2):
        for k in range(2):
            for j, label in enumerate([2, 3, 4] if b == 0 and k == 0 else label_indices[b, k, 1:]):
                source = max(int(label) - 1, 0)
                ref = lm_head(teacher_hidden[b, source].unsqueeze(0)).squeeze(0)
                assert torch.allclose(out[b, k, j], ref, atol=1e-6)


def test_teacher_stream_matches_the_materialized_gather_at_any_offset():
    torch.manual_seed(9)
    lm_head = torch.nn.Linear(6, 13, bias=False)
    teacher_hidden = torch.randn(2, 9, 6)
    label_indices = torch.randint(0, 9, (2, 2, 4))
    stream = TeacherLogitsStream(lm_head, teacher_hidden, label_indices, first_slot=1)
    full = gather_teacher_logits(lm_head, teacher_hidden, label_indices, first_slot=1)
    assert torch.equal(stream.full(), full.reshape(stream.num_rows, -1))
    # A different slab shape may pick a different (still equivalent) matmul
    # kernel, so compare numerically, not bitwise.
    assert torch.allclose(stream.logits(3, 17), full.reshape(stream.num_rows, -1)[3:17], atol=1e-6)
    assert stream.logits(stream.num_rows, stream.num_rows + 5).shape[0] == 0


def test_chunked_teacher_loss_matches_materialized_loss(monkeypatch):
    recipe, student, teacher, targets, mask = _stage_fixture()
    rows = teacher.reshape(-1, teacher.size(-1))
    calls = []

    def chunk_fn(start, stop):
        calls.append((start, stop))
        return rows[start:stop]

    # Force multiple stream chunks that do not align with the 6-slot blocks.
    monkeypatch.setattr(obj, "LK_CHUNK_ROWS", 11)
    monkeypatch.setattr(obj, "E2E_CHUNK_BLOCKS", 2)

    for stage in recipe.stages:
        want_total, want_components = recipe_loss(
            stage,
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=teacher,
            loss_decay_gamma=7.0,
        )
        got_total, got_components = recipe_loss(
            stage,
            student_logits=student,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=None,
            teacher_chunk_fn=chunk_fn,
            loss_decay_gamma=7.0,
        )
        assert float(got_total.detach()) == pytest.approx(float(want_total.detach()), rel=1e-6, abs=1e-7)
        assert got_components == pytest.approx(want_components, rel=1e-6, abs=1e-7)
    assert len(calls) > 4  # streamed, not one materialized gather


def test_lk_and_e2e_teacher_arguments_are_exclusive():
    with pytest.raises(ValueError, match="exactly one of teacher_logits"):
        lk_term(torch.randn(3, 5), None, torch.ones(3), loss_type="hybrid", eta=3.0)
    with pytest.raises(ValueError, match="exactly one of teacher_logits"):
        lk_term(
            torch.randn(3, 5),
            torch.randn(3, 5),
            torch.ones(3),
            loss_type="hybrid",
            eta=3.0,
            teacher_chunk_fn=lambda start, stop: torch.randn(stop - start, 5),
        )
    with pytest.raises(ValueError, match="exactly one of teacher_logits_pb"):
        e2e_tv_term(torch.randn(1, 1, 2, 5), None, torch.ones(1, 1, 2))


def test_checkpointed_chunks_keep_gradients(monkeypatch):
    """The chunk activation checkpointing is memory-only: same gradients."""
    recipe, student, teacher, targets, mask = _stage_fixture()
    stage = recipe.stages[1]

    def grads():
        fresh = student.detach().clone().requires_grad_(True)
        total, _ = recipe_loss(
            stage,
            student_logits=fresh,
            target_ids=targets,
            block_mask=mask,
            teacher_logits=teacher,
            loss_decay_gamma=7.0,
        )
        total.backward()
        return fresh.grad

    checkpointed = grads()
    monkeypatch.setattr(obj, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
    plain = grads()
    assert torch.allclose(checkpointed, plain, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# trainer module integration
# --------------------------------------------------------------------------- #


def _tiny_module(train_recipe=None):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel
    from vlm_dflash.dflash.trainer_core import DFlashTrainerModule

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=8,
        num_target_layers=2,
        dflash_config={"mask_token_id": 255, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(config)
    embed = torch.nn.Embedding(256, 64)
    lm_head = torch.nn.Linear(64, 256, bias=False)
    lm_head.weight = embed.weight
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=255,
        block_size=8,
        attention_backend="sdpa",
        num_anchors=16,
        loss_decay_gamma=7.0,
        train_recipe=train_recipe,
    )


def _tiny_batch(seed: int = 12):
    torch.manual_seed(seed)
    batch_size, seq_len = 2, 48
    input_ids = torch.arange(seq_len).remainder(200).add(10).unsqueeze(0).repeat(batch_size, 1)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :8] = 0
    hidden = torch.randn(batch_size, seq_len, 128)
    teacher_hidden = torch.randn(batch_size, seq_len, 64)
    return input_ids, loss_mask, hidden, teacher_hidden


def test_recipe_trainer_switches_stages_via_progress_hook():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
    assert recipe is not None
    module = _tiny_module(train_recipe=recipe)
    assert module.recipe_needs_teacher
    assert module.recipe_stage_index == 0
    input_ids, loss_mask, hidden, teacher_hidden = _tiny_batch()

    module.set_training_progress(0, 4)
    assert module.recipe_stage_index == 0
    metrics0 = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=teacher_hidden
    )
    assert torch.isfinite(metrics0.loss)
    extra0 = metrics0.extra_metrics or {}
    assert extra0.get("recipe_stage") == 0.0
    assert "lk_loss" in extra0 and "ce_loss" not in extra0 and "e2e_tv_loss" not in extra0
    metrics0.loss.backward()
    assert any(p.grad is not None for p in module.draft_model.parameters())

    module.set_training_progress(2, 4)
    assert module.recipe_stage_index == 1
    metrics1 = module(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, teacher_hidden=teacher_hidden
    )
    extra1 = metrics1.extra_metrics or {}
    assert extra1.get("recipe_stage") == 1.0
    assert {"ce_loss", "e2e_tv_loss", "e2e_accept_len"} <= set(extra1)
    assert torch.isfinite(metrics1.loss)


def test_recipe_trainer_requires_the_teacher_tensor():
    recipe = resolve_recipe({"algorithm": {"train_recipe": "upstream_two_stage"}}, "dflash")
    module = _tiny_module(train_recipe=recipe)
    input_ids, loss_mask, hidden, _ = _tiny_batch()
    module.set_training_progress(0, 4)
    with pytest.raises(ValueError, match="teacher"):
        module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)


def test_default_trainer_is_unchanged():
    module = _tiny_module(train_recipe=None)
    assert module.recipe_needs_teacher is False
    assert module.recipe_stage_index is None
    module.set_training_progress(3, 4)   # no-op without a recipe
    assert module.recipe_stage_index is None
    input_ids, loss_mask, hidden, _ = _tiny_batch()
    metrics = module(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(metrics.loss)
    assert "recipe_stage" not in (metrics.extra_metrics or {})


# --------------------------------------------------------------------------- #
# resume gating
# --------------------------------------------------------------------------- #


def _resume_validate(state: dict, *, recipe_stage=None):
    from vlm_dflash.dflash.train import validate_resume_state

    validate_resume_state(
        state,
        algorithm_name="dfly",
        mode="online",
        feature_source_digest="d" * 64,
        total_steps=6,
        contract={},
        plan_version=FUSION_PLAN_VERSION,
        plan_digest="p" * 64,
        recipe_stage=recipe_stage,
    )


def _resume_state(*, recipe_stage=None, step: int = 3) -> dict:
    state = {
        "format": "vlm-dflash-train-state-v1",
        "step": step,
        "steps_total_planned": 6,
        "algorithm": "dfly",
        "feature_mode": "online",
        "feature_source_digest": "d" * 64,
        "training_plan_version": FUSION_PLAN_VERSION,
        "training_plan_digest": "p" * 64,
    }
    if recipe_stage is not None:
        state["recipe_stage"] = recipe_stage
    return state


def test_resume_recipe_stage_matching_passes():
    _resume_validate(_resume_state(), recipe_stage=None)
    _resume_validate(_resume_state(recipe_stage=1), recipe_stage=1)
    _resume_validate(_resume_state(recipe_stage=0), recipe_stage=0)


def test_resume_recipe_stage_mismatch_is_refused():
    from vlm_dflash.dflash.train import validate_resume_state

    with pytest.raises(ValueError, match="Resume recipe mismatch"):
        _resume_validate(_resume_state(), recipe_stage=1)
    with pytest.raises(ValueError, match="Resume recipe mismatch"):
        _resume_validate(_resume_state(recipe_stage=1), recipe_stage=None)
    with pytest.raises(ValueError, match="Resume recipe-stage mismatch"):
        _resume_validate(_resume_state(recipe_stage=0), recipe_stage=1)
    # recipe_stage defaults to None: legacy callers keep working unchanged.
    validate_resume_state(
        _resume_state(),
        algorithm_name="dfly",
        mode="online",
        feature_source_digest="d" * 64,
        total_steps=6,
        contract={},
        plan_version=FUSION_PLAN_VERSION,
        plan_digest="p" * 64,
    )


def test_expected_stage_matches_gather_train_state():
    from vlm_dflash.dflash.train import _expected_recipe_stage, gather_train_state

    cfg = {"algorithm": {"train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.5}}}
    recipe = resolve_recipe(cfg, "dflash")
    module = _tiny_module(train_recipe=recipe)
    optimizer = torch.optim.AdamW(module.draft_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    masters = {name: p.detach().float().clone() for name, p in module.draft_model.named_parameters()}

    for step in (0, 4, 5, 10):
        state = gather_train_state(
            module,
            masters,
            optimizer,
            scheduler,
            step=step,
            micro_step=step,
            skipped_batches=0,
            history=[],
            steps_total_planned=10,
        )
        expected = _expected_recipe_stage(cfg, "dflash", step=step, total_steps=10)
        assert state["recipe_stage"] == expected
        _resume_validate(_resume_state(recipe_stage=state["recipe_stage"], step=step), recipe_stage=expected)

    legacy = gather_train_state(
        _tiny_module(train_recipe=None),
        masters,
        optimizer,
        scheduler,
        step=3,
        micro_step=3,
        skipped_batches=0,
        history=[],
        steps_total_planned=10,
    )
    assert "recipe_stage" not in legacy


# --------------------------------------------------------------------------- #
# end-to-end through the real train_dflash path (tiny Qwen2.5-VL target)
# --------------------------------------------------------------------------- #


def _tiny_recipe_setup(monkeypatch, tmp_path, *, store_final_hidden: bool):
    """Tiny real target + real capture path (reuses the online-mode fixtures)."""
    import json as _json

    from tests.test_online_mode import (
        _patch_drafter_visual_check,
        _patch_tiny_draft_build,
        _sync_online_run_dir,
        _tiny_contract,
        _tiny_qwen25_vl_model,
        _tiny_train_dflash_cfg,
        _write_tiny_feature_store,
        online_module,
    )
    from vlm_dflash.target.sample_capture import extract_target_head

    model, processor = _tiny_qwen25_vl_model()
    contract = _tiny_contract()
    run_dir = _sync_online_run_dir(tmp_path, model, processor, rows=2, answer_len=12)
    monkeypatch.setattr(online_module, "load_target", lambda cfg, contract_: (model, processor))
    cfg = _tiny_train_dflash_cfg()
    cfg["algorithm"] = {"train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.5}}
    cfg["capture"]["store_final_hidden"] = True
    source = online_module.OnlineFeatureSource(cfg, contract, run_dir)
    if not store_final_hidden:
        # Capture without the teacher signal: the store (and its manifest) then
        # lack final_hidden exactly like a pre-capture-v2 store.
        source.store_final_hidden = False
    records = [source.extract(index) for index in range(len(source.rows))]
    assert (records[0].get("final_hidden") is not None) == store_final_hidden

    embed_weight, _lm_head_weight, _tied = extract_target_head(model, contract)
    features_dir = run_dir / "features"
    _write_tiny_feature_store(features_dir, contract, records, embed_weight)
    manifest_path = features_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stores_final_hidden"] = store_final_hidden
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")

    _patch_tiny_draft_build(monkeypatch)
    _patch_drafter_visual_check(monkeypatch)
    return cfg, contract, run_dir, features_dir


def test_recipe_training_runs_and_resumes_end_to_end(monkeypatch, tmp_path):
    import json as _json

    from vlm_dflash.dflash.train import train_dflash

    cfg, contract, run_dir, _features_dir = _tiny_recipe_setup(
        monkeypatch, tmp_path, store_final_hidden=True
    )
    summary = train_dflash(cfg, run_dir, contract, steps=4, feature_mode="offline")
    assert summary["train_recipe"]["name"] == "upstream_two_stage"
    assert summary["steps"] == 4

    rows = [
        _json.loads(line)
        for line in (run_dir / "train" / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # steps=4, cold_start_fraction=0.5 -> steps 0/1 are stage 0 (LK), 2/3 stage 1.
    stages = [row["recipe_stage"] for row in rows]
    assert stages == [0.0, 0.0, 1.0, 1.0]
    assert all("lk_loss" in row for row in rows[:2])
    assert all("ce_loss" in row and "e2e_tv_loss" in row for row in rows[2:])

    state = torch.load(run_dir / "train" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert state["recipe_stage"] == 1
    final_ckpt = torch.load(run_dir / "train" / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert final_ckpt["train_recipe"]["name"] == "upstream_two_stage"

    # Exact resume: unchanged recipe passes the gate and replays nothing.
    resumed = train_dflash(cfg, run_dir, contract, steps=4, resume=True, feature_mode="offline")
    assert resumed["resumed_from_step"] == 4

    # A changed stage boundary is a different plan: refused before any write.
    changed = _json.loads(_json.dumps(cfg))
    changed["algorithm"]["train_recipe"]["cold_start_fraction"] = 0.25
    with pytest.raises(ValueError, match="training-plan mismatch"):
        train_dflash(changed, run_dir, contract, steps=4, resume=True, feature_mode="offline")


def test_recipe_training_refuses_a_store_without_the_teacher_signal(monkeypatch, tmp_path):
    from vlm_dflash.dflash.train import train_dflash

    cfg, contract, run_dir, _features_dir = _tiny_recipe_setup(
        monkeypatch, tmp_path, store_final_hidden=False
    )
    with pytest.raises(RuntimeError, match="stores_final_hidden"):
        train_dflash(cfg, run_dir, contract, steps=2, feature_mode="offline")
    assert not (run_dir / "train").exists()
