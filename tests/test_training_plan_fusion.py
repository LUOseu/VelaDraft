"""DFlare/DFly training-plan identity: resolved knobs, scheme, resume policy.

The five pre-existing algorithms must keep their v1 payload byte-identical
(their recorded digests stay resumable), while the fusion methods record a v2
scheme that covers every resolved knob of their effective config.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm_dflash.dflash.train import validate_resume_state
from vlm_dflash.training_plan import (
    FUSION_PLAN_VERSION,
    PLAN_VERSION,
    plan_scheme_version,
    training_plan_digest,
)


def _legacy_cfg() -> dict:
    return {
        "drafter": {
            "mask_token_id": 255,
            "block_size": 8,
            "num_anchors": 256,
            "num_hidden_layers": 3,
            "loss_decay_gamma": 7.0,
        },
        "train": {"seed": 42, "micro_batch_size": 2},
    }


# Digests recorded by the v1 scheme (main 3930dbfb) for a fixed config; a
# changed value means the ten pre-existing algorithms lost resume compatibility.
LEGACY_V1_DIGESTS = {
    "dflash": "b1a468221c98d42428f57f529f2b04079e0b23afe9504459928f8cd8e5ed535b",
    "dspark": "5e960d4a51eb3309da34876bb03bcea6191ad00ab09615a02393e20ef2657fc9",
    "domino": "c37c688542fc86f267a300484c1235912e15e4ece734805839ebf96c24a26a2c",
    "jetspec": "8c0d032205470edec763a4116af82ab5223fc4454fc0e12454dd2e55e0c5837d",
    "dflash2": "d8851b49b9033bbafabb2e6d51dc9583f77a28d8d182c5b857fdca278685ea9b",
}


def test_legacy_scheme_payload_is_byte_identical():
    for algorithm, digest in LEGACY_V1_DIGESTS.items():
        assert training_plan_digest(_legacy_cfg(), algorithm) == digest, algorithm
        assert plan_scheme_version(algorithm) == PLAN_VERSION


def _contract() -> dict:
    return {
        "text": {"hidden_size": 32},
        "layers": {"draft_layer_ids": [1, 9, 17, 25, 33]},
    }


def _cfg(algorithm: str, **knobs) -> dict:
    cfg = {"drafter": {"mask_token_id": 255, "block_size": 8, "num_hidden_layers": 3}, "train": {}}
    if knobs:
        cfg["algorithm"] = {algorithm: knobs}
    return cfg


def _digest(cfg, algorithm: str) -> str:
    return training_plan_digest(cfg, algorithm, contract=_contract())


@pytest.mark.parametrize(
    "algorithm,knobs",
    [
        ("dflare", {"fusion_trainable": False}),
        ("dflare", {"fusion_init": "uniform"}),
        ("dflare", {"separate_context_kv": False}),
        ("dflare", {"context_layer_ids": [9, 1, 17, 25, 33]}),  # order changes semantics
        ("dflare", {"context_layer_ids": [1, 17, 33]}),         # subset
        ("dfly", {"fusion_trainable": False}),
        ("dfly", {"fusion_init": "uniform"}),
        ("dfly", {"enable_hidden_correction": False}),
        ("dfly", {"hidden_correction_intermediate_size": 64}),
        ("dfly", {"context_layer_ids": [9, 1, 17, 25, 33]}),
    ],
)
def test_fusion_knobs_change_the_digest(algorithm, knobs):
    base = _digest(_cfg(algorithm), algorithm)
    assert _digest(_cfg(algorithm, **knobs), algorithm) != base


def test_fusion_none_defaults_normalize_against_the_capture():
    contract = _contract()
    for algorithm in ("dflare", "dfly"):
        base = training_plan_digest(_cfg(algorithm), algorithm, contract=contract)
        # null context == the explicit full captured list, in capture order.
        same = _cfg(algorithm, context_layer_ids=[1, 9, 17, 25, 33])
        assert training_plan_digest(same, algorithm, contract=contract) == base
        # Restating every other default is a no-op.
        if algorithm == "dflare":
            defaults = {
                "separate_context_kv": True,
                "fusion_trainable": True,
                "fusion_init": "distance",
            }
        else:
            defaults = {
                "fusion_trainable": True,
                "fusion_init": "distance",
                "enable_hidden_correction": True,
                "hidden_correction_intermediate_size": None,
            }
        restated = _cfg(algorithm, **defaults)
        assert training_plan_digest(restated, algorithm, contract=contract) == base
    # DFly: null correction width == the hidden size; disabled ignores the width.
    base = _digest(_cfg("dfly"), "dfly")
    assert _digest(_cfg("dfly", hidden_correction_intermediate_size=32), "dfly") == base
    disabled = _digest(_cfg("dfly", enable_hidden_correction=False), "dfly")
    assert _digest(
        _cfg("dfly", enable_hidden_correction=False, hidden_correction_intermediate_size=64), "dfly"
    ) == disabled
    # DFly forces the DFlash slot convention; the ignored YAML key is inert.
    forced = _cfg("dfly")
    forced["drafter"] = {**forced["drafter"], "sample_from_anchor": True}
    assert _digest(forced, "dfly") == base


def test_fusion_digest_requires_the_contract():
    for algorithm in ("dflare", "dfly"):
        with pytest.raises(ValueError, match="capture layout"):
            training_plan_digest(_cfg(algorithm), algorithm)


def _resume_state(**overrides) -> dict:
    state = {
        "format": "vlm-dflash-train-state-v1",
        "steps_total_planned": 6,
        "algorithm": "dfly",
        "feature_mode": "online",
        "feature_source_digest": "d" * 64,
        "training_plan_version": PLAN_VERSION,
        "training_plan_digest": "p" * 64,
    }
    state.update(overrides)
    return state


def _validate(state: dict, *, algorithm: str = "dfly", plan_digest: str = "p" * 64) -> None:
    validate_resume_state(
        state,
        algorithm_name=algorithm,
        mode="online",
        feature_source_digest="d" * 64,
        total_steps=6,
        contract={},
        plan_version=plan_scheme_version(algorithm),
        plan_digest=plan_digest,
    )


@pytest.mark.parametrize("algorithm", ["dflare", "dfly"])
def test_v1_fusion_checkpoint_is_refused_explicitly(algorithm):
    """A v1 DFlare/DFly state is refused with a dedicated message: its digest
    never covered the fusion knobs, and the gate must not back-fill one."""
    with pytest.raises(ValueError, match="did not cover the method's fusion knobs"):
        _validate(_resume_state(algorithm=algorithm), algorithm=algorithm)


def test_v2_scheme_compares_digests_within_the_scheme():
    state = _resume_state(training_plan_version=FUSION_PLAN_VERSION)
    _validate(state)
    with pytest.raises(ValueError, match="training-plan mismatch"):
        _validate(state, plan_digest="q" * 64)
    with pytest.raises(ValueError, match="plan-version mismatch"):
        _validate(_resume_state(training_plan_version="vlm-dflash-training-plan-v3"))
    # The legacy five keep the v1 scheme and are untouched by the new branch.
    legacy = _resume_state(algorithm="dflash")
    _validate(legacy, algorithm="dflash")
    with pytest.raises(ValueError, match="plan-version mismatch"):
        _validate(
            _resume_state(algorithm="dflash", training_plan_version=FUSION_PLAN_VERSION),
            algorithm="dflash",
        )


def test_payload_json_round_trip_is_stable():
    contract = _contract()
    for algorithm in ("dflash", "dflare", "dfly"):
        cfg = _cfg(algorithm)
        payload = training_plan_digest(cfg, algorithm, contract=contract)
        reloaded = json.loads(json.dumps(cfg))
        assert training_plan_digest(reloaded, algorithm, contract=contract) == payload
