"""Training-plan identity for exact interrupted-resume checks.

``training_plan_digest`` covers every resolved knob that changes the training
trajectory: optimizer and schedule settings, batch geometry, the data-order
seed, the drafter/loss configuration and the algorithm-namespaced knobs
(resolved defaults included, so a changed default cannot slip through as a
missing key). Non-semantic fields -- device, logging cadence, worker count,
output paths -- are deliberately excluded: changing them does not change the
optimizer steps, and a resume must stay possible when a run is moved or
re-logged.

The digest is versioned. ``vlm-dflash-training-plan-v1`` is the historically
shipped scheme and its payload for the five pre-existing algorithms is
byte-identical to previous releases, so their recorded digests stay resumable.
Algorithms whose resolved knobs were added later use a new scheme
(``vlm-dflash-training-plan-v2`` for DFlare/DFly, whose v1 payloads never
covered the fusion context / K-V / correction knobs): a v1 checkpoint of such
an algorithm is refused explicitly instead of being silently back-filled with
the new digest (see ``validate_resume_state``). Checkpoints without any
recorded version predate the gate and cannot be resumed exactly.

Algorithm-specific knobs are provided by the algorithm plugins through
``DrafterAlgorithm.plan_knobs`` (the legacy four resolve through their
factories below); the resolved values are normalized against the frozen
contract where a knob's meaning depends on the capture layout.
"""

from __future__ import annotations

from .utils import sha256_json

PLAN_VERSION = "vlm-dflash-training-plan-v1"

#: Scheme used by algorithms whose resolved knobs entered the payload after v1
#: (DFlare/DFly fusion context / K-V / correction). Their v1 checkpoints are
#: not exactly resumable and are refused with an explicit message.
FUSION_PLAN_VERSION = "vlm-dflash-training-plan-v2"

#: Algorithms recorded under :data:`FUSION_PLAN_VERSION`.
FUSION_PLAN_ALGORITHMS = frozenset({"dflare", "dfly"})

#: Drafter-section values an algorithm forces regardless of the YAML value
#: (DFly follows the DFlash slot convention in ``dfly.factory.build_draft_config``);
#: the payload records the effective value so toggling an ignored key cannot
#: fake a plan change in either direction.
_FORCED_DRAFTER_VALUES = {"dfly": {"sample_from_anchor": False}}


def plan_scheme_version(algorithm_name: str) -> str:
    """Plan-encoding version recorded and compared for ``algorithm_name``."""
    return FUSION_PLAN_VERSION if algorithm_name in FUSION_PLAN_ALGORITHMS else PLAN_VERSION


def _resolved_algorithm_knobs(cfg: dict, algorithm_name: str, contract: dict | None = None) -> dict:
    """Effective algorithm knobs with defaults filled in (or ``{}``)."""
    if algorithm_name == "dspark":
        from .dspark.factory import resolve_dspark_config
        from .objectives import plan_recipe_knobs

        knobs = resolve_dspark_config(cfg)
        # The recipe block changes the training objective, so it belongs in the
        # digest exactly like for the fusion algorithms ({} when unset, keeping
        # the pre-recipe digests byte-identical).
        knobs.update(plan_recipe_knobs(cfg, algorithm_name))
        return knobs
    if algorithm_name == "domino":
        from .domino.factory import resolve_domino_config

        return resolve_domino_config(cfg)
    if algorithm_name == "jetspec":
        from .jetspec.factory import resolve_jetspec_config

        return resolve_jetspec_config(cfg)
    if algorithm_name == "dflash2":
        from .dflash2.factory import resolve_dflash2_config

        return resolve_dflash2_config(cfg)
    # Registered algorithms provide their own training-plan fields
    # (``DrafterAlgorithm.plan_knobs``; the base default is ``{}``), so adding
    # an algorithm does not require another branch here.
    from .algorithms import get_algorithm

    hook = getattr(get_algorithm(algorithm_name), "plan_knobs", None)
    if hook is None:
        return {}
    knobs = hook(cfg, contract=contract)
    return {} if knobs is None else knobs


def training_plan_payload(
    cfg: dict, algorithm_name: str, dataset_len: int | None = None, *, contract: dict | None = None
) -> dict:
    """Canonical, JSON-serializable description of the effective training plan.

    ``dataset_len`` fixes the row count the loader shuffles over (and therefore
    the data order and the step arithmetic); it is recorded when known so a
    changed ``--limit`` / slice cannot pass an exact-resume gate.

    ``contract`` is the frozen run contract; algorithms whose resolved knobs
    depend on the capture layout (DFlare/DFly: ``context_layer_ids: null`` means
    "every captured layer, in capture order") require it so the payload records
    the effective ordered layer list instead of an ambiguous ``null``.
    """
    algorithm_name = str(algorithm_name)
    train = cfg.get("train") or {}
    drafter = cfg.get("drafter") or {}
    gamma_raw = drafter.get("loss_decay_gamma", 7.0)
    sample_from_anchor = bool(drafter.get("sample_from_anchor", False))
    for forced_key, forced_value in _FORCED_DRAFTER_VALUES.get(algorithm_name, {}).items():
        if forced_key == "sample_from_anchor":
            sample_from_anchor = bool(forced_value)
    payload = {
        "version": plan_scheme_version(algorithm_name),
        "algorithm": algorithm_name,
        "seed": int(train.get("seed", 42)),
        "micro_batch_size": int(train.get("micro_batch_size", 8)),
        "grad_accumulation_steps": max(1, int(train.get("grad_accumulation_steps", 1))),
        "num_epochs": int(train.get("num_epochs", 6)),
        "draft_dtype": str(train.get("draft_dtype", "bfloat16")),
        "optimizer": {
            "lr": float(train.get("lr", 6e-4)),
            "betas": [float(beta) for beta in train.get("betas", (0.9, 0.95))],
            "weight_decay": float(train.get("weight_decay", 0.0)),
            "warmup_ratio": float(train.get("warmup_ratio", 0.04)),
            "min_lr_ratio": float(train.get("min_lr_ratio", 0.1)),
            "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
        },
        "drafter": {
            "block_size": int(drafter.get("block_size", 8)),
            "num_anchors": int(drafter.get("num_anchors", 256)),
            "num_hidden_layers": int(drafter.get("num_hidden_layers", 3)),
            "loss_decay_gamma": None if gamma_raw is None else float(gamma_raw),
            "sample_from_anchor": sample_from_anchor,
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "spatial_rope_enabled": bool(drafter.get("spatial_rope_enabled", True)),
            "spatial_rope_mode": str(drafter.get("spatial_rope_mode", "replace")),
        },
        "algorithm_knobs": _resolved_algorithm_knobs(cfg, algorithm_name, contract=contract),
    }
    if dataset_len is not None:
        payload["dataset_len"] = int(dataset_len)
    return payload


def training_plan_digest(
    cfg: dict, algorithm_name: str, dataset_len: int | None = None, *, contract: dict | None = None
) -> str:
    """Content digest of the effective training plan (see module docstring)."""
    return sha256_json(
        training_plan_payload(cfg, algorithm_name, dataset_len=dataset_len, contract=contract)
    )
