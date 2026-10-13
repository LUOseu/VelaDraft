"""Training-plan identity for exact interrupted-resume checks.

``training_plan_digest`` covers every resolved knob that changes the training
trajectory: optimizer and schedule settings, batch geometry, the data-order
seed, the drafter/loss configuration and the algorithm-namespaced knobs
(resolved defaults included, so a changed default cannot slip through as a
missing key). Non-semantic fields -- device, logging cadence, worker count,
output paths -- are deliberately excluded: changing them does not change the
optimizer steps, and a resume must stay possible when a run is moved or
re-logged.

The digest is versioned: checkpoints without a recorded version predate the
gate and cannot be resumed exactly (see ``validate_resume_state``).
"""

from __future__ import annotations

from .utils import sha256_json

PLAN_VERSION = "vlm-dflash-training-plan-v1"


def _resolved_algorithm_knobs(cfg: dict, algorithm_name: str) -> dict:
    """Effective algorithm knobs with defaults filled in (or ``{}``)."""
    if algorithm_name == "dspark":
        from .dspark.factory import resolve_dspark_config

        return resolve_dspark_config(cfg)
    if algorithm_name == "domino":
        from .domino.factory import resolve_domino_config

        return resolve_domino_config(cfg)
    if algorithm_name == "jetspec":
        from .jetspec.factory import resolve_jetspec_config

        return resolve_jetspec_config(cfg)
    if algorithm_name == "dflash2":
        from .dflash2.factory import resolve_dflash2_config

        return resolve_dflash2_config(cfg)
    return {}


def training_plan_payload(cfg: dict, algorithm_name: str, dataset_len: int | None = None) -> dict:
    """Canonical, JSON-serializable description of the effective training plan.

    ``dataset_len`` fixes the row count the loader shuffles over (and therefore
    the data order and the step arithmetic); it is recorded when known so a
    changed ``--limit`` / slice cannot pass an exact-resume gate.
    """
    train = cfg.get("train") or {}
    drafter = cfg.get("drafter") or {}
    gamma_raw = drafter.get("loss_decay_gamma", 7.0)
    payload = {
        "version": PLAN_VERSION,
        "algorithm": str(algorithm_name),
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
            "sample_from_anchor": bool(drafter.get("sample_from_anchor", False)),
            "attention_backend": str(drafter.get("attention_backend", "flex_attention")),
            "spatial_rope_enabled": bool(drafter.get("spatial_rope_enabled", True)),
            "spatial_rope_mode": str(drafter.get("spatial_rope_mode", "replace")),
        },
        "algorithm_knobs": _resolved_algorithm_knobs(cfg, algorithm_name),
    }
    if dataset_len is not None:
        payload["dataset_len"] = int(dataset_len)
    return payload


def training_plan_digest(cfg: dict, algorithm_name: str, dataset_len: int | None = None) -> str:
    """Content digest of the effective training plan (see module docstring)."""
    return sha256_json(training_plan_payload(cfg, algorithm_name, dataset_len=dataset_len))
