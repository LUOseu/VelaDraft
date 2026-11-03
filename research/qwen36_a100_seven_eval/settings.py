"""Configuration, method table and run layout for the Qwen3.6 seven-method MMSpec benchmark.

Everything a run needs is read from one YAML file (configs/eval/qwen36_a100_seven.yaml) and frozen
into ``manifest.json`` before the first timed request; later stages read the manifest, not the YAML.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / "research/qwen36_seven_20261102/INFERENCE_SETTINGS.json"
CHECKPOINTS = REPO / "research/qwen36_seven_20261102/checkpoint_manifest.json"
METHODS = ("dflash", "dspark", "domino", "jetspec", "dflash2", "dflare", "dfly")
PROTOCOLS = ("natural", "fixed512")
CATEGORIES = (
    "general vqa",
    "text vqa",
    "image captioning",
    "chart understanding",
    "complex reasoning pro",
    "multi-turn conversation",
)
SUITES = ("mmspec-full", "mmspec-smoke")
# The anchor-slot DSpark bridge patches DFlashSpeculator and reuses the native
# DSpark sampler. Selecting "dspark" rewrites its custom architecture to DeepSeek.
SPEC_METHOD = dict.fromkeys(METHODS, "dflash")


def stage_names(repeats: int) -> list[str]:
    """AR0, SD0, SD1, AR1, AR2, SD2, ... (alternating which arm runs first)."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    return [
        f"{arm}_r{r}"
        for r in range(repeats)
        for arm in (("ar", "sd") if r % 2 == 0 else ("sd", "ar"))
    ]


def expand(value):
    """``${NAME}`` / ``${NAME:-default}`` from the environment in every string (site paths stay out of git)."""
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if not isinstance(value, str):
        return value

    def substitute(match):
        name, default = match.group(1), match.group(3)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise ValueError(f"environment variable {name} is required by the config")
        return default

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}", substitute, value)


def load_config(path: Path) -> dict:
    config = expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    for key in (
        "target",
        "data",
        "checkpoints",
        "protocol",
        "engine",
        "run",
        "publish",
    ):
        if key not in config:
            raise ValueError(f"config is missing the {key!r} section")
    protocol = config["protocol"]
    if protocol.get("draft_length") != 3:
        raise ValueError("This benchmark is frozen at draft_length 3 (K=3)")
    if type(protocol.get("repeats")) is not int or protocol["repeats"] < 3:
        raise ValueError("At least three technical repeats are required")
    if (
        set(protocol.get("protocols", [])) != set(PROTOCOLS)
        or len(protocol["protocols"]) != 2
    ):
        raise ValueError("both natural and fixed512 protocols are required")
    for name in (
        "max_new_tokens",
        "fixed_tokens",
        "max_model_len",
        "min_pixels",
        "max_pixels",
        "warmup_sessions_per_topic",
    ):
        if type(protocol.get(name)) is not int or protocol[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if protocol["fixed_tokens"] != 512 or protocol["max_new_tokens"] != 1024:
        raise ValueError("the protocols require fixed512 and natural max1024")
    if protocol["min_pixels"] > protocol["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    return config


def apply_overrides(
    config: dict, *, methods=None, draft_length=None, output_root=None
) -> dict:
    config = json.loads(json.dumps(config))
    if methods is not None:
        unknown = sorted(set(methods) - set(METHODS))
        if unknown or not methods or len(set(methods)) != len(methods):
            raise ValueError(f"unknown methods {unknown}; choose from {METHODS}")
        config["methods"] = list(methods)
    config.setdefault("methods", list(METHODS))
    if draft_length is not None and draft_length != config["protocol"]["draft_length"]:
        raise ValueError(
            f"--draft-length {draft_length} differs from the frozen K={config['protocol']['draft_length']}"
        )
    if output_root is not None:
        config["run"]["root"] = str(output_root)
    return config


def inference_settings() -> dict:
    return json.loads(SETTINGS.read_text())


def checkpoint_manifest() -> dict:
    return json.loads(CHECKPOINTS.read_text())


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    """Atomic JSON write (temporary file + rename in the same directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
