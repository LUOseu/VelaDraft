"""YAML config loading with dotted-path lookup and ``--set`` overrides."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _replace_env(match: re.Match) -> str:
    name, default = match.group(1), match.group(2)
    if name in os.environ:
        return os.environ[name]
    if default is not None:
        return default
    raise ValueError(f"Config references environment variable {name} with no default and it is unset.")


def expand_env_text(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references in a single string.

    Same syntax as YAML configs (unlike ``os.path.expandvars``, which has no
    default-value form); used for manifest prompt paths.
    """
    return _ENV_PATTERN.sub(_replace_env, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_text(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse into a mapping.")
    return _expand_env(cfg)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def parse_scalar(text: str) -> Any:
    """Parse a ``--set key=value`` value as YAML scalar (falls back to string)."""
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        set_dotted(cfg, key, parse_scalar(value))
    return cfg


def resolve_run_dir(cfg: dict, override: str | None = None) -> Path:
    """Run directory: CLI override > ``run.dir`` (relative to CWD)."""
    if override:
        return Path(override).resolve()
    run_dir = get(cfg, "run.dir")
    if not run_dir:
        raise ValueError("Config is missing run.dir and no --run-dir was given.")
    return (Path.cwd() / run_dir).resolve()


def dump_effective_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2, default=str)
