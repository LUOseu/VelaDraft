"""Small shared utilities: seeds, JSON/JSONL IO, hashing, logging."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: os.PathLike | str, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
    tmp.replace(path)


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: os.PathLike | str, rows: Iterable[dict]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: os.PathLike | str, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def gradient_coverage_report(model, *, group_prefixes: Iterable[str] = ()) -> dict:
    """Gradient coverage audit for a trainer module (see GOAL §13).

    Reports, per parameter-name prefix (and per named leaf module when no
    prefixes are given), how many trainable parameters exist and how many have
    received a gradient. This catches the dangerous "head created but never
    updated" failure mode in algorithm zoos: a new head that silently receives
    no gradient still trains the rest of the model and looks plausible.

    Args:
        model: Any ``torch.nn.Module`` (usually the trainer module).
        group_prefixes: Optional parameter-name prefixes to group by; defaults
            to one group per top-level submodule (first dotted segment).

    Returns:
        Dict with ``groups`` (list of per-group dicts), and totals
        ``trainable_params`` / ``params_with_grad`` / ``fraction_with_grad``.
    """
    groups: dict[str, dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            key = "frozen"
        else:
            key = None
            for prefix in group_prefixes:
                if name.startswith(prefix):
                    key = prefix
                    break
            if key is None:
                parts = name.split(".")
                # Top-level submodule granularity: draft_model, markov_head,
                # confidence_head, ... one line per group in the audit.
                key = parts[0]
        group = groups.setdefault(
            key,
            {
                "group": key,
                "trainable_params": 0,
                "params_with_grad": 0,
                "params_with_nonzero_grad": 0,
                "grad_sq_sum": 0.0,
            },
        )
        if parameter.requires_grad:
            group["trainable_params"] += parameter.numel()
            if parameter.grad is not None:
                group["params_with_grad"] += 1
                grad = parameter.grad.detach().float()
                grad_sq = float(grad.pow(2).sum())
                group["grad_sq_sum"] += grad_sq
                if grad_sq > 0:
                    group["params_with_nonzero_grad"] += 1
    for group in groups.values():
        group["grad_norm"] = group.pop("grad_sq_sum") ** 0.5
    ordered = sorted(groups.values(), key=lambda item: item["group"])
    trainable = sum(item["trainable_params"] for item in ordered)
    with_grad = sum(item["params_with_grad"] for item in ordered)
    with_nonzero = sum(item["params_with_nonzero_grad"] for item in ordered)
    return {
        "groups": ordered,
        "trainable_params": trainable,
        "params_with_grad": with_grad,
        "params_with_nonzero_grad": with_nonzero,
        "fraction_with_grad": (with_grad / trainable) if trainable else 0.0,
    }
