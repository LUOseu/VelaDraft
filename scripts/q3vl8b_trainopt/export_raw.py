"""Export run records from ``work/`` into the contract's ``raw/`` layout (section 12), sanitized.

    python3 export_raw.py <campaign_dir> [--redact-gpu]

Run directories (checkpoints, caches) stay in ``work/``; ``raw/`` receives per run:
``updates.jsonl.gz`` (every logged update), ``validation.jsonl``, ``final_test.json``,
``timing_windows.jsonl`` / ``timing_segments.jsonl``, ``run_summary.json``, ``config.json``
and ``stdout.log``. Every text file is sanitized (machine paths, host names, internal
addresses); a file that still matches the leak pattern afterwards is not exported.
Incremental: a file is rewritten only when its source changed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path


def load_rules() -> tuple[list, re.Pattern, re.Pattern]:
    """Sanitize rules of the node (``VELA_SANITIZE``: JSON kept on the node, not in the repository).

    ``{"replace": [[regex, replacement], ...], "leak": regex, "gpu_redact": regex}``: replacements
    run in order; a text that still matches ``leak`` afterwards is never exported.
    """
    path = os.environ.get("VELA_SANITIZE")
    if not path:
        raise SystemExit("set VELA_SANITIZE to the node's sanitize rules (JSON)")
    rules = json.loads(Path(path).read_text())
    replace = [(re.compile(pattern), replacement) for pattern, replacement in rules["replace"]]
    return replace, re.compile(rules["leak"]), re.compile(rules["gpu_redact"])


RULES = None


def rules() -> tuple[list, re.Pattern, re.Pattern]:
    global RULES
    if RULES is None:
        RULES = load_rules()
    return RULES


def sanitize(text: str, redact_gpu: bool) -> str:
    replace, _, gpu = rules()
    for pattern, replacement in replace:
        text = pattern.sub(replacement, text)
    if redact_gpu:
        text = gpu.sub("<gpu>", text)
    return text


def leaks(text: str) -> bool:
    return bool(rules()[1].search(text))


def destination(raw: Path, name: str) -> Path | None:
    parts = name.split("_")
    if name.startswith("fixed_config_") or name.startswith("best_deployment_"):
        track = "fixed_config" if name.startswith("fixed_config_") else "best_deployment"
        rest = name[len(track) + 1 :].split("_")  # <method>_s42_<arm...>
        return raw / "formal" / track / rest[0] / rest[1] / "_".join(rest[2:])
    if parts[0] == "window":  # window_<method>_<stage...>_p<k>_<arm>
        method, pair, arm = parts[1], parts[-2], parts[-1]
        stage = "_".join(parts[2:-2])
        return raw / "windows" / method / stage / "ablation" / pair / arm
    if parts[0] == "replay":  # replay_<method>_<stage...>_p<k>_<arm|record>
        method, pair, arm = parts[1], parts[-2], parts[-1]
        stage = "_".join(parts[2:-2])
        return raw / "windows" / method / stage / "drafter_replay" / pair / arm
    if parts[0] == "scaling":  # scaling_<method>_<stage...>_p<k>_<arm>_mb<n>
        method, pair, arm, mb = parts[1], parts[-3], parts[-2], parts[-1]
        stage = "_".join(parts[2:-3])
        return raw / "windows" / method / stage / "batch_scaling" / pair / f"{arm}_{mb}"
    if parts[0] in ("pilot", "probe", "numcheck", "init", "profile"):
        return raw / "pilot" / parts[0] / name
    return None


def export_text(src: Path, dst: Path, redact_gpu: bool, *, gz: bool = False) -> str:
    if not src.exists():
        return "missing"
    stamp = dst.with_name(dst.name + ".srcmtime")
    mtime = f"{src.stat().st_mtime_ns}:{src.stat().st_size}"
    if dst.exists() and stamp.exists() and stamp.read_text() == mtime:
        return "unchanged"
    text = sanitize(src.read_text(encoding="utf-8", errors="replace"), redact_gpu)
    if leaks(text):
        return "leak_blocked"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with gzip.open(dst, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        dst.write_text(text, encoding="utf-8")
    stamp.write_text(mtime)
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--redact-gpu", action="store_true")
    args = parser.parse_args()
    camp = args.campaign
    raw = camp / "raw"
    status: dict[str, dict] = {}
    for run in sorted((camp / "work").iterdir()):
        dst = destination(raw, run.name)
        if dst is None or not run.is_dir():
            continue
        train = run / "train"
        files = {
            "updates.jsonl.gz": (train / "train_log.jsonl", True),
            "validation.jsonl": (train / "validation.jsonl", False),
            "timing_windows.jsonl": (train / "timing_windows.jsonl", False),
            "timing_segments.jsonl": (train / "timing_segments.jsonl", False),
            "run_summary.json": (train / "train_summary.json", False),
            "config.json": (run / "config.json", False),
            "provenance.json": (run / "provenance.json", False),
            "stdout.log": (run / "stdout.log", False),
        }
        status[run.name] = {
            name: export_text(src, dst / name, args.redact_gpu, gz=gz) for name, (src, gz) in files.items()
        }
        validation = train / "validation.jsonl"
        if validation.exists():
            tests = [json.loads(line) for line in validation.read_text().splitlines()
                     if line.strip() and json.loads(line)["split"] == "test"]
            if tests:
                (dst / "final_test.json").write_text(json.dumps(tests, indent=1))
    samples = raw / "gpu_samples.jsonl"
    if samples.exists():
        export_text(samples, raw / "gpu_samples.jsonl.gz", args.redact_gpu, gz=True)
    for name in ("gpu_time_ledger.jsonl",):
        for ledger in raw.glob(f"scheduler_*/{name}"):
            export_text(ledger, raw / f"{ledger.parent.name}_{name}", args.redact_gpu)
    (raw / "export_status.json").write_text(json.dumps(status, indent=1))
    blocked = [(run, name) for run, files in status.items() for name, value in files.items() if value == "leak_blocked"]
    print(json.dumps({"runs": len(status), "leak_blocked": blocked}))
    if blocked:
        sys.exit(2)


if __name__ == "__main__":
    main()
