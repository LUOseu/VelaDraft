"""Create an isolated native-eval package with vLLM statistics enabled.

This is a separate acceptance-length experiment, never a replacement for the
stats-disabled timing result. No model weights or datasets are copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

OLD = "generation_config='vllm',disable_log_stats=True,"
NEW = "generation_config='vllm',disable_log_stats=False,"


def prepare(source: Path, destination: Path) -> None:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    package = source / 'vlm_dflash'
    if destination == package or package in destination.parents:
        raise ValueError('destination must not be inside the source package')
    original = package / 'eval' / 'native.py'
    text = original.read_text()
    if text.count(OLD) != 1:
        raise ValueError('expected exactly one stats-disabled native configuration')
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.al-harness-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'harness'
        stage.mkdir()
        shutil.copytree(source / 'vlm_dflash', stage / 'vlm_dflash',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pt', '*.safetensors', '*.bin'))
        changed = stage / 'vlm_dflash' / 'eval' / 'native.py'
        changed.write_text(text.replace(OLD, NEW))
        receipt = {
            'scope': 'supplementary acceptance statistics; not a primary speed result',
            'original_native_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'supplement_native_sha256': hashlib.sha256(changed.read_bytes()).hexdigest(),
            'change': 'disable_log_stats=True to False',
        }
        (stage / 'AL_HARNESS_PROVENANCE_V1.json').write_text(json.dumps(receipt, indent=2) + '\n')
        stage.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='repository or harness root')
    parser.add_argument('destination', type=Path, help='new isolated harness directory')
    args = parser.parse_args()
    prepare(args.source, args.destination)
