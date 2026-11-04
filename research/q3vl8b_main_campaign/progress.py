#!/usr/bin/env python3
"""Copy a small, explicit set of live training records to raw/updates."""
import argparse
import os
from pathlib import Path
import tempfile

FILES = {'train_log.jsonl', 'validation.jsonl', 'timing_windows.jsonl', 'timing_segments.jsonl', 'train_summary.json'}


def export(root: Path) -> None:
    work = root / 'work'
    if not work.is_dir() or work.is_symlink():
        return
    for run in sorted(work.iterdir()):
        if not run.is_dir() or run.is_symlink():
            continue
        train = run / 'train'
        if not train.is_dir() or train.is_symlink():
            continue
        for name in FILES:
            src = train / name
            if not src.is_file() or src.is_symlink():
                continue
            data = src.read_bytes()
            if name.endswith('.jsonl') and data and not data.endswith(b'\n'):
                data = data[:data.rfind(b'\n') + 1]
            dst = root / 'raw' / 'updates' / run.name / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=dst.parent, prefix='.progress-')
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                os.replace(temp, dst)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    export(p.parse_args().campaign.resolve())
