#!/usr/bin/env python3
"""Upload the campaign's text evidence, with an explicit allowlist and no deletion."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time

REPO_ID = 'khazic/veladraft-qwen36-400k-dflash'
PREFIX = 'training_system_eval/qwen3vl8b_main'
DIRS = ('manifests', 'configs', 'raw', 'checks', 'tables')
ROOT_FILES = ('STATUS.md', 'REPORT.md', 'run_manifest.json', 'jobs.json')
EXTENSIONS = {'.json', '.jsonl', '.csv', '.tex', '.md', '.txt', '.log'}
FORBIDDEN = re.compile(r'(?:^|[._-])(tokens?|secrets?|credentials?|passwords?|passwd|node_env|\.env)(?:$|[._-])', re.I)
SENSITIVE = re.compile(r'(?i)(?:\bhf_[A-Za-z0-9]{20,}\b|authorization\s*[:=]\s*["\']?bearer\s+\S+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:hf_token|api_key|access_token|password|secret)\s*["\']?\s*[:=]\s*["\']?[^\s"\'\},]{8,})')
SENSITIVE_EXTRA = re.compile(r'''(?ix)(?:["']?(?:token|api[-_]key|access[-_]key[-_]id|aws[-_]access[-_]key[-_]id|secret|password)["']?\s*[:=]\s*["']?[^\s"',}]{8,}|https?://[^\s/@:]+:[^\s/@]+@)''')


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def destination(campaign: str, rel: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', campaign) or campaign in ('.', '..'):
        raise ValueError('invalid campaign name')
    path = PurePosixPath(rel)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('invalid relative path')
    return f'{PREFIX}/{campaign}/{path.as_posix()}'


def collect(root: Path) -> list[Path]:
    root = root.resolve()
    candidates = [root / name for name in ROOT_FILES]
    for directory in DIRS:
        folder = root / directory
        if folder.is_dir() and not folder.is_symlink():
            candidates.extend(folder.rglob('*'))
    files = []
    for p in candidates:
        relative = p.relative_to(root)
        if any(FORBIDDEN.search(part) for part in relative.parts):
            continue
        if p.suffix.lower() not in EXTENSIONS or not p.is_file():
            continue
        if any((root.joinpath(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
            continue
        files.append(p)
    return sorted(set(files))


def safe_bytes(data: bytes) -> bytes | None:
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return None
    return None if SENSITIVE.search(text) or SENSITIVE_EXTRA.search(text) else data


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.sync-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(obj, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cycle(root: Path, state: dict, api, sanitize_rules: dict) -> tuple[dict, list[str]]:
    from huggingface_hub import CommitOperationAdd
    adds, blocked = [], []
    for path in collect(root):
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8')
            for pattern, replacement in sanitize_rules['replace']:
                text = re.sub(pattern, replacement, text)
            data = None if re.search(sanitize_rules['leak'], text) else safe_bytes(text.encode('utf-8'))
        except UnicodeDecodeError:
            data = None
        if data is None:
            blocked.append(rel)
            continue
        digest = sha256(data)
        if state.get('files', {}).get(rel) != digest:
            adds.append((rel, digest, CommitOperationAdd(path_in_repo=destination(root.name, rel), path_or_fileobj=data)))
    for start in range(0, len(adds), 100):
        batch = adds[start:start + 100]
        commit = api.create_commit(repo_id=REPO_ID, repo_type='model', operations=[item[2] for item in batch],
                                   commit_message=f'{PREFIX}/{root.name}: update evidence')
        state.setdefault('files', {}).update({rel: digest for rel, digest, _ in batch})
        state['last_commit'] = commit.oid
        # Persist each successful batch even if a subsequent batch fails.
        atomic_json(root / '.hf_sync_state.json', state)
    return state, blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    root = args.campaign.resolve()
    if not root.is_dir() or args.interval < 30:
        parser.error('campaign must exist; interval must be >= 30 seconds')
    destination(root.name, 'STATUS.md')
    rule_path = os.environ.get('VELA_SANITIZE')
    if not rule_path:
        parser.error('set VELA_SANITIZE to the node-specific sanitization rules JSON')
    sanitize_rules = json.loads(Path(rule_path).read_text())
    if not isinstance(sanitize_rules.get('replace'), list) or not sanitize_rules.get('leak'):
        parser.error('VELA_SANITIZE must define replace (list) and a nonempty leak regex')
    re.compile(sanitize_rules['leak'])
    from huggingface_hub import HfApi
    state_path = root / '.hf_sync_state.json'
    with (root / '.hf_sync.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('sync already running', file=sys.stderr)
            return 2
        api = HfApi()
        while True:
            try:
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                state, blocked = cycle(root, state, api, sanitize_rules)
                state.update(last_success_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), blocked=blocked)
                atomic_json(state_path, state)
                print(f"sync ok: commit={state.get('last_commit', 'unchanged')} blocked={blocked}", flush=True)
                if args.once:
                    return 0 if not blocked else 3
            except Exception as exc:
                # Do not print exception bodies: HTTP errors may include sensitive request details.
                print(f'sync failed: {type(exc).__name__}; will retry', file=sys.stderr, flush=True)
                if args.once:
                    return 1
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
