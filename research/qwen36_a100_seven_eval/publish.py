"""Single publisher: mirror one run to ``<repo>@main:<path>/runs/<run_id>/`` on Hugging Face.

Each cycle uploads files whose content hash changed since the last confirmed upload, in one commit
pinned to the branch head it read (``parent_commit``); a moved head is re-read and retried with
exponential backoff. Completed raw shards are immutable: a shard whose bytes change after upload is
a hard error, never an overwrite. Only paths under this task's directory are written; ``INDEX.json``
is merged with the remote copy, never replaced wholesale. After each commit the remote sizes and
hashes are read back. Pending uploads persist in ``publish_state.json`` so a restart resumes.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import time

from research.qwen36_a100_seven_eval.settings import read_json, save_json, sha256

PUBLISHED = ('manifest.json', 'status.json', 'summary.json', 'summary.csv', 'sessions.json',
             'raw/**/*.jsonl', 'scores/**/*.json', 'stages/**/*.json')
IMMUTABLE = 'raw/**/*.jsonl'


def candidates(run: Path) -> dict[str, Path]:
    out = {}
    for pattern in PUBLISHED:
        for path in run.glob(pattern):
            if path.is_file() and not path.name.endswith(('.partial', '.tmp')) and '.invalid' not in path.name:
                out[path.relative_to(run).as_posix()] = path
    return out


class Publisher:
    def __init__(self, run: Path, repo: str, prefix: str, api=None, max_attempts: int = 6):
        self.run, self.repo, self.prefix = Path(run), repo, prefix.strip('/')
        self.run_id = read_json(self.run / 'manifest.json')['run_id']
        self.remote_root = f'{self.prefix}/runs/{self.run_id}'
        self.state_path = self.run / 'publish_state.json'
        self.state = read_json(self.state_path, {'uploaded': {}, 'commits': []})
        self.max_attempts = max_attempts
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        self.api = api

    def changed(self) -> dict[str, tuple[Path, str]]:
        out = {}
        for rel, path in candidates(self.run).items():
            digest = sha256(path)
            old = self.state['uploaded'].get(rel)
            if old == digest:
                continue
            if old is not None and fnmatch.fnmatch(rel, IMMUTABLE):
                raise ValueError(f'immutable raw shard changed after upload: {rel}')
            out[rel] = (path, digest)
        return out

    def index_operation(self, head: str):
        """INDEX.json merged with the remote copy: this run's entry is added or refreshed."""
        from huggingface_hub import CommitOperationAdd
        index = {'runs': {}}
        try:
            local = self.api.hf_hub_download(self.repo, f'{self.prefix}/INDEX.json', revision=head)
            index = json.loads(Path(local).read_text())
        except Exception:
            pass
        status = read_json(self.run / 'status.json', {})
        index.setdefault('runs', {})[self.run_id] = dict(path=f'runs/{self.run_id}', status=status.get('status'),
                                                         updated=time.strftime('%F %T'), suite=status.get('suite'))
        payload = json.dumps(index, indent=2, sort_keys=True).encode()
        return CommitOperationAdd(path_in_repo=f'{self.prefix}/INDEX.json', path_or_fileobj=payload)

    def cycle(self) -> dict:
        from huggingface_hub import CommitOperationAdd
        todo = self.changed()
        if not todo:
            return dict(uploaded=0)
        delay = 5
        for attempt in range(1, self.max_attempts + 1):
            try:
                head = self.api.repo_info(self.repo, revision='main').sha
                ops = [CommitOperationAdd(path_in_repo=f'{self.remote_root}/{rel}', path_or_fileobj=str(path))
                       for rel, (path, _) in sorted(todo.items())]
                ops.append(self.index_operation(head))
                for op in ops:
                    if not op.path_in_repo.startswith(self.prefix + '/'):
                        raise ValueError(f'refusing to write outside {self.prefix}: {op.path_in_repo}')
                info = self.api.create_commit(self.repo, operations=ops, parent_commit=head,
                                              commit_message=f'eval {self.run_id}: {len(todo)} file(s)')
                self.verify(info.oid, todo)
                for rel, (_, digest) in todo.items():
                    self.state['uploaded'][rel] = digest
                self.state['commits'].append(dict(oid=info.oid, files=len(todo), at=time.strftime('%F %T')))
                save_json(self.state_path, self.state)
                return dict(uploaded=len(todo), commit=info.oid, attempts=attempt)
            except Exception as error:
                self.state['last_error'] = f'{type(error).__name__}: {error}'[:500]
                save_json(self.state_path, self.state)
                if attempt == self.max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 300)
        return dict(uploaded=0)

    def verify(self, revision: str, todo: dict) -> None:
        """Read back sizes (and LFS sha256 where present) of what was just committed."""
        paths = [f'{self.remote_root}/{rel}' for rel in todo]
        infos = {i.path: i for i in self.api.get_paths_info(self.repo, paths, revision=revision)}
        for rel, (path, digest) in todo.items():
            info = infos.get(f'{self.remote_root}/{rel}')
            if info is None or info.size != path.stat().st_size:
                raise ValueError(f'remote copy of {rel} missing or wrong size')
            lfs = getattr(info, 'lfs', None)
            if lfs is not None and getattr(lfs, 'sha256', digest) != digest:
                raise ValueError(f'remote LFS hash of {rel} differs')


def check_write_access(repo: str, api=None) -> dict:
    """Preflight: authenticated user and a dry write probe is not possible on HF; check role instead."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    who = api.whoami()
    info = api.repo_info(repo)
    owner = repo.split('/')[0]
    orgs = {o.get('name') for o in who.get('orgs', [])}
    allowed = who.get('name') == owner or owner in orgs
    token_role = (who.get('auth') or {}).get('accessToken', {}).get('role')
    return dict(user=who.get('name'), repo=repo, head=info.sha, owner_or_org_member=allowed, token_role=token_role,
                ok=bool(allowed and token_role in (None, 'write', 'fineGrained')))


def loop(run: Path, repo: str, prefix: str, interval: int, until_done: bool = True) -> int:
    publisher = Publisher(run, repo, prefix)
    while True:
        result = publisher.cycle()
        print(time.strftime('%F %T'), 'publish', json.dumps(result), flush=True)
        status = read_json(Path(run) / 'status.json', {}).get('status')
        if until_done and status in ('complete', 'failed', 'blocked') and not publisher.changed():
            return 0
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--path', required=True, help='task directory inside the repo, e.g. eval/qwen36_a100_seven_v1')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.once:
        print(json.dumps(Publisher(args.run_dir, args.repo, args.path).cycle()))
        return 0
    return loop(args.run_dir, args.repo, args.path, args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
