import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'hf_sync_safe.py'
spec = importlib.util.spec_from_file_location('hf_sync_safe', MODULE)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_collect_is_allowlisted_and_rejects_symlinks_and_credentials(tmp_path):
    for name in ('STATUS.md', 'raw/updates/progress.jsonl', 'scripts/node_env.sh',
                 'configs/node_env.sh', 'configs/run.json', 'work/model.pt', 'raw/updates/token.json'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
    (tmp_path / 'raw/updates/link.json').symlink_to(tmp_path / 'configs/run.json')
    actual = {p.relative_to(tmp_path).as_posix() for p in sync.collect(tmp_path)}
    assert actual == {'STATUS.md', 'raw/updates/progress.jsonl', 'configs/run.json'}


def test_sensitive_values_are_refused_not_uploaded():
    assert sync.safe_bytes(b'{"step": 2}') == b'{"step": 2}'
    assert sync.safe_bytes(b'{"HF_TOKEN": "hf_abcdefghijklmnopqrstuvwxyz1234"}') is None
    assert sync.safe_bytes(b'Authorization: Bearer abcdefghijklmnopqrstuvwxyz') is None
    assert sync.safe_bytes(b'-----BEGIN PRIVATE KEY-----') is None


def test_hashes_depend_on_content_and_prefix_is_isolated():
    assert sync.sha256(b'one') != sync.sha256(b'two')
    assert sync.destination('run-123', 'raw/updates/a.json') == 'training_system_eval/qwen3vl8b_main/run-123/raw/updates/a.json'
    for invalid in ('../outside', '/absolute', 'a/b'):
        try:
            sync.destination(invalid, 'STATUS.md')
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)


def test_generic_credentials_are_refused():
    for item in (b'{"token":"aaaaaaaaaaaaaaaaaaaaaaaa"}', b'{"AWS_ACCESS_KEY_ID":"AKIA1234567890000000"}', b'{"api-key":"123456789"}', b'https://user:password@example.com'):
        assert sync.safe_bytes(item) is None


def test_cycle_only_commits_changed_safe_content_and_preserves_unrelated_files(tmp_path):
    class Commit:
        oid = 'commit-id'
    class API:
        calls = []
        def create_commit(self, **kwargs):
            self.calls.append(kwargs)
            return Commit()
    (tmp_path / 'STATUS.md').write_text('running /private/server/path')
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/credentials.json').write_text('{"token":"supersecretvalue"}')
    api = API()
    rules = {'replace': [['/private/server/path', '<redacted>']], 'leak': '/private/server'}
    state, blocked = sync.cycle(tmp_path, {}, api, rules)
    assert len(api.calls) == 1
    assert api.calls[0]['operations'][0].path_in_repo.endswith('/STATUS.md')
    assert api.calls[0]['operations'][0].path_or_fileobj == b'running <redacted>'
    assert not blocked
    state, _ = sync.cycle(tmp_path, state, api, rules)
    assert len(api.calls) == 1
