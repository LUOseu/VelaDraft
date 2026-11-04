import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('progress', Path(__file__).resolve().parents[1] / 'progress.py')
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def test_copies_only_existing_logs_with_complete_jsonl_records(tmp_path):
    train = tmp_path / 'work/dfly_s17_ref/train'
    train.mkdir(parents=True)
    (train / 'train_log.jsonl').write_bytes(b'{"step":1}\n{"step":2')
    (train / 'ckpt_last.pt').write_text('weights')
    progress.export(tmp_path)
    raw = tmp_path / 'raw/updates/dfly_s17_ref'
    assert (raw / 'train_log.jsonl').read_bytes() == b'{"step":1}\n'
    assert not (raw / 'ckpt_last.pt').exists()


def test_does_not_follow_linked_run_dir(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'bad').symlink_to(tmp_path, target_is_directory=True)
    progress.export(tmp_path)
    assert not (tmp_path / 'raw/updates/bad').exists()
