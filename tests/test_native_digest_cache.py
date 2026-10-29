import importlib.util,os,hashlib
from pathlib import Path
import pytest

@pytest.fixture
def mod():
    p=Path(__file__).resolve().parents[1]/'vlm_dflash'/'eval'/'digest_cache.py'
    spec=importlib.util.spec_from_file_location('digest_cache',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_reuses_verified_unchanged_file_without_rehash(mod,tmp_path,monkeypatch):
    path=tmp_path/'weight';path.write_bytes(b'a'*128);db=tmp_path/'cache.sqlite'
    assert mod.cached_sha256(path,db)==hashlib.sha256(b'a'*128).hexdigest()
    monkeypatch.setattr(mod,'_hash_stream',lambda stream:pytest.fail('unchanged file must not be rehashed'))
    assert mod.cached_sha256(path,db)==hashlib.sha256(b'a'*128).hexdigest()

def test_changed_bytes_invalidate_even_when_size_and_mtime_are_restored(mod,tmp_path):
    path=tmp_path/'weight';path.write_bytes(b'a'*128);db=tmp_path/'cache.sqlite';original=path.stat()
    assert mod.cached_sha256(path,db)==hashlib.sha256(b'a'*128).hexdigest()
    path.write_bytes(b'b'*128);os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
    assert mod.cached_sha256(path,db)==hashlib.sha256(b'b'*128).hexdigest()

def test_replaced_path_invalidates_cache_even_at_same_size(mod,tmp_path):
    path=tmp_path/'weight';path.write_bytes(b'a'*128);db=tmp_path/'cache.sqlite'
    mod.cached_sha256(path,db)
    other=tmp_path/'other';other.write_bytes(b'c'*128);os.replace(other,path)
    assert mod.cached_sha256(path,db)==hashlib.sha256(b'c'*128).hexdigest()

def test_file_mutation_during_hash_fails_closed(mod,tmp_path,monkeypatch):
    path=tmp_path/'weight';path.write_bytes(b'a'*128);db=tmp_path/'cache.sqlite'
    def concurrent_change(stream):
        data=stream.read();path.write_bytes(b'b'*128)
        return hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(mod,'_hash_stream',concurrent_change)
    with pytest.raises(ValueError,match='changed during integrity check'):
        mod.cached_sha256(path,db)
