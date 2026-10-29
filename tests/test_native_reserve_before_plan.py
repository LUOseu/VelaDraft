"""CPU-only regression: reserve the card before expensive artifact hashing."""
import importlib.util,json,sys,types
from pathlib import Path
from types import SimpleNamespace
import pytest

@pytest.fixture
def harness(monkeypatch):
    pkg=types.ModuleType('reserve_package');pkg.__path__=[]
    monkeypatch.setitem(sys.modules,'reserve_package',pkg)
    spec=importlib.util.spec_from_file_location('reserve_package.native',Path(__file__).resolve().parents[1]/'vlm_dflash'/'eval'/'native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    events=[]
    class Lease:
        def __init__(self,*args): self.uuid='GPU-test'
        def __enter__(self): events.append('reserve'); return self
        def __exit__(self,*args): events.append('release')
    monkeypatch.setitem(sys.modules,'reserve_package.native_resources',SimpleNamespace(GPULease=Lease))
    return module,events

def test_reserve_precedes_plan_hash_and_releases_on_validation_error(harness,monkeypatch,tmp_path):
    mod,events=harness
    def plan(args): events.append('hash_plan'); return {}
    def preserve(plan): events.append('preserve'); raise ValueError('test validation failure')
    monkeypatch.setattr(mod,'make_plan',plan);monkeypatch.setattr(mod,'verify_preservation',preserve)
    args=SimpleNamespace(out=tmp_path/'result.json',dry_run=False,gpu=2)
    assert mod.run(args)==1
    assert events==['reserve','hash_plan','preserve','release']
    assert json.loads(args.out.read_text())['error']=='test validation failure'

def test_plan_failure_releases_card_and_preserves_failure(harness,monkeypatch,tmp_path):
    mod,events=harness
    def plan(args): events.append('hash_plan'); raise ValueError('bad artifact')
    monkeypatch.setattr(mod,'make_plan',plan)
    args=SimpleNamespace(out=tmp_path/'result.json',dry_run=False,gpu=2)
    assert mod.run(args)==1
    assert events==['reserve','hash_plan','release']
    assert json.loads(args.out.read_text())['error']=='bad artifact'

def test_dry_run_does_not_reserve_card(harness,monkeypatch,tmp_path):
    mod,events=harness
    monkeypatch.setattr(mod,'make_plan',lambda args:{'dry_run':True})
    args=SimpleNamespace(out=tmp_path/'dry.json',dry_run=True,gpu=2)
    assert mod.run(args)==0
    assert events==[]


def test_invalid_dry_run_does_not_create_evidence_and_can_retry(harness, monkeypatch, tmp_path):
    mod, events = harness
    args = SimpleNamespace(out=tmp_path/'retry.json', dry_run=True, gpu=2)
    def bad_plan(_):
        raise ValueError('invalid plan')
    monkeypatch.setattr(mod, 'make_plan', bad_plan)
    with pytest.raises(ValueError, match='invalid plan'):
        mod.run(args)
    assert not args.out.exists()
    assert not args.out.with_suffix('.evidence').exists()
    monkeypatch.setattr(mod, 'make_plan', lambda _: {'dry_run': True})
    assert mod.run(args) == 0
