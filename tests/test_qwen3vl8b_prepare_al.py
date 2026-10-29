import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'research' / 'qwen3vl8b_native_vllm' / 'prepare_al_harness.py'


def test_prepare_copies_package_and_only_changes_stats(tmp_path):
    assert SCRIPT.exists(), 'stats harness preparation is missing'
    spec = importlib.util.spec_from_file_location('prepare_al', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src = tmp_path / 'source'
    native = src / 'vlm_dflash' / 'eval' / 'native.py'
    native.parent.mkdir(parents=True)
    native.write_text("x=dict(generation_config='vllm',disable_log_stats=True,)\n")
    other = native.parent / 'other.py'
    other.write_text('unchanged = True\n')
    dest = tmp_path / 'al'
    module.prepare(src, dest)
    assert "disable_log_stats=False" in (dest / 'vlm_dflash' / 'eval' / 'native.py').read_text()
    assert (dest / 'vlm_dflash' / 'eval' / 'other.py').read_text() == other.read_text()
    assert "disable_log_stats=True" in native.read_text()
    with pytest.raises(FileExistsError):
        module.prepare(src, dest)


def test_prepare_rejects_destination_inside_source_package(tmp_path):
    spec = importlib.util.spec_from_file_location('prepare_al_nested', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = tmp_path / 'source' / 'vlm_dflash' / 'eval' / 'native.py'
    native.parent.mkdir(parents=True)
    native.write_text("x=dict(generation_config='vllm',disable_log_stats=True,)\n")
    with pytest.raises(ValueError, match='inside'):
        module.prepare(native.parents[2], native.parent / 'nested-harness')
