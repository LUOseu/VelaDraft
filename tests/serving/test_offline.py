from __future__ import annotations

import builtins
import os
import sys
from types import SimpleNamespace

import pytest

from vlm_dflash.serving import launch, measure, offline


@pytest.fixture
def offline_run(monkeypatch, tmp_path):
    events = []
    state = {"startup_failure": False}
    monkeypatch.setattr(launch, "LOCK_ROOT", str(tmp_path / "locks"))
    monkeypatch.setattr(launch, "gpu_uuid", lambda index: f"GPU-{index}")
    monkeypatch.setattr(launch, "gpu_compute_processes", lambda index: [])
    monkeypatch.setattr(launch, "gpu_free_mib", lambda index: 30000)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,9")

    def locked_event(event):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        with pytest.raises(launch.LaunchError, match="locked"):
            launch.acquire_gpu_lock(2)
        events.append(event)

    def properties(logical_index):
        assert logical_index == 0
        locked_event("properties")
        return SimpleNamespace(name="mock GPU", major=8, minor=0)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: False,
        get_device_properties=properties,
        empty_cache=lambda: locked_event("empty_cache"),
    )))

    class LLM:
        def __init__(self, **kwargs):
            locked_event("llm")
            assert kwargs["tensor_parallel_size"] == 1
            if state["startup_failure"]:
                raise RuntimeError("model initialization failed")
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=lambda: locked_event("shutdown"),
            ))

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "vllm":
            locked_event("import_vllm")
            return SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    class Sampler:
        def __init__(self, gpu_index, interval_s):
            assert gpu_index == 2

        def start(self):
            locked_event("sampler_start")
            return self

        def stop(self):
            locked_event("sampler_stop")

        def peak(self):
            return {"memory_mib": 0}

    monkeypatch.setattr(measure, "GpuSampler", Sampler)
    monkeypatch.setattr(offline, "load_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(offline, "natural_fixtures", lambda manifest: [])
    monkeypatch.setattr(offline, "diagnostic_fixtures", lambda manifest: [])
    monkeypatch.setattr(offline, "model_identity", lambda *args, **kwargs: {})
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = SimpleNamespace(
        gpu_index=2, manifest=str(manifest), skip_hash_check=False, limit=0,
        model=str(tmp_path), dtype="bfloat16", mode="ar", vllm_version="0.22.1",
        code_commit="test", attention_backend=None, cudagraph_mode=None,
        max_new_tokens=8, min_pixels=1, max_pixels=2, max_model_len=1024,
        gpu_memory_utilization=0.5, async_scheduling=None, repeats=1,
        run_id="cpu-mocked", num_speculative_tokens=4, prompt_lookup_min=2,
        prompt_lookup_max=2, out=None,
    )
    return args, events, state


def assert_released():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,9"
    fd = launch.acquire_gpu_lock(2)
    os.close(fd)


def test_offline_pins_gpu_before_import_and_cleans_up_before_unlock(offline_run):
    args, events, _ = offline_run
    document = offline.run_offline(args)
    assert document["meta"]["device"] == [{
        "index": 2, "name": "mock GPU", "uuid": "GPU-2", "compute_capability": "8.0",
    }]
    assert events == [
        "import_vllm", "properties", "sampler_start", "llm",
        "shutdown", "sampler_stop", "empty_cache",
    ]
    assert_released()


@pytest.mark.parametrize("failure", ["busy", "memory"])
def test_offline_checks_gpu_before_import(offline_run, monkeypatch, failure):
    args, events, _ = offline_run
    if failure == "busy":
        monkeypatch.setattr(launch, "gpu_compute_processes", lambda index: ["busy"])
    else:
        monkeypatch.setattr(launch, "gpu_free_mib", lambda index: 0)
    with pytest.raises(launch.LaunchError):
        offline.run_offline(args)
    assert events == []
    assert_released()


def test_offline_stops_sampler_and_releases_lock_on_model_failure(offline_run):
    args, events, state = offline_run
    state["startup_failure"] = True
    with pytest.raises(RuntimeError, match="model initialization failed"):
        offline.run_offline(args)
    assert events[-2:] == ["sampler_stop", "empty_cache"]
    assert "shutdown" not in events
    assert_released()


def test_offline_cleans_up_on_output_failure(offline_run, monkeypatch):
    args, events, _ = offline_run
    args.out = "unused.json"

    def fail_write(*args):
        raise OSError("output failed")

    monkeypatch.setattr(offline, "write_json_atomic", fail_write)
    with pytest.raises(OSError, match="output failed"):
        offline.run_offline(args)
    assert events[-3:] == ["shutdown", "sampler_stop", "empty_cache"]
    assert_released()


def test_offline_rejects_multi_gpu_before_import(offline_run):
    args, events, _ = offline_run
    args.pipeline_parallel_size = 2
    with pytest.raises(launch.LaunchError, match="pipeline_parallel_size"):
        offline.run_offline(args)
    assert events == []
    assert_released()


def test_model_config_does_not_prove_backend_labels():
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype="bfloat16", max_model_len=1024),
        speculative_config=None,
    )
    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config))
    backend = {
        "attention_backend": "FLASH_ATTN",
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "labels_source": "cli_declared_unverified",
        "applied_evidence": False,
    }
    offline._engine_applied_evidence(llm, backend)
    assert backend["labels_source"] == "cli_declared_unverified"
    assert backend["applied_evidence"] is False
    assert backend["applied_config"]["max_model_len"] == "1024"
