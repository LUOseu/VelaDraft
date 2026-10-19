"""target.device_map / max_memory: sharded target loading and multi-device online accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vlm_dflash.adapters import get_adapter
from vlm_dflash.target import load as load_module
from vlm_dflash.target.online import OnlineFeatureSource

CONTRACT = {
    "target": {"model_type": "qwen3_vl", "model_path": "/models/qwen3-vl"},
    "image": {"min_pixels": 3136, "max_pixels": 12544},
}


class _FakeModel:
    def __init__(self):
        self.moved_to = None
        self.frozen = None
        self.devices = ("cuda:0", "cuda:1")

    def parameters(self):
        return [SimpleNamespace(device=torch.device(name)) for name in self.devices]

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self

    def requires_grad_(self, flag):
        self.frozen = not flag
        return self


@pytest.fixture
def captured_load(monkeypatch):
    calls = {}
    model = _FakeModel()

    def fake_load_model(path, **kwargs):
        calls.update(path=path, **kwargs)
        return model

    adapter = get_adapter("qwen3_vl")
    monkeypatch.setattr(adapter, "load_model", fake_load_model)
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right")),
    )
    return calls, model


def test_device_map_is_forwarded_and_skips_single_device_move(captured_load):
    calls, model = captured_load
    cfg = {"target": {"device_map": "auto", "max_memory": {"0": "20GiB", 1: "20GiB", "cpu": "64GiB"}}}
    loaded, _ = load_module.load_target(cfg, CONTRACT)
    assert loaded is model
    assert calls["device_map"] == "auto"
    assert calls["max_memory"] == {0: "20GiB", 1: "20GiB", "cpu": "64GiB"}
    assert model.moved_to is None
    assert model.frozen is True


def test_without_device_map_the_target_moves_to_one_device(captured_load):
    calls, model = captured_load
    load_module.load_target({"target": {"device": "cuda:1"}}, CONTRACT)
    assert "device_map" not in calls and "max_memory" not in calls
    assert model.moved_to == "cuda:1"
    load_module.load_target({"target": {}}, CONTRACT, device="cuda:2")
    assert model.moved_to == "cuda:2"


@pytest.mark.parametrize("offload_device", ["cpu", "meta"])
def test_device_map_refuses_an_offloaded_target(captured_load, offload_device):
    _, model = captured_load
    model.devices = ("cuda:0", offload_device)
    with pytest.raises(ValueError, match=rf"parameters on \['{offload_device}'\] \(offloaded modules: unknown\)"):
        load_module.load_target({"target": {"device_map": "auto", "max_memory": {0: "1GiB"}}}, CONTRACT)


def test_offload_error_names_the_offloaded_modules(captured_load):
    _, model = captured_load
    model.devices = ("cuda:0", "meta")
    model.hf_device_map = {"model.embed_tokens": 0, **{f"model.layers.{i}": "disk" for i in range(5)}, "lm_head": "cpu"}
    with pytest.raises(ValueError, match=r"offloaded modules: model\.layers\.0, .*, model\.layers\.3 \.\.\.\)"):
        load_module.load_target({"target": {"device_map": "auto"}}, CONTRACT)


def test_an_all_cpu_device_map_is_not_an_offload(captured_load):
    _, model = captured_load
    model.devices = ("cpu", "cpu")
    loaded, _ = load_module.load_target({"target": {"device_map": "cpu"}}, CONTRACT)
    assert loaded is model and model.moved_to is None


def test_single_device_target_skips_the_offload_check(captured_load):
    _, model = captured_load
    model.devices = ("cpu",)  # not yet moved: the single-device path calls .to() afterwards
    load_module.load_target({"target": {"device": "cuda:1"}}, CONTRACT)
    assert model.moved_to == "cuda:1"


def test_device_map_and_explicit_device_conflict(captured_load):
    with pytest.raises(ValueError, match="both an explicit device and target.device_map"):
        load_module.load_target({"target": {"device_map": "auto"}}, CONTRACT, device="cuda:0")


def test_adapter_load_model_passes_load_kwargs(monkeypatch):
    import transformers

    seen = {}

    class _FakeClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            seen.update(path=path, **kwargs)
            return "model"

    adapter = get_adapter("qwen3_vl")
    monkeypatch.setattr(transformers, adapter.info.model_class, _FakeClass, raising=False)
    out = adapter.load_model(
        "/models/x", dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        device_map="auto", max_memory={0: "1GiB"},
    )
    assert out == "model"
    assert seen["device_map"] == "auto" and seen["max_memory"] == {0: "1GiB"}


def test_online_peak_and_sync_cover_every_target_device(monkeypatch):
    devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    peaks = {0: 3e9, 1: 7e9}
    calls = {"reset": [], "sync": []}
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: calls["reset"].append(device.index))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls["sync"].append(device.index))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peaks[device.index])

    source = object.__new__(OnlineFeatureSource)
    source.target_devices = devices
    source.stats = {}
    source.phase = "train"
    source._open_peak_window()
    source._sync_target()
    source.set_phase("eval")
    assert calls["reset"] == [0, 1, 0, 1]
    assert calls["sync"] == [0, 1]
    # The largest per-device peak, so a sharded target is not under-reported.
    assert source.stats["train"]["target_device_peak_gb"] == pytest.approx(7.0)


def test_max_memory_without_device_map_is_rejected(captured_load):
    with pytest.raises(ValueError, match="only applies with target.device_map"):
        load_module.load_target({"target": {"max_memory": {0: "20GiB"}}}, CONTRACT)


def test_spec_decode_paths_refuse_a_device_mapped_target(tmp_path):
    from vlm_dflash.benchmark.harness import run_benchmark
    from vlm_dflash.eval.run import evaluate

    cfg = {"target": {"device_map": "auto"}}
    with pytest.raises(RuntimeError, match="eval: target.device_map"):
        evaluate(cfg, tmp_path, CONTRACT)
    with pytest.raises(RuntimeError, match="benchmark: target.device_map"):
        run_benchmark(cfg, tmp_path, CONTRACT, [], out_dir=tmp_path / "out")
    load_module.require_single_device_target({"target": {}}, "eval")


def test_spec_generate_refuses_a_sharded_target():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from vlm_dflash.dflash.draft_model import Qwen3DFlashDraftModel

    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        block_size=4, num_target_layers=2,
        dflash_config={"mask_token_id": 63, "target_layer_ids": [0, 1], "spatial_rope_enabled": False},
    )
    config._attn_implementation = "sdpa"
    target = SimpleNamespace(hf_device_map={"model.embed_tokens": 0, "lm_head": 1}, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="device-mapped over several devices"):
        Qwen3DFlashDraftModel(config).spec_generate(
            target, torch.zeros(1, 3, dtype=torch.long), max_new_tokens=2, stop_token_ids=None, temperature=0.0
        )


def test_capture_fuse_is_per_row_and_device_safe():
    """Per-row fused features equal a plain concat at the row's true length (no padded batch fuse)."""
    hidden = [torch.randn(2, 5, 4) for _ in range(4)]  # embeddings + 3 layers
    layer_ids, lengths = [0, 2], [5, 3]
    fuse_device = hidden[layer_ids[0] + 1].device
    for row, length in enumerate(lengths):
        fused = torch.cat([hidden[layer + 1][row, :length].to(fuse_device) for layer in layer_ids], dim=-1)
        assert fused.shape == (length, 8)
        assert torch.equal(fused[:, :4], hidden[1][row, :length])
        assert torch.equal(fused[:, 4:], hidden[3][row, :length])


def test_online_records_per_device_peaks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: (device.index + 1) * 1e9)
    source = object.__new__(OnlineFeatureSource)
    source.target_devices = [torch.device("cuda", 0), torch.device("cuda", 1)]
    source.stats, source.phase = {}, "train"
    source._open_peak_window()
    source.set_phase("eval")
    assert source.stats["train"]["target_device_peaks_gb"] == {"cuda:0": 1.0, "cuda:1": 2.0}
