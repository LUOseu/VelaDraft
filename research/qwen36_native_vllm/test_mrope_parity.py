"""CPU parity: official vLLM partial interleaved MRoPE vs the trained draft math.

Run with the eval env and the training checkout first on PYTHONPATH:
  PYTHONPATH=<training VelaDraft>:research/qwen36_native_vllm python -m unittest test_mrope_parity
"""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

import contract
from spatial_adapter import rotate_context_keys


def _trained_rotary():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from vlm_dflash.dflash.draft_model import _draft_rotary_embedding
    config = Qwen3Config(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
                         rope_parameters={"rope_type": "default", "rope_theta": contract.ROPE_THETA,
                                          "partial_rotary_factor": contract.PARTIAL_ROTARY_FACTOR})
    rotary, rotary_dim = _draft_rotary_embedding(config)
    assert rotary_dim == 64
    return rotary


def _official_rotary():
    from vllm.config import CompilationConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    config = SimpleNamespace(compilation_config=CompilationConfig(mode=0, custom_ops=["none"]))
    with set_current_vllm_config(config):
        return get_rope(256, max_position=262144, dtype=torch.float32, rope_parameters={
            "rope_type": "default", "rope_theta": contract.ROPE_THETA,
            "partial_rotary_factor": contract.PARTIAL_ROTARY_FACTOR,
            "mrope_section": contract.MROPE_SECTION, "mrope_interleaved": True})


class InterleavedPartialParity(unittest.TestCase):
    def test_official_vllm_mrope_matches_trained_draft_math(self):
        from vlm_dflash.dflash.draft_model import _merge_multiaxis_rotary_embeddings, apply_rotary_pos_emb
        trained, official = _trained_rotary(), _official_rotary()
        self.assertEqual(official.rotary_dim, 64)
        n = 19
        axes = torch.stack((torch.arange(1, 1 + n), torch.arange(30, 30 + n), torch.arange(900, 900 + n)))
        g = torch.Generator().manual_seed(42)
        for dtype, tol in ((torch.float32, 2e-6), (torch.bfloat16, 0.0)):
            q = torch.randn((n, 16 * 256), generator=g).to(dtype)
            k = torch.randn((n, 2 * 256), generator=g).to(dtype)
            x = torch.zeros((1, n, 2048), dtype=dtype)
            cos, sin = _merge_multiaxis_rotary_embeddings(trained, x, axes[:, None, :],
                                                          tuple(contract.MROPE_SECTION), interleaved=True)
            qt = q.reshape(1, n, 16, 256).transpose(1, 2)
            kt = k.reshape(1, n, 2, 256).transpose(1, 2)
            eq, ek = apply_rotary_pos_emb(qt, kt, cos, sin)
            official.cos_sin_cache = official._compute_cos_sin_cache()
            oq, ok = official.forward_native(axes, q.clone(), k.clone())
            torch.testing.assert_close(oq, eq.transpose(1, 2).reshape(n, -1), rtol=0, atol=tol)
            torch.testing.assert_close(ok, ek.transpose(1, 2).reshape(n, -1), rtol=0, atol=tol)
            # Pass-through channels (64..255) must be untouched.
            torch.testing.assert_close(oq.view(n, 16, 256)[..., 64:], q.view(n, 16, 256)[..., 64:], rtol=0, atol=0)

    def test_context_key_rotation_matches_query_rotation(self):
        """rotate_context_keys (fused layer-major K as Q) equals per-layer official K rotation."""
        official = _official_rotary()
        official.cos_sin_cache = official._compute_cos_sin_cache()
        layers, n = 3, 11
        axes = torch.stack((torch.arange(5, 5 + n), torch.arange(7, 7 + n), torch.arange(9, 9 + n)))
        keys = torch.randn((layers * n, 256), generator=torch.Generator().manual_seed(7))
        fused = rotate_context_keys(official, axes, keys.clone(), layers, native=True)
        for layer in range(layers):
            block = keys[layer * n:(layer + 1) * n]
            _, expected = official.forward_native(axes, torch.zeros((n, 256)), block.clone())
            torch.testing.assert_close(fused[layer * n:(layer + 1) * n], expected, rtol=0, atol=0)


class ExportedConfigContract(unittest.TestCase):
    def test_exported_configs_validate(self):
        root = Path(__file__).resolve().parent / "exports"
        for config in sorted(root.glob("*/config.json")) if root.is_dir() else ():
            with self.subTest(config=str(config)):
                contract.validate_draft_config(json.loads(config.read_text()))


if __name__ == "__main__":
    unittest.main()
