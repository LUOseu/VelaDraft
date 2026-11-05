"""CPU parity: official vLLM interleaved MRoPE vs the trained Qwen3-VL-8B draft math.

Run with the eval env and the training checkout first on PYTHONPATH:
  PYTHONPATH=<training VelaDraft>:research/qwen3vl8b_cont_mmspec/native python -m unittest test_mrope_parity
"""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

import contract
from spatial_adapter import rotate_context_keys

HIDDEN = contract.DRAFT_FIELDS["hidden_size"]
HEADS = contract.DRAFT_FIELDS["num_attention_heads"]
KV_HEADS = contract.DRAFT_FIELDS["num_key_value_heads"]
HEAD_DIM = contract.DRAFT_FIELDS["head_dim"]
ROPE = {"rope_type": "default", "rope_theta": contract.ROPE_THETA}


def _trained_rotary():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from vlm_dflash.dflash.draft_model import _draft_rotary_embedding
    config = Qwen3Config(hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
                         head_dim=HEAD_DIM, rope_parameters=dict(ROPE))
    rotary, rotary_dim = _draft_rotary_embedding(config)
    assert rotary_dim == HEAD_DIM
    return rotary


def _official_rotary():
    from vllm.config import CompilationConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    config = SimpleNamespace(compilation_config=CompilationConfig(mode=0, custom_ops=["none"]))
    with set_current_vllm_config(config):
        return get_rope(HEAD_DIM, max_position=262144, dtype=torch.float32, rope_parameters={
            **ROPE, "mrope_section": contract.MROPE_SECTION, "mrope_interleaved": True})


class InterleavedParity(unittest.TestCase):
    def test_official_vllm_mrope_matches_trained_draft_math(self):
        from vlm_dflash.dflash.draft_model import _merge_multiaxis_rotary_embeddings, apply_rotary_pos_emb
        trained, official = _trained_rotary(), _official_rotary()
        self.assertEqual(official.rotary_dim, HEAD_DIM)
        n = 19
        axes = torch.stack((torch.arange(1, 1 + n), torch.arange(30, 30 + n), torch.arange(900, 900 + n)))
        g = torch.Generator().manual_seed(42)
        for dtype, tol in ((torch.float32, 2e-6), (torch.bfloat16, 0.0)):
            q = torch.randn((n, HEADS * HEAD_DIM), generator=g).to(dtype)
            k = torch.randn((n, KV_HEADS * HEAD_DIM), generator=g).to(dtype)
            x = torch.zeros((1, n, HIDDEN), dtype=dtype)
            cos, sin = _merge_multiaxis_rotary_embeddings(trained, x, axes[:, None, :],
                                                          tuple(contract.MROPE_SECTION), interleaved=True)
            qt = q.reshape(1, n, HEADS, HEAD_DIM).transpose(1, 2)
            kt = k.reshape(1, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
            eq, ek = apply_rotary_pos_emb(qt, kt, cos, sin)
            official.cos_sin_cache = official._compute_cos_sin_cache()
            oq, ok = official.forward_native(axes, q.clone(), k.clone())
            torch.testing.assert_close(oq, eq.transpose(1, 2).reshape(n, -1), rtol=0, atol=tol)
            torch.testing.assert_close(ok, ek.transpose(1, 2).reshape(n, -1), rtol=0, atol=tol)

    def test_context_key_rotation_matches_query_rotation(self):
        """rotate_context_keys (fused layer-major K as Q) equals per-layer official K rotation."""
        official = _official_rotary()
        official.cos_sin_cache = official._compute_cos_sin_cache()
        layers, n = 3, 11
        axes = torch.stack((torch.arange(5, 5 + n), torch.arange(7, 7 + n), torch.arange(9, 9 + n)))
        keys = torch.randn((layers * n, HEAD_DIM), generator=torch.Generator().manual_seed(7))
        fused = rotate_context_keys(official, axes, keys.clone(), layers, native=True)
        for layer in range(layers):
            block = keys[layer * n:(layer + 1) * n]
            _, expected = official.forward_native(axes, torch.zeros((n, HEAD_DIM)), block.clone())
            torch.testing.assert_close(fused[layer * n:(layer + 1) * n], expected, rtol=0, atol=0)


class ExportedConfigContract(unittest.TestCase):
    def test_exported_configs_validate(self):
        root = Path(__file__).resolve().parent / "exports"
        for config in sorted(root.glob("*/config.json")) if root.is_dir() else ():
            with self.subTest(config=str(config)):
                contract.validate_draft_config(json.loads(config.read_text()))


if __name__ == "__main__":
    unittest.main()
