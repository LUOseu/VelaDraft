import unittest
from types import SimpleNamespace
import torch
from official_spatial_adapter import rotate_context_keys
from vlm_dflash.dflash.draft_model import _merge_multiaxis_rotary_embeddings,apply_rotary_pos_emb
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

class InterleavedParity(unittest.TestCase):
    def test_official_vllm_mrope_matches_trained_draft_math(self):
        from vllm.config import CompilationConfig,set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope
        config=SimpleNamespace(compilation_config=CompilationConfig(mode=0,custom_ops=['none']))
        with set_current_vllm_config(config):
            official=get_rope(128,max_position=1024,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True},dtype=torch.float32)
        trained=Qwen3RotaryEmbedding(Qwen3Config(hidden_size=4096,num_attention_heads=32,head_dim=128,rope_parameters={'rope_type':'default','rope_theta':5000000.}))
        axes=torch.stack((torch.arange(1,18),torch.arange(30,47),torch.arange(90,107)))
        g=torch.Generator().manual_seed(42)
        for dtype,tol in ((torch.float32,2e-6),(torch.bfloat16,0.0)):
            q=torch.randn((17,32*128),generator=g).to(dtype)
            k=torch.randn((17,8*128),generator=g).to(dtype)
            x=torch.zeros((1,17,4096),dtype=dtype)
            cos,sin=_merge_multiaxis_rotary_embeddings(trained,x,axes[:,None,:],(24,20,20),interleaved=True)
            qt=q.reshape(1,17,32,128).transpose(1,2);kt=k.reshape(1,17,8,128).transpose(1,2)
            eq,ek=apply_rotary_pos_emb(qt,kt,cos,sin)
            official.cos_sin_cache=official._compute_cos_sin_cache()
            oq,ok=official.forward_native(axes,q.clone(),k.clone())
            torch.testing.assert_close(oq,eq.transpose(1,2).reshape(17,-1),rtol=0,atol=tol)
            torch.testing.assert_close(ok,ek.transpose(1,2).reshape(17,-1),rtol=0,atol=tol)
