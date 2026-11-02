"""Original trained Domino head, captured with the native DFlash draft graph."""
from functools import wraps
from pathlib import Path
import sys
import torch
from torch import nn
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM

REFERENCE_DIR=Path(__file__).resolve().parents[1]/'qwen36_native_vllm'
sys.path.insert(0,str(REFERENCE_DIR))
import head_adapters as reference
assert Path(reference.__file__).resolve()==REFERENCE_DIR/'head_adapters.py'
sys.path.remove(str(REFERENCE_DIR))
ARCH='DFlashVelaDominoModel'
MARKER='veladraft_domino_reference'
_installed=False


def sample_anchor_prefix(model,hidden,anchor,k):
    assert hidden.ndim==2 and hidden.shape[0]==k+1 and anchor.shape==(1,)
    return reference.domino_walk(model,hidden[:k],anchor)


class DominoBackbone(DFlashQwen3Model):
    def __init__(self,*,vllm_config,start_layer_id=0,prefix=''):
        super().__init__(vllm_config=vllm_config,start_layer_id=start_layer_id,prefix=prefix)
        dtype=vllm_config.model_config.dtype;cfg=self.config.dflash_config
        expected=dict(projector_type='domino',emb_dim=256,gru_hidden_dim=1024,
                      pure_draft_prefix_len=1,shift_label=True)
        assert all(cfg.get(k)==v for k,v in expected.items())
        self.prefix_gru=nn.GRU(2048,1024,num_layers=1,batch_first=True,bias=False,dtype=dtype)
        self.embed_proj=nn.Sequential(nn.Linear(3072,256,bias=False,dtype=dtype),nn.SiLU(),
                                     nn.Linear(256,248320,bias=False,dtype=dtype))


class DFlashVelaDominoModel(DFlashQwen3ForCausalLM):
    model_cls=DominoBackbone
    @property
    def prefix_gru(self):return self.model.prefix_gru
    @property
    def embed_proj(self):return self.model.embed_proj


def install():
    global _installed
    if _installed:return
    import vllm
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from research.qwen36_v032.spatial_adapter import install as spatial_install
    assert vllm.__version__=='0.32.0'
    spatial_install();ModelRegistry.register_model(ARCH,f'{__name__}:DFlashVelaDominoModel')
    original=DFlashSpeculator._generate_draft
    @wraps(original)
    def generate(self,num_reqs,num_tokens_padded,attn_metadata,slot_mappings,
                 num_tokens_across_dp,cudagraph_runtime_mode=CUDAGraphMode.NONE):
        if not getattr(self.draft_model_config.hf_config,MARKER,False):
            return original(self,num_reqs,num_tokens_padded,attn_metadata,slot_mappings,
                            num_tokens_across_dp,cudagraph_runtime_mode)
        assert num_reqs==1 and self.draft_logits is None and not self.sample_from_anchor
        hidden=self._run_model(num_tokens_padded,attn_metadata,slot_mappings,
                               num_tokens_across_dp,cudagraph_runtime_mode)
        tokens=sample_anchor_prefix(self.model,hidden,self.input_buffers.input_ids[:1],self.num_speculative_steps)
        self.draft_tokens[:1]=tokens.view(1,-1)
    DFlashSpeculator._generate_draft=generate
    _installed=True
