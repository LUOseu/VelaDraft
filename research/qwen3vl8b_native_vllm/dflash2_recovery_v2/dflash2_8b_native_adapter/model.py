"""Strict51-tensor DFlash2 model; official outer forward/head/target remain."""
from torch import nn
import torch
from vllm.model_executor.models import qwen3_dflash as official
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from . import contracts as c
from .backbone import DFlash2OwnedBackbone
from .selector import CandidateSelector
from .logic import selector_walk

EXTRA_KEYS=frozenset(k for k in c.SCHEMA if k.startswith('candidate_selector.'))


class DFlash2NativeQwen3ForCausalLM(official.DFlashQwen3ForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        receipt=c.runtime_receipt();c.validate_launch_config(vllm_config,receipt)
        c.require(str(vllm_config.speculative_config.model)==receipt['export_dir'],'Runtime draft path differs from CPU-verified export')
        nn.Module.__init__(self)
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        self.config.draft_vocab_size=self.config.vocab_size
        self.config.target_layer_count=vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        c.require(self.config.target_layer_count==36,'DFlash2 target must have36 layers')
        self.model=DFlash2OwnedBackbone(vllm_config=vllm_config,start_layer_id=36,prefix=maybe_prefix(prefix,'model'))
        self.lm_head=official.ParallelLMHead(self.config.vocab_size,self.config.hidden_size,prefix=maybe_prefix(prefix,'lm_head'))
        self.logits_processor=official.LogitsProcessor(self.config.vocab_size,scale=1.0)
        self.draft_id_to_target_id=None
        self.candidate_selector=CandidateSelector(vocab_size=151936,hidden_size=4096,rank=256,top_k=16)
        self._dflash2_loaded_tensor_names=()

    def load_weights(self,weights):
        state={}
        for name,tensor in weights:
            c.require(name not in state,'Duplicate DFlash2 checkpoint tensor: '+name);state[name]=tensor
        c.validate_state(state)
        c.validate_runtime_parameters(dict(self.named_parameters()))
        # The official backbone loader handles q/k/v and gate/up packing plus all12 conv tensors.
        # The selector belongs to the outer model and is loaded separately, without filtering unknown keys.
        base={k:v for k,v in state.items() if k not in EXTRA_KEYS}
        super().load_weights(base.items())
        params=dict(self.named_parameters())
        for key in EXTRA_KEYS:
            c.require(params[key].dtype==state[key].dtype,'DFlash2 selector dtype drift: '+key)
            default_weight_loader(params[key],state[key])
        self._dflash2_loaded_tensor_names=tuple(sorted(state))

    def sample_dflash2_prefix(self,hidden_states,actual_anchor_ids):
        c.require(len(self._dflash2_loaded_tensor_names)==51,'All51 DFlash2 tensors must load before real selection')
        c.require(hidden_states.shape==(7,4096) and actual_anchor_ids.shape==(1,),'DFlash2 requires slots1..7 and one actual anchor')
        c.require(hidden_states.dtype==torch.bfloat16==self.candidate_selector.predecessor_codebook.dtype,
            'DFlash2 selector requires canonical BF16 arithmetic')
        # LogitsProcessor's normal FP32 output wraps an original BF16 head GEMM.
        # Recast before the canonical selector addition to retain its BF16 score rounding.
        unary=self.compute_logits(hidden_states).to(dtype=hidden_states.dtype).unsqueeze(0)
        return selector_walk(self.candidate_selector,unary,hidden_states.unsqueeze(0),actual_anchor_ids)
