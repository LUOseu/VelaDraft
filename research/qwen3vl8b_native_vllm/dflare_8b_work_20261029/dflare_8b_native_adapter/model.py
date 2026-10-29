"""Strict42 DFlare model; original target head, query forward and sampler remain."""
from torch import nn
from vllm.model_executor.models import qwen3_dflash as official
from vllm.model_executor.models.utils import maybe_prefix
from . import contracts as c
from .backbone import DFlareOwnedBackbone

class DFlashDFlareNativeQwen3ForCausalLM(official.DFlashQwen3ForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        receipt=c.runtime_receipt();c.validate_launch_config(vllm_config,receipt)
        c.require(str(vllm_config.speculative_config.model)==receipt['export_dir'],'Runtime draft differs from CPU verified export')
        nn.Module.__init__(self)
        self.config=vllm_config.speculative_config.draft_model_config.hf_config
        self.config.draft_vocab_size=self.config.vocab_size
        self.config.target_layer_count=vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        c.require(self.config.target_layer_count==36,'DFlare requires canonical36-layer target')
        self.model=DFlareOwnedBackbone(vllm_config=vllm_config,start_layer_id=36,prefix=maybe_prefix(prefix,'model'))
        self.lm_head=official.ParallelLMHead(151936,4096,prefix=maybe_prefix(prefix,'lm_head'))
        self.logits_processor=official.LogitsProcessor(151936,scale=1.0)
        self.draft_id_to_target_id=None;self._dflare_loaded_tensor_names=()

    def combine_hidden_states(self,*args,**kwargs):
        raise ValueError('DFlare requires explicit raw context proposer; shared FC combine is forbidden')

    def load_weights(self,weights):
        state={}
        for name,value in weights:
            c.require(name not in state,'Duplicate DFlare tensor: '+name);state[name]=value
        c.validate_state(state);c.validate_runtime_parameters(dict(self.named_parameters()))
        # Official auto model loading calls this method; the owned backbone partitions
        # six exact target names, then invokes official QKV/gate-up weight stacking.
        self.model.load_weights(state.items())
        self._dflare_loaded_tensor_names=tuple(sorted(state))
