"""VelaDraft DSpark layout on public vLLM model and Markov sampling components.

The trained variant uses K+1 query rows but predicts from rows 0..K-1.
Keep the native DFlash input/cache kernel in its safe K+1/mask-slot mode;
shift only the hidden rows passed to the official DSpark sampler. Do not turn
on its SAMPLE_FROM_ANCHOR kernel flag with K+1 queries: that native path
expects exactly K queries and would write beyond the K-row sampling buffers.
"""
from functools import wraps
import inspect
import types
import torch
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

ARCH = 'DFlashVelaDSparkModel'
MARKER = 'veladraft_anchor_slot_predictions'
_installed = False


def anchor_slots_for_native_sampler(hidden):
    # Batch one, native sample_indices = 1..K; map these to trained rows 0..K-1.
    # The final query row still participates in the non-causal backbone attention.
    return torch.cat((hidden[:1], hidden[:-1]), dim=0)


class DFlashVelaDSparkModel(Qwen3DSparkForCausalLM):
    def compute_draft_logits(self, hidden_states):
        # The checkpoint reference adds base logits and Markov bias in BF16.
        return super().compute_draft_logits(hidden_states).to(hidden_states.dtype)

    def markov_bias(self, markov_embed):
        return super().markov_bias(markov_embed).to(markov_embed.dtype)


def enabled(config):
    return bool(getattr(config, MARKER, False))


def install():
    global _installed
    if _installed:
        return
    import vllm
    from vllm.config.compilation import CUDAGraphMode
    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from research.qwen36_v032.spatial_adapter import install as spatial_install
    assert vllm.__version__ == '0.32.0'
    spatial_install()
    ModelRegistry.register_model(ARCH, f'{__name__}:DFlashVelaDSparkModel')
    original_init = DFlashSpeculator.__init__
    original_generate = DFlashSpeculator._generate_draft
    assert list(inspect.signature(original_generate).parameters) == [
        'self','num_reqs','num_tokens_padded','attn_metadata','slot_mappings',
        'num_tokens_across_dp','cudagraph_runtime_mode']

    @wraps(original_init)
    def init(self, vllm_config, device):
        original_init(self, vllm_config, device)
        if not enabled(self.draft_model_config.hf_config):
            return
        assert self.max_num_reqs == 1 and not self.sample_from_anchor
        assert self.num_query_per_req == self.num_speculative_steps+1
        assert self.draft_logits is None and not self.enable_adaptive_verification
        self._draft_topk = None
        self._anchor_idx = torch.zeros(1, dtype=torch.int64, device=device)
        self._step_cols = torch.arange(self.num_speculative_steps, dtype=torch.int32, device=device)
        self.use_confidence_head = False
        self._sample_logits = types.MethodType(DSparkSpeculator._sample_logits, self)

    @wraps(original_generate)
    def generate(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings,
                 num_tokens_across_dp, cudagraph_runtime_mode=CUDAGraphMode.NONE):
        if not enabled(self.draft_model_config.hf_config):
            return original_generate(self, num_reqs, num_tokens_padded, attn_metadata,
                slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode)
        assert num_reqs == 1 and num_tokens_padded == self.num_query_per_req
        hidden = self._run_model(num_tokens_padded, attn_metadata, slot_mappings,
                                num_tokens_across_dp, cudagraph_runtime_mode)
        DSparkSpeculator._sample_sequential(self, num_reqs,
                                          anchor_slots_for_native_sampler(hidden))

    DFlashSpeculator.__init__ = init
    DFlashSpeculator._generate_draft = generate
    _installed = True
