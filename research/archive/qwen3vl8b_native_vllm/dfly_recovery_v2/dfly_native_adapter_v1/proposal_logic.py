"""Owned parallel-only copy of official propose: replace only shared-FC boundary."""
from vllm.forward_context import set_forward_context
from . import contracts as c
from .sampling import sequential_greedy


def raw_parallel_propose(self,target_token_ids,target_positions,target_hidden_states,next_token_ids,
                        token_indices_to_sample,common_attn_metadata,sampling_metadata,mm_embed_inputs=None,
                        num_rejected_tokens_gpu=None,slot_mappings=None):
    c.require(self.hidden_size==4096 and target_hidden_states.ndim==2 and target_hidden_states.shape[1]==12288,
        'DFly requires query hidden4096 and explicit raw context12288')
    c.require(self.parallel_drafting and self.num_speculative_tokens==7 and common_attn_metadata.batch_size()==1,
        'Only actual parallel S7/B8, one-request DFly is reviewed')
    c.require(sampling_metadata.all_greedy is True,'DFly native mount requires canonical greedy suffix sampling')
    c.require(next_token_ids.shape==(1,),'DFly requires the actual single target bonus anchor')
    self._last_draft_probs=None
    # target_hidden_states is the raw ordered [1,17,33] payload. It never enters
    # combine_hidden_states or any model-side stash. The query hidden ABI stays4096.
    num_tokens,token_indices_to_sample,common_attn_metadata=self.set_inputs_first_pass(
        target_token_ids=target_token_ids,next_token_ids=next_token_ids,target_positions=target_positions,
        target_hidden_states=target_hidden_states,token_indices_to_sample=token_indices_to_sample,
        cad=common_attn_metadata,num_rejected_tokens_gpu=num_rejected_tokens_gpu)
    _,per_layer_attn_metadata=self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
    cudagraph_runtime_mode,num_input_tokens,num_tokens_across_dp=self._determine_batch_execution_and_padding(num_tokens)
    model_kwargs,slot_mapping_size=self.build_model_inputs_first_pass(num_tokens,num_input_tokens,mm_embed_inputs)
    with set_forward_context(per_layer_attn_metadata,self.vllm_config,num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=self._get_slot_mapping(slot_mapping_size,common_attn_metadata.slot_mapping)):
        ret_hidden_states=self.model(**model_kwargs)
        if not self.model_returns_tuple():last_hidden_states=ret_hidden_states
        else:last_hidden_states,_=ret_hidden_states
    sample_hidden_states=last_hidden_states[token_indices_to_sample]
    # Never enter the stock local-argmax or independent-head sampling shortcut.
    # Every chosen token is fed to the correction at the following suffix row.
    return sequential_greedy(self.model,sample_hidden_states,next_token_ids)
