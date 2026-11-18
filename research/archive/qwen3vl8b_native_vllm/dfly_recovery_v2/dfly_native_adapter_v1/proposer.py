"""Narrow marked DFly proposer with explicit raw12288 context and stock sampling."""
import torch
from vllm.forward_context import set_forward_context
from vllm.v1.spec_decode.dflash import DFlashProposer as OfficialDFlashProposer
from . import contracts as c
from . import spatial
from .proposal_logic import raw_parallel_propose

class DFlyProposerRouter(OfficialDFlashProposer):
    def __init__(self,vllm_config,device,runner=None):
        cfg=vllm_config.speculative_config.draft_model_config.hf_config
        self._dfly_active=c.enabled(cfg)
        c.require(self._dfly_active or not c.near_dfly(cfg),'Near/unmarked DFly cannot fall through to stock DFlash')
        if self._dfly_active:
            self._dfly_approved=c.runtime_receipt();c.validate_launch_config(vllm_config)
        self._dfly_in_propose=False;self._dfly_backend_initialized=False;self._dfly_raw_context=None
        OfficialDFlashProposer.__init__(self,vllm_config,device,runner=runner)
        if self._dfly_active:
            c.require(self.hidden_size==4096 and self.max_query_tokens==8,'DFly query ABI must remain4096/B8')
            c.require(self.block_size in (-1,16),'Unexpected initial physical page sentinel')
            # Dedicated raw profiling buffer; never repurpose hidden_states4096.
            self._raw_context_profile_buffer=torch.zeros((self.max_num_tokens,12288),dtype=self.dtype,device=device)
            self.mrope_positions=torch.zeros((3,9),dtype=torch.int64,device=device)
            self._context_mrope_positions=torch.zeros((3,self.max_num_tokens),dtype=torch.int64,device=device)

    def load_model(self,target_model):
        result=OfficialDFlashProposer.load_model(self,target_model)
        if self._dfly_active:
            from vllm.model_executor.models.interfaces import supports_multimodal
            target=target_model.get_language_model() if supports_multimodal(target_model) else target_model
            c.require(len(self.model._dfly_loaded_tensor_names)==42,'Actual loader must load all42 DFly tensors')
            c.require(self.model.lm_head is target.lm_head and self.model.model.embed_tokens is target.model.embed_tokens,
                'DFly requires exact original target head and embedding objects')
        return result

    def _runtime_guard(self):
        c.require(self.num_speculative_tokens==self.speculative_config.num_speculative_tokens
            ==self.vllm_config.speculative_config.num_speculative_tokens==7,'Public realS7 changed')
        c.require(self.parallel_drafting is True and self.speculative_config.parallel_drafting is True,'Parallel B8 changed')

    def _raise_if_mrope(self):
        if self._dfly_active:c.validate_config(self.draft_model_config.hf_config);return
        return OfficialDFlashProposer._raise_if_mrope(self)

    def _page_guard(self,*,real,slot_mappings=None):
        if real or self._dfly_backend_initialized:
            c.require(self._dfly_backend_initialized and self.block_size==16,'Initialized draft physical page must be16')
        else:c.require(self.block_size==-1 and not slot_mappings,'Only no-write pre-backend dummy allows sentinel-1')

    def initialize_attn_backend(self,*args,**kwargs):
        result=OfficialDFlashProposer.initialize_attn_backend(self,*args,**kwargs)
        if self._dfly_active:
            c.require(self.block_size==16,'Actual initialized draft page is not16');self._dfly_backend_initialized=True
        return result

    def propose(self,*args,**kwargs):
        if not self._dfly_active:return OfficialDFlashProposer.propose(self,*args,**kwargs)
        self._runtime_guard();self._page_guard(real=True)
        c.require(not self._dfly_in_propose,'Nested/stale DFly proposal scope')
        self._dfly_in_propose=True;token=spatial.DUMMY_SCOPE.set(False)
        try:return raw_parallel_propose(self,*args,**kwargs)
        finally:
            self._dfly_raw_context=None;self._dfly_in_propose=False;spatial.DUMMY_SCOPE.reset(token)

    def set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,target_hidden_states,
                              token_indices_to_sample,cad,num_rejected_tokens_gpu):
        if not self._dfly_active:
            return OfficialDFlashProposer.set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,
                target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        self._runtime_guard();self._page_guard(real=True)
        n=target_token_ids.shape[0]
        c.require(self._dfly_in_propose and target_hidden_states.shape==(n,12288),'Explicit raw context must belong to this proposal')
        c.require(target_positions.shape==(3,n),'All3 target spatial axes required')
        physical,axes=spatial.context_query_axes(target_positions,cad.seq_lens,cad.query_start_loc,num_rejected_tokens_gpu)
        result=OfficialDFlashProposer.set_inputs_first_pass(self,target_token_ids,next_token_ids,physical,
            target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        self._dfly_raw_context=target_hidden_states
        # Drop the inherited ambiguous4096 context alias. Only the explicit ABI is used.
        self._dflash_hidden_states=None
        self._context_mrope_positions[:,:n].copy_(target_positions)
        self.mrope_positions.zero_();self.mrope_positions[:,:8].copy_(axes)
        c.require(result[0]==8,'All8 noncausal query rows must execute')
        return result

    def build_model_inputs_first_pass(self,num_tokens,num_input_tokens,mm_embed_inputs):
        if not self._dfly_active:
            return OfficialDFlashProposer.build_model_inputs_first_pass(self,num_tokens,num_input_tokens,mm_embed_inputs)
        self._page_guard(real=True)
        n=self._dflash_num_context
        c.require(self._dfly_raw_context is not None and self._dfly_raw_context.shape==(n,12288),'Raw context payload missing')
        self.model.precompute_and_store_context_kv(self._dfly_raw_context,self._context_mrope_positions[:,:n],
            self._context_slot_mapping_buffer[:n])
        return dict(input_ids=self.input_ids[:num_input_tokens],positions=self._get_positions(num_input_tokens),inputs_embeds=None),num_input_tokens

    @torch.inference_mode()
    def dummy_run(self,num_tokens,use_cudagraphs=True,is_graph_capturing=False,slot_mappings=None):
        if not self._dfly_active:
            return OfficialDFlashProposer.dummy_run(self,num_tokens,use_cudagraphs,is_graph_capturing,slot_mappings)
        self._runtime_guard();self._page_guard(real=False,slot_mappings=slot_mappings)
        token=spatial.DUMMY_SCOPE.set(True)
        try:
            q=min(num_tokens,self.max_query_tokens)
            mode,num_input_tokens,across_dp=self._determine_batch_execution_and_padding(q,use_cudagraphs=use_cudagraphs)
            if self._draft_attn_layer_names and slot_mappings is not None and next(iter(self._draft_attn_layer_names)) in slot_mappings:
                mapping=self._get_slot_mapping(num_input_tokens)
            else:mapping=slot_mappings or {}
            self.model.precompute_and_store_context_kv(self._raw_context_profile_buffer[:num_tokens],self._context_mrope_positions[:,:num_tokens])
            with set_forward_context(None,self.vllm_config,num_tokens=num_input_tokens,num_tokens_across_dp=across_dp,
                cudagraph_runtime_mode=mode,slot_mapping=mapping):
                self.model(input_ids=self.input_ids[:num_input_tokens],positions=self._get_positions(num_input_tokens),inputs_embeds=None)
        finally:spatial.DUMMY_SCOPE.reset(token)

    def _determine_batch_execution_and_padding(self,num_tokens,use_cudagraphs=True):
        result=OfficialDFlashProposer._determine_batch_execution_and_padding(self,num_tokens,use_cudagraphs=use_cudagraphs)
        if self._dfly_active:
            self._runtime_guard();mode,padded,_=result
            spatial.validate_query_batch(num_tokens,padded,mode.name,dummy=spatial.DUMMY_SCOPE.get(),keys_initialized=self.cudagraph_dispatcher.keys_initialized)
        return result

    def initialize_cudagraph_keys(self,cudagraph_mode):
        OfficialDFlashProposer.initialize_cudagraph_keys(self,cudagraph_mode)
        if self._dfly_active:
            self._runtime_guard();mode,descriptor=self.cudagraph_dispatcher.dispatch(8)
            spatial.validate_query_batch(8,descriptor.num_tokens,mode.name,dummy=False,keys_initialized=self.cudagraph_dispatcher.keys_initialized)
