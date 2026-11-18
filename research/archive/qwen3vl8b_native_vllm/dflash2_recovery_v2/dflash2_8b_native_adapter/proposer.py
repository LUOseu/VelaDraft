"""Distinct DFlash2 router; stock propose body/verifier retained unchanged."""
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer as OfficialDFlashProposer

from . import contracts as c
from . import spatial


class DFlash2ProposerRouter(OfficialDFlashProposer):
    def _get_model(self):
        if self._dflash2_active:
            approved=c.runtime_receipt()
            c.validate_draft_loader_selection(self.vllm_config,approved)
        return OfficialDFlashProposer._get_model(self)

    def load_model(self,target_model):
        result=OfficialDFlashProposer.load_model(self,target_model)
        if self._dflash2_active:
            from vllm.model_executor.models.interfaces import supports_multimodal
            target=target_model.get_language_model() if supports_multimodal(target_model) else target_model
            c.require(len(self.model._dflash2_loaded_tensor_names)==51,'Actual official loader did not load all51 DFlash2 tensors')
            c.require(self.model.lm_head is target.lm_head
                and self.model.model.embed_tokens is target.model.embed_tokens,
                'DFlash2 must use the exact standard target head and embedding objects')
        return result

    def __init__(self,vllm_config,device,runner=None):
        cfg=vllm_config.speculative_config.draft_model_config.hf_config
        self._dflash2_active=c.enabled(cfg)
        c.require(self._dflash2_active or not c.near_dflash2(cfg),'Near/unmarked DFlash2 must never fall through to stock DFlash.')
        if self._dflash2_active:
            self._dflash2_approved=c.runtime_receipt();c.validate_launch_config(vllm_config,self._dflash2_approved)
        self._dflash2_in_propose=False;self._dflash2_anchor_ids=None;self._dflash2_backend_initialized=False
        OfficialDFlashProposer.__init__(self,vllm_config,device,runner=runner)
        if self._dflash2_active:
            c.require(self.max_query_tokens==8,'Actual publicS7 must reach stock constructor as realB8.')
            c.require(self.block_size in (-1,16),'Unexpected stock pre-backend page sentinel.')
            self.mrope_positions=torch.zeros((3,9),dtype=torch.int64,device=device)
            self._context_mrope_positions=torch.zeros((3,self.max_num_tokens),dtype=torch.int64,device=device)

    def _runtime_guard(self):
        c.require(self.num_speculative_tokens==self.speculative_config.num_speculative_tokens
            ==self.vllm_config.speculative_config.num_speculative_tokens==7,'PublicS changed after initialization.')
        c.require(self.parallel_drafting is True and self.speculative_config.parallel_drafting is True,'Parallel B8 proposal policy changed after initialization.')

    def _raise_if_mrope(self):
        if self._dflash2_active:
            c.validate_config(self.draft_model_config.hf_config);return
        return OfficialDFlashProposer._raise_if_mrope(self)

    def _page_guard(self,*,real,slot_mappings=None):
        if real or self._dflash2_backend_initialized:
            c.require(self._dflash2_backend_initialized and self.block_size==16,'Actual initialized draft cache page must be16.')
        else:
            c.require(self.block_size==-1 and not slot_mappings,'Only no-write pre-backend dummy permits stock page sentinel-1.')

    def initialize_attn_backend(self,*args,**kwargs):
        result=OfficialDFlashProposer.initialize_attn_backend(self,*args,**kwargs)
        if self._dflash2_active:
            c.require(self.block_size==16,'Actual draft attention backend resolved a non16 page.')
            self._dflash2_backend_initialized=True
        return result

    def propose(self,*args,**kwargs):
        if not self._dflash2_active:return OfficialDFlashProposer.propose(self,*args,**kwargs)
        self._runtime_guard()
        self._page_guard(real=True)
        c.require(not self._dflash2_in_propose,'Nested/stale DFlash2 request scope.')
        self._dflash2_in_propose=True;self._dflash2_anchor_ids=None
        token=spatial.DUMMY_SCOPE.set(False)
        try:return OfficialDFlashProposer.propose(self,*args,**kwargs)
        finally:
            self._dflash2_anchor_ids=None;self._dflash2_in_propose=False;spatial.DUMMY_SCOPE.reset(token)

    def set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,target_hidden_states,
                              token_indices_to_sample,cad,num_rejected_tokens_gpu):
        if not self._dflash2_active:
            return OfficialDFlashProposer.set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,
                target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        self._runtime_guard()
        self._page_guard(real=True)
        c.require(self._dflash2_in_propose and cad.batch_size()==1 and next_token_ids.shape==(1,),
            'One actual per-request anchor must be bound inside a real DFlash2 proposal.')
        n=target_token_ids.shape[0]
        c.require(target_positions.shape==(3,n),'All3 target spatial axes required.')
        physical,axes=spatial.context_query_axes(target_positions,cad.seq_lens,cad.query_start_loc,num_rejected_tokens_gpu)
        result=OfficialDFlashProposer.set_inputs_first_pass(self,target_token_ids,next_token_ids,physical,
            target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        self._dflash2_anchor_ids=next_token_ids
        self._context_mrope_positions[:,:n].copy_(target_positions)
        self.mrope_positions.zero_();self.mrope_positions[:,:8].copy_(axes)
        num_tokens,_,new_cad=result
        c.require(num_tokens==8,'DFlash2 must retain all8 real query computations.')
        return num_tokens,result[1],new_cad

    def build_model_inputs_first_pass(self,num_tokens,num_input_tokens,mm_embed_inputs):
        if not self._dflash2_active:return OfficialDFlashProposer.build_model_inputs_first_pass(self,num_tokens,num_input_tokens,mm_embed_inputs)
        self._page_guard(real=True)
        n=self._dflash_num_context
        self.model.precompute_and_store_context_kv(self._dflash_hidden_states,self._context_mrope_positions[:,:n],
            self._context_slot_mapping_buffer[:n])
        return dict(input_ids=self.input_ids[:num_input_tokens],positions=self._get_positions(num_input_tokens),inputs_embeds=None),num_input_tokens

    def _sample_draft_tokens(self,hidden_states,sampling_metadata):
        if not self._dflash2_active:return OfficialDFlashProposer._sample_draft_tokens(self,hidden_states,sampling_metadata)
        self._runtime_guard()
        self._page_guard(real=True)
        c.require(self._dflash2_in_propose and self._dflash2_anchor_ids is not None,'DFlash2 sampling has no actual request anchor.')
        c.require(sampling_metadata.all_greedy is True,'DFlash2 mount is restricted to actual greedy verification.')
        c.require(hidden_states.shape[0]==7 and hidden_states.dtype==torch.bfloat16,'Canonical slots1..7 BF16 head rows required.')
        tokens=self.model.sample_dflash2_prefix(hidden_states,self._dflash2_anchor_ids)
        c.require(tokens.shape==(1,7),'Exactly7 learned selector proposals required.')
        return tokens.reshape(-1),None

    def dummy_run(self,*args,**kwargs):
        if not self._dflash2_active:return OfficialDFlashProposer.dummy_run(self,*args,**kwargs)
        mappings=kwargs.get('slot_mappings',args[3] if len(args)>3 else None)
        self._page_guard(real=False,slot_mappings=mappings)
        self._runtime_guard();token=spatial.DUMMY_SCOPE.set(True)
        try:return OfficialDFlashProposer.dummy_run(self,*args,**kwargs)
        finally:spatial.DUMMY_SCOPE.reset(token)

    def _determine_batch_execution_and_padding(self,num_tokens,use_cudagraphs=True):
        result=OfficialDFlashProposer._determine_batch_execution_and_padding(self,num_tokens,use_cudagraphs=use_cudagraphs)
        if self._dflash2_active:
            self._runtime_guard();mode,padded,_=result
            spatial.validate_query_batch(num_tokens,padded,mode.name,dummy=spatial.DUMMY_SCOPE.get(),
                keys_initialized=self.cudagraph_dispatcher.keys_initialized)
        return result

    def initialize_cudagraph_keys(self,cudagraph_mode):
        OfficialDFlashProposer.initialize_cudagraph_keys(self,cudagraph_mode)
        if self._dflash2_active:
            self._runtime_guard();mode,descriptor=self.cudagraph_dispatcher.dispatch(8)
            spatial.validate_query_batch(8,descriptor.num_tokens,mode.name,dummy=False,
                keys_initialized=self.cudagraph_dispatcher.keys_initialized)
