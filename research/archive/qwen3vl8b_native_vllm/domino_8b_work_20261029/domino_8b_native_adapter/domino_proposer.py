"""Domino-only class router over the unchanged official parallel propose body."""
from __future__ import annotations
from contextvars import ContextVar
import os
from pathlib import Path
import hashlib,json
import torch
from vllm.v1.spec_decode.dflash import DFlashProposer as OfficialDFlashProposer
from contracts import selected,validate_config,validate_query,require_runtime_binding,validate_loading,TARGET_ROOT,ROOT
from domino_logic import sample_domino
from spatial_math import absolute_context_positions,spatial_query_positions
_DUMMY=ContextVar('domino_dummy',default=False)

def validate_target(mc):
    if str(Path(mc.model).resolve())!=str(Path(TARGET_ROOT).resolve()):raise ValueError('Exact frozen target path required')
    if mc.hf_config.model_type!='qwen3_vl' or mc.hf_config.tie_word_embeddings is not False or mc.dtype!=torch.bfloat16:raise ValueError('Pinned BF16 Qwen3-VL-8B required')
    text=mc.hf_text_config
    for name,value in {'hidden_size':4096,'num_hidden_layers':36,'num_attention_heads':32,'num_key_value_heads':8,'vocab_size':151936,'intermediate_size':12288,'rms_norm_eps':1e-6}.items():
        if getattr(text,name,None)!=value:raise ValueError('Changed target '+name)
    if getattr(text,'head_dim',128)!=128:raise ValueError('Changed target head width')
    rope=text.rope_parameters
    if rope.get('rope_theta')!=5000000. or rope.get('mrope_section')!=[24,20,20] or rope.get('mrope_interleaved') is not True:raise ValueError('Changed target RoPE')
    config=Path(TARGET_ROOT)/'config.json'
    expected=json.loads((ROOT/'source_bindings.json').read_text())['deferred_full_artifacts'][str(config)]
    if hashlib.sha256(config.read_bytes()).hexdigest()!=expected:raise ValueError('Target config bytes changed')


def validate_launch(vc):
    from vllm.config import CUDAGraphMode,CompilationMode
    s=vc.speculative_config
    validate_config(s.draft_model_config.hf_config)
    if type(s.num_speculative_tokens) is not int or s.num_speculative_tokens!=7 or s.method!='dflash':raise ValueError('Domino only supports actual public S7/dflash transport')
    p=vc.parallel_config
    if (p.tensor_parallel_size,p.pipeline_parallel_size,p.data_parallel_size)!=(1,1,1) or vc.scheduler_config.max_num_seqs!=1:raise ValueError('Only TP=PP=DP1 and batch1')
    validate_target(vc.model_config)
    validate_loading(vc)
    if vc.cache_config.block_size!=16:raise ValueError('Physical KV page must remain16')
    if getattr(s.attention_backend,'name',None)!='FLASH_ATTN':raise ValueError('Explicit draft Flash required')
    attention=vc.attention_config
    if getattr(attention.backend,'name',None)!='FLASH_ATTN' or any(getattr(attention,key) not in (None,16) for key in ('flex_attn_block_m','flex_attn_block_n','flex_attn_q_block_size','flex_attn_kv_block_size')):raise ValueError('Target public Flash/page16 config required')
    if not s.draft_model_config.uses_mrope or vc.model_config.enforce_eager or s.enforce_eager:raise ValueError('Official MRoPE and graph path required')
    if s.use_local_argmax_reduction:raise ValueError('Pre-correction local argmax forbidden')
    c=vc.compilation_config
    if c.mode!=CompilationMode.VLLM_COMPILE or c.cudagraph_mode.mixed_mode() not in (CUDAGraphMode.PIECEWISE,CUDAGraphMode.FULL):raise ValueError('Official compile/PIECEWISE draft required')
    if c.cudagraph_capture_sizes is not None and 8 not in c.cudagraph_capture_sizes:raise ValueError('Real8 capture required without padding')
    if vc.scheduler_config.max_num_batched_tokens<8:raise ValueError('Capacity below8')

class DominoProposer(OfficialDFlashProposer):
    def __init__(self,vllm_config,device,runner=None,*,_logic_interface=False):
        self._domino=selected(vllm_config.speculative_config.draft_model_config.hf_config)
        self._domino_logic_interface=_logic_interface
        if self._domino:
            validate_launch(vllm_config)
            if _logic_interface:
                if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.device(device).type!='cpu' or torch.cuda.is_initialized():raise RuntimeError('Non-promotable logic interface requires hidden CUDA and CPU device')
            else:require_runtime_binding(vllm_config)
        super().__init__(vllm_config,device,runner=runner)
        if self._domino:
            if self.max_query_tokens!=8:raise RuntimeError('Official query capacity differs from S7')
            if self.block_size not in (-1,16):raise ValueError('Unexpected initialized physical KV page')
            self._domino_s=self.num_speculative_tokens
            self.mrope_positions=torch.zeros((3,9),dtype=torch.int64,device=device)
            self._context_mrope_positions=torch.zeros((3,self.max_num_tokens),dtype=torch.int64,device=device)
            self._domino_active=False;self._domino_anchor=None

    def load_model(self,target_model):
        result=super().load_model(target_model)
        if self._domino:
            from vllm.model_executor.models.interfaces import supports_multimodal
            target=target_model.get_language_model() if supports_multimodal(target_model) else target_model
            if self.model.lm_head is not target.lm_head or self.model.model.embed_tokens is not target.model.embed_tokens:
                raise ValueError('Domino requires exact target lm_head and embedding objects')
            self._domino_target_alias_verified=True
        return result

    def _runtime_block(self,*,allow_uninitialized=False):
        values=(self.num_speculative_tokens,self._domino_s,self.speculative_config.num_speculative_tokens,self.vllm_config.speculative_config.num_speculative_tokens)
        if any(type(value) is not int or value!=7 for value in values):raise ValueError('S changed during engine lifetime')
        if self.block_size!=16 and not (allow_uninitialized and self.block_size==-1):raise ValueError('Real physical KV page must be16')
        return 8

    def _create_draft_vllm_config(self):
        result=super()._create_draft_vllm_config()
        if self._domino and (getattr(result.attention_config.backend,'name',None)!='FLASH_ATTN' or result.attention_config.use_non_causal is not True):raise ValueError('Derived draft must be actual noncausal Flash')
        return result

    def initialize_attn_backend(self,*args,**kwargs):
        result=super().initialize_attn_backend(*args,**kwargs)
        if self._domino and self.block_size!=16:raise ValueError('Actual draft KV metadata page must be16')
        return result

    def _raise_if_mrope(self):
        if self._domino:validate_config(self.draft_model_config.hf_config);return
        return super()._raise_if_mrope()

    def set_inputs_first_pass(self,target_token_ids,next_token_ids,target_positions,target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu):
        if not self._domino:return super().set_inputs_first_pass(target_token_ids,next_token_ids,target_positions,target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        self._runtime_block()
        if not self._domino_active or cad.batch_size()!=1:raise ValueError('Domino inputs require one active real proposal')
        n=target_token_ids.shape[0]
        if target_positions.shape!=(3,n):raise ValueError('All three target axes required')
        if next_token_ids.numel()!=1:raise ValueError('One actual bonus anchor required')
        self._domino_anchor=next_token_ids.reshape(1).clone()
        physical=absolute_context_positions(cad.seq_lens,cad.query_start_loc,n)
        num_queries,_,new_cad=super().set_inputs_first_pass(target_token_ids,next_token_ids,physical,target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu)
        axes=spatial_query_positions(target_positions,cad.query_start_loc,num_rejected_tokens_gpu,8)
        self._context_mrope_positions[:,:n].copy_(target_positions)
        self.mrope_positions.zero_();self.mrope_positions[:,:axes.shape[1]].copy_(axes)
        # All8 rows reach the head; sampler returns seven shifted proposals.
        return num_queries,torch.arange(8,dtype=torch.int32,device=self.device),new_cad

    def build_model_inputs_first_pass(self,num_tokens,num_input_tokens,mm_embed_inputs):
        if not self._domino:return super().build_model_inputs_first_pass(num_tokens,num_input_tokens,mm_embed_inputs)
        n=self._dflash_num_context
        self.model.precompute_and_store_context_kv(self._dflash_hidden_states,self._context_mrope_positions[:,:n],self._context_slot_mapping_buffer[:n])
        return dict(input_ids=self.input_ids[:num_input_tokens],positions=self._get_positions(num_input_tokens),inputs_embeds=None),num_input_tokens

    def _walk(self,hidden,inputs):
        m=self.model.model
        return sample_domino(hidden,inputs,compute_logits=self.model.compute_logits,embed_tokens=self.model.model.embed_tokens,weight_ih=m.prefix_gru.weight_ih_l0,weight_hh=m.prefix_gru.weight_hh_l0,projector_in=m.embed_proj[0].weight,projector_out=m.embed_proj[2].weight)

    def _sample_draft_tokens(self,hidden_states,sampling_metadata):
        if not self._domino:return super()._sample_draft_tokens(hidden_states,sampling_metadata)
        self._runtime_block()
        if not self._domino_active or self._domino_anchor is None or not sampling_metadata.all_greedy:raise ValueError('Real Domino sampler requires active actual anchor and greedy policy')
        inputs=self.input_ids[:8].clone();inputs[0]=self._domino_anchor[0]
        return self._walk(hidden_states,inputs),None

    def propose(self,*args,**kwargs):
        if not self._domino:return super().propose(*args,**kwargs)
        if getattr(self,'_domino_logic_interface',False):raise RuntimeError('Logic interface cannot execute an engine proposal')
        self._runtime_block()
        if self._domino_active:raise RuntimeError('Reentrant Domino proposal')
        token=_DUMMY.set(False);self._domino_active=True
        try:return super().propose(*args,**kwargs)
        finally:self._domino_anchor=None;self._domino_active=False;_DUMMY.reset(token)

    def dummy_run(self,num_tokens,use_cudagraphs=True,is_graph_capturing=False,slot_mappings=None):
        if not self._domino:return super().dummy_run(num_tokens,use_cudagraphs=use_cudagraphs,is_graph_capturing=is_graph_capturing,slot_mappings=slot_mappings)
        if getattr(self,'_domino_logic_interface',False):raise RuntimeError('Logic interface cannot execute model profiling')
        self._runtime_block(allow_uninitialized=True);token=_DUMMY.set(True)
        try:
            result=super().dummy_run(num_tokens,use_cudagraphs=use_cudagraphs,is_graph_capturing=is_graph_capturing,slot_mappings=slot_mappings)
            if not is_graph_capturing:
                # Memory profiling includes all learned head paths; no proposal/cache claim.
                with torch.inference_mode():self._walk(torch.zeros((8,self.hidden_size),dtype=self.dtype,device=self.device),torch.zeros(8,dtype=torch.int32,device=self.device))
            return result
        finally:_DUMMY.reset(token)

    def _determine_batch_execution_and_padding(self,num_tokens,use_cudagraphs=True):
        result=super()._determine_batch_execution_and_padding(num_tokens,use_cudagraphs=use_cudagraphs)
        if self._domino:
            self._runtime_block(allow_uninitialized=_DUMMY.get());mode,padded,_=result
            validate_query(7,num_tokens,padded,mode.name,dummy=_DUMMY.get(),initialized=self.cudagraph_dispatcher.keys_initialized)
        return result

    def initialize_cudagraph_keys(self,cudagraph_mode):
        result=super().initialize_cudagraph_keys(cudagraph_mode)
        if self._domino:
            self._runtime_block(allow_uninitialized=True);mode,descriptor=self.cudagraph_dispatcher.dispatch(8)
            validate_query(7,8,descriptor.num_tokens,mode.name,dummy=False,initialized=self.cudagraph_dispatcher.keys_initialized)
        return result
