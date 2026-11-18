import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import torch
from torch import nn
from . import contracts as c

def cpu_vllm_config(function):
    def wrapped(*args,**kwargs):
        from vllm.config import VllmConfig,set_current_vllm_config
        with set_current_vllm_config(VllmConfig()):return function(*args,**kwargs)
    return wrapped

class IntegrationTests(unittest.TestCase):
    def require(self,name):self.assertTrue((c.HERE/name).is_file(),'missing independent native module: '+name)

    def test_owned_compile_no_fc_and_standard_decoder_and_head(self):
        self.require('backbone.py');self.require('model.py')
        from .backbone import DFlareOwnedBackbone,DFlareOwnedDecoderLayer
        from .model import DFlashDFlareNativeQwen3ForCausalLM
        from .spatial import validate_compile_ownership
        from vllm.model_executor.models import qwen3_dflash as official
        self.assertIs(DFlareOwnedBackbone.forward,official.DFlashQwen3Model.forward)
        self.assertIs(DFlareOwnedDecoderLayer.forward,official.DFlashQwen3DecoderLayer.forward)
        self.assertIs(DFlashDFlareNativeQwen3ForCausalLM.compute_logits,official.DFlashQwen3ForCausalLM.compute_logits)
        self.assertIs(DFlashDFlareNativeQwen3ForCausalLM.forward,official.DFlashQwen3ForCausalLM.forward)
        proof=validate_compile_ownership(DFlareOwnedBackbone)
        self.assertTrue(proof['stock_mapping_unchanged']);self.assertFalse(proof['shared_with_stock'])
        empty=object.__new__(DFlashDFlareNativeQwen3ForCausalLM)
        with self.assertRaisesRegex(ValueError,'raw context'):empty.combine_hidden_states(torch.empty(2,12288))

    def test_raw_payload_survives_propose_and_standard_slot_sampling(self):
        self.require('proposal_logic.py')
        from .proposal_logic import raw_parallel_propose
        from vllm.config import CUDAGraphMode
        raw=torch.randn(5,12288);seen=[];indices=torch.arange(1,8)
        class Model:
            def __call__(self,**kwargs):return torch.arange(8*4096).view(8,4096)
            def combine_hidden_states(self,*args):raise AssertionError('raw context must never be collapsed')
        proposer=NS(model=Model(),num_speculative_tokens=7,parallel_drafting=True,
            hidden_size=4096,vllm_config=None,_last_draft_probs=None)
        def inputs(**kwargs):
            self.assertIs(kwargs['target_hidden_states'],raw)
            return 8,indices,NS(slot_mapping=torch.arange(8))
        proposer.set_inputs_first_pass=inputs
        proposer.build_per_group_and_layer_attn_metadata=lambda cad:([],{})
        proposer._determine_batch_execution_and_padding=lambda n:(CUDAGraphMode.PIECEWISE,8,None)
        proposer.build_model_inputs_first_pass=lambda *args:({},8)
        proposer._get_slot_mapping=lambda *args:{}
        proposer.model_returns_tuple=lambda:False
        proposer._sample_draft_tokens=lambda h,s:(seen.append(h) or torch.arange(7),None)
        from contextlib import nullcontext
        with mock.patch('dflare_8b_native_adapter.proposal_logic.set_forward_context',return_value=nullcontext()):
            result=raw_parallel_propose(proposer,torch.zeros(5),torch.zeros(3,5),raw,torch.tensor([2]),None,NS(batch_size=lambda:1),NS())
        self.assertEqual(result.tolist(),[list(range(7))]);self.assertTrue(torch.equal(seen[0],proposer.model()[1:8]))
        with self.assertRaisesRegex(ValueError,'12288'):
            raw_parallel_propose(proposer,torch.zeros(5),torch.zeros(3,5),raw[:,:4096],torch.tensor([2]),None,NS(batch_size=lambda:1),NS())

    def test_raw_profile_dummy_executes_full_context_path_without_writes_and_resets_scope(self):
        self.require('proposer.py')
        from .proposer import DFlareProposerRouter
        from . import spatial
        from vllm.config import CUDAGraphMode
        p=object.__new__(DFlareProposerRouter);p._dflare_active=True;p._dflare_backend_initialized=False;p.block_size=-1
        p.num_speculative_tokens=7;p.parallel_drafting=True;p.speculative_config=NS(num_speculative_tokens=7,parallel_drafting=True)
        p.vllm_config=NS(speculative_config=p.speculative_config);p.max_query_tokens=8;p._draft_attn_layer_names=[]
        p._raw_context_profile_buffer=torch.randn(13,12288);p._context_mrope_positions=torch.zeros(3,13,dtype=torch.long)
        p.input_ids=torch.zeros(8,dtype=torch.long);p.mrope_positions=torch.zeros(3,9,dtype=torch.long)
        p._get_positions=lambda n:p.mrope_positions[:,:n]
        p._determine_batch_execution_and_padding=lambda n,use_cudagraphs:(CUDAGraphMode.NONE,n,None)
        observed=[]
        class Model:
            def precompute_and_store_context_kv(self,raw,axes,slots=None):observed.append((raw,axes,slots,spatial.DUMMY_SCOPE.get()))
            def __call__(self,**kw):raise RuntimeError('test forward failure')
        p.model=Model()
        from contextlib import nullcontext
        with mock.patch('dflare_8b_native_adapter.proposer.set_forward_context',return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError,'test forward failure'):p.dummy_run(13,use_cudagraphs=False)
        raw,axes,slots,scope=observed[0]
        self.assertEqual(raw.shape,(13,12288));self.assertEqual(axes.shape,(3,13));self.assertIsNone(slots);self.assertTrue(scope)
        self.assertFalse(spatial.DUMMY_SCOPE.get())

    @cpu_vllm_config
    def test_per_layer_precompute_matches_canonical_norm_mrope_and_cache_order(self):
        self.require('spatial.py')
        from .spatial import precompute
        from .context import build_layer_contexts,project_context
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
        from vllm.model_executor.layers.layernorm import RMSNorm
        torch.manual_seed(721)
        raw=torch.randn(6,12288,dtype=torch.bfloat16);fusion=nn.Parameter(torch.randn(3,3,dtype=torch.bfloat16))
        norm=RMSNorm(4096,eps=1e-6).bfloat16()
        rope=MRotaryEmbedding(128,128,1024,5000000.,True,torch.bfloat16,mrope_section=[24,20,20],mrope_interleaved=True)
        layers=[];writes=[]
        for i in range(3):
            attention=NS(k_proj_target=nn.Linear(4096,1024,bias=False,dtype=torch.bfloat16),
                v_proj_target=nn.Linear(4096,1024,bias=False,dtype=torch.bfloat16),
                k_norm=RMSNorm(128,eps=1e-6).bfloat16(),rotary_emb=rope)
            attention.attn=NS(kv_cache=object())
            attention.attn.impl=NS(do_kv_cache_update=lambda a,k,v,cache,slot,i=i:writes.append((i,k.clone(),v.clone(),slot)))
            layers.append(NS(self_attn=attention))
        axes=torch.tensor([[3,3,4,5,6,7],[9,9,10,11,12,13],[15,16,16,17,18,19]])
        slots=torch.tensor([0,1,2,3,4,-1])
        obj=NS(config={'architectures':[c.ARCH],'dflash_config':{c.MARKER:True}},
            hidden_norm=norm,layer_fusion_weights=fusion,layers=layers,_dflare_weights_loaded=True)
        contexts=build_layer_contexts(raw,fusion,norm)
        for write in (False,True):
            precompute(obj,raw,axes,slots if write else None)
            self.assertEqual(len(writes),3 if write else 0)
        for i,k,v,mapping in writes:
            key,value=project_context(contexts[i],layers[i].self_attn.k_proj_target,layers[i].self_attn.v_proj_target)
            key=layers[i].self_attn.k_norm(key).reshape(6,1024)
            rotated,_=rope.forward_native(axes,key,torch.zeros(6,128,dtype=key.dtype))
            self.assertTrue(torch.equal(k,rotated.view(6,8,128)));self.assertTrue(torch.equal(v,value))
            self.assertIs(mapping,slots)
        self.assertEqual([x[0] for x in writes],[0,1,2]);self.assertFalse(torch.cuda.is_initialized())

    def test_truthful_registry_public_flash_configs_and_no_model_or_cuda(self):
        self.require('model.py');self.require('proposer.py')
        from .install import register_models,verify_official_interfaces
        from .public_config import materialize_public_config
        from vllm.model_executor.models import ModelRegistry
        register_models();register_models();verify_official_interfaces()
        self.assertEqual(ModelRegistry.models[c.ARCH].class_name,c.ARCH)
        self.assertTrue((c.HERE/'hf_step75000_v1/config.json').is_file(),'Actual export must exist')
        with (mock.patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA forbidden')),
              mock.patch('torch.nn.Module.__init__',side_effect=AssertionError('serving construction forbidden'))):
            proof,vc=materialize_public_config(c.HERE/'hf_step75000_v1')
        self.assertEqual(proof['resolved_architectures'],[c.ARCH]);self.assertTrue(proof['draft_non_causal'])
        for field,value in [('block_size',32)]:
            bad=copy.copy(vc);bad.cache_config=copy.copy(vc.cache_config);setattr(bad.cache_config,field,value)
            with self.assertRaises(ValueError):c.validate_launch_config(bad)
        self.assertFalse(torch.cuda.is_initialized())

    def test_marker_stripping_wrong_context_and_runtime_receipt_fail_closed(self):
        self.require('proposer.py')
        from .proposer import DFlareProposerRouter,OfficialDFlashProposer
        meta=json.loads((c.HERE/'CHECKPOINT_METADATA.json').read_text());cfg=copy.deepcopy(meta['draft_config'])
        c.validate_config(cfg,source=True);self.assertTrue(c.near_dflare(cfg));self.assertFalse(c.enabled(cfg))
        with self.assertRaises(ValueError):c.validate_config(cfg)
        cfg.update(architectures=[c.ARCH],source_checkpoint_sha256=c.CHECKPOINT_SHA)
        cfg['dflash_config'].update({c.MARKER:True,'raw_context_width':12288,'raw_context_abi':'ordered_target_layers_12288_v1'})
        cfg['rope_parameters'].update(mrope_section=[24,20,20],mrope_interleaved=True);c.validate_config(cfg)
        cfg['dflash_config']['raw_context_width']=4096
        with self.assertRaises(ValueError):c.validate_config(cfg)
        p=object.__new__(DFlareProposerRouter);p._dflare_active=False
        with mock.patch.object(OfficialDFlashProposer,'propose',return_value='stock') as delegate:
            self.assertEqual(p.propose(),'stock');delegate.assert_called_once()
        with mock.patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'CPU_RECEIPT'):c.runtime_receipt()

    def test_dedicated_loader_never_feeds_target_names_to_stock_substring_packer(self):
        from .backbone import DFlareOwnedBackbone,DEDICATED
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
        fixture=object.__new__(DFlareOwnedBackbone);nn.Module.__init__(fixture)
        fixture.layers=nn.ModuleList()
        for i in range(3):
            layer=nn.Module();layer.self_attn=nn.Module()
            for kind in ('k','v'):setattr(layer.self_attn,kind+'_proj_target',nn.Linear(4096,1024,bias=False,dtype=torch.bfloat16))
            fixture.layers.append(layer)
        state={k:torch.empty(v['shape'],device='meta',dtype=torch.bfloat16) for k,v in c.SCHEMA.items()}
        for i,name in enumerate(sorted(DEDICATED)):state[name]=torch.full(c.SCHEMA[name]['shape'],float(i+1),dtype=torch.bfloat16)
        captured=[]
        def stock(obj,pairs):
            remaining=dict(pairs);captured.append(remaining)
            self.assertFalse(set(remaining)&DEDICATED);return set(remaining)
        with mock.patch.object(DFlashQwen3Model,'load_weights',stock):fixture.load_weights(state.items())
        self.assertTrue(fixture._dflare_weights_loaded);self.assertEqual(len(captured[0]),36)
        for name,param in fixture.named_parameters():self.assertTrue(torch.equal(param,state[name]))
        with self.assertRaises(ValueError):fixture.load_weights(list(state.items())+[next(iter(state.items()))])

    def test_spatial_capture_page_and_request_scope_guards(self):
        from . import spatial
        from .proposer import DFlareProposerRouter
        spatial.validate_query_batch(8,8,'PIECEWISE',dummy=False,keys_initialized=True)
        for args in ((7,8,'PIECEWISE',False,True),(8,16,'PIECEWISE',False,True),(8,8,'NONE',False,True),(8,8,'PIECEWISE',False,False)):
            with self.assertRaises(ValueError):spatial.validate_query_batch(*args[:3],dummy=args[3],keys_initialized=args[4])
        axes=torch.tensor([[2,3,4,5],[8,8,9,10],[12,13,14,15]])
        physical,query=spatial.context_query_axes(axes,torch.tensor([10]),torch.tensor([0,4]),torch.tensor([2]))
        self.assertTrue(torch.equal(physical,torch.arange(6,10)))
        self.assertTrue(torch.equal(query,axes[:,1:2]+torch.arange(1,9)))
        p=object.__new__(DFlareProposerRouter);p._dflare_active=True;p._dflare_backend_initialized=True;p.block_size=16
        p._dflare_in_propose=False;p._dflare_raw_context=None;p.num_speculative_tokens=7;p.parallel_drafting=True
        p.speculative_config=NS(num_speculative_tokens=7,parallel_drafting=True);p.vllm_config=NS(speculative_config=p.speculative_config)
        with mock.patch('dflare_8b_native_adapter.proposer.raw_parallel_propose',side_effect=RuntimeError('request failed')):
            with self.assertRaises(RuntimeError):p.propose()
        self.assertFalse(p._dflare_in_propose);self.assertIsNone(p._dflare_raw_context);self.assertFalse(spatial.DUMMY_SCOPE.get())

    def test_exact_target_head_embedding_alias_required_after_stock_loader(self):
        self.require('proposer.py')
        from .proposer import DFlareProposerRouter,OfficialDFlashProposer
        p=object.__new__(DFlareProposerRouter);p._dflare_active=True
        target=nn.Module();target.model=NS(embed_tokens=nn.Embedding(4,3));target.lm_head=nn.Linear(3,4,bias=False)
        p.model=NS(model=NS(embed_tokens=nn.Embedding(4,3)),lm_head=nn.Linear(3,4,bias=False),_dflare_loaded_tensor_names=tuple(range(42)))
        with mock.patch.object(OfficialDFlashProposer,'load_model',return_value=None):
            with self.assertRaises(ValueError):p.load_model(target)
            p.model.model.embed_tokens=target.model.embed_tokens;p.model.lm_head=target.lm_head;p.load_model(target)

if __name__=='__main__':unittest.main(verbosity=2)
