"""Owned ABI, official operator ownership and production closure checks on CPU."""
import copy
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import torch
from torch import nn
from .test_logic import HERE,cpu_vllm_config

class IntegrationTests(unittest.TestCase):
    def require(self,name):self.assertTrue((HERE/name).is_file(),'missing independent DFly module: '+name)

    def test_owned_compile_and_unchanged_official_query_decoder_head(self):
        self.require('backbone.py');self.require('model.py')
        from .backbone import DFlyOwnedBackbone
        from .model import DFlashDFlyNativeQwen3ForCausalLM
        from .spatial import validate_compile_ownership
        from vllm.model_executor.models import qwen3_dflash as official
        self.assertIs(DFlyOwnedBackbone.forward,official.DFlashQwen3Model.forward)
        self.assertIs(DFlashDFlyNativeQwen3ForCausalLM.forward,official.DFlashQwen3ForCausalLM.forward)
        self.assertIs(DFlashDFlyNativeQwen3ForCausalLM.compute_logits,official.DFlashQwen3ForCausalLM.compute_logits)
        self.assertFalse(validate_compile_ownership(DFlyOwnedBackbone)['shared_with_stock'])
        empty=object.__new__(DFlashDFlyNativeQwen3ForCausalLM)
        with self.assertRaisesRegex(ValueError,'raw context'):empty.combine_hidden_states(torch.empty(1,12288))

    def test_raw_propose_preserves_slots_and_target_bonus_anchor_before_sampling(self):
        self.require('proposal_logic.py')
        from .proposal_logic import raw_parallel_propose
        from vllm.config import CUDAGraphMode
        raw=torch.randn(5,12288);anchor=torch.tensor([2]);observed=[]
        class Model:
            def __call__(self,**kwargs):return torch.arange(8*4096).view(8,4096)
            def combine_hidden_states(self,*args):raise AssertionError('raw payload collapsed')
        p=NS(model=Model(),num_speculative_tokens=7,parallel_drafting=True,hidden_size=4096,vllm_config=None)
        def inputs(**kw):
            self.assertIs(kw['target_hidden_states'],raw);self.assertIs(kw['next_token_ids'],anchor)
            return 8,torch.arange(1,8),NS(slot_mapping=torch.arange(8))
        p.set_inputs_first_pass=inputs;p.build_per_group_and_layer_attn_metadata=lambda cad:([],{})
        p._determine_batch_execution_and_padding=lambda n:(CUDAGraphMode.PIECEWISE,8,None)
        p.build_model_inputs_first_pass=lambda *args:({},8);p._get_slot_mapping=lambda *args:{};p.model_returns_tuple=lambda:False
        p._sample_draft_tokens=lambda *args:(_ for _ in ()).throw(AssertionError('stock draft shortcut used'))
        def sample(model,hidden,previous):observed.append((hidden,previous));return torch.arange(7)[None]
        with mock.patch('dfly_native_adapter_v1.proposal_logic.set_forward_context',return_value=nullcontext()),mock.patch('dfly_native_adapter_v1.proposal_logic.sequential_greedy',side_effect=sample):
            result=raw_parallel_propose(p,torch.zeros(5),torch.zeros(3,5),raw,anchor,None,NS(batch_size=lambda:1),NS(all_greedy=True))
        self.assertEqual(result.tolist(),[list(range(7))]);self.assertTrue(torch.equal(observed[0][0],p.model()[1:8]));self.assertIs(observed[0][1],anchor)
        self.assertIsNone(p._last_draft_probs)
        with self.assertRaises(ValueError):raw_parallel_propose(p,torch.zeros(5),torch.zeros(3,5),raw,anchor,None,NS(batch_size=lambda:1),NS(all_greedy=False))

    @cpu_vllm_config
    def test_shared_qkv_context_precompute_norm_mrope_and_official_writer_order(self):
        self.require('spatial.py');self.require('context.py')
        from . import contracts as c
        from .context import build_layer_contexts,project_shared_context
        from .spatial import precompute
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
        torch.manual_seed(315)
        raw=torch.randn(6,12288,dtype=torch.bfloat16);fusion=nn.Parameter(torch.randn(3,3,dtype=torch.bfloat16))
        fc=nn.Linear(12288,4096,bias=False,dtype=torch.bfloat16);norm=RMSNorm(4096,eps=1e-6).bfloat16()
        rope=MRotaryEmbedding(128,128,256,5000000.,True,torch.bfloat16,mrope_section=[24,20,20],mrope_interleaved=True)
        layers=[];writes=[]
        for i in range(3):
            a=NS(qkv_proj=nn.Linear(4096,6144,bias=False,dtype=torch.bfloat16),q_size=4096,kv_size=1024,
                num_kv_heads=8,head_dim=128,k_norm=RMSNorm(128,eps=1e-6).bfloat16(),rotary_emb=rope)
            a.attn=NS(kv_cache=object());a.attn.impl=NS(do_kv_cache_update=lambda _,k,v,cache,slots,i=i:writes.append((i,k.clone(),v.clone(),slots)))
            layers.append(NS(self_attn=a))
        axes=torch.tensor([[3,3,4,5,6,7],[9,9,10,11,12,13],[15,16,16,17,18,19]]);slots=torch.tensor([0,1,2,3,4,-1])
        obj=NS(config={'architectures':[c.ARCH],'dflash_config':{c.MARKER:True}},fc=fc,hidden_norm=norm,
            layer_fusion_weights=fusion,layers=layers,_dfly_weights_loaded=True)
        contexts=build_layer_contexts(raw,fusion,fc,norm)
        precompute(obj,raw,axes);self.assertEqual(writes,[])
        precompute(obj,raw,axes,slots);self.assertEqual([w[0] for w in writes],[0,1,2])
        for i,k,v,mapping in writes:
            key,value=project_shared_context(contexts[i],layers[i].self_attn)
            key=layers[i].self_attn.k_norm(key).reshape(6,1024)
            expected,_=rope.forward_native(axes,key,torch.zeros(6,128,dtype=key.dtype))
            self.assertTrue(torch.equal(k,expected.view(6,8,128)));self.assertTrue(torch.equal(v,value));self.assertIs(mapping,slots)
        self.assertFalse(torch.cuda.is_initialized())

    @cpu_vllm_config
    def test_correction_loader_partition_preserves_five_original_parameter_names(self):
        self.require('backbone.py')
        from .backbone import DFlyOwnedBackbone,CORRECTION_KEYS
        from .correction import HiddenStatesCorrection
        from . import contracts as c
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
        fixture=object.__new__(DFlyOwnedBackbone);nn.Module.__init__(fixture)
        fixture.hidden_correction=HiddenStatesCorrection(4096,4096,dtype=torch.bfloat16)
        state={k:torch.empty(v['shape'],device='meta',dtype=torch.bfloat16) for k,v in c.SCHEMA.items()}
        for i,key in enumerate(sorted(CORRECTION_KEYS)):state[key]=torch.full(c.SCHEMA[key]['shape'],i+1.,dtype=torch.bfloat16)
        captured=[]
        def stock(obj,pairs):
            remaining=dict(pairs);self.assertFalse(set(remaining)&CORRECTION_KEYS);captured.append(remaining);return set(remaining)
        with mock.patch.object(DFlashQwen3Model,'load_weights',stock):fixture.load_weights(state.items())
        self.assertEqual(len(captured[0]),37);self.assertTrue(fixture._dfly_weights_loaded)
        for name,param in fixture.named_parameters():self.assertTrue(torch.equal(param,state[name]))
        with self.assertRaises(ValueError):fixture.load_weights(list(state.items())+[next(iter(state.items()))])

    def test_truthful_registry_actual_public_parser_and_launch_rejection_matrix(self):
        self.require('public_config.py')
        from . import contracts as c
        from .install import register_models,verify_official_interfaces
        from .public_config import materialize_public_config
        register_models();register_models();verify_official_interfaces()
        with mock.patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA forbidden')),mock.patch('torch.nn.Module.__init__',side_effect=AssertionError('serving construction forbidden')):
            proof,vc=materialize_public_config(c.EXPORT)
        self.assertEqual(proof['resolved_architectures'],[c.ARCH]);self.assertTrue(proof['draft_non_causal'])
        for group,field,value in [('model_config','model','/tmp/wrong-target'),('speculative_config','model','/tmp/wrong-export'),
            ('model_config','model_weights','/tmp/wrong-target-weights'),
            ('cache_config','block_size',32),('scheduler_config','max_num_seqs',2),('load_config','load_format','dummy'),
            ('load_config','load_format','safetensors'),('load_config','model_loader_extra_config',{'anything':1}),
            ('model_config','quantization','fp8')]:
            bad=copy.copy(vc);obj=copy.copy(getattr(vc,group));setattr(bad,group,obj);setattr(obj,field,value)
            with self.assertRaises(ValueError,msg=group+'.'+field):c.validate_launch_config(bad)
        bad=copy.copy(vc);bad.speculative_config=copy.copy(vc.speculative_config)
        bad.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
        bad.speculative_config.draft_model_config.model='/tmp/other-real-model'
        with self.assertRaises(ValueError):c.validate_launch_config(bad)
        bad=copy.copy(vc);bad.speculative_config=copy.copy(vc.speculative_config)
        bad.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
        bad.speculative_config.draft_model_config.model_weights='/tmp/other-real-weights'
        with self.assertRaises(ValueError):c.validate_launch_config(bad)
        bad.speculative_config.draft_model_config.model_weights=''
        bad.speculative_config.draft_model_config.hf_overrides=lambda cfg:cfg
        with self.assertRaises(ValueError):c.validate_launch_config(bad)
        from vllm.model_executor import model_loader
        with mock.patch.dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER,{'auto':object}):
            with self.assertRaisesRegex(ValueError,'custom loader'):c.validate_launch_config(vc)

    def test_exact_three_regular_export_files_and_receipt_artifact_path_closure(self):
        self.require('contracts.py')
        from . import contracts as c
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'export';path.mkdir()
            for name in c.EXPORT_FILES:(path/name).write_text('{}')
            hashes={str(path/name):c.sha(path/name) for name in c.EXPORT_FILES}
            c.validate_export_directory(path,expected_path=path,artifact_hashes=hashes,check_saved_receipt=False)
            for extra in ('model.safetensors.index.json','model-00001-of-00002.safetensors','nested'):
                (path/extra).write_text('bad')
                with self.assertRaises(ValueError):c.validate_export_directory(path,expected_path=path,check_saved_receipt=False)
                (path/extra).unlink()
            original=path/'model.safetensors';original.unlink();original.symlink_to(path/'config.json')
            with self.assertRaises(ValueError):c.validate_export_directory(path,expected_path=path,check_saved_receipt=False)
            original.unlink();original.write_text('{}')
            for badhash in ({},dict(hashes,**{str(path/'other'):'0'*64}),{k.replace('/export/','/other/'):v for k,v in hashes.items()}):
                with self.assertRaises(ValueError):c.validate_export_directory(path,expected_path=path,artifact_hashes=badhash,check_saved_receipt=False)
            with self.assertRaises(ValueError):c.validate_export_directory(path,expected_path=path.parent/'other',check_saved_receipt=False)

    def test_cpu_proof_cannot_bypass_explicit_receipt_sha_and_markers_fail_closed(self):
        self.require('contracts.py');self.require('proposer.py')
        from . import contracts as c
        from .proposer import DFlyProposerRouter,OfficialDFlashProposer
        cfg=json.loads((HERE/'CHECKPOINT_METADATA.json').read_text())['draft_config']
        c.validate_config(cfg,source=True);self.assertTrue(c.near_dfly(cfg));self.assertFalse(c.enabled(cfg))
        with self.assertRaises(ValueError):c.validate_config(cfg)
        with mock.patch.dict('os.environ',{'DFLY_NATIVE_CPU_PROOF':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'CPU_RECEIPT'):c.runtime_receipt()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'receipt.json';path.write_text('{}')
            with mock.patch.dict('os.environ',{'DFLY_NATIVE_CPU_RECEIPT':str(path),'DFLY_NATIVE_CPU_PROOF':'1'},clear=True):
                with self.assertRaisesRegex(ValueError,'SHA256'):c.runtime_receipt()
            with mock.patch.dict('os.environ',{'DFLY_NATIVE_CPU_RECEIPT':str(path),'DFLY_NATIVE_CPU_RECEIPT_SHA256':'0'*64},clear=True):
                with self.assertRaisesRegex(ValueError,'SHA256'):c.runtime_receipt()
        p=object.__new__(DFlyProposerRouter);p._dfly_active=False
        with mock.patch.object(OfficialDFlashProposer,'propose',return_value='stock'):self.assertEqual(p.propose(),'stock')

    def test_spatial_page_capture_and_exception_safe_request_dummy_scopes(self):
        self.require('spatial.py');self.require('proposer.py')
        from . import spatial
        from .proposer import DFlyProposerRouter
        spatial.validate_query_batch(8,8,'PIECEWISE',dummy=False,keys_initialized=True)
        for args in ((7,8,'PIECEWISE',False,True),(8,16,'PIECEWISE',False,True),(8,8,'NONE',False,True),(8,8,'PIECEWISE',False,False)):
            with self.assertRaises(ValueError):spatial.validate_query_batch(*args[:3],dummy=args[3],keys_initialized=args[4])
        axes=torch.tensor([[2,3,4,5],[8,8,9,10],[12,13,14,15]])
        physical,query=spatial.context_query_axes(axes,torch.tensor([10]),torch.tensor([0,4]),torch.tensor([2]))
        self.assertTrue(torch.equal(physical,torch.arange(6,10)));self.assertTrue(torch.equal(query,axes[:,1:2]+torch.arange(1,9)))
        p=object.__new__(DFlyProposerRouter);p._dfly_active=True;p._dfly_backend_initialized=True;p.block_size=16
        p._dfly_in_propose=False;p._dfly_raw_context=None;p.num_speculative_tokens=7;p.parallel_drafting=True
        p.speculative_config=NS(num_speculative_tokens=7,parallel_drafting=True);p.vllm_config=NS(speculative_config=p.speculative_config)
        with mock.patch('dfly_native_adapter_v1.proposer.raw_parallel_propose',side_effect=RuntimeError('request failed')):
            with self.assertRaises(RuntimeError):p.propose()
        self.assertFalse(p._dfly_in_propose);self.assertIsNone(p._dfly_raw_context);self.assertFalse(spatial.DUMMY_SCOPE.get())

    def test_exact_original_target_embedding_head_objects_required_after_stock_loading(self):
        self.require('proposer.py')
        from .proposer import DFlyProposerRouter,OfficialDFlashProposer
        p=object.__new__(DFlyProposerRouter);p._dfly_active=True
        target=nn.Module();target.model=NS(embed_tokens=nn.Embedding(4,3));target.lm_head=nn.Linear(3,4,bias=False)
        p.model=NS(model=NS(embed_tokens=nn.Embedding(4,3)),lm_head=nn.Linear(3,4,bias=False),_dfly_loaded_tensor_names=tuple(range(42)))
        with mock.patch.object(OfficialDFlashProposer,'load_model',return_value=None):
            with self.assertRaises(ValueError):p.load_model(target)
            p.model.model.embed_tokens=target.model.embed_tokens;p.model.lm_head=target.lm_head;p.load_model(target)

    def test_dummy_raw12288_buffer_all_query_sizes_and_exception_safe_no_write_scope(self):
        self.require('proposer.py')
        from .proposer import DFlyProposerRouter
        from . import spatial
        from vllm.config import CUDAGraphMode
        p=object.__new__(DFlyProposerRouter);p._dfly_active=True;p._dfly_backend_initialized=False;p.block_size=-1
        p.num_speculative_tokens=7;p.parallel_drafting=True;p.speculative_config=NS(num_speculative_tokens=7,parallel_drafting=True)
        p.vllm_config=NS(speculative_config=p.speculative_config);p.max_query_tokens=8;p._draft_attn_layer_names=[]
        p._raw_context_profile_buffer=torch.randn(13,12288);p._context_mrope_positions=torch.zeros(3,13,dtype=torch.long)
        p.input_ids=torch.zeros(8,dtype=torch.long);p.mrope_positions=torch.zeros(3,9,dtype=torch.long)
        p._get_positions=lambda n:p.mrope_positions[:,:n]
        p._determine_batch_execution_and_padding=lambda n,use_cudagraphs:(CUDAGraphMode.NONE,n,None)
        observed=[]
        class Model:
            def precompute_and_store_context_kv(self,raw,axes,slots=None):observed.append((raw.shape,axes.shape,slots,spatial.DUMMY_SCOPE.get()))
            def __call__(self,**kwargs):return torch.empty(kwargs['input_ids'].shape[0],4096)
        p.model=Model()
        with mock.patch('dfly_native_adapter_v1.proposer.set_forward_context',return_value=nullcontext()):
            for n in range(1,9):p.dummy_run(n,use_cudagraphs=False)
            with mock.patch.object(Model,'__call__',side_effect=RuntimeError('fixture failure')):
                with self.assertRaises(RuntimeError):p.dummy_run(13,use_cudagraphs=False)
        for n,entry in zip([*range(1,9),13],observed):
            self.assertEqual(entry,((n,12288),(3,n),None,True))
        self.assertFalse(spatial.DUMMY_SCOPE.get())

if __name__=='__main__':unittest.main(verbosity=2)
