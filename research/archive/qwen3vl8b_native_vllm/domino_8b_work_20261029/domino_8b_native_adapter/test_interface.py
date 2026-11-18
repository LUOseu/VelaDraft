from __future__ import annotations
import importlib.util,inspect,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_domino import fixture_config,component
TORCH=importlib.util.find_spec('torch') is not None

@unittest.skipUnless(TORCH,'Remote pinned vLLM/Torch CPU proof')
class InterfaceTests(unittest.TestCase):
    def test_real_EAGLE_wrapper_names_and_no_registry_overwrite(self):
        import torch
        from transformers import Qwen3Config
        from vllm.transformers_utils.configs.eagle import EAGLEConfig
        from vllm.model_executor.models import ModelRegistry
        original=ModelRegistry.models['DFlashDraftModel']
        i=component('install');c=component('contracts')
        with patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('No model construction')):
            i.register_models()
            raw=Qwen3Config.from_dict(fixture_config())
            wrapped=EAGLEConfig(raw,method='dflash',model_type='eagle')
            self.assertEqual(wrapped.architectures,[c.RESOLVED_ARCH]);c.validate_config(wrapped)
        self.assertIs(ModelRegistry.models['DFlashDraftModel'],original)
        self.assertIn(c.RAW_ARCH,ModelRegistry.get_supported_archs());self.assertIn(c.RESOLVED_ARCH,ModelRegistry.get_supported_archs())
        self.assertFalse(torch.cuda.is_initialized())
    def test_actual_public_SpeculativeConfig_resolves_S7_without_model(self):
        import torch
        from vllm.config import ModelConfig,ParallelConfig,SpeculativeConfig,CacheConfig,AttentionConfig
        from vllm.engine.arg_utils import EngineArgs
        i=component('install');c=component('contracts');i.register_models()
        with tempfile.TemporaryDirectory(prefix='domino_CPU_CONFIG_ONLY_') as td:
            Path(td,'config.json').write_text(json.dumps(fixture_config()))
            with patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('No model construction')),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No CUDA initialization')):
                target=ModelConfig(model='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target',dtype='bfloat16',max_model_len=8192)
                parallel=ParallelConfig(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
                spec=SpeculativeConfig(model=td,method='dflash',num_speculative_tokens=7,target_model_config=target,target_parallel_config=parallel,draft_tensor_parallel_size=1,attention_backend='FLASH_ATTN')
                c.validate_config(spec.draft_model_config.hf_config)
                component('domino_proposer').validate_target(target)
                self.assertEqual(spec.draft_model_config.hf_config.architectures,[c.RESOLVED_ARCH])
                self.assertEqual(spec.num_speculative_tokens,7);self.assertTrue(spec.parallel_drafting);self.assertTrue(spec.draft_model_config.uses_mrope)
                self.assertEqual(CacheConfig(block_size=16).block_size,16)
                attention=AttentionConfig(backend='FLASH_ATTN',flex_attn_block_m=16,flex_attn_block_n=16,flex_attn_q_block_size=16,flex_attn_kv_block_size=16)
                args=EngineArgs(model=target.model,dtype='bfloat16',max_num_seqs=1,block_size=16,attention_config=attention,speculative_config={'model':td,'method':'dflash','num_speculative_tokens':7,'attention_backend':'FLASH_ATTN'})
                self.assertEqual(args.speculative_config['num_speculative_tokens'],7)
                self.assertEqual(args.attention_config.backend.name,'FLASH_ATTN');self.assertEqual(args.speculative_config['attention_backend'],'FLASH_ATTN')
                self.assertEqual(spec.attention_backend.name,'FLASH_ATTN')
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_round_anchor_all8_indices_axes_and_exception_scopes(self):
        import torch
        from vllm.v1.spec_decode.dflash import DFlashProposer
        p=component('domino_proposer');x=object.__new__(p.DominoProposer)
        x._domino=True;x._domino_s=7;x.num_speculative_tokens=7;x.speculative_config=NS(num_speculative_tokens=7);x.vllm_config=NS(speculative_config=x.speculative_config)
        x.block_size=16;x._domino_active=False;x._domino_anchor=None;x.device=torch.device('cpu');x.mrope_positions=torch.zeros(3,9,dtype=torch.int64);x._context_mrope_positions=torch.zeros(3,16,dtype=torch.int64);x.input_ids=torch.tensor([9,9,9,9,9,9,9,9])
        cad=NS(batch_size=lambda:1,seq_lens=torch.tensor([8]),query_start_loc=torch.tensor([0,5]),causal=True)
        axes=torch.tensor([[10,11,12,13,14],[20,21,22,23,24],[30,31,32,33,34]])
        seen=[];query_cad=NS(causal=False)
        def official_set(self,ids,anchor,physical,*rest):
            self_test.assertTrue(torch.equal(physical,torch.tensor([3,4,5,6,7])))
            return 8,torch.arange(1,8,dtype=torch.int32),query_cad
        def walk(hidden,inputs):seen.append((tuple(hidden.shape),inputs.clone()));return torch.ones(1,7,dtype=torch.int64)
        self_test=self;x._walk=walk
        def official_propose(self):
            n,indices,new=self.set_inputs_first_pass(torch.arange(5),torch.tensor([2]),axes,torch.zeros(5,4096),None,cad,torch.tensor([2]))
            self_test.assertTrue(torch.equal(indices,torch.arange(8,dtype=torch.int32)));self_test.assertIs(new,query_cad)
            return self._sample_draft_tokens(torch.zeros(8,4),NS(all_greedy=True))
        with patch.object(DFlashProposer,'set_inputs_first_pass',official_set),patch.object(DFlashProposer,'propose',official_propose):result=x.propose()
        self.assertEqual(seen[0][0],(8,4));self.assertEqual(seen[0][1][0].item(),2);self.assertTrue(cad.causal)
        self.assertTrue(torch.equal(x.mrope_positions[:,:8],axes[:,2:3]+torch.arange(1,9)));self.assertIsNone(x._domino_anchor);self.assertFalse(x._domino_active)
        with patch.object(DFlashProposer,'propose',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):x.propose()
        self.assertIsNone(x._domino_anchor);self.assertFalse(x._domino_active);self.assertFalse(p._DUMMY.get())
        with self.assertRaises(ValueError):x._sample_draft_tokens(torch.zeros(8,4),NS(all_greedy=True))
        with patch.object(DFlashProposer,'dummy_run',side_effect=RuntimeError('dummy injected')):
            with self.assertRaises(RuntimeError):x.dummy_run(1)
        self.assertFalse(p._DUMMY.get())

    def test_foreign_wraps_dummy_rejected_before_registry_mutation(self):
        import functools
        from vllm.v1.spec_decode.dflash import DFlashProposer
        i=component('install');original=DFlashProposer.dummy_run
        @functools.wraps(original)
        def foreign(*args,**kwargs):return original(*args,**kwargs)
        with patch.object(DFlashProposer,'dummy_run',foreign),self.assertRaises(RuntimeError):i.check_runtime()

    def test_backbone_owns_shared_axis_metadata_stock_mapping_unchanged(self):
        import torch
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
        before=inspect.getclosurevars(DFlashQwen3Model.__init__).nonlocals['dynamic_arg_dims'].copy()
        with patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('No model construction')):m=component('domino_model')
        self.assertTrue(issubclass(m.DominoForCausalLM,DFlashQwen3ForCausalLM))
        mapping=inspect.getclosurevars(m.DominoBackbone.__init__).nonlocals['dynamic_arg_dims']
        marker=inspect.getclosurevars(m.DominoBackbone.__call__).nonlocals['_mark_dynamic_inputs']
        self.assertIs(mapping,inspect.getclosurevars(marker).nonlocals['dynamic_arg_dims'])
        self.assertEqual(mapping,{'input_ids':{0:'b'},'positions':{-1:'b'},'input_embeds':{0:'b'}})
        self.assertEqual(before,inspect.getclosurevars(DFlashQwen3Model.__init__).nonlocals['dynamic_arg_dims'])
        self.assertIs(m.DominoForCausalLM.forward,DFlashQwen3ForCausalLM.forward)
        self.assertIs(m.DominoForCausalLM.compute_logits,DFlashQwen3ForCausalLM.compute_logits)
    def test_export_is_large_IO_opt_in_and_logic_receipt_ineligible(self):
        e=component('export_domino')
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'unread585MB.pt'
            with self.assertRaises(RuntimeError):e.export(source,Path(td)/'out')
        c=component('contracts')
        with self.assertRaises(RuntimeError):c.validate_full_receipt({'schema':'domino-logic-receipt-v1','classification':'logic_only','gpu_eligible':False})
    def test_future_export_value_gate_detects_bit_shape_dtype_and_key_changes(self):
        import torch
        e=component('export_domino');self.assertTrue(callable(getattr(e,'verify_tensor_values',None)))
        original={'x':torch.tensor([1.,2.],dtype=torch.bfloat16)}
        result=e.verify_tensor_values(original,{'x':original['x'].clone()});self.assertIn('x',result)
        for restored in ({'x':original['x']+1},{'x':original['x'].float()},{'x':original['x'][:1]},{'y':original['x']}):
            with self.assertRaises(ValueError):e.verify_tensor_values(original,restored)
    def test_full_preflight_requires_explicit_large_IO_authorization(self):
        p=component('preflight')
        with tempfile.TemporaryDirectory() as td,self.assertRaises(RuntimeError):p.run(Path(td)/'uncreated.json',full=True,allow_large_io=False)

    def test_router_is_class_and_ordinary_proposer_delegates(self):
        import torch
        from vllm.v1.spec_decode.dflash import DFlashProposer
        p=component('domino_proposer')
        self.assertTrue(isinstance(p.DominoProposer,type));self.assertTrue(issubclass(p.DominoProposer,DFlashProposer))
        sentinel=object();cfg=NS(speculative_config=NS(draft_model_config=NS(hf_config={'architectures':['DFlashDraftModel'],'dflash_config':{}})))
        with patch.object(DFlashProposer,'__init__',return_value=None) as init:
            proposer=p.DominoProposer(cfg,torch.device('cpu'))
            self.assertFalse(proposer._domino);init.assert_called_once_with(cfg,torch.device('cpu'),runner=None)
        with patch.object(DFlashProposer,'_sample_draft_tokens',return_value=sentinel) as original:
            self.assertIs(proposer._sample_draft_tokens('h','s'),sentinel);original.assert_called_once_with('h','s')
    def test_bootstrap_absent_inert_and_gpu_ineligible_before_torch(self):
        import subprocess
        b=Path(__file__).parent/'bootstrap'
        for selected,visible in [(False,'1'),(True,'1')]:
            env={**os.environ,'PYTHONPATH':str(b),'CUDA_VISIBLE_DEVICES':visible,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            for k in list(env):
                if k.startswith('FAIR_DOMINO') or k.startswith('DOMINO_APPROVED'):env.pop(k)
            if selected:env['FAIR_DOMINO_ADAPTER']='1';env['PYTORCH_NVML_BASED_CUDA_CHECK']='1';env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            proc=subprocess.run([sys.executable,'-c','import sys;print("TORCH_IMPORTED", "torch" in sys.modules)'],env=env,text=True,capture_output=True)
            self.assertEqual(proc.returncode,86 if selected else 0,proc.stdout+proc.stderr)
            if not selected:self.assertIn('TORCH_IMPORTED False',proc.stdout)
    def test_production_bootstrap_rejects_all_CPU_proof_flags_and_stays_inert_without_activation(self):
        import subprocess
        bootstrap=Path(__file__).parent/'bootstrap'
        for active in (False,True):
            for cpu_flag in ('0','1'):
                for visible in ('','1'):
                    env={**os.environ,'PYTHONPATH':str(bootstrap),'CUDA_VISIBLE_DEVICES':visible,'FAIR_DOMINO_CPU_PROOF':cpu_flag,'PYTORCH_NVML_BASED_CUDA_CHECK':'1','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
                    env.pop('FAIR_DOMINO_ADAPTER',None)
                    if active:env['FAIR_DOMINO_ADAPTER']='1'
                    proc=subprocess.run([sys.executable,'-c','import sys;print("TORCH", "torch" in sys.modules)'],env=env,text=True,capture_output=True)
                    with self.subTest(active=active,cpu_flag=cpu_flag,visible=visible):
                        self.assertEqual(proc.returncode,86 if active else 0,proc.stdout+proc.stderr)
                        if not active:self.assertIn('TORCH False',proc.stdout)
        i=component('install');self.assertNotIn('cpu_proof',inspect.signature(i.install).parameters)

    def test_logic_registration_never_authorizes_model_construction(self):
        import torch
        c=component('contracts');m=component('domino_model')
        self.assertIsNone(c._VERIFIED_APPROVAL)
        vc=self.metadata_vc()
        with patch.object(torch.nn.Module,'__init__',side_effect=AssertionError('Model construction forbidden')) as initialize:
            for cls in (m.DominoForCausalLM,m.DominoBackbone):
                with self.subTest(cls=cls.__name__),self.assertRaisesRegex(RuntimeError,'No verified full-byte'):cls(vllm_config=vc)
            initialize.assert_not_called()

    def metadata_vc(self):
        import torch
        from vllm.config import CUDAGraphMode,CompilationMode
        text=NS(hidden_size=4096,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,intermediate_size=12288,vocab_size=151936,rms_norm_eps=1e-6,rope_parameters={'rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True})
        attention=NS(backend=NS(name='FLASH_ATTN'),flex_attn_block_m=16,flex_attn_block_n=16,flex_attn_q_block_size=16,flex_attn_kv_block_size=16)
        return NS(speculative_config=NS(model='/cpu_METADATA_ONLY/domino_export',num_speculative_tokens=7,method='dflash',draft_model_config=NS(model='/cpu_METADATA_ONLY/domino_export',hf_config=fixture_config(),uses_mrope=True),enforce_eager=False,use_local_argmax_reduction=False,attention_backend=NS(name='FLASH_ATTN')),parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),scheduler_config=NS(max_num_seqs=1,max_num_batched_tokens=16),model_config=NS(model='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target',hf_text_config=text,hf_config=NS(model_type='qwen3_vl',tie_word_embeddings=False),dtype=torch.bfloat16,enforce_eager=False),load_config=NS(load_format='auto',model_loader_extra_config={}),cache_config=NS(block_size=16),attention_config=attention,compilation_config=NS(mode=CompilationMode.VLLM_COMPILE,cudagraph_mode=CUDAGraphMode.PIECEWISE,cudagraph_capture_sizes=[8],max_cudagraph_capture_size=8))

    def test_actual_target_embedding_and_lm_head_aliases_required(self):
        from vllm.v1.spec_decode.dflash import DFlashProposer
        p=component('domino_proposer'); obj=object.__new__(p.DominoProposer);obj._domino=True
        target=NS(lm_head=object(),model=NS(embed_tokens=object()))
        obj.model=NS(lm_head=object(),model=NS(embed_tokens=object()))
        with patch.object(DFlashProposer,'load_model',return_value=None), patch('vllm.model_executor.models.interfaces.supports_multimodal',return_value=False):
            with self.assertRaises(ValueError):obj.load_model(target)
            obj.model.lm_head=target.lm_head
            with self.assertRaises(ValueError):obj.load_model(target)
            obj.model.model.embed_tokens=target.model.embed_tokens
            obj.load_model(target)

    def test_real_flash_attention_config_with_default_unused_flex_fields(self):
        from vllm.config import AttentionConfig
        p=component('domino_proposer');vc=self.metadata_vc()
        vc.attention_config=AttentionConfig(backend='FLASH_ATTN')
        for key in ('flex_attn_block_m','flex_attn_block_n','flex_attn_q_block_size','flex_attn_kv_block_size'):
            self.assertIsNone(getattr(vc.attention_config,key))
        p.validate_launch(vc)

    def test_target_page_and_Flash_metadata_reject_changed_values(self):
        import copy
        p=component('domino_proposer');vc=self.metadata_vc();p.validate_launch(vc)
        mutations=[('model_config','model','/different_target'),('model_config.hf_text_config','hidden_size',1024),('model_config.hf_text_config','num_hidden_layers',32),('model_config.hf_text_config','vocab_size',11),('cache_config','block_size',2),('cache_config','block_size',8),('cache_config','block_size',32),('speculative_config','attention_backend',NS(name='FLEX_ATTENTION')),('attention_config','backend',NS(name='FLEX_ATTENTION')),('speculative_config','method','eagle')]
        for owner,key,value in mutations:
            bad=copy.deepcopy(vc);node=bad
            for field in owner.split('.'):node=getattr(node,field)
            setattr(node,key,value)
            with self.subTest(owner=owner,key=key,value=value),self.assertRaises(ValueError):p.validate_launch(bad)

    def test_dummy_custom_or_quantized_weight_loading_is_rejected(self):
        import copy
        p=component('domino_proposer');vc=self.metadata_vc()
        vc.load_config=NS(load_format='auto',model_loader_extra_config={});vc.speculative_config.draft_load_config=None
        p.validate_launch(vc)
        for mode in ('dummy','tensorizer'):
            bad=copy.deepcopy(vc);bad.speculative_config.draft_load_config=NS(load_format=mode,model_loader_extra_config={})
            with self.subTest(mode=mode),self.assertRaises(ValueError):p.validate_launch(bad)
        for owner,key,value in [('load_config','model_loader_extra_config',{'weights':'other'}),('model_config','quantization','fp8'),('speculative_config.draft_model_config','model_weights','/other_weights')]:
            bad=copy.deepcopy(vc);node=bad
            for field in owner.split('.'):node=getattr(node,field)
            setattr(node,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):p.validate_launch(bad)

    def test_actual_DefaultModelLoader_auto_consolidated_redirect_and_target_guard(self):
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        c=component('contracts')
        with tempfile.TemporaryDirectory(prefix='CPU_TARGET_LOADER_SELECTION_FIXTURE_') as td:
            target=Path(td).resolve();shards=('model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors')
            for name in shards:(target/name).touch()
            (target/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':0},'weight_map':{f'fixture{i}':name for i,name in enumerate(shards)}}))
            pins={str(target/name):'a'*64 for name in (*shards,'model.safetensors.index.json')}
            expected=sorted(str(target/name) for name in shards)
            for mode in ('auto','safetensors'):
                loader=DefaultModelLoader(LoadConfig(load_format=mode))
                folder,files,safe=loader._prepare_weights(str(target),None,None,True,None)
                self.assertEqual(str(Path(folder).resolve()),str(target));self.assertEqual(sorted(files),expected);self.assertIs(safe,True)
                c.validate_target_layout(target,pins)
        # Stock list_repo_files caches local directory inventories. A fresh root
        # represents process startup with consolidated weights already present.
        # Only filename/index discovery runs; no synthetic weight body is loaded.
        with tempfile.TemporaryDirectory(prefix='CPU_TARGET_LOADER_REDIRECT_FIXTURE_') as td:
            target=Path(td).resolve()
            for name in shards:(target/name).touch()
            (target/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{f'fixture{i}':name for i,name in enumerate(shards)}}))
            pins={str(target/name):'a'*64 for name in (*shards,'model.safetensors.index.json')}
            extra=target/'consolidated.00.safetensors';extra.touch()
            loader=DefaultModelLoader(LoadConfig(load_format='auto'))
            _,files,safe=loader._prepare_weights(str(target),None,None,True,None)
            self.assertEqual(files,[str(extra)]);self.assertIs(safe,True)
            with self.assertRaises(ValueError):c.validate_target_layout(target,pins)

    def test_actual_backend_page_and_derived_noncausal_Flash_guards(self):
        from vllm.v1.spec_decode.dflash import DFlashProposer
        p=component('domino_proposer');x=object.__new__(p.DominoProposer);x._domino=True
        for page in (2,8,32):
            def initialize(self,*args,**kwargs):self.block_size=page
            with patch.object(DFlashProposer,'initialize_attn_backend',initialize),self.assertRaises(ValueError):x.initialize_attn_backend('metadata_only')
        for backend,noncausal in [('FLEX_ATTENTION',True),('FLASH_ATTN',False)]:
            cfg=NS(attention_config=NS(backend=NS(name=backend),use_non_causal=noncausal))
            with patch.object(DFlashProposer,'_create_draft_vllm_config',return_value=cfg),self.assertRaises(ValueError):x._create_draft_vllm_config()

    def test_real_constructor_S7_reaches_stock_unchanged_and_physicalpage16(self):
        # Only official proposer constructor is shimmed; no model/engine is constructed.
        import torch
        from vllm.config import CUDAGraphMode,CompilationMode
        from vllm.v1.spec_decode.dflash import DFlashProposer
        p=component('domino_proposer');cfg=fixture_config()
        vc=self.metadata_vc()
        seen=[]
        def init(self,vllm_config,device,runner=None):
            seen.append(vllm_config.speculative_config.num_speculative_tokens)
            self.vllm_config=vllm_config;self.speculative_config=vc.speculative_config;self.draft_model_config=vc.speculative_config.draft_model_config;self.num_speculative_tokens=7;self.max_query_tokens=8;self.max_num_tokens=16;self.block_size=16;self.device=device;self.dtype=torch.bfloat16;self.hidden_size=4096
        with patch.object(DFlashProposer,'__init__',init):proposer=p.DominoProposer(vc,torch.device('cpu'),_logic_interface=True)
        self.assertEqual(seen,[7]);self.assertEqual(proposer.block_size,16);self.assertEqual(tuple(proposer.mrope_positions.shape),(3,9));self.assertEqual(proposer.mrope_positions[:,:8].stride(),(9,1))
        with self.assertRaisesRegex(RuntimeError,'Logic interface'):proposer.propose()
        with self.assertRaisesRegex(RuntimeError,'Logic interface'):proposer.dummy_run(1)
        for value in (3,7.0):
            vc.speculative_config.num_speculative_tokens=value
            with self.subTest(value=value),self.assertRaises(ValueError):proposer._runtime_block()

if __name__=='__main__':unittest.main(verbosity=2)
