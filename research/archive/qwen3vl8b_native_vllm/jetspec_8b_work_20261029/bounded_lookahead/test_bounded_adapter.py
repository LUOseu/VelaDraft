"""CPU-only bounded public lookahead contracts; never construct a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import ast
import copy
import hashlib
import inspect
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "spatial_adapter"
sys.path.insert(0, str(FROZEN))


class FrozenRegression(unittest.TestCase):
    def test_frozen_constructor_rejects_public_spec1_and_spec3(self):
        import torch
        import official_spatial_graph_adapter as graph
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.v1.spec_decode.dflash import DFlashProposer
        graph.install()
        cfg_data = json.loads((FROZEN / "exported_config_v2.json").read_text())
        hf = NS(**cfg_data)
        hf.to_dict = lambda: cfg_data
        cfg = NS(model_config=NS(enforce_eager=False, hf_config=NS(model_type="qwen2_5_vl")),
            parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
            scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=128),
            compilation_config=NS(mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE, cudagraph_capture_sizes=[8,16]),
            speculative_config=NS(enforce_eager=False, draft_model_config=NS(hf_config=hf, uses_mrope=True)))
        for s in (1,3):
            cfg.speculative_config.num_speculative_tokens = s
            with self.subTest(s=s), self.assertRaisesRegex(ValueError, "num_speculative_tokens=7"):
                DFlashProposer.__init__(object.__new__(DFlashProposer), cfg, torch.device("cpu"))
        self.assertFalse(torch.cuda.is_initialized())


class BoundedPolicy(unittest.TestCase):
    def test_supported_public_policy(self):
        import official_spatial_bounded_adapter as bounded
        for s, b in ((1,2),(3,4),(7,8)):
            self.assertEqual(bounded.query_block(s), b)
        for s in (0,2,4,8,-1,True,1.0,"1",None):
            with self.subTest(s=s), self.assertRaises(ValueError):
                bounded.query_block(s)

    def test_real_and_dummy_boundaries(self):
        import official_spatial_bounded_adapter as bounded
        for s in (1,3,7):
            b = bounded.query_block(s)
            bounded.validate_query_batch(s,b,b,"PIECEWISE",dummy=False,keys_initialized=True)
            for n in range(1,b+1):
                for padded in range(n,b+1):
                    for mode in ("NONE","PIECEWISE"):
                        bounded.validate_query_batch(s,n,padded,mode,dummy=True,keys_initialized=False)
            for n,padded,mode,initialized in ((b,b,"NONE",True),(b,b,"FULL",True),
                    (b,b,"PIECEWISE",False),(1,b,"PIECEWISE",True),(b,b*2,"PIECEWISE",True)):
                with self.subTest(s=s,n=n,padded=padded,mode=mode), self.assertRaises(ValueError):
                    bounded.validate_query_batch(s,n,padded,mode,dummy=False,keys_initialized=initialized)
            for n,padded,mode in ((0,0,"NONE"),(b+1,b+1,"NONE"),(2,1,"NONE"),(1,b+1,"NONE"),(1,1,"FULL")):
                with self.assertRaises(ValueError):
                    bounded.validate_query_batch(s,n,padded,mode,dummy=True,keys_initialized=True)

    def test_config_semantics_are_frozen_and_launch_checks_are_bounded(self):
        import official_spatial_bounded_adapter as bounded
        for s in (1,3,7):
            cfg = launch_config(s)
            self.assertEqual(bounded.validate_launch_config(cfg), 1+s)
            self.assertEqual(cfg.speculative_config.draft_model_config.hf_config.block_size,8)
            mutations = (
                (cfg.parallel_config,"tensor_parallel_size",2),
                (cfg.parallel_config,"pipeline_parallel_size",2),
                (cfg.parallel_config,"data_parallel_size",2),
                (cfg.scheduler_config,"max_num_seqs",2),
                (cfg.scheduler_config,"max_num_batched_tokens",s),
                (cfg.model_config.hf_config,"model_type","qwen3"),
                (cfg.speculative_config.draft_model_config,"uses_mrope",False),
                (cfg.model_config,"enforce_eager",True),
                (cfg.speculative_config,"enforce_eager",True),
                (cfg.compilation_config,"cudagraph_capture_sizes",[2*(1+s)]),
            )
            for obj,name,value in mutations:
                before = getattr(obj,name)
                setattr(obj,name,value)
                with self.subTest(s=s,field=name), self.assertRaises(ValueError):
                    bounded.validate_launch_config(cfg)
                setattr(obj,name,before)
            cfg.speculative_config.draft_model_config.hf_config.to_dict = lambda: dict(config_data(),block_size=4)
            with self.assertRaisesRegex(ValueError,"block_size=8"):
                bounded.validate_launch_config(cfg)
        for path,value in (("spatial_rope_mode","gated"),("target_layer_ids",[0,16,32]),
                ("spatial_rope_sections",[24,24,16]),("spatial_rope_interleaved",True)):
            data = config_data()
            data["dflash_config"][path] = value
            with self.assertRaises(ValueError):
                bounded.validate_draft_config(data)


def config_data():
    return json.loads((FROZEN / "exported_config_v2.json").read_text())


def launch_config(s):
    from vllm.config import CUDAGraphMode, CompilationMode
    data = config_data()
    hf = NS(**data)
    hf.to_dict = lambda: data
    return NS(model_config=NS(enforce_eager=False,hf_config=NS(model_type="qwen2_5_vl")),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),
        scheduler_config=NS(max_num_seqs=1,max_num_batched_tokens=128),lora_config=None,
        speculative_config=NS(num_speculative_tokens=s,method="dflash",enforce_eager=False,
            draft_model_config=NS(hf_config=hf,uses_mrope=True)),
        compilation_config=NS(mode=CompilationMode.VLLM_COMPILE,cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1+s,2*(1+s)],max_cudagraph_capture_size=2*(1+s),
            compile_sizes=None,is_attention_compiled_piecewise=lambda: True))


class OfficialDispatch(unittest.TestCase):
    def make_proposer(self,s):
        import official_spatial_bounded_adapter as bounded
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.config import CUDAGraphMode
        cfg = launch_config(s)
        dispatcher = CudagraphDispatcher(cfg)
        dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE)
        class Proposer:
            def dummy_run(self,n,fail=False,nested_real=False):
                result = self._determine_batch_execution_and_padding(n,use_cudagraphs=False)
                if fail: raise RuntimeError("dummy failed")
                if nested_real: return self.propose(n)
                return result
            def propose(self,n):
                return self._determine_batch_execution_and_padding(n)
            _determine_batch_execution_and_padding = SpecDecodeBaseProposer._determine_batch_execution_and_padding
        bounded.install_scoped_query_guards(Proposer,SpecDecodeBaseProposer._determine_batch_execution_and_padding)
        proposer = Proposer()
        proposer.vllm_config = cfg
        proposer.speculative_config = cfg.speculative_config
        proposer.draft_model_config = cfg.speculative_config.draft_model_config
        proposer.num_speculative_tokens = s
        proposer._bounded_num_speculative_tokens = s
        proposer.cudagraph_dispatcher = dispatcher
        return proposer

    def test_actual_dispatcher_b2_b4_b8_and_dummy1(self):
        import official_spatial_bounded_adapter as bounded
        for s in (1,3,7):
            proposer = self.make_proposer(s)
            for n in range(1,s+2):
                proposer.dummy_run(n)
            mode,padded,_ = proposer.propose(s+1)
            self.assertEqual((mode.name,padded),("PIECEWISE",s+1))
            for kwargs in ({"fail":True},{"nested_real":True}):
                with self.assertRaises((ValueError,RuntimeError)):
                    proposer.dummy_run(1,**kwargs)
                self.assertFalse(bounded.in_dummy_scope())
            with self.assertRaises(ValueError): proposer.propose(1)
            self.assertFalse(bounded.in_dummy_scope())
            proposer.num_speculative_tokens = 7 if s!=7 else 1
            with self.assertRaisesRegex(ValueError,"changed"):
                proposer.propose(s+1)

    def test_actual_dispatcher_padding_is_rejected(self):
        import official_spatial_bounded_adapter as bounded
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.config import CUDAGraphMode
        for s in (1,3,7):
            cfg = launch_config(s)
            cfg.compilation_config.cudagraph_capture_sizes = [2*(1+s)]
            dispatcher = CudagraphDispatcher(cfg)
            dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE)
            mode,desc = dispatcher.dispatch(s+1)
            self.assertEqual(desc.num_tokens,2*(s+1))
            with self.assertRaisesRegex(ValueError,"padding"):
                bounded.validate_query_batch(s,s+1,desc.num_tokens,mode.name,dummy=False,keys_initialized=True)


class PublicConfig(unittest.TestCase):
    def test_actual_public_config_accepts_s_without_rewriting_trained_b8(self):
        import torch
        from vllm.config import ModelConfig,SpeculativeConfig,ParallelConfig
        from vllm.model_executor.models import ModelRegistry
        target=ModelConfig(model="/home/lyh/models/Qwen2.5-VL-3B-Instruct",dtype="bfloat16",max_model_len=4096)
        pc=ParallelConfig(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
        for s in (1,3,7):
            spec=SpeculativeConfig(model=str(FROZEN/"frozen_parent37500_hf_v2"),method="dflash",
                num_speculative_tokens=s,target_model_config=target,target_parallel_config=pc,
                draft_tensor_parallel_size=1)
            cfg=spec.draft_model_config.hf_config
            self.assertEqual(spec.num_speculative_tokens,s)
            self.assertTrue(spec.parallel_drafting)
            self.assertEqual(cfg.block_size,8)
            self.assertTrue(spec.draft_model_config.uses_mrope)
            self.assertEqual(cfg.architectures,["DFlashDraftModel"])
            self.assertFalse(hasattr(cfg,"n_predict"))
            self.assertFalse(hasattr(cfg,"num_lookahead_tokens"))
            self.assertIn(cfg.architectures[0],ModelRegistry.get_supported_archs())
        self.assertFalse(torch.cuda.is_initialized())


class ProvenanceContracts(unittest.TestCase):
    def test_added_foreign_dummy_wrapper_is_rejected_before_class_mutation(self):
        import subprocess
        code = """
import functools
import torch
import official_spatial_bounded_adapter as bounded
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
stock = DFlashProposer.dummy_run
original_init = DFlashProposer.__init__
original_context = DFlashQwen3Model.precompute_and_store_context_kv
@functools.wraps(stock)
def foreign_dummy(self, *args, **kwargs):
    return stock(self, *args, **kwargs)
DFlashProposer.dummy_run = foreign_dummy
try:
    bounded.install()
except RuntimeError as error:
    assert 'dummy_run' in str(error), str(error)
else:
    raise AssertionError('Additional foreign dummy_run wrapper was accepted')
assert not bounded._INSTALLED
assert DFlashProposer.__init__ is original_init
assert DFlashProposer.dummy_run is foreign_dummy
assert DFlashQwen3Model.precompute_and_store_context_kv is original_context
DFlashProposer.dummy_run = stock
bounded.install()
assert bounded._INSTALLED
assert not torch.cuda.is_initialized()
"""
        env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE="1")
        env.pop("VLLM_OFFICIAL_SPATIAL_BOUNDED_ADAPTER",None)
        subprocess.run([sys.executable,"-c",code],env=env,check=True)

    def test_stock_torch_dummy_wrapper_has_exactly_one_official_leaf(self):
        import torch
        import official_spatial_bounded_adapter as bounded
        from vllm.v1.spec_decode import dflash
        stock=dflash.DFlashProposer.dummy_run
        bounded._require_official_dummy_run(stock,dflash)
        self.assertIs(inspect.getclosurevars(stock).nonlocals["func"],stock.__wrapped__)
        self.assertFalse(hasattr(stock.__wrapped__,"__wrapped__"))
        self.assertFalse(torch.cuda.is_initialized())


class StaticContracts(unittest.TestCase):
    def test_frozen_context_body_is_byte_and_ast_identical(self):
        def nested(path,name):
            source = path.read_text()
            tree = ast.parse(source)
            install = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="install")
            node = next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name==name)
            return ast.get_source_segment(source,node),ast.dump(node)
        for name in ("precompute","build_inputs","guarded_raise"):
            self.assertEqual(nested(FROZEN/"official_spatial_adapter.py",name),
                nested(ROOT/"official_spatial_bounded_adapter.py",name))

    def test_no_page_size_assignment_or_kernel_or_sampler_replacement(self):
        tree = ast.parse((ROOT/"official_spatial_bounded_adapter.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target,ast.Attribute):
                        self.assertNotIn(target.attr,("block_size","forward","__call__","_sample_draft_tokens","_greedy_sample"))
            if isinstance(node,ast.Call):
                func = ast.unparse(node.func)
                self.assertNotIn(func,("triton.jit","torch.library.custom_op","torch.library.triton_op"))

    def test_frozen_files_remain_unchanged(self):
        manifest = json.loads((ROOT/"FROZEN_FILES_BEFORE.json").read_text())
        for name,digest in manifest.items():
            path = ROOT.parent/name
            # Local manifest can include local-only receipts; remote has its own full manifest.
            if path.exists():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,name)

    def test_bootstrap_without_flag_has_no_effect(self):
        import subprocess
        code = "import sys; before=list(sys.path); import sitecustomize; assert sys.path==before; assert 'official_spatial_bounded_adapter' not in sys.modules; assert 'torch' not in sys.modules"
        env = dict(os.environ,PYTHONPATH=str(ROOT/"bootstrap"),PYTHONDONTWRITEBYTECODE="1")
        env.pop("VLLM_OFFICIAL_SPATIAL_BOUNDED_ADAPTER",None)
        subprocess.run([sys.executable,"-c",code],env=env,check=True)


class InstalledContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import official_spatial_bounded_adapter as bounded
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
        from vllm.v1.spec_decode.dflash import DFlashProposer
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        cls.bounded = bounded
        cls.Proposer,cls.Model = DFlashProposer,DFlashQwen3Model
        cls.original_init = DFlashProposer.__init__
        cls.original_forward = DFlashQwen3Model.forward
        cls.original_call = DFlashQwen3Model.__call__
        cls.original_sample = SpecDecodeBaseProposer._sample_draft_tokens
        cls.original_context_combine = DFlashQwen3ForCausalLM.combine_hidden_states
        cls.bindings = bounded.install()

    def test_installer_pins_official_sources_preserves_math_and_is_idempotent(self):
        import official_spatial_adapter as frozen
        import official_spatial_graph_adapter as graph1
        import official_spatial_graph_adapter_v2 as graph2
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
        self.assertEqual(self.bindings,self.bounded.install())
        self.assertIn("cudagraph_dispatcher",self.bindings)
        self.assertFalse(any(m._INSTALLED for m in (frozen,graph1,graph2)))
        self.assertIs(self.Model.forward,type(self).original_forward)
        self.assertIs(self.Model.__call__,type(self).original_call)
        self.assertIs(SpecDecodeBaseProposer._sample_draft_tokens,type(self).original_sample)
        self.assertIs(DFlashQwen3ForCausalLM.combine_hidden_states,type(self).original_context_combine)
        for name in ("absolute_context_positions","spatial_query_positions","rotate_context_keys","validate_draft_config"):
            self.assertIs(getattr(self.bounded,name),getattr(frozen,name))

    def test_real_s_reaches_official_constructor_and_buffers_have_bounded_stride(self):
        import torch
        # Replace only the original constructor's globals with a CPU base shim;
        # the installed hook still invokes the exact original DFlash constructor.
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        seen=[]
        def cpu_base(self,vllm_config,device,pass_hidden_states_to_model,runner=None):
            seen.append((vllm_config.speculative_config.num_speculative_tokens,
                vllm_config.speculative_config.draft_model_config.hf_config.block_size))
            self.vllm_config=vllm_config
            self.speculative_config=vllm_config.speculative_config
            self.draft_model_config=self.speculative_config.draft_model_config
            self.num_speculative_tokens=self.speculative_config.num_speculative_tokens
            self.max_batch_size=1
            self.max_num_tokens=128
            self.block_size=32
        with patch.object(SpecDecodeBaseProposer,"__init__",cpu_base):
            for s in (1,3,7):
                proposer = self.Proposer(launch_config(s),torch.device("cpu"))
                self.assertEqual(proposer.mrope_positions.shape,(3,s+2))
                self.assertEqual(proposer.mrope_positions[:,:s+1].stride(),(s+2,1))
                self.assertEqual(proposer._context_mrope_positions.shape,(3,128))
                self.assertEqual(proposer.positions.shape,(s+1,))
                self.assertEqual(proposer._slot_mapping_buffer.shape,(s+1,))
                self.assertEqual(proposer.block_size,32)
        self.assertEqual(seen,[(1,8),(3,8),(7,8)])

    def test_actual_initialize_keys_uses_selected_b_and_rejects_missing_capture(self):
        from vllm.config import CUDAGraphMode
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        for s in (1,3,7):
            for missing in (False,True):
                cfg=launch_config(s)
                if missing:
                    cfg.compilation_config.cudagraph_capture_sizes=[2*(s+1)]
                proposer=object.__new__(self.Proposer)
                proposer.vllm_config=cfg
                proposer.speculative_config=cfg.speculative_config
                proposer.draft_model_config=cfg.speculative_config.draft_model_config
                proposer.num_speculative_tokens=s
                proposer._bounded_num_speculative_tokens=s
                proposer.cudagraph_dispatcher=CudagraphDispatcher(cfg)
                if missing:
                    with self.assertRaisesRegex(ValueError,"padding"):
                        proposer.initialize_cudagraph_keys(CUDAGraphMode.FULL_AND_PIECEWISE)
                else:
                    proposer.initialize_cudagraph_keys(CUDAGraphMode.FULL_AND_PIECEWISE)
                    mode,desc=proposer.cudagraph_dispatcher.dispatch(s+1)
                    self.assertEqual((mode.name,desc.num_tokens),("PIECEWISE",s+1))

    def test_actual_official_expansion_shapes_spatial_rejections_and_cache_page(self):
        import numpy as np
        import torch
        from vllm.v1.spec_decode import dflash
        from vllm.v1.attention.backend import CommonAttentionMetadata
        # No kernel is executed. The CPU stand-in records arguments at the
        # unchanged official kernel boundary and fills an independent oracle.
        records=[]
        class CPUExpansion:
            def __getitem__(self,grid):
                def boundary(**kw):
                    records.append((grid,kw["num_speculative_tokens"],kw["block_size"]))
                    n=kw["total_input_tokens"]
                    b=kw["num_query_per_req"]
                    r=int(kw["num_rejected_tokens_ptr"][0]) if kw["HAS_NUM_REJECTED"] else 0
                    physical=kw["target_positions_ptr"]
                    kw["out_context_positions_ptr"][:n].copy_(physical)
                    q=torch.arange(int(physical[-1])+1-r,int(physical[-1])+1-r+b)
                    kw["out_query_positions_ptr"][:b].copy_(q)
                    kw["out_input_ids_ptr"][:b].fill_(kw["parallel_drafting_token_id"])
                    kw["out_input_ids_ptr"][0]=kw["next_token_ids_ptr"][0]
                    for positions,out in ((physical,kw["out_context_slot_mapping_ptr"]),(q,kw["out_query_slot_mapping_ptr"])):
                        slots=kw["block_table_ptr"][0,positions//kw["block_size"]]*kw["block_size"]+positions%kw["block_size"]
                        out[:len(positions)].copy_(slots)
                    kw["out_context_slot_mapping_ptr"][n-r:n].fill_(-1)
                    kw["out_token_indices_ptr"][:b-1].copy_(torch.arange(b-1))
                return boundary
        axes=torch.tensor([[145,146,147,148,149],[143,144,145,146,147],[144,145,146,147,148]])
        for s in (1,3,7):
            for r in (0,1,2,4):
                b=s+1
                proposer=object.__new__(self.Proposer)
                proposer.vllm_config=launch_config(s)
                proposer.speculative_config=proposer.vllm_config.speculative_config
                proposer.draft_model_config=proposer.speculative_config.draft_model_config
                proposer.num_speculative_tokens=s
                proposer._bounded_num_speculative_tokens=s
                proposer.max_num_tokens=128
                proposer.max_query_tokens=b
                proposer.device=torch.device("cpu")
                proposer.block_size=32
                proposer.parallel_drafting_token_id=151669
                proposer.input_ids=torch.zeros(128,dtype=torch.int32)
                proposer.positions=torch.zeros(b,dtype=torch.long)
                proposer.mrope_positions=torch.zeros((3,b+1),dtype=torch.long)
                proposer._context_mrope_positions=torch.zeros((3,128),dtype=torch.long)
                proposer._context_positions_buffer=torch.zeros(128,dtype=torch.long)
                proposer._context_slot_mapping_buffer=torch.zeros(128,dtype=torch.long)
                proposer._slot_mapping_buffer=torch.zeros(b,dtype=torch.long)
                proposer.arange=torch.arange(130,dtype=torch.int32)
                proposer.token_arange_np=np.arange(128,dtype=np.int32)
                cad=CommonAttentionMetadata(query_start_loc=torch.tensor([0,5],dtype=torch.int32),
                    query_start_loc_cpu=torch.tensor([0,5],dtype=torch.int32),seq_lens=torch.tensor([1905]),
                    seq_lens_cpu_upper_bound=torch.tensor([1905]),num_reqs=1,num_actual_tokens=5,
                    max_query_len=5,max_seq_len=1905,block_table_tensor=torch.arange(100,200).view(1,-1),
                    slot_mapping=torch.arange(5),causal=True,_seq_lens_cpu=None,_num_computed_tokens_cpu=None)
                ptr=proposer.mrope_positions.data_ptr()
                with patch.object(dflash,"copy_and_expand_dflash_inputs_kernel",CPUExpansion()):
                    n,indices,new_cad=proposer.set_inputs_first_pass(torch.arange(5),torch.tensor([42]),axes,
                        torch.zeros((5,2048)),None,cad,torch.tensor([r]))
                self.assertEqual((n,indices.shape,new_cad.slot_mapping.shape),(b,(s,),(b,)))
                self.assertEqual(new_cad.max_query_len,b)
                self.assertFalse(new_cad.causal)
                self.assertEqual(new_cad.seq_lens.tolist(),[1905-r+b])
                torch.testing.assert_close(indices,torch.arange(s,dtype=torch.int32))
                self.assertEqual(proposer.input_ids[:b].tolist(),[42]+[151669]*s)
                torch.testing.assert_close(proposer.mrope_positions[:,:b],axes[:,4-r:5-r]+torch.arange(1,b+1))
                torch.testing.assert_close(proposer.mrope_positions[:,:b],
                    self.bounded.spatial_query_positions(axes,torch.tensor([0,5]),torch.tensor([r]),8)[:,:b])
                self.assertEqual(proposer.mrope_positions.data_ptr(),ptr)
                self.assertEqual(proposer.block_size,32)
                torch.testing.assert_close(proposer.positions,torch.arange(1905-r,1905-r+b))
                self.assertTrue(torch.all(proposer._slot_mapping_buffer>=0))
                cad.seq_lens=torch.tensor([1905,5])
                with self.assertRaisesRegex(ValueError,"batch"):
                    proposer.set_inputs_first_pass(torch.arange(5),torch.tensor([42]),axes,
                        torch.zeros((5,2048)),None,cad,torch.tensor([r]))
        self.assertEqual({(s,page) for _,s,page in records},{(1,32),(3,32),(7,32)})
        for rejected in (-1,5,6):
            with self.assertRaises((RuntimeError,IndexError)):
                self.bounded.spatial_query_positions(axes,torch.tensor([0,5]),torch.tensor([rejected]),2)

    def test_metadata_repair_reuses_frozen_shared_last_token_axis(self):
        import official_spatial_graph_adapter_v2 as graph2
        binding=inspect.getclosurevars(self.Model.__init__).nonlocals["binding"]
        instance=NS(_dynamic_arg_dims=binding.mapping,do_not_compile=False,compiled=False)
        before=id(binding.mapping)
        graph2.repair_metadata_mapping(instance,binding)
        self.assertEqual(binding.mapping,{"input_ids":{0:"b"},"positions":{-1:"b"},"input_embeds":{0:"b"}})
        self.assertEqual(id(binding.mapping),before)
        binding.mapping.clear()
        binding.mapping.update(graph2.original_metadata())

    def test_real_official_metadata_marker_compiles_last_axis_for_b2_b4_b8(self):
        import torch
        import official_spatial_graph_adapter_v2 as graph2
        from vllm.config.compilation import DynamicShapesType
        binding=inspect.getclosurevars(self.Model.__init__).nonlocals["binding"]
        graph2.repair_metadata_mapping(NS(_dynamic_arg_dims=binding.mapping,
            do_not_compile=False,compiled=False),binding)
        class TokenContract(torch.nn.Module):
            def forward(self,input_ids,positions,input_embeds=None):
                return input_ids[:,None].float()+positions.T.float()
        calls=[]
        def backend(gm,inputs):
            calls.append(gm)
            return gm.forward
        try:
            probe=TokenContract()
            compiled=torch.compile(probe,backend=backend,fullgraph=True)
            for b in (2,4,8):
                ids=torch.arange(b)
                positions=torch.arange(3*(b+1)).reshape(3,b+1)[:,:b]
                binding.mark_function(probe,DynamicShapesType.BACKED,ids,positions)
                torch.testing.assert_close(compiled(ids,positions),probe(ids,positions))
            self.assertEqual(len(calls),1)
        finally:
            binding.mapping.clear()
            binding.mapping.update(graph2.original_metadata())
            torch._dynamo.reset()

    def test_context_hook_matches_frozen_projection_rotary_v_and_slot_math(self):
        import torch
        from vllm import _custom_ops as ops
        from vllm.config import CompilationConfig,set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from test_cpu import trained_reference
        config=NS(compilation_config=CompilationConfig(mode=0,custom_ops=["none"]))
        with set_current_vllm_config(config):
            rope=get_rope(128,max_position=1024,rope_parameters={"rope_type":"default",
                "rope_theta":1000000.0,"mrope_section":[16,24,24],"mrope_interleaved":False},dtype=torch.float32)
        def cpu_norm(out,x,weight,eps):
            out.copy_(torch.nn.functional.rms_norm(x,(x.shape[-1],),weight,eps))
        for b in (2,4,8):
            # Official MRoPE casts its cache in place; restore fp32 after the
            # preceding BF16 query check, as in the frozen CPU test fixture.
            rope.cos_sin_cache=rope._compute_cos_sin_cache()
            g=torch.Generator().manual_seed(46+b)
            L,n,nkv,hd,hidden=3,b,2,128,32
            kv=nkv*hd
            states=torch.randn((n,hidden),generator=g)
            hidden_norm=torch.randn(hidden,generator=g)
            weights=torch.randn((L*2*kv,hidden),generator=g)/hidden**0.5
            knorm=[torch.randn(hd,generator=g) for _ in range(L)]
            axes=torch.stack((torch.arange(n),torch.arange(13,13+n),torch.arange(29,29+n)))
            slots=torch.arange(1900,1900+n)
            captured=[]
            def update(attn,key,value,cache,slot_mapping):
                captured.append((key.clone(),value.clone(),slot_mapping.clone()))
            model=NS(config=NS(dflash_config={self.bounded.MARKER:True}),
                _num_attn_layers=L,_kv_size=kv,_head_dim=hd,_num_kv_heads=nkv,
                _hidden_norm_weight=hidden_norm,_rms_norm_eps=1e-6,
                _fused_kv_weight=weights,_fused_kv_bias=None,_k_norm_weights=knorm,
                layers=[NS(self_attn=NS(rotary_emb=rope))],
                _attn_layers=[NS(impl=NS(do_kv_cache_update=update),kv_cache=None) for _ in range(L)])
            with patch.object(ops,"rms_norm",cpu_norm):
                self.Model.precompute_and_store_context_kv(model,states,axes,slots)
                with self.assertRaisesRegex(ValueError,"real context"):
                    self.Model.precompute_and_store_context_kv(model,states,axes[0],slots)
            self.assertEqual(len(captured),L)
            normed=torch.nn.functional.rms_norm(states,(hidden,),hidden_norm,1e-6)
            for i,(actual_k,actual_v,actual_slots) in enumerate(captured):
                projected=torch.nn.functional.linear(normed,weights[i*2*kv:(i+1)*2*kv])
                expected_k,expected_v=projected.view(n,2,nkv,hd).unbind(1)
                expected_k=torch.nn.functional.rms_norm(expected_k,(hd,),knorm[i],1e-6)
                expected_k=trained_reference(expected_k.flatten(1),expected_k.flatten(1),axes)[0]
                torch.testing.assert_close(actual_k.flatten(1),expected_k,rtol=1e-5,atol=2e-6)
                torch.testing.assert_close(actual_v,expected_v,rtol=1e-5,atol=2e-6)
                torch.testing.assert_close(actual_slots,slots)
            for dtype,tol in ((torch.float32,2e-6),(torch.bfloat16,0)):
                rope.cos_sin_cache=rope._compute_cos_sin_cache()
                q=torch.randn((b,16*128),generator=g).to(dtype)
                k=torch.randn((b,2*128),generator=g).to(dtype)
                actual=rope.forward_native(axes,q.clone(),k.clone())
                expected=trained_reference(q,k,axes)
                for a,e in zip(actual,expected):
                    torch.testing.assert_close(a,e,rtol=0,atol=tol)

    def test_bootstrap_enabled_installs_only_bounded_chain_without_cuda(self):
        import subprocess
        code="import torch; import official_spatial_bounded_adapter as b; import official_spatial_adapter as f; import official_spatial_graph_adapter as g; import official_spatial_graph_adapter_v2 as v; assert b._INSTALLED; assert not f._INSTALLED and not g._INSTALLED and not v._INSTALLED; assert not torch.cuda.is_initialized()"
        env=dict(os.environ,PYTHONPATH=str(ROOT/"bootstrap"),PYTHONDONTWRITEBYTECODE="1",
            VLLM_OFFICIAL_SPATIAL_BOUNDED_ADAPTER="1")
        for name in ("VLLM_OFFICIAL_SPATIAL_ADAPTER","VLLM_OFFICIAL_SPATIAL_GRAPH_ADAPTER",
                "VLLM_OFFICIAL_SPATIAL_GRAPH_ADAPTER_V2","VLLM_OFFICIAL_SPATIAL_GRAPH_ADAPTER_V3"):
            env.pop(name,None)
        subprocess.run([sys.executable,"-c",code],env=env,check=True)

    def test_no_cuda_initialization(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    if "--frozen-regression" in sys.argv:
        sys.argv.remove("--frozen-regression")
        unittest.main(defaultTest="FrozenRegression", verbosity=2)
    elif "--static" in sys.argv:
        sys.argv.remove("--static")
        unittest.main(defaultTest="StaticContracts",verbosity=2)
    else:
        unittest.main(defaultTest=("BoundedPolicy","OfficialDispatch","PublicConfig","ProvenanceContracts","StaticContracts","InstalledContracts"), verbosity=2)
