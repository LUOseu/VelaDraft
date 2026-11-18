"""CPU-only tests for the local shared-block JetSpec mount; no model or forward."""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def config_data():
    return json.loads((ROOT/'exported_config.json').read_text())


def hf(data=None):
    data = config_data() if data is None else data
    result = NS(**data)
    result.to_dict = lambda: data
    return result


def launch(s=7):
    from vllm.config import CompilationMode, CUDAGraphMode
    return NS(model_config=NS(enforce_eager=False,model='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target',hf_config=NS(model_type='qwen3_vl')),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),
        scheduler_config=NS(max_num_seqs=1,max_num_batched_tokens=128),lora_config=None,
        speculative_config=NS(num_speculative_tokens=s,method='dflash',enforce_eager=False,
            draft_model_config=NS(hf_config=hf(),uses_mrope=True)),
        compilation_config=NS(mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,cudagraph_capture_sizes=[s+1,2*(s+1)],
            max_cudagraph_capture_size=2*(s+1),compile_sizes=None,
            is_attention_compiled_piecewise=lambda:True))


class ContractTests(unittest.TestCase):
    def test_actual_target_embedding_and_head_alias_guard(self):
        import official_jetspec_adapter as j
        target=NS(lm_head=object(),model=NS(embed_tokens=object()))
        draft=NS(lm_head=object(),model=NS(embed_tokens=object()))
        with self.assertRaises(ValueError):j.validate_actual_target_aliases(draft,target)
        draft.lm_head=target.lm_head
        with self.assertRaises(ValueError):j.validate_actual_target_aliases(draft,target)
        draft.model.embed_tokens=target.model.embed_tokens
        j.validate_actual_target_aliases(draft,target)

    def test_marked_config_and_all_flags_rejected_if_incompatible(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('official_jetspec_adapter'),
            'JetSpec causal/config contract implementation is absent')
        import official_jetspec_adapter as j
        data=config_data()
        self.assertEqual(j.validate_config(data),data)
        for path,value in ((('algorithm',),'dflash'),(('architectures',),['Qwen3DFlashDraftModel']),
                (('block_size',),4),(('dflash_config',j.MARKER),False),
                (('dflash_config','inference_in_block_causal'),False),
                (('dflash_config','sample_from_anchor'),True),
                (('dflash_config','shift_label'),True),(('dflash_config','pure_draft_prefix_len'),1),
                (('dflash_config','projector_type'),'domino'),(('dflash_config','layer_routing_enabled'),True),
                (('dflash_config','visual_num_query_tokens'),1),
                (('dflash_config','spatial_rope_interleaved'),False),
                (('jetspec_inference_contract','sample_from_anchor'),True),
                (('jetspec_provenance','source_sha256'),'bad')):
            bad=copy.deepcopy(data); obj=bad
            for key in path[:-1]:obj=obj[key]
            obj[path[-1]]=value
            with self.subTest(path=path),self.assertRaises((ValueError,RuntimeError)):
                j.validate_config(bad)
        bad=copy.deepcopy(data);bad['dflash_config']['unexpected_learned_path']=True
        with self.assertRaisesRegex(ValueError,'unsupported'):
            j.validate_config(bad)
        bad=copy.deepcopy(data);bad['model_type']='eagle'
        with self.assertRaisesRegex(ValueError,'wrapper'):j.validate_config(hf(bad))

    def test_marker_selection_is_fail_closed_and_plain_unmarked_is_inert(self):
        import official_jetspec_adapter as j
        self.assertFalse(j.selected(NS(dflash_config={})))
        for data in ({'algorithm':'jetspec','dflash_config':{}},
                     {'dflash_config':{j.SPATIAL_MARKER:True}},
                     {'algorithm':'jetspec','dflash_config':{j.MARKER:False}}):
            with self.assertRaises(ValueError):j.selected(hf(data))

    def test_additional_backbone_or_learned_branch_settings_are_rejected(self):
        import official_jetspec_adapter as j
        for key,value in (('hidden_act','gelu'),('attention_dropout',0.2),
                ('use_sliding_window',True),('layer_types',['sliding_attention']*3),
                ('max_position_embeddings',4096),('dspark_config',{'enabled':True}),
                ('quantization_config',{'quant_method':'awq'})):
            bad=config_data();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):j.validate_config(bad)

    def test_secondary_eagle_config_and_feature_overrides_are_rejected(self):
        import official_jetspec_adapter as j
        # An EAGLEConfig wrapper is supported; an extra eagle_config branch is
        # absent in the checkpoint and is mutated in place by the constructor.
        for value in (None,{},[],False,True,{'use_aux_hidden_state':False},
                {'use_aux_hidden_state':True},{'use_aux_hidden_state':1},
                {'use_aux_hidden_state':'true'},{'extra_learned_branch':True}):
            bad=config_data();bad['eagle_config']=value
            with self.subTest(eagle_config=value),self.assertRaises(ValueError):j.validate_config(bad)
        for value in (None,[],[2,18,34],[3,9],True,'2,18,34'):
            bad=config_data();bad['eagle_aux_hidden_state_layer_ids']=value
            with self.subTest(aux_override=value),self.assertRaises(ValueError):j.validate_config(bad)
        for value in (False,True,None,0,1,'true'):
            bad=config_data();bad['dflash_config']['use_aux_hidden_state']=value
            with self.subTest(dflash_aux=value),self.assertRaises(ValueError):j.validate_config(bad)

    def test_draft_vocab_preserves_full_external_head_and_rejects_other_types(self):
        import official_jetspec_adapter as j
        for value in (None,151936):
            data=config_data();data['draft_vocab_size']=value;before=copy.deepcopy(data)
            self.assertEqual(j.validate_config(data),before);self.assertEqual(data,before)
        for value in (64,0,-1,True,False,151936.0,'151936',[],{}):
            bad=config_data();bad['draft_vocab_size']=value
            with self.subTest(draft_vocab_size=value),self.assertRaises(ValueError):j.validate_config(bad)

    def test_logit_scale_preserves_unit_default_and_rejects_other_types(self):
        import official_jetspec_adapter as j
        for value in (1,1.0):
            data=config_data();data['logit_scale']=value;before=copy.deepcopy(data)
            self.assertEqual(j.validate_config(data),before);self.assertEqual(data,before)
        for value in (-1.0,0,0.5,None,True,False,'1.0',[],{},float('nan'),float('inf')):
            bad=config_data();bad['logit_scale']=value
            with self.subTest(logit_scale=value),self.assertRaises(ValueError):j.validate_config(bad)

    def test_explicit_bootstrap_flag_installs_cpu_only(self):
        code="import official_jetspec_adapter as j; import torch; assert j._INSTALLED; assert not torch.cuda.is_initialized(); print('explicit local-block JetSpec bootstrap installed; CUDA uninitialized')"
        env=dict(os.environ,PYTHONPATH=str(ROOT/'bootstrap'),VLLM_OFFICIAL_JETSPEC_LOCAL_BLOCK_ADAPTER='1')
        subprocess.run([sys.executable,'-c',code],env=env,check=True)

    def test_launch_fixes_target_and_greedy_draft_and_keeps_three_sizes_distinct(self):
        import official_jetspec_adapter as j
        for s in (1,3,7):
            cfg=launch(s);self.assertEqual(j.validate_launch_config(cfg),s+1)
            self.assertEqual(cfg.speculative_config.draft_model_config.hf_config.block_size,8)
            cfg.model_config.model='/tmp/another_qwen_target'
            with self.assertRaisesRegex(ValueError,'target'):j.validate_launch_config(cfg)
            cfg.model_config.model='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target'
            cfg.speculative_config.draft_sample_method='probabilistic'
            with self.assertRaisesRegex(ValueError,'greedy'):j.validate_launch_config(cfg)

    def test_exact_schema_rejects_extra_missing_wrong_dtype_or_shape(self):
        import torch
        import export_jetspec as e
        # Meta tensors test schema only; no ~0.5GB allocation or model exists.
        state={n:torch.empty(shape,device='meta',dtype=torch.bfloat16) for n,shape in e.expected_schema().items()}
        e.validate_state(state)
        for variant in ('extra','missing','dtype','shape'):
            bad=dict(state);name=next(iter(bad))
            if variant=='extra':bad['aux_head.weight']=torch.empty((1,),device='meta')
            if variant=='missing':bad.pop(name)
            if variant=='dtype':bad[name]=torch.empty(bad[name].shape,device='meta',dtype=torch.float32)
            if variant=='shape':bad[name]=torch.empty((1,),device='meta',dtype=torch.bfloat16)
            with self.subTest(variant=variant),self.assertRaises(ValueError):e.validate_state(bad)

    def test_payload_flags_unknown_paths_and_export_overwrite_are_rejected(self):
        import torch
        import export_jetspec as e
        payload=json.loads((ROOT/'checkpoint_metadata.json').read_text())
        payload['model']={n:torch.empty(shape,device='meta',dtype=torch.bfloat16) for n,shape in e.expected_schema().items()}
        run=json.loads((e.SOURCE/'config.json').read_text());contract=json.loads((e.SOURCE/'contract.json').read_text())
        e.validate_payload(payload,run,contract)
        for path,value in ((('unused_learned_branch',),{}),(('draft_config','dflash_config','shift_label'),True),
                (('draft_config','dflash_config','sample_from_anchor'),True),(('jetspec','causal_in_block'),False),
                (('head',),'independent_head')):
            bad=copy.deepcopy(payload);obj=bad
            for key in path[:-1]:obj=obj[key]
            obj[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):e.validate_payload(bad,run,contract)
        with self.assertRaises(FileExistsError):e.export(ROOT/'hf_export_step75000_v1')
        with self.assertRaises(ValueError):e.export(ROOT.parent/'foreign_export')

    def test_bootstrap_without_flag_is_inert(self):
        code="import sys; before=list(sys.path); import sitecustomize; assert sys.path==before; assert 'official_jetspec_adapter' not in sys.modules; assert 'torch' not in sys.modules"
        env=dict(os.environ,PYTHONPATH=str(ROOT/'bootstrap'))
        env.pop('VLLM_OFFICIAL_JETSPEC_LOCAL_BLOCK_ADAPTER',None)
        subprocess.run([sys.executable,'-c',code],env=env,check=True)


class MaskTests(unittest.TestCase):
    @staticmethod
    def local_mask():
        # Extract the exact pinned function with AST. Do not import or construct
        # the local HF model: only its independent CPU mask helper executes.
        import torch
        source=(ROOT/'source_evidence/draft_model.py').read_bytes()
        manifest=json.loads((ROOT/'source_bindings.json').read_text())
        assert hashlib.sha256(source).hexdigest()==manifest['local_mask_sha256']
        tree=ast.parse(source)
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_inference_block_mask')
        module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
        namespace={'torch':torch};exec(compile(module,'pinned_local_mask','exec'),namespace)
        return namespace['_inference_block_mask']

    def test_actual_local_mask_equals_official_bottom_right_relation(self):
        import torch
        from torch.nn.attention.bias import causal_lower_right
        local=self.local_mask()
        for b in (2,4,8):
            for prefix in (0,3,7,15,16,17,31,32,33):
                for cached in range(prefix+1):
                    cache=NS(get_seq_length=lambda:cached)
                    m=local(NS(inference_in_block_causal=True),cache,
                        torch.empty((1,prefix-cached,2)),torch.empty((1,b,2)),torch.empty((1,b),dtype=torch.long))
                    allowed=m[0,0]==0
                    actual=causal_lower_right(b,prefix+b)._materialize()
                    self.assertTrue(torch.equal(allowed,actual),(b,prefix,cached))
                    self.assertTrue(allowed[:,:prefix].all())
                    self.assertEqual(int(allowed[:,prefix:].sum()),b*(b+1)//2)
                    self.assertFalse(torch.equal(allowed,torch.ones_like(allowed)))

    def test_official_cpu_sdpa_observes_noncausal_mismatch(self):
        import torch
        import torch.nn.functional as F
        from torch.nn.attention.bias import causal_lower_right
        for b in (2,4,8):
            for prefix in (0,3,7):
                # Official CPU SDPA, toy arrays only. No model/attention layer.
                q=torch.zeros((1,1,b,4));k=torch.zeros((1,1,prefix+b,4))
                v=torch.arange(prefix+b,dtype=torch.float32).view(1,1,-1,1)
                local=self.local_mask()(NS(inference_in_block_causal=True),NS(get_seq_length=lambda:prefix),
                    torch.empty((1,0,4)),torch.empty((1,b,4)),torch.empty((1,b),dtype=torch.long))
                reference=F.scaled_dot_product_attention(q,k,v,attn_mask=local)
                official=F.scaled_dot_product_attention(q,k,v,attn_mask=causal_lower_right(b,prefix+b))
                ordinary=F.scaled_dot_product_attention(q,k,v)
                torch.testing.assert_close(reference,official,rtol=0,atol=0)
                self.assertFalse(torch.equal(reference,ordinary))

    def test_synthetic_paged_cache_visibility_rejection_and_context_overwrite(self):
        import torch
        for b in (2,4,8):
            for page in (16,32):
                for prefix in (3,7,15,16,17,31,32,33):
                    for rejected in range(min(7,prefix-1)+1):
                        valid=prefix-rejected
                        table=torch.tensor([5,1,7,3,9,0,8,6]);storage=torch.full((10*page,),-999)
                        def slots(pos):return table[pos//page]*page+pos%page
                        # Previously stored query/stale suffix rows differ from fresh target features.
                        old=torch.arange(prefix+b);storage[slots(old)]=10000+old
                        context=torch.arange(prefix);keep=context<valid
                        storage[slots(context[keep])]=context[keep]*10
                        query=torch.arange(valid,valid+b);storage[slots(query)]=20000+torch.arange(b)
                        logical=torch.cat((torch.arange(valid)*10,20000+torch.arange(b)))
                        self.assertTrue(torch.equal(storage[slots(torch.arange(valid+b))],logical))
                        relation=torch.arange(valid+b)[None,:]<=valid+torch.arange(b)[:,None]
                        for i in range(b):
                            visible=storage[slots(torch.arange(valid+b))][relation[i]]
                            self.assertTrue(torch.equal(visible,logical[:valid+i+1]))
                            self.assertFalse((visible>=10000).logical_and(visible<20000).any())


class InstalledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import official_jetspec_adapter as j
        from vllm.v1.spec_decode.dflash import DFlashProposer
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
        cls.j=j;cls.P=DFlashProposer
        cls.originals=(DFlashQwen3Model.forward,DFlashQwen3Model.__call__,
            SpecDecodeBaseProposer._sample_draft_tokens,DFlashQwen3ForCausalLM.combine_hidden_states)
        cls.bindings=j.install()

    def test_operator_forward_sampler_and_spatial_helpers_unchanged(self):
        import official_spatial_bounded_adapter as bounded
        import official_spatial_adapter as frozen
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
        self.assertEqual(self.bindings,self.j.install())
        self.assertEqual((DFlashQwen3Model.forward,DFlashQwen3Model.__call__,
            SpecDecodeBaseProposer._sample_draft_tokens,DFlashQwen3ForCausalLM.combine_hidden_states),self.originals)
        for n in ('absolute_context_positions','spatial_query_positions','rotate_context_keys'):
            self.assertIs(getattr(bounded,n),getattr(frozen,n))
        self.assertEqual(self.bindings['bounded_adapter'],json.loads((ROOT/'source_bindings.json').read_text())['frozen']['bounded_lookahead/official_spatial_bounded_adapter.py'])
        self.assertEqual(Path(inspect.getsourcefile(DFlashQwen3Model.precompute_and_store_context_kv)).resolve(),
            (ROOT.parent/'bounded_lookahead/official_spatial_bounded_adapter.py').resolve())

    def test_feature_order_and_official_external_head_embedding_binding_sentinels(self):
        # Python namespaces and identity sentinels, never model instances.
        from vllm.v1.spec_decode import llm_base_proposer as base
        embedding,head=object(),object()
        target=NS(model=NS(embed_tokens=embedding),lm_head=head)
        draft=NS(has_own_embed_tokens=False,has_own_lm_head=False,model=NS(embed_tokens=object()),lm_head=object())
        p=NS(model=draft,use_local_argmax_reduction=False)
        with patch.object(base,'get_pp_group',lambda:NS(world_size=1)):
            base.SpecDecodeBaseProposer._maybe_share_embeddings(p,target)
        base.SpecDecodeBaseProposer._maybe_share_lm_head(p,target)
        self.assertIs(draft.model.embed_tokens,embedding);self.assertIs(draft.lm_head,head)
        source=(ROOT/'source_evidence/gpu_model_runner.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),'690998ae7d9353ce42d7fc43f755fdbd72097c6e1f7a364dde435743408b00f2')
        tree=ast.parse(source)
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_eagle3_aux_layers_from_config')
        module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module);namespace={}
        exec(compile(module,'pinned_official_feature_order','exec'),namespace)
        indices=namespace['_get_eagle3_aux_layers_from_config'](NS(speculative_config=launch().speculative_config))
        self.assertEqual(indices,(2,18,34))
        alternate=launch();alternate.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids=[3,9]
        # The pinned official helper otherwise gives this extra field priority.
        self.assertEqual(namespace['_get_eagle3_aux_layers_from_config'](NS(speculative_config=alternate.speculative_config)),(3,9))
        data=config_data();data['eagle_aux_hidden_state_layer_ids']=[3,9]
        with self.assertRaises(ValueError):self.j.validate_config(data)

    def test_draft_config_only_causal_policy_changes_before_dummy_backend_selection(self):
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        @dataclass
        class Attention:
            use_non_causal:bool=False
            backend:str='FLASH_ATTN'
            other:int=73
        @dataclass
        class Config:
            attention_config:Attention
            sentinel:object
        p=object.__new__(self.P);p.draft_model_config=NS(hf_config=hf())
        base=Config(Attention(),object())
        with patch.object(SpecDecodeBaseProposer,'_create_draft_vllm_config',lambda self:base):
            result=p._create_draft_vllm_config()
            self.assertFalse(result.attention_config.use_non_causal)
            self.assertEqual(result.attention_config.backend,'FLASH_ATTN')
            self.assertEqual(result.attention_config.other,73)
            self.assertIs(result.sentinel,base.sentinel)
            self.assertFalse(base.attention_config.use_non_causal)
            p.draft_model_config=NS(hf_config=NS(dflash_config={}))
            self.assertTrue(p._create_draft_vllm_config().attention_config.use_non_causal)
        # Stock dummy_run calls set_forward_context(None): the official backend
        # resolves profiling/dummy causality from the draft AttentionConfig.
        from vllm.v1.spec_decode import dflash
        tree=ast.parse(Path(dflash.__file__).read_text())
        dummy=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='dummy_run')
        context=next(n for n in ast.walk(dummy) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='set_forward_context')
        self.assertIsNone(context.args[0].value)

    def test_official_base_metadata_builder_accepts_only_causal_all_layers(self):
        calls=[]
        class Builder:
            def __init__(self,value):self.value=value
            def build_for_drafting(self,common_attn_metadata,draft_index):
                calls.append((common_attn_metadata,draft_index))
                return NS(causal=self.value,sentinel='unchanged')
        p=object.__new__(self.P);p.draft_model_config=NS(hf_config=hf())
        for scope in ('real','profile_fixture'):
            cad=NS(causal=True,scope=scope)
            p.draft_attn_groups=[NS(layer_names=['layer0','layer1'],get_metadata_builder=lambda:Builder(True)),
                NS(layer_names=['layer2'],get_metadata_builder=lambda:Builder(True))]
            groups,layers=p.build_per_group_and_layer_attn_metadata(cad,3)
            self.assertEqual(len(groups),2);self.assertEqual(len(layers),3)
            self.assertTrue(all(v.causal is True for v in layers.values()))
            self.assertIs(calls[-1][0],cad);self.assertEqual(calls[-1][1],3)
            for bad in (False,None,1):
                p.draft_attn_groups=[NS(layer_names=['layer0'],get_metadata_builder=lambda:Builder(bad))]
                with self.assertRaisesRegex(RuntimeError,'causal'):p.build_per_group_and_layer_attn_metadata(cad)
            with self.assertRaisesRegex(ValueError,'causal'):p.build_per_group_and_layer_attn_metadata(NS(causal=False))
        p.draft_model_config=NS(hf_config=NS(dflash_config={}))
        p.draft_attn_groups=[NS(layer_names=['plain'],get_metadata_builder=lambda:Builder(False))]
        self.assertFalse(p.build_per_group_and_layer_attn_metadata(NS(causal=False))[1]['plain'].causal)

    def test_real_official_expansion_rejection_offsets_and_anchor_exclusion(self):
        import torch,numpy as np
        from vllm.v1.spec_decode import dflash
        from vllm.v1.attention.backend import CommonAttentionMetadata
        seen=[]
        class CPUExpansion:
            def __getitem__(self,grid):
                def boundary(**kw):
                    seen.append(kw)
                    n=kw['total_input_tokens'];b=kw['num_query_per_req']
                    r=int(kw['num_rejected_tokens_ptr'][0]) if kw['HAS_NUM_REJECTED'] else 0
                    physical=kw['target_positions_ptr'];q=torch.arange(int(physical[-1])+1-r,int(physical[-1])+1-r+b)
                    kw['out_context_positions_ptr'][:n].copy_(physical);kw['out_query_positions_ptr'][:b].copy_(q)
                    kw['out_input_ids_ptr'][:b].fill_(kw['parallel_drafting_token_id']);kw['out_input_ids_ptr'][0]=kw['next_token_ids_ptr'][0]
                    for positions,out in ((physical,kw['out_context_slot_mapping_ptr']),(q,kw['out_query_slot_mapping_ptr'])):
                        out[:len(positions)]=kw['block_table_ptr'][0,positions//kw['block_size']]*kw['block_size']+positions%kw['block_size']
                    if r:kw['out_context_slot_mapping_ptr'][n-r:n].fill_(-1)
                    kw['out_token_indices_ptr'][:b-1]=torch.arange(1,b,dtype=torch.int32)
                return boundary
        for s in (1,3,7):
            b=s+1
            for r in range(8):
                n=9;p=object.__new__(self.P);cfg=launch(s)
                p.vllm_config=cfg;p.speculative_config=cfg.speculative_config;p.draft_model_config=cfg.speculative_config.draft_model_config
                p.num_speculative_tokens=s;p._bounded_num_speculative_tokens=s;p.device=torch.device('cpu');p.block_size=16
                p.parallel_drafting_token_id=151669;p.max_num_tokens=128;p.max_query_tokens=b
                p.input_ids=torch.zeros(128,dtype=torch.int32);p.positions=torch.zeros(b,dtype=torch.long)
                p.mrope_positions=torch.zeros((3,b+1),dtype=torch.long);p._context_mrope_positions=torch.zeros((3,128),dtype=torch.long)
                p._context_positions_buffer=torch.zeros(128,dtype=torch.long);p._context_slot_mapping_buffer=torch.zeros(128,dtype=torch.long)
                p._slot_mapping_buffer=torch.zeros(b,dtype=torch.long);p.arange=torch.arange(130,dtype=torch.int32);p.token_arange_np=np.arange(128,dtype=np.int32)
                axes=torch.stack([torch.arange(n)+x for x in (100,200,300)])
                target_causal=(r%2)==0  # Synthetic immutability probe, not a new target policy.
                cad=CommonAttentionMetadata(query_start_loc=torch.tensor([0,n],dtype=torch.int32),
                    query_start_loc_cpu=torch.tensor([0,n],dtype=torch.int32),seq_lens=torch.tensor([33]),
                    seq_lens_cpu_upper_bound=torch.tensor([33]),num_reqs=1,num_actual_tokens=n,max_query_len=n,max_seq_len=33,
                    block_table_tensor=torch.tensor([[6,2,7,1,3]]),slot_mapping=torch.arange(n),causal=target_causal,
                    _seq_lens_cpu=None,_num_computed_tokens_cpu=None)
                with patch.object(dflash,'copy_and_expand_dflash_inputs_kernel',CPUExpansion()):
                    count,indices,new=p.set_inputs_first_pass(torch.arange(n),torch.tensor([42]),axes,
                        torch.zeros((n,2)),None,cad,torch.tensor([r]))
                self.assertEqual(count,b);self.assertIsNot(new,cad);self.assertTrue(new.causal);self.assertEqual(cad.causal,target_causal)
                self.assertEqual(new.seq_lens.tolist(),[33-r+b]);self.assertIs(new.block_table_tensor,cad.block_table_tensor)
                self.assertEqual(indices.tolist(),list(range(1,b)));self.assertNotIn(0,indices.tolist())
                self.assertEqual(p.input_ids[:b].tolist(),[42]+[151669]*s)
                torch.testing.assert_close(p.positions,torch.arange(33-r,33-r+b))
                torch.testing.assert_close(p.mrope_positions[:,:b],axes[:,n-r-1:n-r]+torch.arange(1,b+1))
                torch.testing.assert_close(p._context_mrope_positions[:,:n],axes)
                self.assertEqual(p.block_size,16);self.assertEqual(p.draft_model_config.hf_config.block_size,8)
                self.assertEqual(p.mrope_positions[:,:b].stride(),(b+1,1))
        self.assertEqual({(x['num_speculative_tokens'],x['block_size']) for x in seen},{(1,16),(3,16),(7,16)})

    def test_inherited_dummy_graph_scope_restores_and_real_queries_remain_strict(self):
        import official_spatial_bounded_adapter as bounded
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        from vllm.config import CUDAGraphMode
        for s in (1,3,7):
            class P:
                def dummy_run(self,n,fail=False):
                    result=self._determine_batch_execution_and_padding(n,use_cudagraphs=False)
                    if fail:raise RuntimeError('fixture failure')
                    return result
                def propose(self,n):return self._determine_batch_execution_and_padding(n)
                _determine_batch_execution_and_padding=SpecDecodeBaseProposer._determine_batch_execution_and_padding
            bounded.install_scoped_query_guards(P,SpecDecodeBaseProposer._determine_batch_execution_and_padding)
            p=P();cfg=launch(s);p.vllm_config=cfg;p.speculative_config=cfg.speculative_config
            p.draft_model_config=cfg.speculative_config.draft_model_config;p.num_speculative_tokens=s;p._bounded_num_speculative_tokens=s
            p.cudagraph_dispatcher=CudagraphDispatcher(cfg);p.cudagraph_dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE)
            for n in range(1,s+2):p.dummy_run(n)
            with self.assertRaises(RuntimeError):p.dummy_run(1,fail=True)
            self.assertFalse(bounded.in_dummy_scope())
            self.assertEqual(p.propose(s+1)[1],s+1)
            with self.assertRaises(ValueError):p.propose(1)
            self.assertFalse(bounded.in_dummy_scope())


class PublicConfigTests(unittest.TestCase):
    def test_actual_registry_and_speculative_config_loading_without_model(self):
        import torch
        from vllm.config import ModelConfig,ParallelConfig,SpeculativeConfig
        from vllm.model_executor.models import ModelRegistry
        target=ModelConfig(model='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target',dtype='bfloat16',max_model_len=4096)
        pc=ParallelConfig(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
        for s in (1,3,7):
            spec=SpeculativeConfig(model=str(ROOT/'hf_export_step75000_v1'),method='dflash',
                num_speculative_tokens=s,target_model_config=target,target_parallel_config=pc,draft_tensor_parallel_size=1)
            self.assertEqual(spec.draft_model_config.hf_config.architectures,['DFlashDraftModel'])
            self.assertEqual(spec.draft_model_config.hf_config.algorithm,'jetspec')
            self.assertEqual(spec.draft_model_config.hf_config.block_size,8)
            self.assertEqual(spec.num_speculative_tokens,s);self.assertTrue(spec.parallel_drafting)
            self.assertTrue(spec.draft_model_config.uses_mrope)
            self.assertIn('DFlashDraftModel',ModelRegistry.get_supported_archs())
            import official_jetspec_adapter as j
            before=copy.deepcopy(spec.draft_model_config.hf_config.to_dict())
            j.validate_config(spec.draft_model_config.hf_config)
            self.assertEqual(spec.draft_model_config.hf_config.to_dict(),before)
            # Execute only the pinned constructor's metadata-normalization IF,
            # on copied real public config objects. Never call a constructor.
            import vllm
            source=(Path(vllm.__file__).parent/'model_executor/models/qwen3_dflash.py').read_bytes()
            self.assertEqual(hashlib.sha256(source).hexdigest(),'7c99f26d24ac1f31dc317284009d05165155d32e05b3cf1277914ad3f747d82a')
            tree=ast.parse(source)
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DFlashQwen3ForCausalLM')
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            normalize=next(n for n in init.body if isinstance(n,ast.If))
            self.assertIn('draft_vocab_size',ast.unparse(normalize))
            code=ast.Module(body=[normalize],type_ignores=[]);ast.fix_missing_locations(code)
            for start in ('absent',None):
                normalized=copy.deepcopy(spec.draft_model_config.hf_config)
                if start is None:normalized.draft_vocab_size=None
                else:normalized.__dict__.pop('draft_vocab_size',None)
                exec(compile(code,'pinned_official_vocab_normalization','exec'),{'self':NS(config=normalized)})
                self.assertIs(type(normalized.draft_vocab_size),int)
                self.assertEqual(normalized.draft_vocab_size,151936)
                normalized_before=copy.deepcopy(normalized.to_dict())
                j.validate_config(normalized)
                self.assertTrue(j.selected(normalized))
                self.assertEqual(normalized.to_dict(),normalized_before)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_export_all_36_tensors_bitwise_and_target_tied_head(self):
        import torch
        from safetensors.torch import load_file
        import export_jetspec as e
        cp=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/jetspec/ckpt_final.pt')
        original=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
        restored=load_file(str(ROOT/'hf_export_step75000_v1/model.safetensors'),device='cpu')
        e.validate_state(original['model']);self.assertEqual(set(restored),set(original['model']))
        for name,t in restored.items():
            self.assertTrue(torch.equal(t.view(torch.uint8),original['model'][name].view(torch.uint8)),name)
        target=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/target')
        data=json.loads((target/'config.json').read_text());index=json.loads((target/'model.safetensors.index.json').read_text())
        self.assertFalse(data['tie_word_embeddings']);self.assertIn('model.language_model.embed_tokens.weight',index['weight_map'])
        self.assertIn('lm_head.weight',index['weight_map'])
        self.assertFalse(torch.cuda.is_initialized())


class ProvenanceTests(unittest.TestCase):
    def test_foreign_prior_install_and_foreign_causal_hook_rejected_before_mutation(self):
        variants=("j._bounded().install()",
            "from vllm.v1.spec_decode.dflash import DFlashProposer as P; P._create_draft_vllm_config=lambda self:None",
            "from vllm.v1.spec_decode.dflash import DFlashProposer as P; P.build_per_group_and_layer_attn_metadata=lambda self,*a:None")
        for prepare in variants:
            code="import official_jetspec_adapter as j\n"+prepare+"\nfrom vllm.v1.spec_decode.dflash import DFlashProposer as P\nbefore=P.__init__\ntry:j.install()\nexcept RuntimeError:pass\nelse:raise AssertionError('foreign installation accepted')\nassert P.__init__ is before\nimport torch;assert not torch.cuda.is_initialized()\n"
            subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)

    def test_all_captured_frozen_source_hashes_unchanged(self):
        for name,digest in json.loads((ROOT/'REMOTE_FROZEN_BEFORE.json').read_text()).items():
            p=Path(name)
            # Full target/weight hashing is recorded separately in final receipt.
            if p.suffix=='.safetensors' or p.name=='ckpt_final.pt':continue
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest,name)


if __name__=='__main__':
    import torch
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if torch.cuda.is_initialized():raise RuntimeError('CPU tests initialized CUDA')
    print(json.dumps({'scope':'local shared-block JetSpec only; no upstream tree runtime',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'cuda_initialized':torch.cuda.is_initialized(),'model_construction':False,'model_forward':False}))
    sys.exit(not result.wasSuccessful())
