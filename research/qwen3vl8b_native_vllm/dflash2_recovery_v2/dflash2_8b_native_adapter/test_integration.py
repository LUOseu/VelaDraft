import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import torch

from . import contracts as c


class IntegrationTests(unittest.TestCase):
    def require(self,name):self.assertTrue((c.HERE/name).is_file(),'missing independent vLLM module: '+name)

    def test_owned_backbone_compile_dictionary_and_stock_operators(self):
        self.require('backbone.py')
        from .backbone import DFlash2OwnedBackbone,DFlash2OwnedDecoderLayer
        from .spatial import compile_mapping,validate_compile_ownership
        from .layer_logic import decoder_forward
        from vllm.model_executor.models import qwen3_dflash as official
        self.assertIs(DFlash2OwnedBackbone.forward,official.DFlashQwen3Model.forward)
        self.assertIs(DFlash2OwnedBackbone.load_weights,official.DFlashQwen3Model.load_weights)
        self.assertIs(DFlash2OwnedDecoderLayer.forward,decoder_forward)
        self.assertTrue(issubclass(DFlash2OwnedDecoderLayer,official.DFlashQwen3DecoderLayer))
        proof=validate_compile_ownership(DFlash2OwnedBackbone)
        self.assertTrue(proof['stock_mapping_unchanged']);self.assertFalse(proof['shared_with_stock'])

    def test_truthful_registry_and_unmodified_outer_head(self):
        self.require('model.py');self.require('install.py')
        from .install import register_models,verify_official_interfaces
        from .model import DFlash2NativeQwen3ForCausalLM
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
        register_models();register_models()
        self.assertEqual(ModelRegistry.models[c.ARCH].class_name,c.ARCH)
        for name in ('forward','compute_logits','combine_hidden_states','precompute_and_store_context_kv'):
            self.assertIs(getattr(DFlash2NativeQwen3ForCausalLM,name),getattr(DFlashQwen3ForCausalLM,name))
        self.assertTrue(verify_official_interfaces()['official_callable_origins_verified'])

    def test_spatial_rejection_axes_and_query_capture_guards(self):
        self.require('spatial.py')
        from . import spatial
        axes=torch.tensor([[7,7,8,9,10],[12,12,12,13,14],[17,18,18,19,20]])
        physical,query=spatial.context_query_axes(axes,torch.tensor([12]),torch.tensor([0,5]),torch.tensor([2]))
        self.assertTrue(torch.equal(physical,torch.arange(7,12)))
        self.assertTrue(torch.equal(query,torch.tensor([[8],[12],[18]])+torch.arange(1,9)))
        spatial.validate_query_batch(8,8,'PIECEWISE',dummy=False,keys_initialized=True)
        spatial.validate_query_batch(1,1,'NONE',dummy=True,keys_initialized=False)
        for args in ((7,8,'PIECEWISE',False,True),(8,16,'PIECEWISE',False,True),(8,8,'NONE',False,True),(8,8,'PIECEWISE',False,False)):
            with self.assertRaises(ValueError):spatial.validate_query_batch(*args[:3],dummy=args[3],keys_initialized=args[4])

    def test_marked_selector_actual_anchor_and_nonmarker_delegation(self):
        self.require('proposer.py')
        from .proposer import DFlash2ProposerRouter,OfficialDFlashProposer
        p=object.__new__(DFlash2ProposerRouter);p._dflash2_active=True;p._dflash2_in_propose=True
        p._dflash2_backend_initialized=True;p.block_size=16;p._dflash2_anchor_ids=torch.tensor([13])
        p.num_speculative_tokens=7;p.parallel_drafting=True;p.speculative_config=NS(num_speculative_tokens=7,parallel_drafting=True)
        p.vllm_config=NS(speculative_config=p.speculative_config)
        observed=[]
        p.model=NS(sample_dflash2_prefix=lambda h,a:(observed.append((h,a)) or torch.arange(7).view(1,7)))
        hidden=torch.zeros(7,32,dtype=torch.bfloat16)
        tokens,probs=p._sample_draft_tokens(hidden,NS(all_greedy=True))
        self.assertEqual(tokens.tolist(),list(range(7)));self.assertIsNone(probs)
        self.assertIs(observed[0][1],p._dflash2_anchor_ids)
        with self.assertRaises(ValueError):p._sample_draft_tokens(hidden,NS(all_greedy=False))
        p._dflash2_active=False
        with mock.patch.object(OfficialDFlashProposer,'_sample_draft_tokens',return_value='stock') as original:
            self.assertEqual(p._sample_draft_tokens(hidden,NS()),'stock');original.assert_called_once()
        p._dflash2_active=True;p._dflash2_in_propose=False
        with mock.patch.object(OfficialDFlashProposer,'propose',side_effect=RuntimeError('request failure')):
            with self.assertRaises(RuntimeError):p.propose()
        self.assertFalse(p._dflash2_in_propose);self.assertIsNone(p._dflash2_anchor_ids)

    def test_marker_stripping_and_config_changes_fail_closed(self):
        meta=json.loads((c.HERE/'CHECKPOINT_METADATA.json').read_text());config=meta['draft_config']
        c.validate_config(config,source=True)
        self.assertTrue(c.near_dflash2(config));self.assertFalse(c.enabled(config))
        with self.assertRaises(ValueError):c.validate_config(config)
        config.update(architectures=[c.ARCH],source_checkpoint_sha256=c.CHECKPOINT_SHA)
        config['dflash_config'][c.MARKER]=True;config['rope_parameters'].update(mrope_section=[24,20,20],mrope_interleaved=True)
        c.validate_config(config)
        for key,val in [('block_size',7),('vocab_size',11),('sample_from_anchor',True)]:
            wrong=copy.deepcopy(config);wrong[key]=val
            with self.assertRaises(ValueError):c.validate_config(wrong)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_export_public_configs_parse_without_model_or_cuda(self):
        self.require('public_config.py')
        from .public_config import materialize_public_config
        from .install import verify_official_interfaces
        verify_official_interfaces()  # Import-time activation modules precede the no-model guard.
        export=c.HERE/'hf_step75000_v1'
        self.assertTrue((export/'config.json').is_file(),'Actual export is required for configuration proof')
        with (mock.patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA must stay hidden')),
              mock.patch('torch.nn.Module.__init__',side_effect=AssertionError('No serving model construction'))):
            proof,vc=materialize_public_config(export)
        self.assertEqual(proof['resolved_architectures'],[c.ARCH]);self.assertEqual(proof['public_S'],7)
        c.validate_launch_config(vc)
        for field,bad in [('block_size',32)]:
            wrong=copy.copy(vc);wrong.cache_config=copy.copy(vc.cache_config);setattr(wrong.cache_config,field,bad)
            with self.assertRaises(ValueError):c.validate_launch_config(wrong)
        for role in ('target','draft'):
            from vllm.config import LoadConfig
            wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
            if role=='target':wrong.load_config=LoadConfig(load_format='dummy')
            else:wrong.speculative_config.draft_load_config=LoadConfig(load_format='dummy')
            with self.assertRaises(ValueError):c.validate_launch_config(wrong)
        wrong=copy.copy(vc);wrong.model_config=copy.copy(vc.model_config);wrong.model_config.model='/some/other/target'
        with self.assertRaises(ValueError):c.validate_launch_config(wrong)
        wrong=copy.copy(vc);wrong.speculative_config=copy.copy(vc.speculative_config)
        wrong.speculative_config.draft_model_config=copy.copy(vc.speculative_config.draft_model_config)
        wrong.speculative_config.draft_model_config.dtype=torch.float16
        with self.assertRaises(ValueError):c.validate_launch_config(wrong)
        self.assertFalse(torch.cuda.is_initialized())

    def test_marker_only_or_unmarked_dflash2_never_falls_through(self):
        self.assertTrue(c.near_dflash2({'architectures':['Qwen3ForCausalLM'],'dflash_config':{c.MARKER:True}}))
        self.assertFalse(c.near_dflash2({'architectures':['DFlashQwen3ForCausalLM'],'dflash_config':None}))

    def test_standard_loader_must_share_exact_target_head_and_embedding(self):
        from .proposer import DFlash2ProposerRouter,OfficialDFlashProposer
        p=object.__new__(DFlash2ProposerRouter);p._dflash2_active=True
        target=torch.nn.Module();target.model=NS(embed_tokens=torch.nn.Embedding(4,3));target.lm_head=torch.nn.Linear(3,4,bias=False)
        p.model=NS(model=NS(embed_tokens=torch.nn.Embedding(4,3)),lm_head=torch.nn.Linear(3,4,bias=False),
            _dflash2_loaded_tensor_names=tuple(range(51)))
        with mock.patch.object(OfficialDFlashProposer,'load_model',return_value=None):
            with self.assertRaises(ValueError):p.load_model(target)
            p.model.model.embed_tokens=target.model.embed_tokens;p.model.lm_head=target.lm_head
            p.load_model(target)

if __name__=='__main__':unittest.main(verbosity=2)
