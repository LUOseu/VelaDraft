import unittest
import torch
from types import SimpleNamespace as NS
from .context import build_layer_contexts,project_shared_context
from . import contracts

class DFly8B(unittest.TestCase):
    def test_target_config_requires_interleaved_8b(self):
        cfg={'model_type':'qwen3','hidden_act':'silu','hidden_size':4096,'head_dim':128,
            'num_hidden_layers':3,'num_attention_heads':32,'num_key_value_heads':8,
            'target_hidden_size':4096,'vocab_size':151936,'block_size':8,'intermediate_size':12288,
            'attention_bias':False,'rms_norm_eps':1e-6,'rope_parameters':{'rope_type':'default','rope_theta':5000000.},
            'dflash_config':{'mask_token_id':151669,'target_layer_ids':[1,17,33],
            'spatial_rope_enabled':True,'spatial_rope_mode':'replace','spatial_rope_sections':[24,20,20],
            'spatial_rope_interleaved':True,'visual_num_query_tokens':0,'context_layer_ids':[1,17,33],
            'separate_context_kv':False,'sample_from_anchor':False,'fusion_trainable':True,'fusion_init':'distance',
            'enable_hidden_correction':True,'hidden_correction_intermediate_size':None}}
        contracts.validate_config(cfg,source=True)
        cfg['dflash_config']['spatial_rope_interleaved']=False
        with self.assertRaises(ValueError):contracts.validate_config(cfg,source=True)
    def test_ordered_context_uses_all_three_4096_vectors(self):
        class FC:
            weight=torch.zeros(1,dtype=torch.float32)
            def __call__(self,x):return x[:,:4096]
        features=torch.randn(2,12288,generator=torch.Generator().manual_seed(4))
        fusion=torch.tensor([[0.,0.,0.],[1.,2.,3.],[3.,1.,2.]])
        out=build_layer_contexts(features,fusion,FC(),lambda x:x)
        stacked=features.reshape(2,3,4096)
        for i in range(3):
            expected=features[:,:4096]+torch.einsum('t,ctd->cd',fusion[i].softmax(-1),stacked)
            torch.testing.assert_close(out[i],expected)
    def test_context_kv_has_eight_heads(self):
        attention=NS(qkv_proj=NS(weight=torch.randn(6144,4),bias=None),q_size=4096,kv_size=1024)
        key,value=project_shared_context(torch.randn(2,4),attention)
        self.assertEqual(tuple(key.shape),(2,8,128))
        self.assertEqual(tuple(value.shape),(2,8,128))
