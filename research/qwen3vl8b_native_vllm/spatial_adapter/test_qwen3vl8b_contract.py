import unittest
from types import SimpleNamespace as NS
from official_spatial_adapter import validate_draft_config, MARKER


def draft_config():
    return {'hidden_size':4096,'head_dim':128,'num_hidden_layers':3,
        'num_attention_heads':32,'num_key_value_heads':8,'target_hidden_size':4096,
        'vocab_size':151936,'intermediate_size':12288,'block_size':8,'attention_bias':False,
        'dflash_config':{'mask_token_id':151669,'target_layer_ids':[1,17,33],
          'spatial_rope_enabled':True,'spatial_rope_mode':'replace',
          'spatial_rope_sections':[24,20,20],'spatial_rope_interleaved':True,MARKER:True},
        'rope_parameters':{'rope_type':'default','rope_theta':5000000.,
          'mrope_section':[24,20,20],'mrope_interleaved':True}}

class Qwen3VL8BContract(unittest.TestCase):
    def test_full_expected_checkpoint_configuration_is_admitted(self):
        self.assertEqual(validate_draft_config(draft_config())['hidden_size'],4096)
    def test_noninterleaved_position_layout_is_rejected(self):
        cfg=draft_config();cfg['dflash_config']['spatial_rope_interleaved']=False
        with self.assertRaises(ValueError):validate_draft_config(cfg)
    def test_wrong_target_width_is_rejected(self):
        cfg=draft_config();cfg['target_hidden_size']=2048
        with self.assertRaises(ValueError):validate_draft_config(cfg)
    def test_untied_target_head_and_embedding_are_both_shared(self):
        from official_spatial_graph_adapter import validate_target_head_binding
        embedding=object();head=object()
        target=NS(model=NS(embed_tokens=embedding),lm_head=head)
        draft=NS(model=NS(embed_tokens=embedding),lm_head=head)
        validate_target_head_binding(draft,target)
        draft.lm_head=embedding
        with self.assertRaises(ValueError):validate_target_head_binding(draft,target)

if __name__=='__main__':unittest.main()
