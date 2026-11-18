from __future__ import annotations
import ast,copy,importlib.util,json,sys,textwrap,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def component(name):
    spec=importlib.util.find_spec(name)
    if spec is None:raise AssertionError(f'{name} feature is not implemented')
    return __import__(name)

def fixture_config():
    inv=json.loads((ROOT/'reference_metadata.json').read_text())
    c=copy.deepcopy(inv['common_draft_config'])
    c.update(algorithm='domino',architectures=['Qwen3DominoDraftModel'],model_type='qwen3',intermediate_size=12288,hidden_act='silu',rms_norm_eps=1e-6,tie_word_embeddings=False,attention_bias=False)
    c['dflash_config']=copy.deepcopy(inv['dflash_config'])
    c['dflash_config']['official_domino_public_adapter_v1']=True
    c['rope_parameters']={'rope_type':'default','rope_theta':5000000.0,'mrope_section':[24,20,20],'mrope_interleaved':True}
    c['source_checkpoint_sha256']='f3fdfcd68f4eeb1aab74649c71c9cdb5c117f5c5822cf54b9c0f749b1dde85f7'
    return c

class ContractTests(unittest.TestCase):
    def test_truthful_raw_config_and_exact_40_schema(self):
        c=component('contracts');cfg=fixture_config()
        self.assertTrue(c.selected(cfg));self.assertEqual(c.validate_config(cfg)['algorithm'],'domino')
        self.assertEqual(len(c.expected_schema()),40)
        self.assertEqual(c.expected_schema()['prefix_gru.weight_ih_l0'],[3072,4096])
        self.assertEqual(c.expected_schema()['embed_proj.2.weight'],[151936,256])
    def test_foreign_configs_never_selected_and_near_domino_fail_closed(self):
        c=component('contracts')
        self.assertFalse(c.selected({'architectures':['DFlashDraftModel'],'dflash_config':{}}))
        try:result=c.selected(NS(to_dict=lambda:{'architectures':['DFlashDraftModel'],'dflash_config':{}}))
        except Exception as error:result=str(error)
        self.assertIs(result,False)
        for field,val in [('algorithm','dflash'),('architectures',['DFlashDraftModel'])]:
            cfg=fixture_config();cfg[field]=val
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_config(cfg)
        cfg=fixture_config();cfg['dflash_config']['official_domino_public_adapter_v1']=False
        with self.assertRaises(ValueError):c.selected(cfg)
    def test_nonmarker_optional_dflash_mapping_is_left_to_stock(self):
        c=component('contracts')
        for fields in (None,[],{}):
            cfg={'architectures':['DFlashDraftModel'],'dflash_config':fields,'pard_token':151669}
            try:result=c.selected(cfg)
            except Exception as error:result=str(error)
            self.assertIs(result,False)
        cfg=fixture_config();cfg['dflash_config']=None
        with self.assertRaises(ValueError):c.selected(cfg)

    def test_unsupported_checkpoint_branches_and_types_rejected(self):
        c=component('contracts')
        for key,value in [('eagle_config',{}),('eagle_config',None),('eagle_aux_hidden_state_layer_ids',[3,4,5]),('draft_vocab_size',4),('draft_vocab_size',True),('logit_scale',True),('logit_scale',-1),('logit_scale','1')]:
            cfg=fixture_config();cfg[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):c.validate_config(cfg)
        for key,value in [('shift_label',False),('pure_draft_prefix_len',0),('sample_from_anchor',True),('inference_in_block_causal',True),('gru_hidden_dim',512),('enable_hidden_correction',True)]:
            cfg=fixture_config();cfg['dflash_config'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_config(cfg)
    def test_schema_rejects_missing_extra_wrong_shape_and_dtype(self):
        c=component('contracts');s=c.expected_schema()
        state={k:NS(shape=tuple(v),dtype='torch.bfloat16') for k,v in s.items()}
        c.validate_state(state)
        for mode in ('missing','extra','shape','dtype'):
            x=state.copy()
            if mode=='missing':x.pop('prefix_gru.weight_ih_l0')
            elif mode=='extra':x['lm_head.weight']=NS(shape=(1,),dtype='torch.bfloat16')
            elif mode=='shape':x['prefix_gru.weight_ih_l0']=NS(shape=(1,),dtype='torch.bfloat16')
            else:x['prefix_gru.weight_ih_l0']=NS(shape=(3072,4096),dtype='torch.float32')
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.validate_state(x)
    def test_query_guards_preserve_real8_dummy1_to8(self):
        c=component('contracts')
        c.validate_query(7,8,8,'PIECEWISE',dummy=False,initialized=True)
        for n in range(1,9):c.validate_query(7,n,n,'NONE',dummy=True,initialized=False)
        for args in [(3,8,8,'PIECEWISE',False,True),(7,7,8,'PIECEWISE',False,True),(7,8,16,'PIECEWISE',False,True),(7,8,8,'NONE',False,True),(7,8,8,'PIECEWISE',False,False),(7,0,1,'NONE',True,False),(7,8,9,'NONE',True,False)]:
            s,n,p,m,d,i=args
            with self.subTest(args=args),self.assertRaises(ValueError):c.validate_query(s,n,p,m,dummy=d,initialized=i)
    def test_raw_and_source_model_types_are_exact_qwen3(self):
        c=component('contracts')
        for source in (False,True):
            for model_type in ('eagle','llama',None):
                cfg=fixture_config();cfg['model_type']=model_type
                with self.subTest(source=source,model_type=model_type),self.assertRaises(ValueError):c.validate_config(cfg,source=source)

    def test_approved_design_rehashed_not_merely_recorded(self):
        import tempfile,hashlib
        from unittest.mock import patch
        c=component('contracts')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'new';root.mkdir();design=Path(td)/'approved.md';design.write_text('approved')
            source=Path(td)/'source.py';source.write_text('source')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            b={'frozen':{},'upstream':{},'design_sha256':{'approved.md':digest(design)},'canonical_reference':{'path':str(source),'sha256':digest(source)}}
            (root/'source_bindings.json').write_text(json.dumps(b))
            with patch.object(c,'ROOT',root):
                self.assertIn(str(design),c.verify_dependencies())
                design.write_text('changed')
                with self.assertRaises(RuntimeError):c.verify_dependencies()

    def test_direct_loader_sources_are_pinned(self):
        b=json.loads((ROOT/'source_bindings.json').read_text())
        for suffix in ('vllm/model_executor/models/utils.py','vllm/model_executor/model_loader/weight_utils.py','vllm/model_executor/layers/vocab_parallel_embedding.py'):
            self.assertTrue(any(name.endswith(suffix) for name in b['upstream']),suffix)

    def test_approved_receipt_is_bound_to_actual_target_and_export_paths(self):
        c=component('contracts');self.assertTrue(callable(getattr(c,'validate_model_binding',None)))
        target='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target';draft='/cpu_METADATA_ONLY/domino_export'
        r={'target_model_path':target,'draft_model_path':draft,'full_artifacts':{draft+'/config.json':'a'*64,draft+'/model.safetensors':'b'*64,draft+'/export_receipt.json':'c'*64}}
        vc=NS(model_config=NS(model=target),speculative_config=NS(model=draft,draft_model_config=NS(model=draft)))
        c.validate_model_binding(vc,r)
        for actual in ('/different_target','/different_export'):
            bad=copy.deepcopy(vc)
            if actual=='/different_target':bad.model_config.model=actual
            else:bad.speculative_config.draft_model_config.model=actual
            with self.assertRaises(ValueError):c.validate_model_binding(bad,r)
        bad=copy.deepcopy(r);bad['full_artifacts'].pop(draft+'/model.safetensors')
        with self.assertRaises(ValueError):c.validate_model_binding(vc,bad)

    def test_legacy_full_receipt_boolean_cannot_authorize_new_Flash_clone(self):
        from unittest.mock import patch
        c=component('contracts');draft='/cpu_METADATA_ONLY/approved'
        base={'schema':'domino-full-preflight-v1','classification':'full_export_and_bytes_verified','gpu_eligible':True,'cuda_initialized':False,'model_constructed':False,'LLM_constructed':False,'forward_executed':False,'all40_values_bitwise_equal':True,'target_weight_bodies_read':True,'dependency_binary_bodies_rehashed':True,'real_checkpoint_contents_read':True,'own_sources':{},'target_model_path':'/home/lyh/qwen3vl8b_eval_20261027/artifacts/target','draft_model_path':draft,'full_artifacts':{draft+'/'+n:'a'*64 for n in ('config.json','model.safetensors','export_receipt.json')}}
        with patch.object(c,'own_source_hashes',return_value={}):
            with self.assertRaises(RuntimeError):c.validate_full_receipt(base)
            for key,value in [('real_checkpoint_contents_read',False),('forward_executed',True),('LLM_constructed',True)]:
                r=copy.deepcopy(base);r[key]=value
                with self.subTest(key=key),self.assertRaises(RuntimeError):c.validate_full_receipt(r)

    def test_export_layout_rejects_unapproved_loader_index_or_extra_weights(self):
        import tempfile
        c=component('contracts');self.assertTrue(callable(getattr(c,'validate_export_layout',None)))
        with tempfile.TemporaryDirectory(prefix='CPU_METADATA_ONLY_layout_') as td:
            root=Path(td)
            for name in ('config.json','model.safetensors','export_receipt.json'):(root/name).write_text('tiny metadata stand-in; not real weights')
            c.validate_export_layout(root)
            for extra in ('model.safetensors.index.json','other.safetensors'):
                (root/extra).write_text('metadata-only')
                with self.assertRaises(ValueError):c.validate_export_layout(root)
                (root/extra).unlink()

    def test_initial_receipt_never_authorizes_hardware(self):
        c=component('contracts')
        with self.assertRaises(RuntimeError):c.validate_full_receipt({'classification':'logic_only','gpu_eligible':False})

TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'Torch unavailable locally; run remote CPU-only suite')
class MathTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;torch.manual_seed(73)
        self.h=torch.randn(8,4);self.ids=torch.tensor([2,9,9,9,9,9,9,9]);self.e=torch.randn(11,4);self.head=torch.randn(11,4)
        self.wi=torch.randn(9,4);self.wh=torch.randn(9,3);self.p0=torch.randn(2,7);self.p2=torch.randn(11,2)
    def run_ours(self,h=None,ids=None):
        l=component('domino_logic');t=self.t;calls=[]
        def logits(x):calls.append(tuple(x.shape));return t.nn.functional.linear(x,self.head)
        out=l.sample_domino(self.h if h is None else h,self.ids if ids is None else ids,compute_logits=logits,embed_tokens=lambda ids:t.nn.functional.embedding(ids,self.e),weight_ih=self.wi,weight_hh=self.wh,projector_in=self.p0,projector_out=self.p2)
        return out,calls
    def reference(self,return_trace=False):
        t=self.t
        p=Path('/home/lyh/VelaDraft-a6000-v2/vlm_dflash/dflash/draft_model.py')
        if not p.exists():self.skipTest('exact canonical source is remote')
        tree=ast.parse(p.read_text());trace=[]
        def sample(x):trace.append(x.clone());return x.argmax(-1)
        ns={'torch':t,'nn':t.nn,'F':t.nn.functional,'sample':sample}
        for name in ('_domino_gru_step','_sample_draft_tokens'):
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
        owner=NS(block_size=8,projector_type='domino',pure_draft_prefix_len=1,shift_label=True,gru_hidden_dim=3,prefix_gru=NS(weight_ih_l0=self.wi,weight_hh_l0=self.wh),embed_proj=lambda x:t.nn.functional.linear(t.nn.functional.silu(t.nn.functional.linear(x,self.p0)),self.p2))
        owner._domino_gru_step=lambda emb,state:ns['_domino_gru_step'](owner,emb,state)
        target=NS(lm_head=lambda x:t.nn.functional.linear(x,self.head),get_input_embeddings=lambda:lambda ids:t.nn.functional.embedding(ids,self.e))
        out=ns['_sample_draft_tokens'](owner,target,self.h[None],self.ids[None])
        return (out,trace) if return_trace else out
    def test_exact_canonical_gru_step_and_full_walk(self):
        out,calls=self.run_ours();self.assertEqual(calls,[(8,4)])
        self.assertTrue(self.t.equal(out,self.reference()))
    def test_round_reset_and_no_input_mutation(self):
        original_h=self.h.clone();original_ids=self.ids.clone()
        out,_=self.run_ours();again,_=self.run_ours()
        self.assertTrue(self.t.equal(out,again));self.assertTrue(self.t.equal(original_h,self.h));self.assertTrue(self.t.equal(original_ids,self.ids))
    def test_first_proposal_uses_row0_base_but_updates_gru(self):
        t=self.t;out,_=self.run_ours();base=t.nn.functional.linear(self.h,self.head)
        self.assertEqual(out[0,0].item(),base[0].argmax().item())
        # Reference compares all later corrected steps, including the first GRU state.
        self.assertTrue(t.equal(out,self.reference()))
        changed=self.h.clone();changed[7].fill_(1e5)
        self.assertTrue(t.equal(out,self.run_ours(h=changed)[0]))
    def test_official_int32_inputs_preserve_original_Long_proposal_dtype(self):
        out,_=self.run_ours(ids=self.ids.to(self.t.int32))
        self.assertEqual(out.dtype,self.t.int64)
        self.assertTrue(self.t.equal(out,self.reference()))

    def test_each_corrected_logit_matches_canonical_BF16_and_FP32(self):
        from unittest.mock import patch
        t=self.t
        for dtype in (t.float32,t.bfloat16):
            for name in ('h','e','head','wi','wh','p0','p2'):setattr(self,name,getattr(self,name).to(dtype))
            trace=[];original=t.Tensor.argmax
            def argmax(x,*args,**kwargs):trace.append(x.clone());return original(x,*args,**kwargs)
            with patch.object(t.Tensor,'argmax',argmax):out,_=self.run_ours()
            expected,expected_trace=self.reference(return_trace=True)
            self.assertTrue(t.equal(out,expected));self.assertEqual(len(trace),7)
            for got,want in zip(trace,expected_trace):self.assertTrue(t.equal(got,want),str(dtype))

    def test_sampler_rejects_wrong_query_and_anchor_layouts(self):
        for h,ids in [(self.h[:7],self.ids),(self.h,self.ids[:7]),(self.h[None],self.ids)]:
            with self.subTest(shape=h.shape),self.assertRaises(ValueError):self.run_ours(h,ids)
    def test_actual_predecessor_chain_not_masks_or_parallel_base(self):
        out,_=self.run_ours();self.assertTrue(self.t.equal(out,self.reference()))
        # At least one selected predecessor differs from incoming mask9.
        self.assertTrue(bool((out[:,:-1]!=9).any()))

if __name__=='__main__':unittest.main(verbosity=2)
