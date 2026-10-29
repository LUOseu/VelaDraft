"""Small real-file contract fixtures; these never authorize a production model."""
from __future__ import annotations
import copy,hashlib,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from test_domino import component

class RuntimeApprovalTests(unittest.TestCase):
    def setUp(self):
        self.c=component('contracts');self.tmp=tempfile.TemporaryDirectory(prefix='domino_APPROVAL_UNIT_FIXTURE_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.draft=self.root/'export';self.draft.mkdir()
        self.schema=self.c.expected_schema()
        self.sources={'contracts.py':'1'*64}
        self.patches=[patch.object(self.c,'ROOT',self.root),patch.object(self.c,'own_source_hashes',return_value=self.sources)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.assertTrue(callable(getattr(self.c,'runtime_contract',None)),'runtime approval contract is not implemented')
        (self.root/'TENSOR_SCHEMA.json').write_text(json.dumps(self.schema))
        target=self.root/'target';target.mkdir()
        target_files=('model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors','model.safetensors.index.json')
        for name in target_files:(target/name).write_bytes(('unit fixture target '+name).encode())
        checkpoint=self.root/'checkpoint.pt';checkpoint.write_bytes(b'unit fixture checkpoint')
        binary=self.root/'dependency.so';binary.write_bytes(b'unit fixture binary')
        for name in ('config.json','model.safetensors','export_receipt.json'):(self.draft/name).write_bytes(name.encode())
        for attr,value in [('TARGET_ROOT',str(target)),('CHECKPOINT_PATH',str(checkpoint)),('CHECKPOINT_SHA',self.sha(checkpoint))]:
            p=patch.object(self.c,attr,value);p.start();self.addCleanup(p.stop)
        required={str(p):self.sha(p) for p in (*[target/name for name in target_files],checkpoint,binary)}
        (self.root/'source_bindings.json').write_text(json.dumps({'deferred_full_artifacts':required}))
        self.proof_path=self.root/'proof.json';self.approval_path=self.root/'approval.json'
        artifacts={**required,**{str(self.draft/n):self.sha(self.draft/n) for n in ('config.json','model.safetensors','export_receipt.json')}}
        self.proof={'schema':'domino-flash-full-preflight-v1','classification':'full_export_and_bytes_verified','gpu_eligible':False,'requires_independent_review':True,'cpu_only':True,'cuda_initialized':False,'model_constructed':False,'LLM_constructed':False,'forward_executed':False,'all40_values_bitwise_equal':True,'real_checkpoint_contents_read':True,'target_weight_bodies_read':True,'dependency_binary_bodies_rehashed':True,'own_sources':self.sources,'source_root':str(self.root),'receipt_path':str(self.proof_path),'target_model_path':str(target),'draft_model_path':str(self.draft),'full_artifacts':artifacts,'runtime_contract':self.c.runtime_contract(),'checkpoint_path':str(checkpoint),'checkpoint_sha256':self.sha(checkpoint),'checkpoint_step':75000,'tensor_count':40,'tensors':{name:{'shape':shape,'dtype':'torch.bfloat16','sha256':'2'*64} for name,shape in self.schema.items()},'tests_run':40,'tests_passed':40,'tests_skipped':0,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','actual_public_config_and_class_interface_checks':True,'production_CPU_escape_present':False}
        expected_shards=sorted(str(target/name) for name in target_files if name.endswith('.safetensors'))
        self.proof.update(target_weight_layout_validated=True,target_loader_selection={'auto':expected_shards,'safetensors':expected_shards})
        self.write_proof();self.write_approval()

    @staticmethod
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def write_proof(self):
        self.proof_path.write_text(json.dumps(self.proof));self.proof_sha=self.sha(self.proof_path)
    def write_approval(self):
        self.binding={'full_cpu_proof':{'path':str(self.proof_path),'sha256':self.proof_sha},'source_root':str(self.root),'own_sources':self.sources,'runtime_contract':self.c.runtime_contract(),'target_model_path':self.proof['target_model_path'],'draft_model_path':str(self.draft),'full_artifacts':self.proof['full_artifacts']}
        reviews={}
        for role in ('spec','quality'):
            path=self.root/(role+'.json')
            path.write_text(json.dumps({'schema':'domino-flash-independent-review-v1','role':role,'decision':'PASS','reviewer':'unit_fixture_'+role,'independent':True,'blocking_findings':[],'reviewed_binding':self.binding}))
            reviews[role]={'path':str(path),'sha256':self.sha(path)}
        self.approval={'schema':'domino-flash-runtime-approval-v1','classification':'independently_reviewed_runtime_authorization','gpu_eligible':True,'decision':'APPROVE','approved_by':'root','binding':self.binding,'independent_reviews':reviews}
        self.write_approval_bytes()
    def write_approval_bytes(self):
        self.approval_path.write_text(json.dumps(self.approval));self.approval_sha=self.sha(self.approval_path)
    def check(self):return self.c.check_full_header(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)

    def test_separate_approval_accepts_false_CPU_full_proof_only_with_both_reviews(self):
        self.assertEqual(self.check(),self.proof)
        for path,digest in ((None,None),(self.approval_path,None),(None,self.approval_sha)):
            with self.subTest(path=path),self.assertRaises(RuntimeError):self.c.check_full_header(self.proof_path,self.proof_sha,path,digest)
    def test_mutating_CPU_proof_gpu_flag_does_not_authorize(self):
        self.proof['gpu_eligible']=True;self.write_proof();self.write_approval()
        with self.assertRaises(RuntimeError):self.check()
    def test_skipped_CPU_checks_cannot_count_as_full_pass(self):
        self.proof['tests_skipped']=1;self.write_proof();self.write_approval()
        with self.assertRaises(RuntimeError):self.check()
    def test_source_mutation_during_full_rehash_cannot_create_capability(self):
        with patch.object(self.c,'_VERIFIED_APPROVAL',None),patch.object(self.c,'own_source_hashes',side_effect=[self.sources,{}]):
            with self.assertRaises(RuntimeError):self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
    def test_full_proof_requires_exact_target_loader_selection(self):
        self.proof['target_loader_selection']['auto']=[str(Path(self.proof['target_model_path'])/'consolidated.safetensors')]
        self.write_proof();self.write_approval()
        with self.assertRaises(RuntimeError):self.check()
    def test_target_alternate_weights_indexes_and_missing_shards_are_rejected(self):
        self.assertTrue(callable(getattr(self.c,'validate_target_layout',None)),'target layout guard is not implemented')
        target=Path(self.proof['target_model_path']);self.c.validate_target_layout(target,self.proof['full_artifacts'])
        for name in ('consolidated.safetensors','consolidated.00.safetensors','alternate.safetensors','consolidated.safetensors.index.json','pytorch_model.bin','model.pt'):
            (target/name).write_bytes(b'unit fixture alternate')
            with self.subTest(name=name),self.assertRaises(ValueError):self.c.validate_target_layout(target,self.proof['full_artifacts'])
            (target/name).unlink()
        shard=target/'model-00002-of-00004.safetensors';data=shard.read_bytes();shard.unlink()
        with self.assertRaises(ValueError):self.c.validate_target_layout(target,self.proof['full_artifacts'])
        shard.write_bytes(data)
        index=target/'model.safetensors.index.json';data=index.read_bytes();index.unlink()
        with self.assertRaises(ValueError):self.c.validate_target_layout(target,self.proof['full_artifacts'])
        index.write_bytes(data)
    def test_added_target_consolidated_rejected_at_full_gate_and_runtime(self):
        vc=NS(model_config=NS(model=self.proof['target_model_path']),speculative_config=NS(model=str(self.draft),draft_model_config=NS(model=str(self.draft))),load_config=NS(load_format='auto',model_loader_extra_config={}))
        with patch.object(self.c,'_VERIFIED_APPROVAL',None),patch.dict(os.environ,{'DOMINO_APPROVED_PREFLIGHT_SHA256':self.proof_sha,'DOMINO_APPROVED_RUNTIME_SHA256':self.approval_sha}):
            self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
            self.c.require_runtime_binding(vc)
            for name in ('consolidated.safetensors','other.safetensors.index.json','pytorch_model.bin'):
                extra=Path(self.proof['target_model_path'])/name;extra.write_bytes(b'unit fixture alternate')
                with self.subTest(name=name,phase='gate'),self.assertRaises(ValueError):self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
                with self.subTest(name=name,phase='runtime'),self.assertRaises(ValueError):self.c.require_runtime_binding(vc)
                extra.unlink()
    def test_approval_rejects_wrong_method_source_root_proof_path_sha_and_artifacts(self):
        for kind in ('method','source_root','proof_path','proof_sha','own_source','artifact'):
            self.write_approval()
            binding=self.approval['binding']
            if kind=='method':binding['runtime_contract']['method']='eagle'
            elif kind=='source_root':binding['source_root']=str(self.root/'other_clone')
            elif kind=='proof_path':binding['full_cpu_proof']['path']=str(self.root/'copied_proof.json')
            elif kind=='proof_sha':binding['full_cpu_proof']['sha256']='a'*64
            elif kind=='own_source':binding['own_sources']={'contracts.py':'b'*64}
            else:binding['full_artifacts']={}
            self.write_approval_bytes()
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.check()
    def test_review_must_be_actual_pinned_PASS_for_exact_binding(self):
        for kind in ('absent','sha','unreviewed','wrong_role','foreign_source','not_independent','duplicate_reviewer'):
            self.write_approval();ref=self.approval['independent_reviews']['quality'];path=Path(ref['path'])
            review=json.loads(path.read_text())
            if kind=='absent':self.approval['independent_reviews'].pop('quality')
            elif kind=='sha':ref['sha256']='c'*64
            else:
                if kind=='unreviewed':review['decision']='PENDING'
                elif kind=='wrong_role':review['role']='spec'
                elif kind=='foreign_source':review['reviewed_binding']['own_sources']={}
                elif kind=='not_independent':review['independent']=False
                else:review['reviewer']='unit_fixture_spec'
                path.write_text(json.dumps(review));ref['sha256']=self.sha(path)
            self.write_approval_bytes()
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.check()
    def test_full_proof_rejects_stale_sources_methods_paths_and_missing_artifacts(self):
        original=copy.deepcopy(self.proof)
        for kind in ('sources','source_root','method','path','checkpoint','missing_artifact','extra_artifact','tensor','logic_only','tests_failed','real_checkpoint_contents_read','target_weight_bodies_read','dependency_binary_bodies_rehashed','cuda_initialized','model_constructed','LLM_constructed','forward_executed'):
            self.proof=copy.deepcopy(original)
            if kind=='sources':self.proof['own_sources']={}
            elif kind=='source_root':self.proof['source_root']=str(self.root/'old')
            elif kind=='method':self.proof['runtime_contract']['method']='eagle'
            elif kind=='path':self.proof['receipt_path']=str(self.root/'elsewhere.json')
            elif kind=='checkpoint':self.proof['checkpoint_sha256']='0'*64
            elif kind=='missing_artifact':self.proof['full_artifacts'].pop(next(iter(self.proof['full_artifacts'])))
            elif kind=='extra_artifact':self.proof['full_artifacts']['/unexpected']='f'*64
            elif kind=='tensor':self.proof['tensors'].pop(next(iter(self.proof['tensors'])))
            elif kind=='logic_only':self.proof['classification']='logic_only'
            elif kind=='tests_failed':self.proof['tests_passed']=0
            else:self.proof[kind]=not self.proof[kind]
            self.write_proof();self.write_approval()
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):self.check()
    def test_gate_rehashes_current_bytes_rejects_extra_index_and_cached_mutation(self):
        with patch.object(self.c,'_VERIFIED_APPROVAL',None):
            (self.draft/'model.safetensors.index.json').write_text('{}')
            with self.assertRaises(ValueError):self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
            (self.draft/'model.safetensors.index.json').unlink()
            weights=self.draft/'model.safetensors';data=weights.read_bytes();weights.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
            weights.write_bytes(data)
            self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
            Path(self.approval['independent_reviews']['quality']['path']).write_text('{}')
            with self.assertRaises(RuntimeError):self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
    def test_runtime_capability_needs_both_pins_actual_paths_and_real_loader(self):
        vc=NS(model_config=NS(model=self.proof['target_model_path']),speculative_config=NS(model=str(self.draft),draft_model_config=NS(model=str(self.draft))),load_config=NS(load_format='auto',model_loader_extra_config={}))
        with patch.object(self.c,'_VERIFIED_APPROVAL',None),patch.dict(os.environ,{'DOMINO_APPROVED_PREFLIGHT_SHA256':self.proof_sha,'DOMINO_APPROVED_RUNTIME_SHA256':self.approval_sha}):
            with self.assertRaises(RuntimeError):self.c.require_runtime_binding(vc)
            self.c.require_full_gate(self.proof_path,self.proof_sha,self.approval_path,self.approval_sha)
            self.c.require_runtime_binding(vc)
            for kind in ('missing_runtime_pin','missing_proof_pin','wrong_model','dummy_loader','extra_index'):
                bad=copy.deepcopy(vc)
                with patch.dict(os.environ,dict(os.environ)):
                    if kind=='missing_runtime_pin':os.environ.pop('DOMINO_APPROVED_RUNTIME_SHA256')
                    elif kind=='missing_proof_pin':os.environ.pop('DOMINO_APPROVED_PREFLIGHT_SHA256')
                    elif kind=='wrong_model':bad.speculative_config.model=str(self.root/'elsewhere')
                    elif kind=='dummy_loader':bad.load_config.load_format='dummy'
                    else:(self.draft/'model.safetensors.index.json').write_text('{}')
                    with self.subTest(kind=kind),self.assertRaises((RuntimeError,ValueError)):self.c.require_runtime_binding(bad)
                (self.draft/'model.safetensors.index.json').unlink(missing_ok=True)

if __name__=='__main__':unittest.main(verbosity=2)
