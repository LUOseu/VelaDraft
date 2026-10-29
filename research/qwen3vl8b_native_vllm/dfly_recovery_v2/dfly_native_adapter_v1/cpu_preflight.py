"""Frozen canonical/actual42 CPU and public configuration receipt, no GPU claims."""
import argparse
import json
import os
from pathlib import Path
import time
import unittest
from . import contracts as c


def run(export_dir,output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight requires CUDA hidden')
    output=Path(output);export_dir=Path(export_dir).resolve()
    c.require(not output.exists(),'CPU receipt destination must be new')
    c.require(export_dir==c.EXPORT,'Reviewed CPU tests bind exact independent v1 export')
    c.validate_export_directory(export_dir,expected_path=c.EXPORT)
    frozen=c.own_sources();bindings=c.source_checks()
    from .install import verify_official_interfaces,install
    interfaces=verify_official_interfaces()
    tests=unittest.defaultTestLoader.loadTestsFromNames([
        'dfly_native_adapter_v1.test_logic','dfly_native_adapter_v1.test_integration','dfly_native_adapter_v1.test_actual_weights'])
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    c.require(result.wasSuccessful() and not result.skipped,'Every CPU/source/actual-weight test must run and pass')
    import torch
    from safetensors.torch import load_file
    from .export_checkpoint import verify_tensor_values
    from .test_actual_weights import actual_weight_functional_proof
    from .public_config import materialize_public_config
    c.require(c.sha(c.CHECKPOINT)==c.CHECKPOINT_SHA,'Actual canonical checkpoint SHA changed')
    payload=torch.load(c.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    restored=load_file(str(export_dir/'model.safetensors'),device='cpu')
    c.validate_state(payload['model']);c.validate_state(restored)
    tensors=verify_tensor_values(payload['model'],restored)
    saved=json.loads((export_dir/'export_receipt.json').read_text())
    c.require(saved['config_sha256']==c.sha(export_dir/'config.json') and saved['weights_sha256']==c.sha(export_dir/'model.safetensors')
        and saved['checkpoint_sha256']==c.CHECKPOINT_SHA,'Saved export receipt differs from actual bytes')
    c.validate_config(json.loads((export_dir/'config.json').read_text()))
    component=actual_weight_functional_proof();public_config,_=materialize_public_config(export_dir)
    from vllm.v1.sample import rejection_sampler
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from .spatial import compile_mapping
    before=(rejection_sampler.rejection_sample,DFlashQwen3Model.__init__,DFlashQwen3Model.forward,
        DFlashQwen3ForCausalLM.compute_logits,DFlashProposer.propose,DFlashProposer._sample_draft_tokens,
        compile_mapping(DFlashQwen3Model))
    c.require(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    c.require(frozen==c.own_sources() and bindings==c.source_checks(),'Source changed during CPU preflight')
    artifacts={str(export_dir/name):c.sha(export_dir/name) for name in ('config.json','model.safetensors','export_receipt.json')}
    receipt={'schema':'dfly-native-cpu-v1','status':'complete','created_unix':time.time(),
        'CPU_evidence_complete':True,'GPU_eligible':False,'GPU_numerical_or_performance_gate_passed':False,
        'cuda_initialized':False,'serving_model_constructed':False,'full_backbone_forward_executed':False,
        'actual_official_model_load_executed':False,'target_weight_bytes_verified':False,
        'target_or_head_forward_executed':False,'timing_or_speedup_measured':False,
        'checkpoint_path':c.CHECKPOINT,'checkpoint_sha256':c.CHECKPOINT_SHA,'algorithm':'dfly','step':75000,
        'export_dir':str(export_dir),'tensor_count':42,'all42_values_bitwise_equal':True,'tensors':tensors,
        'export_artifacts':artifacts,'own_source_sha256':frozen,'official_source_bindings':bindings,
        'source_provenance':json.loads((c.HERE/'SOURCE_PROVENANCE.json').read_text()),
        'CPU_test_count':result.testsRun,'CPU_failures':0,'CPU_errors':0,'CPU_skips':0,
        'official_interfaces':interfaces,'actual_trained_component_proof':component,
        'actual_public_config':public_config,'raw_context_abi':'ordered_target_layers_12288_v1',
        'query_hidden_size':4096,'context_feature_width':12288,'original_FC_preserved':True,
        'separate_target_KV_created':False,'model_raw_context_sidechannel':False,
        'CPU_official_linear_prerequisite':'real single-process CPU Gloo TP1; CUDA _lazy_init forbidden',
        'proposal_sampler':'slots1..7 with original five-parameter hidden correction and actual selected-token feedback',
        'production_receipt_binding':'explicit DFLY_NATIVE_CPU_RECEIPT_SHA256 required; no CPU_PROOF bypass',
        'production_path_closure':'spec.model and draft ModelConfig.model/model_weights equal exact verified three-regular-file export',
        'installer_process_local_only':True,'official_target_head_verifier_and_class_compile_objects_unchanged':True,
        'limitations':['No CUDA/serving model construction/full official weight load or original target/head forward has run.',
            'Actual extra-module CPU checks use synthetic raw/query features with real trained tensors.',
            'Context norm/key norm and official MRoPE/cache order are tested on CPU fixtures; real GPU caches remain unverified.',
            'EngineArgs constructor parsed; create_engine_config/LLM construction deferred to GPU stage.',
            'Normal BF16 official fused residual/head arithmetic may differ from canonical eager arithmetic.',
            'Root specification and quality review, actual loading/graph capture, standard GPU validation then timing remain required.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    old={k:os.environ.get(k) for k in ('DFLY_NATIVE_CPU_RECEIPT','DFLY_NATIVE_CPU_RECEIPT_SHA256')}
    os.environ['DFLY_NATIVE_CPU_RECEIPT']=str(output.resolve());os.environ['DFLY_NATIVE_CPU_RECEIPT_SHA256']=c.sha(output)
    try:
        c.runtime_receipt();install();install()
        after=(rejection_sampler.rejection_sample,DFlashQwen3Model.__init__,DFlashQwen3Model.forward,
            DFlashQwen3ForCausalLM.compute_logits,DFlashProposer.propose,DFlashProposer._sample_draft_tokens,
            compile_mapping(DFlashQwen3Model))
        c.require(all(a is b for a,b in zip(before,after)),'Installer changed official class/head/sampler/verifier/compile object')
        c.require(not torch.cuda.is_initialized(),'Installation initialized CUDA')
    except BaseException as error:
        failure=output.with_name(output.stem+'_FAILED_INSTALL.json')
        with failure.open('x') as stream:json.dump({'status':'failed_install','error':repr(error),
            'retained_receipt_path':str(output.resolve()),'retained_receipt_sha256':c.sha(output),
            'CPU_tests_passed_but_install_not_completed':True,'GPU_eligible':False},stream,indent=2)
        raise
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    print(json.dumps({'status':'complete','CPU_test_count':result.testsRun,'all42_values_bitwise_equal':True,
        'cuda_initialized':False,'GPU_numerical_or_performance_gate_passed':False,'receipt_sha256':c.sha(output)},indent=2))
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export-dir',type=Path,default=c.HERE/'hf_step75000_v1')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.export_dir,args.output)
