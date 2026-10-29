"""Complete CPU/source/actual51-weight receipt, explicitly without GPU claims."""
import argparse
import json
import os
from pathlib import Path
import time
import unittest

from . import contracts as c


def run(export_dir,output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight requires CUDA hidden')
    output=Path(output);export_dir=c.validate_export_layout(export_dir)
    c.require(not output.exists(),'CPU receipt destination must be new')
    source_freeze=c.own_sources();bindings=c.source_checks()
    from .install import verify_official_interfaces,install
    interfaces=verify_official_interfaces()
    tests=unittest.defaultTestLoader.loadTestsFromNames([
        'dflash2_8b_native_adapter.test_logic','dflash2_8b_native_adapter.test_integration',
        'dflash2_8b_native_adapter.test_actual_weights','dflash2_8b_native_adapter.test_weight_selection'])
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    c.require(result.wasSuccessful() and not result.skipped,'All actual CPU tests must execute and pass')
    import torch
    from safetensors.torch import load_file
    from .export_checkpoint import verify_tensor_values
    from .test_actual_weights import actual_weight_functional_proof
    from .public_config import materialize_public_config
    c.require(c.sha(c.CHECKPOINT)==c.CHECKPOINT_SHA,'Canonical actual checkpoint SHA changed')
    payload=torch.load(c.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
    state=load_file(str(export_dir/'model.safetensors'),device='cpu')
    c.validate_state(state);values=verify_tensor_values(payload['model'],state)
    saved=json.loads((export_dir/'export_receipt.json').read_text())
    c.require(saved['config_sha256']==c.sha(export_dir/'config.json')
        and saved['weights_sha256']==c.sha(export_dir/'model.safetensors')
        and saved['checkpoint_sha256']==c.CHECKPOINT_SHA,'Saved export receipt differs from actual bytes')
    c.validate_config(json.loads((export_dir/'config.json').read_text()))
    component_proof=actual_weight_functional_proof()
    public_config,envelope=materialize_public_config(export_dir)
    from vllm.v1.sample import rejection_sampler
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model,DFlashQwen3ForCausalLM
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from .spatial import compile_mapping
    before=(rejection_sampler.rejection_sample,DFlashQwen3Model.__init__,DFlashQwen3Model.forward,
        DFlashQwen3ForCausalLM.compute_logits,DFlashProposer.propose,compile_mapping(DFlashQwen3Model))
    install();install()
    after=(rejection_sampler.rejection_sample,DFlashQwen3Model.__init__,DFlashQwen3Model.forward,
        DFlashQwen3ForCausalLM.compute_logits,DFlashProposer.propose,compile_mapping(DFlashQwen3Model))
    c.require(all(a is b for a,b in zip(before,after)),'Installer changed official class/head/rejection or compile objects')
    c.require(not torch.cuda.is_initialized(),'CPU preflight initialized CUDA')
    c.require(source_freeze==c.own_sources() and bindings==c.source_checks(),'Sources changed during CPU preflight')
    artifacts={str(export_dir/name):c.sha(export_dir/name) for name in ('config.json','model.safetensors','export_receipt.json')}
    receipt={'schema':c.RECEIPT_SCHEMA,'status':'complete','created_unix':time.time(),
        'CPU_evidence_complete':True,'GPU_eligible':False,'GPU_numerical_or_performance_gate_passed':False,
        'cuda_initialized':False,'serving_model_constructed':False,'full_backbone_forward_executed':False,
        'actual_official_model_load_executed':False,'target_weight_bytes_verified':False,
        'target_or_head_forward_executed':False,'timing_or_speedup_measured':False,
        'checkpoint_path':c.CHECKPOINT,'checkpoint_sha256':c.CHECKPOINT_SHA,'algorithm':'dflash2','step':75000,
        'export_dir':str(export_dir),'tensor_count':51,'all51_values_bitwise_equal':True,'tensors':values,
        'export_artifacts':artifacts,'own_source_sha256':source_freeze,'official_source_bindings':bindings,
        'source_provenance':json.loads((c.HERE/'SOURCE_PROVENANCE.json').read_text()),
        'CPU_test_count':result.testsRun,'CPU_failures':0,'CPU_errors':0,'CPU_skips':0,
        'official_interfaces':interfaces,'actual_trained_component_proof':component_proof,
        'actual_public_config':public_config,'installer_process_local_only':True,
        'official_target_head_verifier_and_class_compile_objects_unchanged':True,
        'limitations':['No CUDA/model construction/full official weight load or target/head forward has run.',
            'Canonical selector batch>1 assignment is unsupported upstream; reviewed runtime is max_num_seqs1.',
            'Actual trained conv/selector CPU checks use synthetic inputs, not benchmark generations.',
            'EngineArgs constructor parsed; create_engine_config/LLM construction deferred to GPU stage.',
            'Standard BF16 fused residual/head arithmetic can differ from canonical eager execution.',
            'Root specification and quality review plus GPU numerical/standard validation remain required.']}
    receipt['actual_public_loader_selection']=c.validate_draft_loader_selection(envelope,receipt)
    receipt['export_directory_exact_regular_files']=sorted(c.EXPORT_FILES)
    receipt['production_receipt_requires_explicit_expected_sha256']=True
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    previous=os.environ.get('DFLASH2_NATIVE_CPU_RECEIPT')
    previous_sha=os.environ.get('DFLASH2_NATIVE_CPU_RECEIPT_SHA256')
    os.environ['DFLASH2_NATIVE_CPU_RECEIPT']=str(output.resolve())
    os.environ['DFLASH2_NATIVE_CPU_RECEIPT_SHA256']=c.sha(output)
    try:c.runtime_receipt()
    finally:
        if previous is None:os.environ.pop('DFLASH2_NATIVE_CPU_RECEIPT',None)
        else:os.environ['DFLASH2_NATIVE_CPU_RECEIPT']=previous
        if previous_sha is None:os.environ.pop('DFLASH2_NATIVE_CPU_RECEIPT_SHA256',None)
        else:os.environ['DFLASH2_NATIVE_CPU_RECEIPT_SHA256']=previous_sha
    print(json.dumps({'status':'complete','CPU_test_count':result.testsRun,'all51_values_bitwise_equal':True,
        'cuda_initialized':False,'GPU_numerical_or_performance_gate_passed':False,'receipt_sha256':c.sha(output)},indent=2))
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export-dir',type=Path,default=c.HERE/'hf_step75000_v1')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.export_dir,args.output)
