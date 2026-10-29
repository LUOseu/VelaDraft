"""CPU artifact audit for local shared-block JetSpec only, not upstream tree.

Reads frozen sources/weights and writes receipts only under jetspec_adapter/.
No target/draft/LLM construction, model forward, CUDA initialization or kernels.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from safetensors.torch import load_file
import export_jetspec as e
import official_jetspec_adapter as j

ROOT=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,payload):
    (ROOT/name).write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    assert not torch.cuda.is_initialized()
    predecessor=json.loads((ROOT/'evidence_v1/PRESERVED_V1_MANIFEST.json').read_text())
    preserved={name:digest(ROOT/'evidence_v1'/name) for name in predecessor['files_sha256']}
    assert preserved==predecessor['files_sha256'],'Preserved v1 source/evidence changed'
    write('PRESERVED_V1_VERIFICATION.json',{'all_unchanged':True,'checked_files':len(preserved),
        'revision_sha256':predecessor['revision_sha256']})
    before=json.loads((ROOT/'REMOTE_FROZEN_BEFORE.json').read_text())
    after={name:digest(name) for name in before}
    changed=[name for name in before if before[name]!=after[name]]
    write('REMOTE_FROZEN_AFTER.json',after)
    write('REMOTE_FROZEN_VERIFICATION.json',{'scope':j.SCOPE,'checked_files':len(before),
        'all_unchanged':not changed,'changed':changed,
        'includes':'all captured target files including full weight-shard SHA256; old spatial/B887/benchmark files and old DFlash export; JetSpec source checkpoint/config/contract'})
    assert not changed,changed
    expected=json.loads((ROOT/'EXPECTED_SOURCE_CHECKPOINTS.json').read_text())
    source_after={name:digest(name) for name in expected}
    assert source_after==expected,'Existing source checkpoint/config file changed'
    write('SOURCE_CHECKPOINT_VERIFICATION.json',{'all_unchanged':True,'checked_files':len(expected),'sha256':source_after})
    bindings=j._check_frozen_sources()
    runtime_bindings=j._check_runtime_sources()
    checkpoint=e.SOURCE/'train/ckpt_final.pt'
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    e.validate_payload(payload,json.loads((e.SOURCE/'config.json').read_text()),json.loads((e.SOURCE/'contract.json').read_text()))
    export=ROOT/'hf_export_local_block_parent37500'
    config=json.loads((export/'config.json').read_text())
    assert config==json.loads(json.dumps(e.exported_config(payload)))
    j.validate_config(config)
    restored=load_file(str(export/'model.safetensors'),device='cpu')
    e.validate_state(restored)
    tensor_hashes={}
    for name,value in restored.items():
        assert torch.equal(value.view(torch.uint8),payload['model'][name].view(torch.uint8)),name
        tensor_hashes[name]=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    receipt=json.loads((export/'export_receipt.json').read_text())
    assert tensor_hashes=={name:value['sha256'] for name,value in receipt['tensors'].items()}
    assert digest(export/'config.json')==receipt['config_sha256']
    assert digest(export/'model.safetensors')==receipt['model_safetensors_sha256']
    assert not torch.cuda.is_initialized()
    versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers','safetensors')}
    versions['torch_runtime']=str(torch.__version__)
    versions['torch_cuda_build']=torch.version.cuda
    write('RUNTIME_VERSIONS.json',{'versions':versions,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'cuda_initialized':False,'model_construction':False,'model_forward':False})
    core=['official_jetspec_adapter.py','export_jetspec.py','verify_artifacts.py','test_jetspec_adapter.py',
        'bootstrap/sitecustomize.py','source_bindings.json','checkpoint_metadata.json','exported_config.json',
        'EXPECTED_SOURCE_CHECKPOINTS.json','source_evidence/draft_model.py','source_evidence/gpu_model_runner.py',
        'README.md','B887_CPU_ORACLE_NOTE.md','B887_CPU_ORACLE_NOTE.json',
        'CONSTRUCTOR_CONTRACT_REVIEW_FIX_V2.json']
    owned={name:digest(ROOT/name) for name in core}
    revision=hashlib.sha256(json.dumps(owned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write('SOURCE_MANIFEST.json',{'schema':'jetspec-local-shared-block-source-manifest-v2',
        'scope':j.SCOPE,'date_utc':datetime.now(timezone.utc).isoformat(),'revision_sha256':revision,
        'predecessor_revision_sha256':predecessor['revision_sha256'],
        'review_change':'I1 constructor context/feature/full-vocabulary/unit-logit-scale guards; no numerical/runtime hook/export change',
        'git_repository':False,'owned_source_sha256':owned,'frozen_dependency_sha256':bindings,
        'official_runtime_source_sha256':runtime_bindings,'runtime_versions':versions,
        'source_checkpoint_sha256':j.CHECKPOINT_SHA256,'export_config_sha256':receipt['config_sha256'],
        'export_weights_sha256':receipt['model_safetensors_sha256'],'model_or_operator_forward_changed':False})
    write('EXPORT_REVERIFICATION.json',{'scope':j.SCOPE,'revision_sha256':revision,
        'final_source_validation':True,'all_36_tensors_bitwise_equal':True,'tensor_count':len(restored),
        'config_matches_final_exporter':True,'config_sha256':receipt['config_sha256'],
        'weights_sha256':receipt['model_safetensors_sha256'],'tensor_sha256':tensor_hashes,
        'cuda_initialized':False,'model_construction':False,'model_forward':False})
    print(json.dumps({'scope':j.SCOPE,'frozen_files_checked':len(before),'all_frozen_unchanged':True,
        'preserved_v1_files_unchanged':len(preserved),
        'source_checkpoint_and_config_files_checked':len(expected),'tensor_count':len(restored),
        'all_36_bitwise_equal':True,'cuda_initialized':False,'revision_sha256':revision,'versions':versions},indent=2))
