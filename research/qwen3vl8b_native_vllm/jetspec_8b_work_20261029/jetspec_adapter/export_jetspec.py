"""Mechanical CPU export, local shared-block JetSpec only, not upstream tree.

Never constructs a target/draft model. All 36 tensors keep names and bytes.
The official public DFlash loader retains external target embedding/head sharing.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from safetensors.torch import save_file,load_file
import official_jetspec_adapter as j

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source_run'
TARGET=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/target')
CONFIG_SHA='4f39862bb3ddec2374b8cb81730befd4e0b3b73321a5ae523f01b899eef7aacc'
CONTRACT_SHA='9e49ddbcab982d3b556448d3944ef2fc9e209d00a83250485a4cb6ae2c958712'
TARGET_FINGERPRINT='3925a38557fb5f5f21e0ee2b003a9bafe0ac452d8aac2e6ee702e2288b31a1ac'


def sha256(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def expected_schema()->dict[str,tuple]:
    result={}
    layer={'self_attn.q_proj.weight':(4096,4096),'self_attn.k_proj.weight':(1024,4096),
        'self_attn.v_proj.weight':(1024,4096),'self_attn.o_proj.weight':(4096,4096),
        'self_attn.q_norm.weight':(128,),'self_attn.k_norm.weight':(128,),
        'mlp.gate_proj.weight':(12288,4096),'mlp.up_proj.weight':(12288,4096),
        'mlp.down_proj.weight':(4096,12288),'input_layernorm.weight':(4096,),
        'post_attention_layernorm.weight':(4096,)}
    for i in range(3):result.update({f'layers.{i}.{n}':shape for n,shape in layer.items()})
    result.update({'norm.weight':(4096,),'fc.weight':(4096,12288),'hidden_norm.weight':(4096,)})
    return result


def validate_state(state:dict)->None:
    schema=expected_schema()
    if set(state)!=set(schema):
        raise ValueError(f'Expected all 36 tensors exactly; extra={sorted(set(state)-set(schema))}, missing={sorted(set(schema)-set(state))}')
    for name,shape in schema.items():
        value=state[name]
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=torch.bfloat16:
            raise ValueError(f'Unsupported tensor dtype/shape: {name}')


def validate_payload(payload:dict,run:dict,contract:dict)->None:
    metadata=json.loads((ROOT/'checkpoint_metadata.json').read_text())
    if set(payload)!=set(metadata)|{'model'}:
        raise ValueError('Unexpected/missing checkpoint payload paths; nothing may be silently dropped')
    actual={k:v for k,v in payload.items() if k!='model'}
    # JSON normalization accounts only for integer id2label keys, not flags.
    if json.loads(json.dumps(actual))!=metadata:
        raise ValueError('Checkpoint metadata/config differs from inspected final checkpoint')
    if payload['algorithm']!='jetspec' or payload['step']!=75000 or payload['head']!='target_lm_head':
        raise ValueError('Expected actual final local-block JetSpec checkpoint with external target head')
    cfg=payload['draft_config'];d=cfg['dflash_config']
    j._bounded().validate_draft_config(cfg,require_marker=False)
    if set(d)!=j._BASE_DFLASH_FIELDS:raise ValueError('Unsupported extra/missing learned/inference config path')
    for key,value in {'inference_in_block_causal':True,'visual_num_query_tokens':0,**j._DEFAULTS}.items():
        actual=d.get(key,j._DEFAULTS.get(key))
        if type(actual) is not type(value) or actual!=value:raise ValueError(f'Unsupported inference flag {key}')
    if cfg.get('intermediate_size')!=12288 or cfg.get('num_target_layers')!=36 or cfg.get('rms_norm_eps')!=1e-6:
        raise ValueError('Unexpected JetSpec backbone dimensions/normalization')
    if (run['algorithm']!={'name':'jetspec','jetspec':{'kd_temperature':1.0,'kd_chunk_size':1024}}
            or run['drafter'].get('sample_from_anchor',False) is not False
            or run['drafter']['loss_decay_gamma'] is not None
            or run['drafter']['block_size']!=8
            or contract['layers']['capture_layer_ids']!=[1,17,33]):
        raise ValueError('Actual run algorithm/anchor/training config mismatch')
    if payload['jetspec']!={'kd_temperature':1.0,'kd_chunk_size':1024,'causal_in_block':True}:
        raise ValueError('JetSpec causal training configuration mismatch')
    if (contract['target']['model_path']!='/data/Qwen3-VL-8B-Instruct' or contract['target']['model_fingerprint']!=TARGET_FINGERPRINT
            or contract['head']['tie_word_embeddings'] is not False
            or contract['layers']['capture_layer_ids']!=[1,17,33]
            or contract['rope']['sections']!=[24,20,20] or contract['rope']['interleaved'] is not True):
        raise ValueError('Target/head/layer/spatial contract mismatch')
    validate_state(payload['model'])


def exported_config(payload:dict)->dict:
    cfg=copy.deepcopy(payload['draft_config'])
    cfg['architectures']=['DFlashDraftModel']
    cfg['dflash_config'][j.SPATIAL_MARKER]=True
    cfg['dflash_config'][j.MARKER]=True
    cfg['rope_parameters']['mrope_section']=[24,20,20]
    cfg['rope_parameters']['mrope_interleaved']=True
    cfg['algorithm']='jetspec'
    cfg['jetspec_inference_contract']=copy.deepcopy(j.INFERENCE_CONTRACT)
    cfg['jetspec_provenance']={'source_sha256':j.CHECKPOINT_SHA256,'step':75000,'scope':j.SCOPE,
        'algorithm':'jetspec','original_architectures':payload['draft_config']['architectures'],
        'source_run_config_sha256':CONFIG_SHA,'source_contract_sha256':CONTRACT_SHA,
        'registry_dispatch_disclosure':'DFlashDraftModel requires the separate JetSpec causal adapter; not native JetSpec'}
    j.validate_config(cfg)
    return cfg


def export(output:Path)->dict:
    if torch.cuda.is_initialized():raise RuntimeError('Exporter requires uninitialized CUDA')
    if output.resolve().parent!=ROOT or output.name!='hf_export_step75000_v1':
        raise ValueError('Export must use the separate new JetSpec adapter destination')
    if output.exists():raise FileExistsError('Export refuses to overwrite any existing destination')
    checkpoint=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/jetspec/ckpt_final.pt')
    for path,digest in ((checkpoint,j.CHECKPOINT_SHA256),(SOURCE/'config.json',CONFIG_SHA),(SOURCE/'contract.json',CONTRACT_SHA)):
        if sha256(path)!=digest:raise ValueError(f'Frozen source hash mismatch: {path}')
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    run=json.loads((SOURCE/'config.json').read_text());contract=json.loads((SOURCE/'contract.json').read_text())
    validate_payload(payload,run,contract)
    target_config=json.loads((TARGET/'config.json').read_text())
    index=json.loads((TARGET/'model.safetensors.index.json').read_text())
    for name,digest in contract['target']['model_files'].items():
        if sha256(TARGET/name)!=digest:raise ValueError(f'Target contract hash changed: {name}')
    if target_config['tie_word_embeddings'] is not False or 'model.language_model.embed_tokens.weight' not in index['weight_map'] or 'lm_head.weight' not in index['weight_map']:
        raise ValueError('Target does not match the expected separate embedding and lm_head')
    cfg=exported_config(payload)
    from vllm.model_executor.models import ModelRegistry
    if 'DFlashDraftModel' not in ModelRegistry.get_supported_archs():raise RuntimeError('Public registry dispatch key is unavailable')
    weights=payload['model']
    if any(t.device.type!='cpu' for t in weights.values()):raise ValueError('Only CPU checkpoint tensors may be exported')
    tensor_bytes=sum(t.numel()*t.element_size() for t in weights.values())
    free=shutil.disk_usage(ROOT).free
    # At least data bytes plus 100MiB for config/header/receipts; recorded before writes.
    if free<tensor_bytes+100*1024*1024:raise RuntimeError('Insufficient disk space for a separate JetSpec export')
    output.mkdir()
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    save_file(weights,str(output/'model.safetensors'),metadata={'format':'pt','algorithm':'jetspec','scope':'local shared-block only'})
    restored=load_file(str(output/'model.safetensors'),device='cpu')
    if set(restored)!=set(weights):raise AssertionError('Mechanical export changed tensor names')
    tensors={}
    for name,value in weights.items():
        other=restored[name]
        if value.dtype!=other.dtype or value.shape!=other.shape or not torch.equal(value.view(torch.uint8),other.view(torch.uint8)):
            raise AssertionError(f'Export tensor changed: {name}')
        tensors[name]={'shape':list(value.shape),'dtype':str(value.dtype),'bitwise_equal':True,
            'sha256':hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if torch.cuda.is_initialized():raise RuntimeError('Exporter initialized CUDA')
    receipt={'schema':'jetspec-local-shared-block-export-v1','scope':j.SCOPE,
        'source_checkpoint':str(checkpoint),'source_sha256':j.CHECKPOINT_SHA256,
        'source_run_config_sha256':CONFIG_SHA,'source_contract_sha256':CONTRACT_SHA,
        'output':str(output),'config_sha256':sha256(output/'config.json'),
        'model_safetensors_sha256':sha256(output/'model.safetensors'),
        'tensor_count':len(tensors),'tensor_data_bytes':tensor_bytes,'all_36_tensors_bitwise_equal':True,
        'disk_free_before_export_bytes':free,'trained_block_size':8,'runtime_query_sizes_are_separate':[2,4,8],
        'registered_architecture':'DFlashDraftModel','checked_installed_registry':True,
        'algorithm':'jetspec','inference_contract':j.INFERENCE_CONTRACT,
        'target_embedding_head':'external actual target embedding and lm_head; no checkpoint head was dropped',
        'config_changes':['architectures','dflash_config.'+j.SPATIAL_MARKER,'dflash_config.'+j.MARKER,
            'rope_parameters.mrope_section','rope_parameters.mrope_interleaved','algorithm',
            'jetspec_inference_contract','jetspec_provenance'],
        'numerical_weights_changed':False,'cuda_initialized':False,'model_constructed':False,'model_forward':False,
        'tensors':tensors}
    (output/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'exported_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (ROOT/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'hf_export_step75000_v1')
    args=p.parse_args();receipt=export(args.output)
    print(json.dumps({k:v for k,v in receipt.items() if k!='tensors'},indent=2))
