"""Fail-closed DFly schema, source, config, and reviewed CPU evidence binding."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import re
import stat

HERE = Path(__file__).resolve().parent
ARCH = 'DFlashDFlyNativeQwen3ForCausalLM'
MARKER = 'dfly_native_adapter_v1'
CHECKPOINT = '/home/lyh/qwen3vl8b_eval_20261027/artifacts/dfly/ckpt_final.pt'
EXPORT = HERE/'hf_step75000_v1'
EXPORT_FILES = frozenset({'config.json','model.safetensors','export_receipt.json'})
CHECKPOINT_SHA = '39ae6a7db1cc8b9920e7703ef0db80149c0eac27ca682864e94ac4e3348ba919'
TARGET = '/home/lyh/qwen3vl8b_eval_20261027/artifacts/target'
SCHEMA = json.loads((HERE/'TENSOR_SCHEMA.json').read_text())


def require(condition, message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def as_dict(config):return config if isinstance(config,dict) else config.to_dict()


def enabled(config):
    cfg=as_dict(config)
    d=cfg.get('dflash_config') or {}
    return cfg.get('architectures')==[ARCH] and isinstance(d,dict) and d.get(MARKER) is True


def near_dfly(config):
    cfg=as_dict(config)
    d=cfg.get('dflash_config') or {}
    return ((isinstance(d,dict) and ('dfly' in d or MARKER in d or 'raw_context_abi' in d or d.get('enable_hidden_correction') is True))
        or cfg.get('algorithm')=='dfly' or any('dfly' in x.lower() for x in cfg.get('architectures',[])))


def validate_config(config, source=False):
    cfg=as_dict(config);d=cfg.get('dflash_config',{})
    expected={'hidden_size':4096,'head_dim':128,'num_hidden_layers':3,'num_attention_heads':32,
        'num_key_value_heads':8,'target_hidden_size':4096,'vocab_size':151936,'block_size':8,
        'intermediate_size':12288,'attention_bias':False,'rms_norm_eps':1e-6}
    for k,v in expected.items():require(cfg.get(k)==v,'DFly config mismatch: '+k)
    for k,v in {'mask_token_id':151669,'target_layer_ids':[1,17,33], 'spatial_rope_enabled':True,
        'spatial_rope_mode':'replace','spatial_rope_sections':[24,20,20],
        'spatial_rope_interleaved':True,'visual_num_query_tokens':0,
        'context_layer_ids':[1,17,33],'separate_context_kv':False,'sample_from_anchor':False,'fusion_trainable':True,'fusion_init':'distance',
        'enable_hidden_correction':True,'hidden_correction_intermediate_size':None}.items():
        require(d.get(k)==v,'DFly config mismatch: dflash_config.'+k)
    require(not cfg.get('sample_from_anchor',False) and cfg.get('projector_type') is None,'DFly requires canonical slot and head policy')
    inner=getattr(config,'model',None)
    source_type=inner.model_type if inner is not None else cfg.get('model_type')
    require(source_type=='qwen3' and cfg.get('hidden_act')=='silu','DFly requires actual canonical qwen3/silu config')
    require(d.get('use_aux_hidden_state',True) is True and not d.get('layer_routing_enabled',False),'DFly requires ordered auxiliary capture without a second routing mechanism')
    rope=cfg['rope_parameters']
    require(rope.get('rope_type')=='default' and rope.get('rope_theta')==5000000.,'DFly rope mismatch')
    if not source:
        require(enabled(cfg),'Missing truthful DFly architecture and marker')
        require(rope.get('mrope_section')==[24,20,20] and rope.get('mrope_interleaved') is True,'Official MRoPE is required')
        require(cfg.get('source_checkpoint_sha256')==CHECKPOINT_SHA,'DFly checkpoint binding changed')
        require(d.get('raw_context_width')==12288 and d.get('raw_context_abi')=='ordered_target_layers_12288_v1','DFly raw context ABI changed')
    return cfg


def validate_state(state):
    require(set(state)==set(SCHEMA),'DFly requires exactly all42 tensor keys; missing='+str(sorted(set(SCHEMA)-set(state)))+' unknown='+str(sorted(set(state)-set(SCHEMA))))
    for k,v in state.items():
        require(list(v.shape)==SCHEMA[k]['shape'] and str(v.dtype)==SCHEMA[k]['dtype'],'DFly tensor shape/dtype mismatch: '+k)


def runtime_parameter_schema():
    result={}
    for name,entry in SCHEMA.items():
        target='model.'+name
        packed=False
        for source,destination in (('.q_proj.','.qkv_proj.'),('.k_proj.','.qkv_proj.'),('.v_proj.','.qkv_proj.'),
                                   ('.gate_proj.','.gate_up_proj.'),('.up_proj.','.gate_up_proj.')):
            if source in target and not name.startswith('hidden_correction.'):
                target=target.replace(source,destination);packed=True;break
        shape=list(entry['shape'])
        if packed and target in result:result[target][0]+=shape[0]
        else:result[target]=shape
    return result


def validate_runtime_parameters(parameters):
    expected=runtime_parameter_schema()
    require(not (set(parameters)-set(expected)-{'model.embed_tokens.weight','lm_head.weight'}),'Unexpected runtime learned parameter')
    for name,shape in expected.items():
        require(name in parameters and list(parameters[name].shape)==shape
            and str(parameters[name].dtype)=='torch.bfloat16','Runtime parameter layout/dtype mismatch: '+name)


def own_sources():
    paths=list(HERE.glob('*.py'))+list((HERE/'bootstrap').glob('*.py'))
    paths += [HERE/name for name in ('TENSOR_SCHEMA.json','OFFICIAL_SOURCE_PINS.json','SOURCE_PROVENANCE.json','SOURCE_MAP.json','SOURCE_COMPATIBILITY_REVIEW.json','CHECKPOINT_METADATA.json')]
    paths += list((HERE/'canonical_source').rglob('*.py'))
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}


def source_checks():
    import vllm
    require(vllm.__version__=='0.22.1','Official vLLM0.22.1 is required')
    pins=json.loads((HERE/'OFFICIAL_SOURCE_PINS.json').read_text());actual={}
    for name,pin in pins.items():
        path=Path(importlib.import_module(name).__file__)
        require(path.resolve()==Path(pin['path']).resolve() and sha(path)==pin['sha256'],'Official source origin/bytes changed: '+name)
        actual[name]=pin['sha256']
    for rel,pin in json.loads((HERE/'SOURCE_PROVENANCE.json').read_text()).items():
        require(sha(HERE/'canonical_source'/rel)==pin['sha256'],'Canonical source snapshot changed: '+rel)
        path=Path(pin['remote_path'])
        require(path.is_file() and sha(path)==pin['sha256'],'Actual canonical source changed: '+rel)
    return actual


def validate_launch_config(vc):
    validate_config(vc.speculative_config.draft_model_config.hf_config)
    s=vc.speculative_config;p=vc.parallel_config
    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from transformers import Qwen3Config
    wrapper=s.draft_model_config.hf_config
    require(type(wrapper) is EAGLEConfig and type(wrapper.model) is Qwen3Config
        and wrapper.model.architectures==[ARCH],'DFly requires the actual unmodified public EAGLEConfig wrapper over canonical Qwen3Config')
    require((p.tensor_parallel_size,p.pipeline_parallel_size,p.data_parallel_size)==(1,1,1),'DFly requires TP/PP/DP1')
    require(vc.scheduler_config.max_num_seqs==1 and vc.cache_config.block_size==16,'DFly requires max_num_seqs1/page16')
    require(s.method=='dflash' and s.num_speculative_tokens==7 and s.parallel_drafting,'DFly requires real publicS7/B8')
    require(str(s.attention_backend).split('.')[-1]=='FLASH_ATTN','DFly draft requires FLASH_ATTN')
    require(str(vc.attention_config.backend).split('.')[-1]=='FLASH_ATTN','DFly target requires FLASH_ATTN')
    cc=vc.compilation_config
    require(cc.cudagraph_mode.name=='PIECEWISE' and list(cc.cudagraph_capture_sizes)==[8] and cc.max_cudagraph_capture_size==8,'DFly requires public PIECEWISE capture[8]')
    require(not vc.model_config.enforce_eager,'DFly requires the reviewed public compile route')
    require(str(vc.model_config.dtype)=='torch.bfloat16','DFly requires BF16')
    require(str(s.draft_model_config.dtype)=='torch.bfloat16','DFly draft requires BF16')
    require(str(vc.model_config.model)==TARGET and vc.model_config.hf_config.model_type=='qwen3_vl','Wrong canonical DFly target path or architecture')
    require(str(s.model)==str(EXPORT) and str(s.draft_model_config.model)==str(EXPORT),
        'Actual DFly speculative and draft ModelConfig paths must equal the CPU verified export path')
    require(Path(TARGET).is_dir() and Path(TARGET).resolve()==Path(TARGET),'Canonical target path cannot be a symlink/alias')
    validate_export_directory(EXPORT,expected_path=EXPORT)
    require(vc.model_config.quantization is None and s.draft_model_config.quantization is None,'DFly requires unquantized models')
    require(s.use_local_argmax_reduction is False,'DFly hidden correction cannot use a local-argmax shortcut')
    from vllm.config import LoadConfig
    from vllm.model_executor import model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    require(model_loader._LOAD_FORMAT_TO_MODEL_LOADER.get('auto') is DefaultModelLoader,
        'Stock auto loader registry was replaced by custom loader')
    for cfg in (vc.load_config,s.draft_load_config or vc.load_config):
        require(type(cfg) is LoadConfig and str(cfg.load_format)=='auto' and not cfg.model_loader_extra_config,
            'DFly requires stock auto weight loading without overrides')
        require(cfg.safetensors_load_strategy in (None,'lazy','eager','prefetch') and cfg.ignore_patterns==['original/**/*']
            and cfg.download_dir is None and cfg.device is None,'DFly loader strategy/path overrides are forbidden')
    for cfg in (vc.model_config,s.draft_model_config):
        require(not cfg.trust_remote_code and getattr(cfg,'model_impl','auto')=='auto',
            'DFly requires native model implementation without remote-code overrides')
        require(not as_dict(cfg.hf_config).get('quantization_config'),'DFly HF quantization override is forbidden')
        require(cfg.model_weights in ('',cfg.model),'DFly model_weights must resolve to the exact verified model path')
    from vllm.config import SpeculativeConfig
    require(not vc.model_config.hf_overrides and s.draft_model_config.hf_overrides is SpeculativeConfig.hf_config_override,
        'DFly only permits the exact official SpeculativeConfig draft config override')


def validate_export_directory(path,*,expected_path,artifact_hashes=None,check_saved_receipt=True):
    path=Path(path);expected=Path(expected_path)
    require(path.is_absolute() and path==path.resolve() and path==expected and not path.is_symlink(),
        'DFly export path must equal the exact canonical verified path')
    require(path.is_dir() and {p.name for p in path.iterdir()}==EXPORT_FILES,
        'DFly export must contain exactly config.json/model.safetensors/export_receipt.json')
    for name in EXPORT_FILES:
        entry=path/name
        require(stat.S_ISREG(entry.lstat().st_mode) and not entry.is_symlink(),
            'DFly export contains a non-regular or symlinked artifact: '+name)
    if artifact_hashes is not None:
        expected_keys={str(path/name) for name in EXPORT_FILES}
        require(set(artifact_hashes)==expected_keys,'DFly receipt export artifact paths do not close over the exact export directory')
        for name,digest in artifact_hashes.items():require(sha(name)==digest,'DFly export artifact changed: '+name)
    if check_saved_receipt:
        saved=json.loads((path/'export_receipt.json').read_text())
        require(saved.get('schema')=='dfly-native-export-v1' and saved.get('checkpoint_sha256')==CHECKPOINT_SHA
            and saved.get('algorithm')=='dfly' and saved.get('step')==75000 and saved.get('tensor_count')==42
            and saved.get('all42_values_bitwise_equal') is True and saved.get('cuda_initialized') is False,
            'DFly export receipt does not attest canonical actual42 export')
        require(saved.get('config_sha256')==sha(path/'config.json') and saved.get('weights_sha256')==sha(path/'model.safetensors'),
            'DFly saved export receipt disagrees with actual config/weights')
        require(set(saved.get('tensors',{}))==set(SCHEMA),'DFly export receipt tensor keys differ')
        validate_config(json.loads((path/'config.json').read_text()))
    return path


def runtime_receipt():
    path=os.environ.get('DFLY_NATIVE_CPU_RECEIPT')
    require(bool(path),'DFLY_NATIVE_CPU_RECEIPT must bind completed CPU/export evidence before model construction')
    expected_sha=os.environ.get('DFLY_NATIVE_CPU_RECEIPT_SHA256')
    require(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha) is not None,
        'DFLY_NATIVE_CPU_RECEIPT_SHA256 must explicitly bind the reviewed receipt SHA256')
    path=Path(path)
    require(path.is_absolute() and path==path.resolve() and stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
        'DFly CPU receipt must be one exact regular canonical path')
    require(sha(path)==expected_sha,'DFly CPU receipt SHA256 differs from explicitly supplied SHA256')
    receipt=json.loads(path.read_text())
    require(receipt.get('schema')=='dfly-native-cpu-v1' and receipt.get('status')=='complete'
        and receipt.get('all42_values_bitwise_equal') is True and receipt.get('cuda_initialized') is False,
        'Incomplete DFly CPU/export evidence')
    require(receipt.get('CPU_test_count',0)>=16 and receipt.get('CPU_failures')==0
        and receipt.get('CPU_errors')==0 and receipt.get('CPU_skips')==0
        and receipt.get('checkpoint_sha256')==CHECKPOINT_SHA and receipt.get('tensor_count')==42,
        'DFly CPU receipt does not bind complete actual42 verification')
    require(receipt.get('export_dir')==str(EXPORT) and receipt.get('checkpoint_path')==CHECKPOINT,
        'DFly CPU receipt export/checkpoint path differs from the actual production path')
    require(receipt['own_source_sha256']==own_sources(),'CPU receipt source freeze differs')
    source_checks()
    validate_export_directory(EXPORT,expected_path=EXPORT,artifact_hashes=receipt['export_artifacts'])
    require(sha(Path(TARGET)/'config.json')==receipt['actual_public_config']['target_config_sha256'],'DFly target configuration changed')
    return receipt
