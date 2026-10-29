"""Fail-closed DFlare schema, source, config, and reviewed CPU evidence binding."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import importlib
import json
import os
import stat

HERE = Path(__file__).resolve().parent
ARCH = 'DFlashDFlareNativeQwen3ForCausalLM'
MARKER = 'dflare_8b_native_adapter'
CHECKPOINT = '/home/lyh/qwen3vl8b_eval_20261027/artifacts/dflare/ckpt_final.pt'
CHECKPOINT_SHA = 'cfdbcb9f4481d5e201d2aad78553ffc7eb54dbe61c50bf1da9f934dc950345af'
TARGET = '/home/lyh/qwen3vl8b_eval_20261027/artifacts/target'
EXPORT_FILES = frozenset(('config.json','model.safetensors','export_receipt.json'))
RECEIPT_SCHEMA = 'dflare-native-cpu-v2'
SCHEMA = json.loads((HERE/'TENSOR_SCHEMA.json').read_text())


def require(condition, message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def _canonical_path(path):
    path=Path(path)
    require(path.is_absolute() and path==path.resolve(),'Canonical absolute non-symlink path required: '+str(path))
    return path

def validate_export_layout(export_dir,receipt=None):
    export_dir=_canonical_path(export_dir)
    require(stat.S_ISDIR(export_dir.lstat().st_mode),'Export must be a real directory')
    require({p.name for p in export_dir.iterdir()}==EXPORT_FILES,'Export must contain exactly config.json/model.safetensors/export_receipt.json')
    for name in EXPORT_FILES:
        path=export_dir/name
        require(stat.S_ISREG(path.lstat().st_mode),'Export artifact must be a regular non-symlink file: '+name)
        require(path.resolve().parent==export_dir,'Export artifact escaped its bound parent: '+name)
    if receipt is not None:
        require(receipt.get('export_dir')==str(export_dir),'Receipt export_dir does not match actual draft directory')
        expected={str(export_dir/name) for name in EXPORT_FILES}
        require(set(receipt.get('export_artifacts',{}))==expected,'All and only the three export artifacts must share export_dir as parent')
        for file,digest in receipt['export_artifacts'].items():
            require(isinstance(digest,str) and len(digest)==64 and all(x in '0123456789abcdef' for x in digest),'Invalid export artifact SHA256')
            require(sha(file)==digest,'DFlare export artifact changed: '+file)
    return export_dir

def draft_config_digest(config):
    value=json.loads(json.dumps(as_dict(config),default=str))
    # Official model initialization adds these two fixed derived values.
    for field,expected in (('draft_vocab_size',151936),('target_layer_count',36)):
        if field in value:
            require(value.pop(field)==expected,'Changed derived DFlare config value: '+field)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_effective_load_configs(vc):
    from vllm.config import LoadConfig
    reference=asdict(LoadConfig())
    configs={'target':vc.load_config,'draft':vc.speculative_config.draft_load_config or vc.load_config}
    for role,config in configs.items():
        require(type(config) is LoadConfig and asdict(config)==reference,'Unsupported '+role+' public loader options; exact stock LoadConfig required')
    for role,model in (('target',vc.model_config),('draft',vc.speculative_config.draft_model_config)):
        require(not model.trust_remote_code,'Remote-code loader override is forbidden: '+role)
        for field in ('revision','code_revision','tokenizer_revision'):
            require(getattr(model,field,None) is None,'Revision loader override is forbidden: '+role+'.'+field)
    return {role:asdict(config) for role,config in configs.items()}

def as_dict(config):return config if isinstance(config,dict) else config.to_dict()


def enabled(config):
    cfg=as_dict(config)
    d=cfg.get('dflash_config') or {}
    return cfg.get('architectures')==[ARCH] and isinstance(d,dict) and d.get(MARKER) is True


def near_dflare(config):
    cfg=as_dict(config)
    d=cfg.get('dflash_config') or {}
    return ((isinstance(d,dict) and ('dflare' in d or MARKER in d or d.get('separate_context_kv') is True or 'raw_context_abi' in d))
        or cfg.get('algorithm')=='dflare' or any('dflare' in x.lower() for x in cfg.get('architectures',[])))


def validate_config(config, source=False):
    cfg=as_dict(config);d=cfg.get('dflash_config',{})
    expected={'hidden_size':4096,'head_dim':128,'num_hidden_layers':3,'num_attention_heads':32,
        'num_key_value_heads':8,'target_hidden_size':4096,'vocab_size':151936,'block_size':8,
        'intermediate_size':12288,'attention_bias':False,'rms_norm_eps':1e-6}
    for k,v in expected.items():require(cfg.get(k)==v,'DFlare config mismatch: '+k)
    for k,v in {'mask_token_id':151669,'target_layer_ids':[1,17,33], 'spatial_rope_enabled':True,
        'spatial_rope_mode':'replace','spatial_rope_sections':[24,20,20],
        'spatial_rope_interleaved':True,'visual_num_query_tokens':0,
        'context_layer_ids':[1,17,33],'separate_context_kv':True,'fusion_trainable':True,'fusion_init':'distance'}.items():
        require(d.get(k)==v,'DFlare config mismatch: dflash_config.'+k)
    require(not cfg.get('sample_from_anchor',False) and cfg.get('projector_type') is None,'DFlare requires canonical slot and head policy')
    inner=getattr(config,'model',None)
    source_type=inner.model_type if inner is not None else cfg.get('model_type')
    require(source_type=='qwen3' and cfg.get('hidden_act')=='silu','DFlare requires actual canonical qwen3/silu config')
    require(d.get('use_aux_hidden_state',True) is True and not d.get('layer_routing_enabled',False),'DFlare requires ordered auxiliary capture without a second routing mechanism')
    rope=cfg['rope_parameters']
    require(rope.get('rope_type')=='default' and rope.get('rope_theta')==5000000.,'DFlare rope mismatch')
    if not source:
        require(enabled(cfg),'Missing truthful DFlare architecture and marker')
        require(rope.get('mrope_section')==[24,20,20] and rope.get('mrope_interleaved') is True,'Official MRoPE is required')
        require(cfg.get('source_checkpoint_sha256')==CHECKPOINT_SHA,'DFlare checkpoint binding changed')
        require(d.get('raw_context_width')==12288 and d.get('raw_context_abi')=='ordered_target_layers_12288_v1','DFlare raw context ABI changed')
    return cfg


def validate_state(state):
    require(set(state)==set(SCHEMA),'DFlare requires exactly all42 tensor keys; missing='+str(sorted(set(SCHEMA)-set(state)))+' unknown='+str(sorted(set(state)-set(SCHEMA))))
    for k,v in state.items():
        require(list(v.shape)==SCHEMA[k]['shape'] and str(v.dtype)==SCHEMA[k]['dtype'],'DFlare tensor shape/dtype mismatch: '+k)


def runtime_parameter_schema():
    result={}
    for name,entry in SCHEMA.items():
        target='model.'+name
        packed=False
        for source,destination in (('.q_proj.','.qkv_proj.'),('.k_proj.','.qkv_proj.'),('.v_proj.','.qkv_proj.'),
                                   ('.gate_proj.','.gate_up_proj.'),('.up_proj.','.gate_up_proj.')):
            if source in target:target=target.replace(source,destination);packed=True;break
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


def validate_launch_config(vc,receipt=None):
    validate_config(vc.speculative_config.draft_model_config.hf_config)
    s=vc.speculative_config;p=vc.parallel_config
    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from transformers import Qwen3Config
    wrapper=s.draft_model_config.hf_config
    require(type(wrapper) is EAGLEConfig and type(wrapper.model) is Qwen3Config
        and wrapper.model.architectures==[ARCH],'DFlare requires the actual unmodified public EAGLEConfig wrapper over canonical Qwen3Config')
    require((p.tensor_parallel_size,p.pipeline_parallel_size,p.data_parallel_size)==(1,1,1),'DFlare requires TP/PP/DP1')
    require(vc.scheduler_config.max_num_seqs==1 and vc.cache_config.block_size==16,'DFlare requires max_num_seqs1/page16')
    require(s.method=='dflash' and s.num_speculative_tokens==7 and s.parallel_drafting,'DFlare requires real publicS7/B8')
    require(str(s.attention_backend).split('.')[-1]=='FLASH_ATTN','DFlare draft requires FLASH_ATTN')
    require(str(vc.attention_config.backend).split('.')[-1]=='FLASH_ATTN','DFlare target requires FLASH_ATTN')
    cc=vc.compilation_config
    require(cc.cudagraph_mode.name=='PIECEWISE' and list(cc.cudagraph_capture_sizes)==[8] and cc.max_cudagraph_capture_size==8,'DFlare requires public PIECEWISE capture[8]')
    require(not vc.model_config.enforce_eager,'DFlare requires the reviewed public compile route')
    require(str(vc.model_config.dtype)=='torch.bfloat16','DFlare requires BF16')
    require(str(s.draft_model_config.dtype)=='torch.bfloat16','DFlare draft requires BF16')
    require(str(vc.model_config.model)==TARGET and vc.model_config.hf_config.model_type=='qwen3_vl','Wrong canonical DFlare target path or architecture')
    require(vc.model_config.quantization is None and s.draft_model_config.quantization is None,'DFlare requires unquantized models')
    for cfg in (vc.load_config,s.draft_load_config or vc.load_config):
        require(str(cfg.load_format)=='auto' and not cfg.model_loader_extra_config,'DFlare requires stock auto weight loading without overrides')

    validate_effective_load_configs(vc)
    spec_path=_canonical_path(s.model);draft_path=_canonical_path(s.draft_model_config.model)
    require(spec_path==draft_path,'Actual SpeculativeConfig.model and draft ModelConfig.model differ')
    validate_export_layout(spec_path,receipt)
    if receipt is not None:
        require(draft_config_digest(s.draft_model_config.hf_config)==receipt['actual_public_config']['resolved_draft_hf_config_sha256'],
            'Actual resolved draft ModelConfig differs from the pinned CPU public configuration')


def validate_draft_loader_selection(vc,receipt):
    validate_launch_config(vc,receipt)
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    s=vc.speculative_config
    loader=get_model_loader(s.draft_load_config or vc.load_config)
    require(type(loader) is DefaultModelLoader,'Only the actual standard DefaultModelLoader is allowed')
    folder,files,safe=loader._prepare_weights(s.draft_model_config.model,subfolder=None,
        revision=s.draft_model_config.revision,fall_back_to_pt=True,allow_patterns_overrides=None)
    export=Path(receipt['export_dir'])
    require(Path(folder)==export and safe is True and files==[str(export/'model.safetensors')],
        'Actual public loader selected an unbound directory/weight/index source')
    return {'loader_class':type(loader).__module__+'.'+type(loader).__name__,
        'selected_folder':folder,'selected_files':files,'use_safetensors':safe,
        'model_constructed':False,'weights_deserialized':False,'cuda_initialized':False}

def runtime_receipt():
    path=os.environ.get('DFLARE_NATIVE_CPU_RECEIPT')
    require(bool(path),'DFLARE_NATIVE_CPU_RECEIPT must bind completed CPU/export evidence before model construction')
    expected=os.environ.get('DFLARE_NATIVE_CPU_RECEIPT_SHA256','')
    require(len(expected)==64 and all(x in '0123456789abcdef' for x in expected),
        'DFLARE_NATIVE_CPU_RECEIPT_SHA256 must explicitly pin the reviewed receipt')
    path=_canonical_path(path)
    require(stat.S_ISREG(path.lstat().st_mode),'CPU receipt must be a regular non-symlink file')
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'CPU receipt differs from explicitly expected SHA256')
    receipt=json.loads(raw)
    require(receipt.get('schema')==RECEIPT_SCHEMA and receipt.get('status')=='complete'
        and receipt.get('all42_values_bitwise_equal') is True and receipt.get('cuda_initialized') is False,
        'Incomplete DFlare CPU/export evidence')
    require(receipt.get('checkpoint_sha256')==CHECKPOINT_SHA and receipt.get('tensor_count')==42,
        'CPU receipt canonical checkpoint or tensor count changed')
    require(receipt.get('CPU_test_count',0)>=24 and receipt.get('CPU_failures')==0
        and receipt.get('CPU_errors')==0 and receipt.get('CPU_skips')==0,'Incomplete DFlare CPU tests')
    require(receipt['own_source_sha256']==own_sources(),'CPU receipt source freeze differs')
    source_checks()
    export=validate_export_layout(receipt['export_dir'],receipt)
    proof=receipt['actual_public_config']
    require(proof['draft_model_config_model']==proof['speculative_model']==str(export)
        ==proof['engine_kwargs']['speculative_config']['model'],'CPU receipt actual ModelConfig/spec/export binding differs')
    require(sha(Path(TARGET)/'config.json')==receipt['actual_public_config']['target_config_sha256'],'DFlare target configuration changed')
    return receipt
