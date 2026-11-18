"""Strict, stdlib-only Domino parent and source/gate contracts."""
from __future__ import annotations
import hashlib,json,math,os
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MARKER='official_domino_public_adapter_v1'
RAW_ARCH='Qwen3DominoDraftModel'
RESOLVED_ARCH='DFlashQwen3DominoDraftModel'
CHECKPOINT_SHA='f3fdfcd68f4eeb1aab74649c71c9cdb5c117f5c5822cf54b9c0f749b1dde85f7'
CHECKPOINT_PATH='/home/lyh/qwen3vl8b_eval_20261027/artifacts/domino/ckpt_final.pt'
TARGET_ROOT='/home/lyh/qwen3vl8b_eval_20261027/artifacts/target'
OWN_FILES=('contracts.py','domino_logic.py','domino_model.py','domino_proposer.py','spatial_math.py','install.py','export_domino.py','preflight.py','bootstrap/sitecustomize.py','TENSOR_SCHEMA.json','CHECKPOINT_METADATA.json','reference_metadata.json','source_bindings.json','test_domino.py','test_interface.py','test_runtime_approval.py','README.md')
BASE={'mask_token_id':151669,'target_layer_ids':[1,17,33],'spatial_rope_enabled':True,'spatial_rope_mode':'replace','spatial_rope_sections':[24,20,20],'spatial_rope_interleaved':True,'visual_num_query_tokens':0,'projector_type':'domino','emb_dim':256,'gru_hidden_dim':1024,'pure_draft_prefix_len':1,'shift_label':True}
OPTIONAL={'sample_from_anchor':False,'inference_in_block_causal':False,'layer_routing_enabled':False}

def _dict(config):
    if type(config) is dict:return config
    from transformers import Qwen3Config
    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    if type(config) not in (Qwen3Config,EAGLEConfig):raise ValueError('Only raw Qwen3Config or exact official EAGLEConfig accepted')
    return config.to_dict()

def selected(config):
    cfg=config if type(config) is dict else config.to_dict()
    d=cfg.get('dflash_config',{})
    near=cfg.get('algorithm')=='domino' or (isinstance(d,dict) and MARKER in d) or any('domino' in str(a).lower() for a in (cfg.get('architectures') or ()))
    if not near:return False
    validate_config(config)
    return True

def validate_config(config,*,source=False):
    cfg=_dict(config);d=cfg.get('dflash_config',{})
    if not isinstance(d,dict):raise ValueError('Missing Domino config')
    resolved=type(config) is not dict and type(config).__name__=='EAGLEConfig'
    if cfg.get('model_type')!=('eagle' if resolved else 'qwen3'):raise ValueError('Wrong raw/resolved model_type; no spoofing')
    if resolved:
        from transformers import Qwen3Config
        if type(config.model) is not Qwen3Config or config.model.model_type!='qwen3':raise ValueError('Resolved EAGLEConfig must wrap the real Qwen3Config')
    if any(k in cfg for k in ('eagle_config','eagle_aux_hidden_state_layer_ids')):raise ValueError('EAGLE override branches are unsupported')
    if not source:
        if cfg.get('algorithm')!='domino' or d.get(MARKER) is not True:raise ValueError('Truthful Domino algorithm and marker required')
        arch=cfg.get('architectures')
        resolved=type(config) is not dict and type(config).__name__=='EAGLEConfig'
        if arch!=[RESOLVED_ARCH if resolved else RAW_ARCH]:raise ValueError('Wrong raw/resolved Domino architecture')
        if cfg.get('source_checkpoint_sha256')!=CHECKPOINT_SHA:raise ValueError('Wrong Domino parent')
    for k,v in {'hidden_size':4096,'target_hidden_size':4096,'head_dim':128,'num_hidden_layers':3,'num_attention_heads':32,'num_key_value_heads':8,'intermediate_size':12288,'vocab_size':151936,'block_size':8,'rms_norm_eps':1e-6,'attention_bias':False}.items():
        if cfg.get(k)!=v or (type(v) in (int,bool) and type(cfg.get(k)) is not type(v)):raise ValueError(f'Unsupported {k}')
    if cfg.get('hidden_act','silu')!='silu':raise ValueError('SwiGLU activation must remain silu')
    if cfg.get('quantization_config') is not None:raise ValueError('Quantized checkpoint configuration is unsupported')
    if set(d)-set(BASE)-set(OPTIONAL)-{MARKER}:raise ValueError('Unknown or foreign draft inference flags')
    for k,v in BASE.items():
        if d.get(k)!=v or (type(v) in (int,bool) and type(d.get(k)) is not type(v)):raise ValueError(f'Unsupported dflash_config.{k}')
    for k,v in OPTIONAL.items():
        if k in d and d[k] is not v:raise ValueError(f'Unsupported {k}')
    vocab=cfg.get('draft_vocab_size')
    if vocab is not None and (type(vocab) is not int or vocab!=151936):raise ValueError('Only full target vocabulary accepted')
    scale=cfg.get('logit_scale',1.0)
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale!=1:raise ValueError('Only numeric unit logit scale accepted')
    rope=cfg.get('rope_parameters') or cfg.get('rope_scaling') or {}
    if rope.get('rope_type','default')!='default' or rope.get('rope_theta')!=5000000.0:raise ValueError('Wrong RoPE frequencies')
    if not source and (rope.get('mrope_section')!=[24,20,20] or rope.get('mrope_interleaved') is not True):raise ValueError('Three-axis replacement RoPE required')
    return cfg

def expected_schema():return json.loads((ROOT/'TENSOR_SCHEMA.json').read_text())

def validate_state(state):
    schema=expected_schema()
    if set(state)!=set(schema):raise ValueError('Checkpoint must contain exactly all40 Domino tensors')
    for name,shape in schema.items():
        t=state[name]
        if list(t.shape)!=shape or str(t.dtype)!='torch.bfloat16':raise ValueError(f'Wrong shape/dtype: {name}')

def validate_query(s,n,p,mode,*,dummy,initialized):
    if type(s) is not int or s!=7:raise ValueError('Only immutable public S7/queryB8 accepted')
    if dummy:
        if not 1<=n<=p<=8 or mode not in ('NONE','PIECEWISE'):raise ValueError('Unsafe dummy query/padding')
    elif n!=8 or p!=8 or mode!='PIECEWISE' or not initialized:raise ValueError('Real Domino requires initialized exact8 PIECEWISE without padding')

def own_source_hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in OWN_FILES}

def verify_dependencies():
    b=json.loads((ROOT/'source_bindings.json').read_text());out={}
    paths={**{str(ROOT.parent/name):sha for name,sha in {**b['frozen'],**b.get('design_sha256',{})}.items()},**b['upstream'],b['canonical_reference']['path']:b['canonical_reference']['sha256']}
    for name,expected in paths.items():
        p=Path(name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Pinned dependency changed: {name}')
        out[name]=expected
    return out

@dataclass(frozen=True)
class VerifiedApproval:
    receipt_sha256: str
    runtime_approval_sha256: str
    target_model_path: str
    draft_model_path: str
    full_artifacts: object
    own_sources: object
    stamps: object

_VERIFIED_APPROVAL=None

def validate_loading(vc):
    spec=vc.speculative_config
    for model in (vc.model_config,spec.draft_model_config):
        if getattr(model,'quantization',None) is not None:raise ValueError('Quantized target/draft loading is unsupported')
        alternate=getattr(model,'model_weights',None)
        if alternate and str(Path(alternate).resolve())!=str(Path(model.model).resolve()):raise ValueError('Alternate model_weights path is unsupported')
    if getattr(spec,'quantization',None) is not None:raise ValueError('Draft quantization is unsupported')
    target_load=vc.load_config
    for load in (target_load,getattr(spec,'draft_load_config',None) or target_load):
        if load.load_format not in ('auto','safetensors') or getattr(load,'model_loader_extra_config',None) not in (None,{}):raise ValueError('Only stock real-weight auto/safetensors loading without extra overrides is supported')

def validate_export_layout(path):
    path=Path(path)
    expected={'config.json','model.safetensors','export_receipt.json'}
    if not path.is_dir() or {p.name for p in path.iterdir()}!=expected or any(not (path/name).is_file() for name in expected):raise ValueError('Approved export must contain exactly config/weights/receipt; no alternate index or weights')

def validate_target_layout(path,artifacts):
    target=Path(path).resolve()
    expected={'model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors'}
    pinned={Path(name).name for name in artifacts if Path(name).parent==target and name.endswith('.safetensors')}
    if pinned!=expected or str(target/'model.safetensors.index.json') not in artifacts:raise ValueError('Fixed target shard/index pins changed')
    if not target.is_dir() or {p.name for p in target.glob('*.safetensors')}!=expected or any(not (target/name).is_file() for name in expected):raise ValueError('Target loader has alternate or missing safetensors weights')
    indexes=list(target.glob('*.safetensors.index.json'))
    if {p.name for p in indexes}!={'model.safetensors.index.json'} or any(not p.is_file() for p in indexes):raise ValueError('Target safetensors index selection changed')
    if any(list(target.glob(pattern)) for pattern in ('*.bin','*.pt','*.pth')):raise ValueError('Target loader has unapproved alternate weight files')


def validate_model_binding(vc,receipt):
    target=str(Path(receipt.get('target_model_path','')).resolve())
    draft=str(Path(receipt.get('draft_model_path','')).resolve())
    if target!=str(Path(TARGET_ROOT).resolve()) or not receipt.get('draft_model_path'):raise ValueError('Unbound approved target/draft root')
    if str(Path(vc.model_config.model).resolve())!=target:raise ValueError('Actual target differs from approved target')
    spec=vc.speculative_config
    if str(Path(spec.model).resolve())!=draft or str(Path(spec.draft_model_config.model).resolve())!=draft:raise ValueError('Actual draft differs from approved all40-value export')
    artifacts=receipt.get('full_artifacts',{})
    for name in ('config.json','model.safetensors','export_receipt.json'):
        digest=artifacts.get(str(Path(draft)/name))
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Missing approved export config/weights/receipt pin')

def runtime_contract():
    return {'algorithm':'domino','method':'dflash','target_attention_backend':'FLASH_ATTN','draft_attention_backend':'FLASH_ATTN','num_speculative_tokens':7,'query_tokens':8,'physical_kv_page_size':16,'tensor_parallel_size':1,'pipeline_parallel_size':1,'data_parallel_size':1,'max_num_seqs':1,'graph_route':'official_compile_piecewise_draft','weight_loading':'stock_real_auto_or_safetensors'}

def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def _absolute(value):
    return isinstance(value,str) and bool(value) and Path(value).is_absolute() and str(Path(value).resolve())==value

def validate_full_receipt(receipt):
    # A CPU proof remains ineligible forever. A separate, explicitly pinned root
    # approval records independent reviews; changing this boolean never promotes it.
    if receipt.get('schema')!='domino-flash-full-preflight-v1' or receipt.get('classification')!='full_export_and_bytes_verified' or receipt.get('gpu_eligible') is not False or receipt.get('requires_independent_review') is not True:raise RuntimeError('Expected non-promotable Flash full CPU proof')
    if receipt.get('cuda_initialized') is not False or receipt.get('model_constructed') is not False or receipt.get('LLM_constructed') is not False or receipt.get('forward_executed') is not False or receipt.get('real_checkpoint_contents_read') is not True or receipt.get('all40_values_bitwise_equal') is not True:raise RuntimeError('Missing full CPU evidence')
    if receipt.get('source_root')!=str(ROOT) or receipt.get('own_sources')!=own_source_hashes():raise RuntimeError('Receipt source attestation differs from current files')
    if receipt.get('runtime_contract')!=runtime_contract():raise RuntimeError('Wrong Flash runtime method/config binding')
    if receipt.get('cpu_only') is not True or receipt.get('OMP_NUM_THREADS')!='1' or receipt.get('MKL_NUM_THREADS')!='1' or receipt.get('production_CPU_escape_present') is not False or receipt.get('actual_public_config_and_class_interface_checks') is not True:raise RuntimeError('Missing non-promotable CPU/config evidence')
    if type(receipt.get('tests_run')) is not int or receipt['tests_run']<1 or type(receipt.get('tests_passed')) is not int or receipt.get('tests_passed')!=receipt['tests_run'] or type(receipt.get('tests_skipped')) is not int or receipt['tests_skipped']!=0:raise RuntimeError('Full CPU tests did not pass without skips')
    if receipt.get('target_weight_bodies_read') is not True or receipt.get('dependency_binary_bodies_rehashed') is not True or not receipt.get('full_artifacts'):raise RuntimeError('Missing full current-byte evidence')
    if receipt.get('target_model_path')!=str(Path(TARGET_ROOT).resolve()) or not _absolute(receipt.get('draft_model_path')) or not _absolute(receipt.get('receipt_path')):raise RuntimeError('Missing exact full CPU model/proof paths')
    target=Path(TARGET_ROOT).resolve();shards=sorted(str(target/name) for name in ('model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors'))
    if receipt.get('target_weight_layout_validated') is not True or receipt.get('target_loader_selection')!={'auto':shards,'safetensors':shards}:raise RuntimeError('Missing exact actual target loader selection evidence')
    if receipt.get('checkpoint_path')!=CHECKPOINT_PATH or receipt.get('checkpoint_sha256')!=CHECKPOINT_SHA or receipt.get('checkpoint_step')!=75000:raise RuntimeError('Wrong original Domino checkpoint')
    required=json.loads((ROOT/'source_bindings.json').read_text())['deferred_full_artifacts']
    export={str(Path(receipt['draft_model_path'])/name) for name in ('config.json','model.safetensors','export_receipt.json')}
    artifacts=receipt['full_artifacts']
    if not isinstance(artifacts,dict) or set(artifacts)!=set(required)|export or any(artifacts.get(name)!=sha for name,sha in required.items()) or artifacts.get(CHECKPOINT_PATH)!=CHECKPOINT_SHA or any(not _absolute(name) or not _digest(sha) for name,sha in artifacts.items()):raise RuntimeError('Missing, extra, or changed full-artifact pins')
    tensors=receipt.get('tensors');schema=expected_schema()
    if receipt.get('tensor_count')!=40 or not isinstance(tensors,dict) or set(tensors)!=set(schema):raise RuntimeError('Missing complete all40 value evidence')
    for name,shape in schema.items():
        entry=tensors[name]
        if not isinstance(entry,dict) or entry.get('shape')!=shape or entry.get('dtype')!='torch.bfloat16' or not _digest(entry.get('sha256')):raise RuntimeError('Invalid tensor evidence: '+name)

def approval_binding(receipt,path,digest):
    return {'full_cpu_proof':{'path':str(Path(path).resolve()),'sha256':digest},'source_root':receipt['source_root'],'own_sources':receipt['own_sources'],'runtime_contract':receipt['runtime_contract'],'target_model_path':receipt['target_model_path'],'draft_model_path':receipt['draft_model_path'],'full_artifacts':receipt['full_artifacts']}

def _pinned_json(path,digest,label):
    if not _absolute(str(path)) or not _digest(digest):raise RuntimeError('Explicit absolute path and SHA required: '+label)
    p=Path(path)
    raw=p.read_bytes() if p.is_file() else b''
    if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError('Pinned evidence changed or absent: '+label)
    doc=json.loads(raw)
    if not isinstance(doc,dict):raise RuntimeError('Evidence must be an object: '+label)
    return doc

def check_runtime_approval(receipt,path,digest,approval_path,approval_digest):
    approval=_pinned_json(approval_path,approval_digest,'root runtime approval')
    if approval.get('schema')!='domino-flash-runtime-approval-v1' or approval.get('classification')!='independently_reviewed_runtime_authorization' or approval.get('gpu_eligible') is not True or approval.get('decision')!='APPROVE' or approval.get('approved_by')!='root':raise RuntimeError('No explicit root runtime approval')
    binding=approval_binding(receipt,path,digest)
    if approval.get('binding')!=binding:raise RuntimeError('Runtime approval differs from current full CPU binding')
    refs=approval.get('independent_reviews')
    if not isinstance(refs,dict) or set(refs)!={'spec','quality'}:raise RuntimeError('Independent spec and quality PASS evidence required')
    evidence={str(Path(approval_path).resolve()):approval_digest,str(Path(path).resolve()):digest};reviewers=set()
    for role,ref in refs.items():
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise RuntimeError('Exact review path and SHA required')
        review=_pinned_json(ref['path'],ref['sha256'],role+' independent review')
        reviewer=review.get('reviewer')
        if review.get('schema')!='domino-flash-independent-review-v1' or review.get('role')!=role or review.get('decision')!='PASS' or review.get('independent') is not True or review.get('blocking_findings')!=[] or not isinstance(reviewer,str) or not reviewer.strip() or reviewer=='root' or reviewer in reviewers or review.get('reviewed_binding')!=binding:raise RuntimeError('Unreviewed, non-independent, or stale '+role+' evidence')
        if ref['path'] in evidence:raise RuntimeError('Review evidence paths must be distinct')
        reviewers.add(reviewer);evidence[ref['path']]=ref['sha256']
    return evidence

def check_full_header(path,digest,approval_path=None,approval_digest=None):
    # Only small proof/source/review reads here, before production imports Torch.
    receipt=_pinned_json(path,digest,'full CPU proof')
    validate_full_receipt(receipt)
    if receipt['receipt_path']!=str(Path(path).resolve()):raise RuntimeError('Full proof path differs from attested path')
    check_runtime_approval(receipt,path,digest,approval_path,approval_digest)
    return receipt

def _stamp(path):
    st=Path(path).stat()
    return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

def require_full_gate(path,digest,approval_path=None,approval_digest=None):
    # Single full-byte verification per process; later model construction uses this
    # immutable capability and rejects changed sources, paths or artifact stat stamps.
    global _VERIFIED_APPROVAL
    receipt=check_full_header(path,digest,approval_path,approval_digest)
    evidence=check_runtime_approval(receipt,path,digest,approval_path,approval_digest)
    validate_export_layout(receipt['draft_model_path'])
    validate_target_layout(receipt['target_model_path'],receipt['full_artifacts'])
    if _VERIFIED_APPROVAL is not None:
        a=_VERIFIED_APPROVAL
        if a.receipt_sha256!=digest or a.runtime_approval_sha256!=approval_digest or dict(a.full_artifacts)!=receipt['full_artifacts'] or any(_stamp(name)!=stamp for name,stamp in a.stamps.items()):raise RuntimeError('Approved artifact identity changed')
        return a
    stamps={}
    for name,expected in {**receipt['full_artifacts'],**evidence}.items():
        before=_stamp(name);h=hashlib.sha256()
        with Path(name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected or before!=_stamp(name):raise RuntimeError('Full artifact changed: '+name)
        stamps[name]=before
    if own_source_hashes()!=receipt['own_sources'] or any(_stamp(name)!=stamp for name,stamp in stamps.items()):raise RuntimeError('Sources or artifacts changed during full authorization')
    validate_target_layout(receipt['target_model_path'],receipt['full_artifacts'])
    _VERIFIED_APPROVAL=VerifiedApproval(digest,approval_digest,str(Path(receipt['target_model_path']).resolve()),str(Path(receipt['draft_model_path']).resolve()),MappingProxyType(dict(receipt['full_artifacts'])),MappingProxyType(dict(receipt['own_sources'])),MappingProxyType(stamps))
    return _VERIFIED_APPROVAL

def require_runtime_binding(vc):
    a=_VERIFIED_APPROVAL
    if a is None or os.environ.get('DOMINO_APPROVED_PREFLIGHT_SHA256')!=a.receipt_sha256 or os.environ.get('DOMINO_APPROVED_RUNTIME_SHA256')!=a.runtime_approval_sha256:raise RuntimeError('No verified full-byte hardware authorization in this process')
    validate_target_layout(a.target_model_path,a.full_artifacts)
    if own_source_hashes()!=dict(a.own_sources) or any(_stamp(name)!=stamp for name,stamp in a.stamps.items()):raise RuntimeError('Authorized source/artifact changed')
    validate_model_binding(vc,{'target_model_path':a.target_model_path,'draft_model_path':a.draft_model_path,'full_artifacts':a.full_artifacts})
    validate_loading(vc);validate_export_layout(a.draft_model_path)
    return a
