"""Source-bound causal policy for the local shared-block JetSpec reference.

This is NOT the upstream parallel-tree JetSpec runtime. It reuses unchanged
B887 spatial/graph helpers and official vLLM DFlash operators. No new kernel,
model/attention/target forward, output head or sampler is implemented here.
Import is inert; install explicitly before backend/model initialization.
"""
from __future__ import annotations
from dataclasses import replace
from functools import wraps
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
MARKER='official_jetspec_local_block_adapter_v1'
SPATIAL_MARKER='official_spatial_adapter_v1'
CHECKPOINT_SHA256='afb8fdb1d14df3b3d003003fa12a30d29e2d6b08fa7cc45d24728be929437391'
SCOPE='local shared-block JetSpec reference; upstream parallel-tree runtime is not ported'
_INSTALLED=False
_BINDINGS={}

INFERENCE_CONTRACT={
    'trained_block_size':8,'inference_in_block_causal':True,'sample_from_anchor':False,
    'shift_label':False,'pure_draft_prefix_len':0,'projector_type':None,
    'layer_routing_enabled':False,'visual_num_query_tokens':0,'causal_in_block_training':True,
    'head':'external_target_lm_head','capture_layer_ids':[1,17,33],
    'hf_capture_hidden_indices':[2,18,34],'draft_sampling':'greedy',
}
_BASE_DFLASH_FIELDS={'mask_token_id','target_layer_ids','spatial_rope_enabled','spatial_rope_mode',
    'spatial_rope_sections','spatial_rope_interleaved','visual_num_query_tokens','inference_in_block_causal'}
_DEFAULTS={'sample_from_anchor':False,'shift_label':False,'pure_draft_prefix_len':0,
    'projector_type':None,'layer_routing_enabled':False}


def _dict(config:Any)->dict:
    return config if isinstance(config,dict) else config.to_dict()


def _check_frozen_sources()->dict[str,str]:
    manifest=json.loads((ROOT/'source_bindings.json').read_text())
    hashes={}
    for name,digest in manifest['frozen'].items():
        actual=hashlib.sha256((CAMPAIGN/name).read_bytes()).hexdigest()
        if actual!=digest:raise RuntimeError(f'Frozen source changed: {name}; re-review required')
        hashes[name]=actual
    return hashes


def _bounded():
    _check_frozen_sources()
    path=CAMPAIGN/'bounded_lookahead'
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    module=importlib.import_module('official_spatial_bounded_adapter')
    if Path(module.__file__).resolve()!=(path/'official_spatial_bounded_adapter.py').resolve():
        raise RuntimeError('B887 import resolved outside pinned campaign')
    return module


def _check_runtime_sources()->dict[str,str]:
    hashes={}
    for name,entry in json.loads((ROOT/'source_bindings.json').read_text())['upstream'].items():
        module=importlib.import_module(entry['module']);path=Path(module.__file__)
        if 'relative_path' in entry:path=path.parent/entry['relative_path']
        contents=path.read_bytes()
        if entry['strip_final_newlines']:contents=contents.rstrip(b'\n')
        actual=hashlib.sha256(contents).hexdigest()
        if actual!=entry['sha256']:raise RuntimeError(f'Official source changed: {name}; re-review required')
        hashes[name]=actual
    return hashes


def selected(config:Any)->bool:
    """Reject partial/wrong markers instead of silently falling back to DFlash."""
    d=config if isinstance(config,dict) else vars(config)
    fields=d.get('dflash_config',{})
    marker=fields.get(MARKER)
    if marker is True:
        validate_config(config)
        return True
    if MARKER in fields or fields.get(SPATIAL_MARKER) is True or d.get('algorithm')=='jetspec':
        raise ValueError('JetSpec installer requires its own verified local-block marker; wrong/partial marker')
    return False


def validate_config(config:Any)->dict:
    cfg=_dict(config);fields=cfg.get('dflash_config',{})
    if cfg.get('algorithm')!='jetspec' or fields.get(MARKER) is not True:
        raise ValueError('Marked algorithm=jetspec local-block export is required')
    if cfg.get('architectures')!=['DFlashDraftModel']:
        raise ValueError('JetSpec adapter requires the registered DFlashDraftModel dispatch key')
    allowed=_BASE_DFLASH_FIELDS|set(_DEFAULTS)|{MARKER,SPATIAL_MARKER}
    unknown=set(fields)-allowed
    if unknown:raise ValueError(f'unsupported JetSpec inference configuration: {sorted(unknown)}')
    _bounded().validate_draft_config(config)
    expected={'inference_in_block_causal':True,'visual_num_query_tokens':0,**_DEFAULTS}
    for key,value in expected.items():
        actual=fields.get(key,_DEFAULTS.get(key))
        if type(actual) is not type(value) or actual!=value:
            raise ValueError(f'JetSpec requires dflash_config.{key}={value!r}')
    if cfg.get('intermediate_size')!=12288 or cfg.get('rms_norm_eps')!=1e-6 or cfg.get('num_target_layers')!=36:
        raise ValueError('Unsupported JetSpec backbone dimensions/normalization')
    if cfg.get('model_type')!='qwen3':
        # SpeculativeConfig officially wraps DFlash HF config as EAGLEConfig;
        # this is public loader normalization, not temporary config spoofing.
        from vllm.transformers_utils.configs.eagle import EAGLEConfig
        if cfg.get('model_type')!='eagle' or type(config) is not EAGLEConfig:
            raise ValueError('Unsupported JetSpec model_type/public config wrapper')
    backbone={'hidden_act':'silu','max_position_embeddings':262144,
        'attention_dropout':0.0,'use_sliding_window':False,'sliding_window':None,
        'layer_types':['full_attention']*3}
    for key,value in backbone.items():
        if cfg.get(key)!=value:raise ValueError(f'Unsupported JetSpec backbone configuration {key}')
    for key in ('dspark_config','domino_config','dflash2_config','dflare_config','dfly_config',
                'quantization_config','eagle_config','eagle_aux_hidden_state_layer_ids',*_DEFAULTS):
        if key in cfg:raise ValueError(f'Unsupported top-level learned/inference configuration {key}')
    # The checkpoint has no secondary eagle_config branch; unlike the official
    # EAGLEConfig wrapper, that branch can override auxiliary context/FC policy.
    # Its absence plus the exact dflash field allowlist preserves aux=True.
    # The official head constructor fills an absent/None draft_vocab_size with
    # the full vocab. Accept that exact normalization on subsequent real hooks.
    draft_vocab=cfg.get('draft_vocab_size')
    if draft_vocab is not None and (type(draft_vocab) is not int or draft_vocab!=151936):
        raise ValueError('JetSpec requires full-head draft_vocab_size=151936 or absent/None')
    logit_scale=cfg.get('logit_scale',1.0)
    if type(logit_scale) not in (int,float) or logit_scale!=1.0:
        raise ValueError('JetSpec requires numeric logit_scale=1/default; bool and other types are unsupported')
    if cfg.get('tie_word_embeddings') is not False:
        raise ValueError('Draft export must preserve original tie_word_embeddings=False; served head is external target')
    if cfg.get('jetspec_inference_contract')!=INFERENCE_CONTRACT:
        raise ValueError('JetSpec inference flags/head/layer contract differs from verified checkpoint')
    provenance=cfg.get('jetspec_provenance',{})
    if (provenance.get('source_sha256')!=CHECKPOINT_SHA256 or provenance.get('step')!=75000
            or provenance.get('scope')!=SCOPE or provenance.get('algorithm')!='jetspec'):
        raise ValueError('JetSpec checkpoint provenance does not identify the verified local-block source')
    return cfg


def validate_launch_config(config:Any)->int:
    cfg=config.speculative_config.draft_model_config.hf_config
    validate_config(cfg)
    target=Path('/home/lyh/qwen3vl8b_eval_20261027/artifacts/target')
    if str(getattr(config.model_config,'model',''))!=str(target):
        raise ValueError('Local-block JetSpec mount requires the fixed actual target path')
    target_hashes={'config.json':'5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661',
        'model.safetensors.index.json':'520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070'}
    for name,digest in target_hashes.items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Fixed target source binding changed: {name}')
    if getattr(config.speculative_config,'draft_sample_method','greedy')!='greedy':
        raise ValueError('Local-block JetSpec mount requires public greedy draft sampling')
    return _bounded().validate_launch_config(config)


def validate_actual_target_aliases(draft,target):
    if draft.lm_head is not target.lm_head or draft.model.embed_tokens is not target.model.embed_tokens:
        raise ValueError('JetSpec requires exact target lm_head and embedding objects')


def install()->dict[str,str]:
    """Install the separate local-block JetSpec chain before any model creation.

    Three policy hooks: draft AttentionConfig, returned query CAD, and the
    unchanged official base metadata builder with a causal-True requirement.
    The constructor wrapper only rejects incompatible/partial configurations.
    Target metadata, target model, verifier and sampling operators are untouched.
    """
    global _INSTALLED,_BINDINGS
    b=_bounded()
    from vllm.v1.spec_decode import dflash, llm_base_proposer
    P=dflash.DFlashProposer
    if _INSTALLED:
        if any(getattr(P,name) is not function for name,function in _HOOK_IDENTITIES.items()):
            raise RuntimeError('JetSpec hooks changed after installation')
        b._check_source_bindings();_check_runtime_sources()
        return dict(_BINDINGS)
    for name in ('official_spatial_adapter','official_spatial_graph_adapter','official_spatial_graph_adapter_v2',
                 'official_spatial_graph_adapter_v3','official_spatial_bounded_adapter'):
        module=sys.modules.get(name)
        if module is not None and module._INSTALLED:
            raise RuntimeError('JetSpec requires a fresh process without prior spatial/bounded adapter installation')
    # All checks precede the first class mutation by B887.
    b._check_source_bindings();extra_bindings=_check_runtime_sources()
    original_create=P._create_draft_vllm_config
    original_load=P.load_model
    original_metadata=P.build_per_group_and_layer_attn_metadata
    base_metadata=llm_base_proposer.SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata
    for fn in (original_create,original_metadata):b._require_official_function(fn,dflash)
    b._require_official_function(base_metadata,llm_base_proposer)
    b._require_official_function(original_load,llm_base_proposer)
    bindings=b.install()
    original_init=P.__init__
    spatial_set=P.set_inputs_first_pass

    @wraps(original_init)
    def guarded_init(self,vllm_config,device,runner=None):
        cfg=vllm_config.speculative_config.draft_model_config.hf_config
        if selected(cfg):validate_launch_config(vllm_config)
        return original_init(self,vllm_config,device,runner=runner)

    @wraps(original_load)
    def guarded_load_model(self,target_model):
        result=original_load(self,target_model)
        if selected(self.draft_model_config.hf_config):
            from vllm.model_executor.models.interfaces import supports_multimodal
            target=target_model.get_language_model() if supports_multimodal(target_model) else target_model
            validate_actual_target_aliases(self.model,target)
            self._jetspec_target_alias_verified=True
        return result

    @wraps(original_create)
    def causal_draft_config(self):
        enabled=selected(self.draft_model_config.hf_config)
        resolved=original_create(self)
        if not enabled:return resolved
        # The official resolver independently chooses the draft backend. Only
        # its DFlash-specific causal policy value changes, before backend init.
        return replace(resolved,attention_config=replace(resolved.attention_config,use_non_causal=False))

    @wraps(spatial_set)
    def causal_query_inputs(self,target_token_ids,next_token_ids,target_positions,
            target_hidden_states,token_indices_to_sample,cad,num_rejected_tokens_gpu):
        enabled=selected(self.draft_model_config.hf_config)
        result=spatial_set(self,target_token_ids,next_token_ids,target_positions,target_hidden_states,
            token_indices_to_sample,cad,num_rejected_tokens_gpu)
        if not enabled:return result
        num_queries,indices,query_cad=result
        if query_cad is cad:raise RuntimeError('Official expansion unexpectedly reused target attention metadata')
        return num_queries,indices,replace(query_cad,causal=True)

    @wraps(original_metadata)
    def causal_layer_metadata(self,cad,draft_index=0):
        if not selected(self.draft_model_config.hf_config):
            return original_metadata(self,cad,draft_index)
        if getattr(cad,'causal',None) is not True:
            raise ValueError('JetSpec query metadata must be causal=True before the official builder')
        # Stock DFlash's override asserts False. This calls the unchanged
        # official base operation directly; no attention builder is replaced.
        per_group,per_layer=base_metadata(self,cad,draft_index)
        if not per_layer:raise RuntimeError('JetSpec requires real causal per-layer attention metadata')
        for name,metadata in per_layer.items():
            if getattr(metadata,'causal',None) is not True:
                raise RuntimeError(f'JetSpec backend metadata for {name} is not causal=True; unsupported backend')
        return per_group,per_layer

    P.__init__=guarded_init
    P.load_model=guarded_load_model
    P._create_draft_vllm_config=causal_draft_config
    P.set_inputs_first_pass=causal_query_inputs
    P.build_per_group_and_layer_attn_metadata=causal_layer_metadata
    _HOOK_IDENTITIES.update({name:getattr(P,name) for name in ('__init__','load_model','_create_draft_vllm_config',
        'set_inputs_first_pass','build_per_group_and_layer_attn_metadata')})
    _BINDINGS=dict(bindings,**extra_bindings,bounded_adapter=_check_frozen_sources()['bounded_lookahead/official_spatial_bounded_adapter.py'])
    _INSTALLED=True
    return dict(_BINDINGS)


_HOOK_IDENTITIES={}
