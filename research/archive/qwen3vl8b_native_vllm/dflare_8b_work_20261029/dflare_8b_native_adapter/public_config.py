"""Actual public vLLM config parsing with CUDA hidden and no serving model."""
from dataclasses import dataclass
from pathlib import Path

from . import contracts as c


def engine_kwargs(export_dir):
    return dict(model=c.TARGET,dtype='bfloat16',max_model_len=8192,max_num_seqs=1,
        max_num_batched_tokens=8192,tensor_parallel_size=1,block_size=16,enforce_eager=False,
        attention_config={'backend':'FLASH_ATTN'},
        compilation_config={'mode':3,'cudagraph_mode':'PIECEWISE','cudagraph_capture_sizes':[8],'max_cudagraph_capture_size':8},
        speculative_config={'model':str(Path(export_dir).resolve()),'method':'dflash','num_speculative_tokens':7,
            'draft_tensor_parallel_size':1,'attention_backend':'FLASH_ATTN','use_local_argmax_reduction':False})


@dataclass
class ConfigEnvelope:
    model_config:object
    speculative_config:object
    attention_config:object
    load_config:object
    cache_config:object
    parallel_config:object
    scheduler_config:object
    compilation_config:object
    kernel_config:object=None


def materialize_public_config(export_dir):
    from vllm.config import (ModelConfig,ParallelConfig,SpeculativeConfig,AttentionConfig,LoadConfig,
        CacheConfig,SchedulerConfig,CompilationConfig)
    from vllm.engine.arg_utils import EngineArgs
    from .install import register_models
    from .proposer import DFlareProposerRouter
    register_models();kwargs=engine_kwargs(export_dir)
    args=EngineArgs(**kwargs)
    target=ModelConfig(model=c.TARGET,dtype='bfloat16',max_model_len=8192)
    parallel=ParallelConfig(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
    spec=SpeculativeConfig(**kwargs['speculative_config'],target_model_config=target,target_parallel_config=parallel)
    envelope=ConfigEnvelope(target,spec,AttentionConfig(**kwargs['attention_config']),LoadConfig(),
        CacheConfig(block_size=16),parallel,
        SchedulerConfig(max_num_seqs=1,max_num_batched_tokens=8192,max_model_len=8192,is_encoder_decoder=False),
        CompilationConfig(**kwargs['compilation_config']))
    c.validate_launch_config(envelope)
    proposer=object.__new__(DFlareProposerRouter);proposer.vllm_config=envelope;proposer.speculative_config=spec
    clone=proposer._create_draft_vllm_config()
    c.validate_launch_config(clone)
    c.require(clone.attention_config.use_non_causal and not envelope.attention_config.use_non_causal,
        'Official DFlash clone must enable only draft non-causal query attention')
    c.require(spec.draft_model_config.uses_mrope,'Actual draft ModelConfig did not select MRoPE')
    return {'EngineArgs_parsed':True,'ModelConfig_parsed':True,'SpeculativeConfig_parsed':True,
        'draft_model_config_model':spec.draft_model_config.model,'speculative_model':spec.model,
        'resolved_draft_hf_config_sha256':c.draft_config_digest(spec.draft_model_config.hf_config),
        'effective_stock_load_configs':c.validate_effective_load_configs(envelope),
        'raw_architectures':[c.ARCH],'resolved_architectures':spec.draft_model_config.hf_config.architectures,
        'actual_wrapper_class':type(spec.draft_model_config.hf_config).__name__,
        'actual_inner_class':type(spec.draft_model_config.hf_config.model).__name__,
        'actual_model_type':spec.draft_model_config.hf_config.model_type,
        'actual_inner_model_type':spec.draft_model_config.hf_config.model.model_type,
        'public_S':7,'query_B':8,'target_backend':'FLASH_ATTN','draft_backend':spec.attention_backend.name,
        'parallel_drafting':spec.parallel_drafting,'uses_mrope':spec.draft_model_config.uses_mrope,
        'page_size':16,'max_num_seqs':1,'tensor_parallel_size':1,'compile_mode':envelope.compilation_config.mode.name,
        'cudagraph_mode':envelope.compilation_config.cudagraph_mode.name,
        'cudagraph_capture_sizes':envelope.compilation_config.cudagraph_capture_sizes,
        'max_cudagraph_capture_size':envelope.compilation_config.max_cudagraph_capture_size,
        'official_draft_config_clone_executed':True,'target_causal_unchanged':True,'draft_non_causal':True,
        'target_config_sha256':c.sha(Path(c.TARGET)/'config.json'),'engine_kwargs':kwargs,
        'serving_model_constructed':False,'standard_EngineArgs_create_engine_config_executed':False},envelope
