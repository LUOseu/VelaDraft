"""Explicit process-local marked registration; no installed vLLM source mutation."""
import inspect
from pathlib import Path
from . import contracts as c
_INSTALLED=False

def _official(function,module):
    if Path(inspect.getsourcefile(function)).resolve()==Path(module.__file__).resolve():return
    # Only the exact native Torch context wrapper is admitted, never blind unwrap.
    import torch
    wrapped=getattr(function,'__wrapped__',None)
    admitted=False
    for context in (torch.no_grad(),torch.inference_mode()):
        reference=context(lambda:None)
        if getattr(function,'__code__',None) is not reference.__code__:continue
        captured=inspect.getclosurevars(function).nonlocals;expected=inspect.getclosurevars(reference).nonlocals
        factory=captured.get('ctx_factory');wanted=expected['ctx_factory']
        if (set(captured)=={'ctx_factory','func'} and captured['func'] is wrapped
                and getattr(factory,'__func__',None) is wanted.__func__
                and type(getattr(factory,'__self__',None)) is type(wanted.__self__)
                and getattr(getattr(factory,'__self__',None),'mode',None)==getattr(wanted.__self__,'mode',None)):
            admitted=True
    c.require(admitted and wrapped is not None and not hasattr(wrapped,'__wrapped__')
        and Path(inspect.getsourcefile(wrapped)).resolve()==Path(module.__file__).resolve(),
        'Official callable/native wrapper already replaced: '+function.__name__)


def _official_dummy(function,module):
    import torch
    reference=torch.inference_mode()(lambda:None)
    c.require(getattr(function,'__code__',None) is reference.__code__,'Foreign dummy_run inference wrapper.')
    captured=inspect.getclosurevars(function).nonlocals;expected=inspect.getclosurevars(reference).nonlocals
    wrapped=getattr(function,'__wrapped__',None);factory=captured.get('ctx_factory');expected_factory=expected['ctx_factory']
    context=getattr(factory,'__self__',None)
    c.require(set(captured)=={'ctx_factory','func'} and captured['func'] is wrapped
        and getattr(factory,'__func__',None) is expected_factory.__func__ and type(context) is type(expected_factory.__self__)
        and getattr(context,'mode',None) is True and not hasattr(wrapped,'__wrapped__')
        and getattr(getattr(wrapped,'__code__',None),'co_qualname',None)=='DFlashProposer.dummy_run',
        'Stock dummy_run wrapper chain changed.')
    _official(wrapped,module)


def _official_auto_loader(model_utils):
    from vllm.model_executor.model_loader.reload import torchao_decorator
    factory=torchao_decorator.support_quantized_model_reload_from_hp_weights
    _official(factory,torchao_decorator)
    reference=factory(lambda:None);function=model_utils.AutoWeightsLoader.load_weights
    captured=inspect.getclosurevars(function).nonlocals;leaf=getattr(function,'__wrapped__',None)
    c.require(function.__code__ is reference.__code__ and set(captured)=={'original_load_weights'}
        and captured['original_load_weights'] is leaf and leaf is not None and not hasattr(leaf,'__wrapped__')
        and leaf.__code__.co_qualname=='AutoWeightsLoader.load_weights','Foreign AutoWeightsLoader wrapper chain.')
    _official(leaf,model_utils)


def verify_official_interfaces():
    from vllm.v1.spec_decode import dflash, llm_base_proposer
    from vllm.v1.worker import gpu_model_runner
    from vllm.model_executor.models import qwen3_dflash,utils as model_utils
    from vllm.model_executor.layers import vocab_parallel_embedding,logits_processor
    from vllm.model_executor.model_loader import weight_utils
    import vllm.model_executor.model_loader as model_loader
    from .proposer import DFlareProposerRouter
    from . import spatial
    stock=dflash.DFlashProposer
    c.require(gpu_model_runner.DFlashProposer is stock,'Runner proposer dispatch was replaced by foreign code.')
    for function in (stock.__init__,stock.set_inputs_first_pass,stock.build_model_inputs_first_pass):_official(function,dflash)
    for function in (stock.propose,stock._sample_draft_tokens,stock._determine_batch_execution_and_padding,stock.initialize_cudagraph_keys):_official(function,llm_base_proposer)
    for function in (stock.load_model,stock._get_model,stock._maybe_share_embeddings,stock._maybe_share_lm_head,stock.initialize_attn_backend):_official(function,llm_base_proposer)
    for cls,names in ((qwen3_dflash.DFlashQwen3Model,('forward','load_weights','_build_fused_kv_buffers','precompute_and_store_context_kv')),
            (qwen3_dflash.DFlashQwen3ForCausalLM,('forward','load_weights','compute_logits','combine_hidden_states'))):
        for name in names:_official(getattr(cls,name),qwen3_dflash)
    _official(model_utils.process_eagle_weight,model_utils)
    _official_auto_loader(model_utils)
    for function in (weight_utils.default_weight_loader,):_official(function,weight_utils)
    _official(model_loader.get_model,model_loader)
    _official(model_loader.get_model_loader,model_loader)
    from vllm.model_executor.model_loader import default_loader
    _official(default_loader.DefaultModelLoader._prepare_weights,default_loader)
    _official(logits_processor.LogitsProcessor.forward,logits_processor)
    _official(logits_processor.LogitsProcessor._get_logits,logits_processor)
    _official(vocab_parallel_embedding.VocabParallelEmbedding.forward,vocab_parallel_embedding)
    _official_dummy(stock.dummy_run,dflash)
    from .backbone import DFlareOwnedBackbone
    ownership=spatial.validate_compile_ownership(DFlareOwnedBackbone)
    return {'official_callable_origins_verified':True,'stock_compile_objects_unchanged':True,
        'owned_compile_mapping':ownership,'actual_alias_or_loading_executed':False}


def register_models():
    from vllm.model_executor.models import ModelRegistry
    existing=ModelRegistry.models.get(c.ARCH)
    if existing is not None:
        c.require(getattr(existing,'module_name',None)=='dflare_8b_native_adapter.model'
            and getattr(existing,'class_name',None)==c.ARCH,'Foreign DFlare registry entry')
    else:ModelRegistry.register_model(c.ARCH,'dflare_8b_native_adapter.model:'+c.ARCH)


def install():
    global _INSTALLED
    c.runtime_receipt()
    bindings=c.source_checks()
    if _INSTALLED:return bindings
    verify_official_interfaces()
    from vllm.v1.worker import gpu_model_runner
    from .proposer import DFlareProposerRouter
    register_models()
    gpu_model_runner.DFlashProposer=DFlareProposerRouter
    _INSTALLED=True
    return bindings

