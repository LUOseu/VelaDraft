"""Source-bound registration and class router; no ordinary DFlash overwrite."""
from __future__ import annotations
import hashlib,inspect,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
if 'contracts' in sys.modules and Path(sys.modules['contracts'].__file__).resolve()!=ROOT/'contracts.py':raise RuntimeError('Foreign contracts module')
import contracts
_REGISTERED={}
_INSTALLED=False
_APPROVAL=None

def _own_origins():
    if not sys.path or Path(sys.path[0]).resolve()!=ROOT:raise RuntimeError('Own adapter root must lead sys.path')
    for name in ('contracts','domino_logic','domino_model','domino_proposer','spatial_math','install','export_domino','preflight','test_domino','test_interface','test_runtime_approval'):
        m=sys.modules.get(name)
        if m is not None and Path(m.__file__).resolve()!=ROOT/(name+'.py'):raise RuntimeError(f'Foreign module origin: {name}')

def _official(f,module):
    path=Path(module.__file__).resolve()
    pins=json.loads((ROOT/'source_bindings.json').read_text())['upstream']
    if str(path) not in pins or hashlib.sha256(path.read_bytes()).hexdigest()!=pins[str(path)] or Path(f.__code__.co_filename).resolve()!=path:raise RuntimeError('Prior foreign official hook/source detected')

def check_runtime():
    contracts.verify_dependencies();_own_origins()
    import torch
    import vllm
    from vllm.model_executor.models import qwen3_dflash as m
    from vllm.v1.spec_decode import dflash as p,llm_base_proposer as b
    if vllm.__version__!='0.22.1':raise RuntimeError('Pinned vLLM0.22.1 required')
    if torch.cuda.is_initialized():raise RuntimeError('Install before CUDA initialization')
    site=Path('/home/lyh/miniforge3/envs/c1vllm/lib/python3.12/site-packages')
    if Path(vllm.__file__).resolve().parent!=Path('/home/lyh/qwen3vl8b_eval_20261027/runtime_frozen_v1/lib/python3.12/site-packages/vllm') or Path(torch.__file__).resolve().parent!=site/'torch':raise RuntimeError('Wrong installed package origin')
    from vllm.compilation import decorators
    _official(m.DFlashQwen3Model.__init__,decorators)
    init_capture=inspect.getclosurevars(m.DFlashQwen3Model.__init__).nonlocals
    _official(init_capture['old_init'],m)
    mark=inspect.getclosurevars(m.DFlashQwen3Model.__call__).nonlocals['_mark_dynamic_inputs']
    if init_capture['dynamic_arg_dims']!={'input_ids':0,'positions':0,'input_embeds':0} or init_capture['dynamic_arg_dims'] is not inspect.getclosurevars(mark).nonlocals['dynamic_arg_dims']:raise RuntimeError('Ordinary DFlash metadata was changed')
    for name in ('official_spatial_adapter','official_spatial_graph_adapter','official_spatial_graph_adapter_v2','official_spatial_graph_adapter_v3','official_spatial_bounded_adapter','official_jetspec_adapter'):
        if getattr(sys.modules.get(name),'_INSTALLED',False):raise RuntimeError('Prior adapter installation is forbidden')
    for f in (p.DFlashProposer.__init__,p.DFlashProposer.set_inputs_first_pass,p.DFlashProposer.build_model_inputs_first_pass,p.DFlashProposer._create_draft_vllm_config,p.DFlashProposer.build_per_group_and_layer_attn_metadata):_official(f,p)
    for f in (p.DFlashProposer.propose,p.DFlashProposer._sample_draft_tokens,p.DFlashProposer._raise_if_mrope,p.DFlashProposer._determine_batch_execution_and_padding,p.DFlashProposer.initialize_cudagraph_keys,p.DFlashProposer.initialize_attn_backend,p.DFlashProposer.load_model,p.DFlashProposer._maybe_share_embeddings,p.DFlashProposer._maybe_share_lm_head):_official(f,b)
    for f in (m.DFlashQwen3Model.precompute_and_store_context_kv,m.DFlashQwen3Model.forward,m.DFlashQwen3Model._build_fused_kv_buffers,m.DFlashQwen3Model.load_weights,m.DFlashQwen3ForCausalLM.forward,m.DFlashQwen3ForCausalLM.compute_logits,m.DFlashQwen3ForCausalLM.combine_hidden_states,m.DFlashQwen3ForCausalLM.load_weights,m.DFlashQwen3DecoderLayer.forward,m.DFlashQwen3Attention.forward):_official(f,m)
    from vllm.model_executor.models import utils as model_utils
    from vllm.model_executor.model_loader import weight_utils
    from vllm.model_executor.layers import vocab_parallel_embedding as embeddings
    for f in (model_utils.process_eagle_weight,model_utils.get_draft_quant_config,model_utils.maybe_prefix):_official(f,model_utils)
    from vllm.model_executor.model_loader.reload import torchao_decorator
    factory=torchao_decorator.support_quantized_model_reload_from_hp_weights
    _official(factory,torchao_decorator)
    reference=factory(lambda:None);loader=model_utils.AutoWeightsLoader.load_weights
    leaf=getattr(loader,'__wrapped__',None);captured=inspect.getclosurevars(loader).nonlocals
    if loader.__code__ is not reference.__code__ or set(captured)!={'original_load_weights'} or captured['original_load_weights'] is not leaf or hasattr(leaf,'__wrapped__') or leaf.__code__.co_qualname!='AutoWeightsLoader.load_weights':raise RuntimeError('Foreign loader wrapper chain')
    _official(leaf,model_utils)
    _official(weight_utils.default_weight_loader,weight_utils)
    for f in (embeddings.VocabParallelEmbedding.__init__,embeddings.VocabParallelEmbedding.forward,embeddings.ParallelLMHead.__init__):_official(f,embeddings)
    from vllm.model_executor import model_loader
    from vllm.model_executor.model_loader import default_loader
    for f in (model_loader.get_model_loader,model_loader.get_model):_official(f,model_loader)
    _official(default_loader.DefaultModelLoader.get_all_weights,default_loader)
    ref=torch.inference_mode()(lambda:None);fn=p.DFlashProposer.dummy_run
    if fn.__code__ is not ref.__code__:raise RuntimeError('Foreign dummy wrapper')
    captured=inspect.getclosurevars(fn).nonlocals;expected=inspect.getclosurevars(ref).nonlocals
    factory=captured.get('ctx_factory');context=getattr(factory,'__self__',None);leaf=getattr(fn,'__wrapped__',None)
    if set(captured)!={'ctx_factory','func'} or captured['func'] is not leaf or getattr(factory,'__func__',None) is not expected['ctx_factory'].__func__ or type(context) is not type(expected['ctx_factory'].__self__) or getattr(context,'mode',None) is not True or hasattr(leaf,'__wrapped__') or leaf.__code__.co_qualname!='DFlashProposer.dummy_run':raise RuntimeError('Foreign dummy wrapper chain')
    _official(leaf,p)

def register_models():
    check_runtime()
    from vllm.model_executor.models import ModelRegistry
    for arch in (contracts.RAW_ARCH,contracts.RESOLVED_ARCH):
        if arch in _REGISTERED:
            if ModelRegistry.models.get(arch) is not _REGISTERED[arch]:raise RuntimeError('Domino registry entry changed')
            continue
        if arch in ModelRegistry.models:raise RuntimeError(f'Architecture already registered: {arch}')
        ModelRegistry.register_model(arch,'domino_model:DominoForCausalLM')
        _REGISTERED[arch]=ModelRegistry.models[arch]
    return dict(_REGISTERED)

def install():
    global _INSTALLED,_APPROVAL
    _APPROVAL=contracts.require_full_gate(os.environ.get('DOMINO_FULL_PREFLIGHT',str(ROOT/'FULL_PREFLIGHT.json')),os.environ.get('DOMINO_APPROVED_PREFLIGHT_SHA256'),os.environ.get('DOMINO_RUNTIME_APPROVAL'),os.environ.get('DOMINO_APPROVED_RUNTIME_SHA256'))
    register_models()
    from vllm.v1.worker import gpu_model_runner as runner
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from domino_proposer import DominoProposer
    if _INSTALLED:
        if runner.DFlashProposer is not DominoProposer:raise RuntimeError('Domino router replaced')
        return
    if runner.DFlashProposer is not DFlashProposer:raise RuntimeError('Prior proposer router detected')
    runner.DFlashProposer=DominoProposer
    _INSTALLED=True
