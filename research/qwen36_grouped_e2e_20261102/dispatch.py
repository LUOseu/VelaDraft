"""Select an existing vLLM assignment path; no kernels or expert choices change."""
from functools import wraps
import hashlib
import inspect
from pathlib import Path

_state={'installed':False,'policy':'stock','min_query_tokens':3,'max_query_tokens':8,'captured_grouped_calls':{}}


def eligible(num_tokens,top_k_num,global_num_experts,expert_map,*,use_int8_w8a16=False,use_int4_w4a16=False,block_shape=None):
    return (3<=num_tokens<=8 and top_k_num==8 and global_num_experts==256 and expert_map is None
            and not use_int8_w8a16 and not use_int4_w4a16 and block_shape is None)


def make_dispatcher(original,aligned,is_capturing=lambda:False):
    @wraps(original)
    def dispatch(topk_ids,config,num_tokens,top_k_num,global_num_experts,expert_map,*,
                 use_int8_w8a16=False,use_int4_w4a16=False,block_shape=None,ignore_invalid_experts=False):
        if eligible(num_tokens,top_k_num,global_num_experts,expert_map,use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=use_int4_w4a16,block_shape=block_shape):
            if is_capturing():
                key=str(num_tokens);calls=_state['captured_grouped_calls'];calls[key]=calls.get(key,0)+1
            return aligned(topk_ids,config['BLOCK_SIZE_M'],global_num_experts,expert_map,
                           ignore_invalid_experts=ignore_invalid_experts)
        return original(topk_ids,config,num_tokens,top_k_num,global_num_experts,expert_map,
            use_int8_w8a16=use_int8_w8a16,use_int4_w4a16=use_int4_w4a16,
            block_shape=block_shape,ignore_invalid_experts=ignore_invalid_experts)
    return dispatch


def install():
    if _state['installed']:return
    import torch
    import vllm
    import importlib
    fm=importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    tm=importlib.import_module('vllm.model_executor.layers.fused_moe.experts.triton_moe')
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    assert vllm.__version__=='0.32.0'
    original=fm._prepare_expert_assignment
    assert tm._prepare_expert_assignment is original
    assert list(inspect.signature(original).parameters)[:6]==['topk_ids','config','num_tokens','top_k_num','global_num_experts','expert_map']
    wrapper=make_dispatcher(original,moe_align_block_size,torch.cuda.is_current_stream_capturing)
    fm._prepare_expert_assignment=wrapper;tm._prepare_expert_assignment=wrapper
    _state.update(installed=True,policy='official_grouping_for_q3_to_q8',
        native_source_sha256=hashlib.sha256(Path(fm.__file__).read_bytes()).hexdigest(),
        dispatcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def metadata():
    return dict(_state,captured_grouped_calls=dict(_state['captured_grouped_calls']))
