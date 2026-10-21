"""Shared, backend-neutral mechanics for the bounded PR30 audit experiments.

These helpers use production factories, feature collation and FP32-master AdamW.
They never change the algorithm objective or synthesize missing teacher records.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
from pathlib import Path
import torch


def load_bundle(path):
    path=Path(path)
    if path.is_dir():
        from vlm_dflash.dflash.features import FeatureDataset
        features=path/'features'
        head=torch.load(features/'target_head.pt',map_location='cpu',weights_only=False)
        contract=json.loads((path/'contract.json').read_text())
        manifest=json.loads((features/'manifest.json').read_text())
        if head['model_fingerprint'] != contract['target']['model_fingerprint']:
            raise ValueError('Target-head identity mismatch')
        ds=FeatureDataset(features)
        return {'records':[ds[i] for i in range(len(ds))], 'contract':contract,
                'embed_weight':head['embed_tokens'], 'lm_head_weight':head.get('lm_head'),
                'manifest':manifest}
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    if not bundle.get('records') or any(r.get('final_hidden') is None for r in bundle['records']):
        raise ValueError('Audit requires nonempty real capture-v2 records')
    return bundle


def build_module(bundle,cfg,device,seed=42,checkpoint=None):
    from vlm_dflash.algorithms import get_algorithm
    if str(device).startswith('npu'):
        import torch_npu  # explicit backend registration, no CUDA substitution
        torch.npu.set_device(device)
    elif str(device).startswith('cuda'):
        torch.cuda.set_device(device)
    torch.manual_seed(seed)
    contract=copy.deepcopy(bundle['contract'])
    if cfg.get('target',{}).get('model'):
        contract['target']['model_path']=cfg['target']['model']
    algorithm=get_algorithm(cfg['algorithm'])
    algorithm.validate_config(cfg)
    draft_config=algorithm.build_draft_config(contract,cfg)
    module=algorithm.build_trainer_module(draft_config,contract,cfg,bundle['embed_weight'],lm_head_weight=bundle.get('lm_head_weight'))
    module.configure_feature_layout(bundle['manifest']['capture_layer_ids'],bundle['manifest']['hidden_size'])
    if checkpoint is not None:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False) if isinstance(checkpoint,(str,Path)) else checkpoint
        if state.get('identity',{}).get('model_fingerprint') not in (None,contract['target']['model_fingerprint']):
            raise ValueError('Checkpoint target mismatch')
        module.draft_model.load_state_dict(state.get('model',state),strict=True)
    dtype=getattr(torch,cfg.get('train',{}).get('draft_dtype','bfloat16'))
    module.to(device);module.draft_model.to(dtype)
    module.embed_tokens.to(device=device,dtype=dtype);module.lm_head.to(device=device,dtype=dtype)
    module.train()
    return module,draft_config


def make_batch(bundle,indices,device,dtype,context_layer_ids=None):
    from vlm_dflash.dflash.features import collate_features
    records=[bundle['records'][i] for i in indices]
    if any(r.get('final_hidden') is None for r in records):
        raise ValueError('Missing teacher final hidden')
    batch=collate_features(records,bundle['contract']['tokens']['image_token_id'])
    # Keep the source layout intact: configure_feature_layout selects by IDs.
    return {'input_ids':batch['input_ids'].to(device),
            'hidden_states':batch['hidden'].to(device=device,dtype=dtype),
            'loss_mask':batch['loss_mask'].to(device),
            'teacher_hidden':batch['final_hidden'].to(device=device,dtype=dtype),
            'image_mask':batch['image_mask'].to(device),
            'multimodal_position_ids':batch['position3'].to(device)}


def create_optimizer(module,cfg,horizon):
    from vlm_dflash.dflash.train import _make_optimizer_and_scheduler
    masters={n:p.detach().float().clone().requires_grad_(True) for n,p in module.draft_model.named_parameters()}
    fn=_make_optimizer_and_scheduler
    kwargs={'device':next(module.draft_model.parameters()).device} if 'device' in inspect.signature(fn).parameters else {}
    optimizer,scheduler=fn(list(masters.values()),cfg.get('train',{}),horizon,**kwargs)
    return masters,optimizer,scheduler


def optimizer_step(module,masters,optimizer,scheduler,max_grad_norm=1.0):
    with torch.no_grad():
        for n,p in module.draft_model.named_parameters():
            masters[n].grad=None if p.grad is None else p.grad.detach().float()
            p.grad=None
    grad_norm=torch.nn.utils.clip_grad_norm_(list(masters.values()),max_grad_norm)
    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n,p in module.draft_model.named_parameters():p.copy_(masters[n])
    return float(grad_norm)


def digest_tensors(mapping):
    digest=hashlib.sha256()
    for name,value in sorted(mapping.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
