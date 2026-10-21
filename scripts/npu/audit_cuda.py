#!/usr/bin/env python3
"""Run one pinned CUDA regression leg with the production trainer.

Invoke this same script in separate processes with --repo BASE / HEAD and the
same bundle/config/device. The JSON plus tensor probes retain finite-gradient,
initial-state and final-state evidence. No environment installation or fallback.
This is a short regression, not a quality or performance benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def probe_indices(numel, device):
    import torch
    count=min(32,numel)
    return torch.arange(count,dtype=torch.long,device=device)*(numel-1)//max(1,count-1)


def tensor_digest(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['dflash','dspark','dfly','domino'], required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-budget-gib', type=float, default=22.0)
    parser.add_argument('--stage-step', type=int, default=75)
    parser.add_argument('--horizon', type=int, default=100)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    # The harness is identical on both legs, while production imports resolve
    # from the explicitly selected checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import transformers
    import vlm_dflash
    from audit_common import load_bundle, build_module, make_batch, create_optimizer, optimizer_step
    assert Path(vlm_dflash.__file__).resolve().is_relative_to(args.repo.resolve())
    assert args.device.startswith('cuda') and torch.cuda.is_available()
    assert 'torch_npu' not in sys.modules
    torch.cuda.set_per_process_memory_fraction(min(1.0,args.memory_budget_gib*2**30/torch.cuda.get_device_properties(args.device).total_memory),args.device)
    cfg = json.loads(args.config.read_text())
    cfg['algorithm'] = {'name':args.method}
    if args.method == 'dspark':
        cfg['algorithm'].update(train_recipe='upstream_dspark',dspark={'markov_rank':256,'enable_confidence_head':True})
    if args.method == 'dfly':
        cfg['algorithm'].update(train_recipe={'name':'upstream_two_stage','cold_start_fraction':0.5},
                                dfly={'context_layer_ids':[1,17,33],'fusion_trainable':True,'fusion_init':'distance','enable_hidden_correction':True})
    cfg['drafter']['attention_backend'] = 'flex_attention'
    cfg['train'].update(feature_mode='offline',micro_batch_size=1,grad_accumulation_steps=1)
    bundle = load_bundle(args.bundle)
    # Capture may contain five layers while this pinned regression uses three
    # draft layers; retain the captured layout and resolve production routing.
    from vlm_dflash.contract import build_target_layer_ids
    bundle['contract']['layers']['draft_layer_ids'] = build_target_layer_ids(
        bundle['contract']['layers']['num_target_layers'],cfg['drafter']['num_hidden_layers'])
    bundle['contract']['layers']['num_draft_layers'] = cfg['drafter']['num_hidden_layers']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    module, draft_config = build_module(bundle,cfg,args.device,seed=args.seed)
    dtype = next(module.draft_model.parameters()).dtype
    initial = tensor_digest(module.draft_model.state_dict().items())
    master, optimizer, scheduler = create_optimizer(module,cfg,args.horizon)
    history, probes = [], {}
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        torch.manual_seed(args.seed+step)
        torch.cuda.manual_seed_all(args.seed+step)
        module.set_training_progress(args.stage_step+step,args.horizon)
        batch = make_batch(bundle,[step % len(bundle['records'])],args.device,dtype,
                           cfg.get('algorithm',{}).get('dfly',{}).get('context_layer_ids',cfg['drafter'].get('capture_layer_ids')))
        optimizer.zero_grad(set_to_none=True)
        module.zero_grad(set_to_none=True)
        query_shapes=[]
        hook=module.draft_model.register_forward_hook(lambda m,a,o:query_shapes.append(list(o.shape)))
        metrics = module(**batch)
        hook.remove()
        assert torch.isfinite(metrics.loss), 'nonfinite loss'
        metrics.loss.backward()
        gradients = [(n,p.grad) for n,p in module.draft_model.named_parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for _,g in gradients)
        required={'dflash':(), 'dspark':('markov_head','confidence_head'),
                  'dfly':('layer_fusion_weights','hidden_correction'),
                  'domino':('prefix_gru','embed_proj')}[args.method]
        for fragment in required:
            assert any(fragment in name and bool(grad.ne(0).any()) for name,grad in gradients), fragment
        for name, grad in gradients:
            flat=grad.detach().float().flatten()
            index=probe_indices(flat.numel(),flat.device)
            probes[f'{step}:{name}']=torch.cat([flat[index],flat.norm().view(1),flat.abs().max().view(1)]).cpu()
        history.append({'step':step,'input_shape':list(batch['input_ids'].shape),'draft_shapes':query_shapes,'loss':float(metrics.loss),'valid_tokens':float(metrics.valid_tokens),
                        'gradient_sha256':tensor_digest(gradients),
                        'extra_metrics':{k:float(v) for k,v in (metrics.extra_metrics or {}).items() if isinstance(v,(int,float))}})
        optimizer_step(module,master,optimizer,scheduler,float(cfg['train'].get('max_grad_norm',1.0)))
    torch.cuda.synchronize()
    result={'scope':'formal model geometry and recipe / natural features / diagnostic microbatch1 CUDA regression',
            'source_config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'scope_overrides':{'feature_mode':'offline','micro_batch_size':1,'grad_accumulation_steps':1,'steps':args.steps,'fixed_stage_step':args.stage_step,'fixed_horizon':args.horizon},'repo':str(args.repo),
            'method':args.method,'config':cfg,'draft_config':draft_config.to_dict(),'steps':args.steps,'seed':args.seed,
            'torch':torch.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(),
            'contended_functional_only':True,'allocator_budget_gib':args.memory_budget_gib,
            'torch_npu_imported':'torch_npu' in sys.modules,'initial_sha256':initial,
            'final_sha256':tensor_digest(module.draft_model.state_dict().items()),'history':history,
            'wall_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    args.output.write_text(json.dumps(result,indent=2))
    torch.save(probes,args.output.with_suffix('.probes.pt'))
    print(json.dumps(result))

def configure_export_verifier():
    """Use the same generate-parity backend policy as production evaluation."""
    from vlm_dflash.algorithms import get_algorithm
    algorithm=get_algorithm('dfly')
    algorithm.configure_verify_backend(False)
    return algorithm


def export64():
    """Compare native and exported weights through the real sequential decoder."""
    parser = argparse.ArgumentParser()
    for name in ['repo','corpus','checkpoint','export-dir','target','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--memory-budget-gib',type=float,default=22.0)
    args=parser.parse_args(sys.argv[2:])
    sys.path.insert(0,str(args.repo.resolve()))
    import torch
    import transformers
    import importlib.util
    from vlm_dflash import runtime
    from vlm_dflash.adapters import adapter_for_model
    from vlm_dflash.algorithms import get_algorithm
    from vlm_dflash.eval.run import load_eval_rows, prepare_prompts, _stops, check_checkpoint_head, _target_head_is_tied
    from vlm_dflash.eval.accept import summarize_samples
    from vlm_dflash.eval.equivalence import sequences_equal
    from vlm_dflash.export.hf import load_export
    from vlm_dflash.target.load import load_target
    assert torch.cuda.is_available() and 'torch_npu' not in sys.modules
    assert importlib.util.find_spec('torch_npu') is None, 'export acceptance requires an environment without torch_npu'
    torch.cuda.set_per_process_memory_fraction(min(1.0,args.memory_budget_gib*2**30/torch.cuda.get_device_properties(0).total_memory),0)
    cfg=json.loads((args.corpus/'r3/config.json').read_text())
    contract=json.loads((args.corpus/'r3/contract.json').read_text())
    cfg['target'].update(model=str(args.target),device='cuda:0')
    contract['target']['model_path']=str(args.target)
    target,processor=load_target(cfg,contract)
    algorithm=configure_export_verifier()
    native,meta=algorithm.load_decoder(args.checkpoint)
    portable,export_meta=load_export(args.export_dir)
    for field in ['model_fingerprint','processor_fingerprint']:
        assert meta['identity'][field]==contract['target'][field], field
        assert export_meta['target'][field]==contract['target'][field], field
    check_checkpoint_head(str(args.checkpoint),meta,_target_head_is_tied(target))
    native_digest=tensor_digest(native.state_dict().items())
    portable_digest=tensor_digest(portable.state_dict().items())
    assert native_digest==portable_digest, 'native/export weights differ'
    rows,media=load_eval_rows({'type':'sharegpt_jsonl','path':str(args.corpus/'data/test.jsonl'),
                              'meta':str(args.corpus/'data/meta.json')},args.corpus)
    assert len(rows)==64
    prompts=prepare_prompts(target,processor,rows,media,max_new_tokens=64,device=target.device)
    stops=_stops(target)
    adapter=adapter_for_model(target)
    results={}; outputs={}; acceptance_signatures={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.with_suffix('.samples.jsonl').open('w') as stream:
        for label,draft in [('native',native),('portable',portable)]:
            draft.to(device=target.device,dtype=torch.bfloat16).eval()
            samples=[]; outputs[label]=[]; acceptance_signatures[label]=[]
            for entry in prompts:
                started=time.monotonic()
                with torch.inference_mode():
                    ids,stats=draft.spec_generate(target=target,input_ids=entry['inputs']['input_ids'],
                        max_new_tokens=64,stop_token_ids=stops,temperature=0,
                        target_kwargs=adapter.forward_kwargs_from_inputs(entry['inputs']),
                        return_stats=True,sequential_target_verification=True)
                runtime.synchronize(target.device)
                tokens=ids[0,entry['prompt_len']:].tolist()
                sample={'id':entry['id'],'new_tokens':len(tokens),'tokens':tokens,
                        'equivalent':sequences_equal(tokens,entry['tokens'],set(stops)),
                        'wall_seconds':time.monotonic()-started,**stats}
                samples.append(sample);outputs[label].append(tokens)
                acceptance_signatures[label].append({k:stats[k] for k in ['accepted_tokens','draft_tokens','verify_steps','acceptance_events']})
                stream.write(json.dumps({'format':label,**sample})+'\n');stream.flush()
            results[label]={**summarize_samples(samples,8),'token_exact':all(s['equivalent'] for s in samples)}
            draft.cpu()
    from vlm_dflash.dflash.draft_model import verify_sdpa_backend_switch_enabled
    report={'scope':'64 fixed natural heldout / max_new_tokens64 / sequential verification',
            'verification':'sequential','verify_sdpa_backend':'parity',
            'verify_sdpa_backend_switch_enabled':verify_sdpa_backend_switch_enabled(),
            'torch':torch.__version__,'transformers':transformers.__version__,'torch_npu_imported':'torch_npu' in sys.modules,'torch_npu_installed':False,
            'weight_sha256':native_digest,'target_fingerprint':contract['target']['model_fingerprint'],
            'processor_fingerprint':contract['target']['processor_fingerprint'],
            'train_recipe':export_meta.get('train_recipe'),'allocator_budget_gib':args.memory_budget_gib,
            'format_outputs_equal':outputs['native']==outputs['portable'],
            'format_acceptance_equal':acceptance_signatures['native']==acceptance_signatures['portable'],
            'results':results}
    args.output.write_text(json.dumps(report,indent=2))
    assert report['format_outputs_equal'] and report['format_acceptance_equal'] and all(r['token_exact'] and r['accepted_tokens']>0 for r in results.values())


def shared_forward():
    """Record the same frozen teacher-feature/anchor forward on NPU or CUDA."""
    parser=argparse.ArgumentParser()
    for name in ['repo','bundle','config','checkpoint','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--memory-budget-gib',type=float,default=22.0)
    args=parser.parse_args(sys.argv[2:])
    sys.path.insert(0,str(args.repo.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import torch
    from audit_common import load_bundle,build_module,make_batch,digest_tensors
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(min(1.0,args.memory_budget_gib*2**30/torch.cuda.get_device_properties(args.device).total_memory),args.device)
    cfg=json.loads(args.config.read_text())
    cfg['drafter']['attention_backend']='sdpa_chunked'
    cfg['drafter']['attention_chunk_size']=32
    bundle=load_bundle(args.bundle)
    module,_=build_module(bundle,cfg,args.device,seed=42,checkpoint=args.checkpoint)
    module.set_training_progress(75,100)
    compute_dtype=next(module.draft_model.parameters()).dtype
    batch=make_batch(bundle,[0],args.device,compute_dtype)
    torch.manual_seed(42)
    anchors=module._sample_anchor_positions(batch['input_ids'].shape[1],batch['loss_mask'].cpu(),torch.device('cpu'))
    batch['anchor_table']=anchors
    captured=[]
    student_head=[]
    hook=module.draft_model.register_forward_hook(lambda m,a,o:captured.append(o.detach().cpu()))
    def capture_student_head(head, inputs, output):
        # Production trainer projects corrected student hidden states before
        # any teacher-distribution streaming invokes this same frozen head.
        if not student_head:
            student_head.append((inputs[0].detach().cpu(),output.detach().cpu()))
    head_hook=module.lm_head.register_forward_hook(capture_student_head)
    with torch.no_grad():
        metrics=module(**batch)
    hook.remove()
    head_hook.remove()
    assert len(captured)==1 and len(student_head)==1
    source_files=['vlm_dflash/dflash/trainer_core.py','vlm_dflash/dfly/trainer.py',
                  'vlm_dflash/dfly/model.py','vlm_dflash/dflash/draft_model.py']
    source_sha={name:hashlib.sha256((args.repo/name).read_bytes()).hexdigest() for name in source_files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'draft_hidden':captured[0],'post_correction_hidden':student_head[0][0],
                'student_logits':student_head[0][1], 'dtype':str(captured[0].dtype),
                'student_logits_dtype':str(student_head[0][1].dtype),'source_sha256':source_sha,
                'anchor_table':anchors,'loss':float(metrics.loss),
                'weights_sha256':digest_tensors(module.draft_model.state_dict()),
                'input_sha256':digest_tensors({k:v for k,v in batch.items() if torch.is_tensor(v)}),
                'device':args.device,'config':cfg,'scope':'shared natural teacher features; sample0; frozen CPU anchor selection'},args.output)


def materialize_lite():
    """Reattach an independently verified local teacher embedding to a subset."""
    parser=argparse.ArgumentParser()
    for name in ['lite','embed','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-embed-sha256',required=True)
    args=parser.parse_args(sys.argv[2:])
    import torch
    embed=torch.load(args.embed,map_location='cpu',weights_only=False)
    digest=hashlib.sha256(embed.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest==args.expected_embed_sha256, 'teacher embedding mismatch'
    bundle=torch.load(args.lite,map_location='cpu',weights_only=False)
    assert len(bundle['records'])==8, 'expected fixed first8 training records'
    assert bundle.get('lm_head_weight') is None, 'this materializer requires the tied teacher head'
    bundle['embed_weight']=embed
    bundle['lm_head_weight']=None
    info={'source_lite_sha256':hashlib.sha256(args.lite.read_bytes()).hexdigest(),
          'embedding_raw_tensor_sha256':digest,'samples':8,
          'sample_ids':[str(r.get('sample_id')) for r in bundle['records']],
          'scope':'first8 natural training capture records; original teacher embedding reconstructed locally'}
    bundle['materialization']=info
    temporary=args.output.with_suffix('.materializing')
    torch.save(bundle,temporary)
    temporary.replace(args.output)
    args.output.with_suffix('.materialization.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info))


def compare_shared_forward():
    """Compare every saved value against the pre-existing H1 BF16 envelope."""
    parser=argparse.ArgumentParser()
    for name in ['npu','cuda','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(sys.argv[2:])
    import torch
    npu=torch.load(args.npu,map_location='cpu',weights_only=False)
    cuda=torch.load(args.cuda,map_location='cpu',weights_only=False)
    identity={key:npu[key]==cuda[key] for key in ['weights_sha256','input_sha256','dtype','student_logits_dtype','source_sha256']}
    identity['anchors_equal']=all(torch.equal(a,b) for a,b in zip(npu['anchor_table'],cuda['anchor_table']))
    checks={}
    for key in ['draft_hidden','post_correction_hidden','student_logits']:
        a=npu[key].float();b=cuda[key].float()
        same_shape=a.shape==b.shape
        if not same_shape:
            checks[key]={'passed':False,'npu_shape':list(a.shape),'cuda_shape':list(b.shape)};continue
        delta=(a-b).abs();tolerance=.02+.02*b.abs()
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        failures=(delta>tolerance)|~torch.isfinite(delta)
        checks[key]={'passed':finite and not bool(failures.any()),'shape':list(a.shape),
                     'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
                     'violating_elements':int(failures.sum()),'total_elements':a.numel(),
                     'violating_fraction':float(failures.sum())/a.numel(),
                     'relative_l2_error':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-12))}
    loss_error=abs(npu['loss']-cuda['loss'])
    checks['loss']={'passed':loss_error<=.02+.02*abs(cuda['loss']),'npu':npu['loss'],'cuda':cuda['loss'],'abs_error':loss_error,'exact':npu['loss']==cuda['loss']}
    passed=all(identity.values()) and all(check['passed'] for check in checks.values())
    metadata_keys=['device','dtype','student_logits_dtype','weights_sha256','input_sha256','source_sha256']
    metadata={'npu':{k:npu.get(k) for k in metadata_keys},'cuda':{k:cuda.get(k) for k in metadata_keys}}
    report={'status':'PASS' if passed else 'FAILED','metadata':metadata,
            'scope':('FP32 precision diagnostic only; does not replace BF16 gate' if npu['dtype']=='torch.float32' else 'BF16 cross-backend forward gate'),
            'identity_checks':identity,'checks':checks,
            'tolerance':{'rtol':.02,'atol':.02,'reference':'CUDA','source':'pre-existing scripts/npu/operator_smoke.py BF16 envelope'},
            'artifact_sha256':{'npu':hashlib.sha256(args.npu.read_bytes()).hexdigest(),'cuda':hashlib.sha256(args.cuda.read_bytes()).hexdigest()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    return 0 if passed else 1


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='compare-forward':
        raise SystemExit(compare_shared_forward())
    elif len(sys.argv)>1 and sys.argv[1]=='materialize-lite':
        materialize_lite()
    elif len(sys.argv)>1 and sys.argv[1]=='shared-forward':
        shared_forward()
    elif len(sys.argv)>1 and sys.argv[1]=='export64':
        export64()
    else:
        main()
