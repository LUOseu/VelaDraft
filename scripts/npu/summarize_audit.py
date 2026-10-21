#!/usr/bin/env python3
"""Recompute bounded audit tables/profile records from checked-in small raw JSON.

Does not rerun hardware or reconstruct gradients from missing tensor artifacts.
--check verifies generated JSON/Markdown and the existing matrix's hash bindings.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = '4625ef676cd77b37a7fa9ed1f0e9f3864ee5f5ca'
GEOMETRY = dict(num_hidden_layers=3, hidden_size=2048, intermediate_size=11008,
                num_attention_heads=16, num_key_value_heads=2, head_dim=128, block_size=8)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_pass(record, label):
    if str(record.get('status', '')).lower() != 'pass':
        raise ValueError(f'{label}: raw artifact is not PASS')


def require(condition, label):
    if not condition:
        raise ValueError(label)



def require_geometry(actual, label, *, include_block=True):
    keys = GEOMETRY if include_block else {k:v for k,v in GEOMETRY.items() if k != 'block_size'}
    require(isinstance(actual, dict), label+' missing geometry')
    for key, expected in keys.items():
        require(type(actual.get(key)) is int and actual[key] == expected,
                f'{label} geometry.{key} contradicts formal geometry')


def measured_attention_chunks(record, mode, label):
    """Read completed execution counters, never the analytical memory plan."""
    require_geometry(record.get('geometry'), label)
    require(record.get('mode') == mode and record.get('dtype') == 'bfloat16', label+' mode/dtype mismatch')
    pilots = record.get('warmup_updates', [])
    require(len(pilots) == 5, label+' requires five pilot updates')
    estimate = record.get('first5_estimate', {})
    require(estimate.get('seconds') == [step['seconds'] for step in pilots], label+' first-five estimate mismatch')
    measured = record.get('measured_updates', [])
    require(len(measured) == 3, label+' requires three measured updates')
    expected_backend = 'sdpa_chunked' if mode == 'chunked_checkpoint' else mode
    expected_chunks = math.ceil(record['rows']['allocated_query_rows']/128) if mode == 'chunked_checkpoint' else 1
    for step in pilots + measured:
        require(math.isfinite(step['seconds']) and step['seconds'] > 0, label+' invalid measured time')
        require(type(step['peak_hbm_bytes']) is int and step['peak_hbm_bytes'] > 0, label+' invalid measured HBM')
    for step in measured:
        attention = step['attention']
        window = attention['measurement_window']
        calls, chunks = window['completed_attention_calls'], window['chunks']
        require(type(calls) is int and calls > 0 and type(chunks) is int, label+' invalid attention counters')
        require(attention['effective_backends'] == [expected_backend] and window['effective_backends'] == [expected_backend], label+' actual attention backend mismatch')
        require(chunks == calls * expected_chunks, label+' actual chunk count mismatch')
    return chunks // calls



def require_attention_geometry(plan, label):
    for field, expected in (('num_heads',16), ('head_dim',128), ('block_size',8)):
        require(type(plan.get(field)) is int and plan[field] == expected,
                f'{label} attention geometry.{field} mismatch')


def require_recipe_observation(observation, method):
    if method == 'dspark':
        name = 'upstream_dspark'
        stages = [dict(name='verifier_kl_confidence', weighting='decay', ce_coef=0.,
                       lk_coef=1., e2e_tv_coef=0., confidence_coef=1., lk_loss_type='kl')]
    else:
        name = 'upstream_two_stage'
        stages = [dict(name='cold_start_lk_dpace', weighting='dpace', ce_coef=0., lk_coef=1., e2e_tv_coef=0.),
                  dict(name='e2e_tv', weighting='dpace', ce_coef=1., lk_coef=0., e2e_tv_coef=1.)]
    recipe = observation['recipe']
    require(recipe.get('name') == name and len(recipe.get('stages',[])) == len(stages), method+' recipe identity/stage count')
    def check(actual, expected):
        require(all(actual.get(key) == value for key,value in expected.items()), method+' recipe coefficients/weighting mismatch')
    for actual,expected in zip(recipe['stages'],stages):
        check(actual,expected)
    for row in observation['observations']:
        index=row['stage_index']
        require(type(index) is int and 0 <= index < len(stages), method+' recipe observed stage index')
        check(row['stage'],stages[index])


def require_dfly_checkpoint_geometry(metadata):
    shapes=metadata['model_tensor_shapes']
    expected={'layer_fusion_weights':[3,3], 'norm.weight':[2048],
              'hidden_norm.weight':[2048], 'fc.weight':[2048,6144],
              'hidden_correction.hidden_norm.weight':[2048],
              'hidden_correction.embed_norm.weight':[2048],
              'hidden_correction.gate_proj.weight':[2048,4096],
              'hidden_correction.up_proj.weight':[2048,4096],
              'hidden_correction.down_proj.weight':[2048,2048]}
    layer_shapes={'self_attn.q_proj.weight':[2048,2048],
                  'self_attn.k_proj.weight':[256,2048], 'self_attn.v_proj.weight':[256,2048],
                  'self_attn.o_proj.weight':[2048,2048], 'self_attn.q_norm.weight':[128],
                  'self_attn.k_norm.weight':[128], 'mlp.gate_proj.weight':[11008,2048],
                  'mlp.up_proj.weight':[11008,2048], 'mlp.down_proj.weight':[2048,11008],
                  'input_layernorm.weight':[2048], 'post_attention_layernorm.weight':[2048]}
    for index in range(3):
        expected.update({f'layers.{index}.{key}':value for key,value in layer_shapes.items()})
    require(shapes == expected, 'DFly actual checkpoint tensor geometry mismatch')


def shared_forward_failure(record, path):
    require(str(record.get('status')).upper() in ('FAIL', 'FAILED'), 'expected failed shared-forward evidence')
    return dict(id='dfly-npu-export-cuda', method='dfly', objective_profile='upstream_two_stage',
                stage='portable_export', backend='cuda', mode='inference', status='FAILED',
                reason='Frozen BF16 cross-backend forward tolerance failed. Portable-format/usability evaluation is independent and cannot override this failed numerical gate.',
                evidence=[path])


def phase_work(backend, sides):
    rows=[]
    for side,steps in sides.items():
        require(len(steps)==30 and all(s['measured'] is True for s in steps), 'A measured phase window')
        row=dict(backend=backend,compaction=side,updates=len(steps))
        for field in ('forward','backward','update'):
            row[field+'_ms']=1000*statistics.mean(s[field+'_seconds'] for s in steps)
        for field in ('student_projected_rows_forward','teacher_projected_rows_forward','projected_rows_including_backward_recomputation'):
            row[field]=sum(s[field] for s in steps)
        rows.append(row)
    return rows


def completed_cuda_compaction(directory, npu_directory):
    """Validate a complete guarded run before any CUDA performance promotion."""
    directory,npu_directory=Path(directory),Path(npu_directory)
    require(read(directory/'guard.json')['returncode']==0, 'CUDA A full-run guard failed')
    require_pass(read(directory/'result.json'),'CUDA A final result')
    parity=read(directory/'parity.json');require_pass(parity,'CUDA A parity')
    require(len(parity['cases'])==8 and all(c['passed'] and c['bf16_envelope_pass'] for c in parity['cases']), 'CUDA A eight BF16 parity cases')
    npu_parity=read(npu_directory/'parity.json')
    require(parity['binding']['plan']==npu_parity['binding']['plan'], 'CUDA A original NPU plan')
    require(read(directory/'batches.json')==read(npu_directory/'batches.json'), 'CUDA A original batch plan')
    require(parity['tolerance']['bf16_atol']==parity['tolerance']['bf16_rtol']==0.02, 'CUDA A frozen tolerance')
    phase_sides={'off':[],'on':[]};pairs=[]
    for i in range(3):
        off,on=[read(directory/f'pair{i}_{side}.json') for side in ('off','on')]
        npu_off=read(npu_directory/f'pair{i}_off.json')
        require(off['compaction'] is False and on['compaction'] is True, 'CUDA A switches')
        require(off['initial_state_hashes']==on['initial_state_hashes'], 'CUDA A paired initial states')
        require(off['initial_sha256']==npu_off['initial_sha256'], 'CUDA A same NPU checkpoint weights')
        require(off['binding']==on['binding']==parity['binding'], 'CUDA A source/input binding')
        for side,leg in (('off',off),('on',on)):
            require_geometry(leg['model_geometry'],'CUDA A '+side,include_block=False)
            require([s['measured'] for s in leg['steps']]==[False]*5+[True]*10, 'CUDA A complete measurement window')
            for key in ('seed','optimizer','optimizer_horizon','optimizer_start_step','objective_progress_start'):
                require(leg[key]==npu_off[key], 'CUDA A matched '+key)
            phase_sides[side].extend(s for s in leg['steps'] if s['measured'])
        for a,b,n in zip(off['steps'],on['steps'],npu_off['steps']):
            for key in ('batch','valid_tokens','allocated_rows','kept_rows','padding_fraction'):
                require(a[key]==b[key]==n[key], 'CUDA A same actual work '+key)
        times=[statistics.mean(s['wall_seconds'] for s in leg['steps'] if s['measured']) for leg in (off,on)]
        peaks=[leg['peak_hbm_bytes']/2**30 for leg in (off,on)]
        pairs.append(dict(pair=i,off_seconds=times[0],on_seconds=times[1],time_ratio=times[0]/times[1],off_gib=peaks[0],on_gib=peaks[1]))
    return dict(pairs=pairs,phase_work=phase_work('cuda',phase_sides),binding=parity['binding'],
                strict_bitwise_cases=sum(c['strict_bitwise_contract_pass'] is True for c in parity['cases']),bf16_envelope_cases=8)


def generate(directory, cuda_directory):
    directory, cuda_directory = Path(directory), Path(cuda_directory)
    raw = directory / 'raw'
    profiles = []
    lines = ['# PR30 审稿补强：可重算证据报告（2026-10-21）', '',
             '本报告区分正式模型几何、实际训练工作量与证据适用范围。NPU recipe、DFly compaction、attention 三路径以及 CUDA base/head 功能回归已有结果；跨后端匹配性能对照与 CUDA 上完整导出验收仍按实际状态列出。', '',
             '所有训练几何为 3 层、hidden 2048、FFN 11008、16 Q / 2 KV heads、head_dim 128、block8、BF16；诊断 recipe / CUDA / attention 使用 microbatch1，A compaction 使用正式 microbatch8。自然数据来自预先固定的 64 条 capture-v2 子集，未把合成张量或生成几何图片混作自然数据。', '',
             '## 请求与证据对照', '',
             '| 请求 | 状态与边界 |', '|---|---|',
             '| 正式 DSpark / DFly objective 与阶段 | PASS：真实 teacher 分布、有效监督、专属 head 梯度；短程诊断，不是质量训练 |',
             '| DFly stage 边界恢复 | PASS：固定 horizon12，在已提交 step5/6 重新加载，完整最终状态逐项相等 |',
             '| A：NPU compaction 同工作负载对照 | PASS：8 个固定 batch 数值检查，3 对、每 leg 5 warmup +10 measured |',
             '| A：匹配 GPU 性能对照 | PENDING：当前无独占窗口，同正式最坏batch的CUDA限额容量资格检查尚未运行；CUDA功能回归不代替本实验 |',
             '| B：attention 实测 | PASS：自然短上下文与 shape-only 分列；未发现 OOM 容量边界，不能称容量胜出 |',
             '| CUDA base/head 回归 | PASS：4 方法、各 3 update；同种子初始化、同数据与配置，非吞吐对比 |',
             '| 可控proposal接受路径 | 28个CPU fixtures：零/部分/整块接受、EOS/尾块截断、KV commit；仅控制流正确性，不证明模型质量 |',
             '| C：NPU continuation / export → CUDA | NPU continuation/export 已完成；CUDA forward 与64 held-out真实接受由独立原始结果确认，未完成前保持 PENDING |', '',
             '## 来源与复算边界', '',
             f'部署基线为 `{BASE}`，但硬件运行含未提交修改，不能把该 commit 当作已运行源码的完整身份。NPU 部署没有 `.git`，原始 JSON 中的 `source_hashes` 和 A binding.source 才标识实际源码；CUDA 每方法记录对应 dirty-tree SHA256。不同时间点源码哈希不同，报告保留这些区别。', '',
             '`profiles.json` 同时固定所有小型 raw artifact 的 SHA256；矩阵 PASS 行逐字段绑定对应 `profile_records`。本生成器重算时间均值、比值、行数和声明一致性，不重新执行硬件，也不能仅凭 JSON 重建梯度张量。大型 checkpoint / features / 输出梯度文件留在执行主机，未加入仓库。', '']

    def profile(id, method, objective, stage, backend, shape, samples, updates, source, evidence, **extra):
        p = dict(id=id, method=method, objective_profile=objective, stage=stage, backend=backend,
                 mode='offline', status='PASS', dtype='bfloat16', model_geometry=GEOMETRY,
                 shape=shape, data_kind='natural_image_cached_teacher_features', samples=samples,
                 updates=updates, verification_mode='not_applicable_training', source_revision=BASE,
                 source_revision_kind='base_commit_plus_dirty_source_hashes', source_hashes=source,
                 evidence=evidence, **extra)
        profiles.append(p)
        return p

    ds = read(raw / 'recipes/dspark.json'); require_pass(ds, 'DSpark')
    ds_source = read(raw / 'recipes/results.json')['source_hashes']
    ds_geometry=ds['geometry']
    for field in ('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads'):
        require(type(ds_geometry.get(field)) is int and ds_geometry[field] == GEOMETRY[field], 'DSpark actual geometry.'+field+' mismatch')
    require_attention_geometry(ds['attention']['memory_plan'], 'DSpark')
    require_recipe_observation(ds['continuous_observation'], 'dspark')
    obs = ds['continuous_observation']['observations']
    require(len(obs) == 10 and all(r['teacher_rows'] > 0 and r['valid_tokens'] > 0 for r in obs), 'DSpark supervision/teacher coverage')
    require(ds['continuous_observation']['recipe']['name'] == 'upstream_dspark', 'DSpark recipe')
    require(all(v > 0 for v in ds['continuous_observation']['max_head_gradient_norm_by_stage']['0'].values()), 'DSpark gradients')
    plan = ds['attention']['memory_plan']; last = obs[-1]
    shape = dict(context=plan['context_len'], anchors=plan['num_blocks'], query_rows=plan['query_len'],
                 chunks=plan['chunks'], valid_supervision_rows=last['valid_tokens'],
                 padding_fraction=1-last['valid_tokens']/plan['query_len'])
    profile('dspark-upstream-npu','dspark','upstream_dspark','verifier_kl_confidence','npu',shape,10,10,
            ds_source,['raw/recipes/dspark.json'], shape_scope='last update; per-update supervision retained in raw', requested_anchors=256)
    lines += ['## 正式配方及恢复', '',
              f'DSpark 执行 10 updates，loss {obs[0]["loss"]:.6g} → {obs[-1]["loss"]:.6g}；实际每步有效监督 {min(r["valid_tokens"] for r in obs)}–{max(r["valid_tokens"] for r in obs)} rows。teacher 流每步实际投影，KL 与 analytic confidence BCE 系数均为1，CE 为0。Markov/confidence head 非零梯度见 [原始记录](raw/recipes/dspark.json)。', '']
    resumes = [read(raw / f'resume-{step}/results.json') for step in (5,6)]
    continuous = read(raw / 'recipes-dfly/continuous_checkpoint_metadata.json')
    require_dfly_checkpoint_geometry(continuous)
    require(continuous['step'] == 12 and continuous['recipe_stage'] == 1, 'DFly continuous horizon')
    for report in resumes:
        require_pass(report, 'DFly resume')
        result = report['results'][0]
        require_pass(result, 'DFly resume row')
        require_recipe_observation(result['observation'], 'dfly')
        require_attention_geometry(result['summary']['attention']['memory_plan'], 'DFly')
        require(result['config']['drafter']['num_hidden_layers'] == 3 and result['config']['drafter']['block_size'] == 8, 'DFly configured geometry mismatch')
        require(result['exact_final_state'] is True and result['history'] == continuous['history'], 'DFly final state/history')
    result = resumes[0]['results'][0]
    rows = result['observation']['observations']
    require({r['stage_index'] for r in rows} == {0,1}, 'DFly stage observation')
    selection = read(raw / 'corpus/selection_manifest.json')['train']
    for stage_index, stage_name in ((0,'cold_start_lk_dpace'),(1,'e2e_tv')):
        stage_rows = [r for r in rows if r['stage_index'] == stage_index]
        require(all(r['teacher_rows'] > 0 and r['valid_tokens'] > 0 for r in stage_rows), 'DFly teacher/supervision')
        require(all(v>0 for v in result['observation']['max_head_gradient_norm_by_stage'][str(stage_index)].values()), 'DFly gradients')
        first = stage_rows[0]; sample_index = 5 if stage_index == 0 else 6
        context = selection[sample_index]['prompt_tokens'] + selection[sample_index]['answer_tokens']
        q = first['valid_blocks']*8
        shape = dict(context=context, anchors=first['valid_blocks'], query_rows=q, chunks=math.ceil(q/128),
                     valid_supervision_rows=first['valid_tokens'], padding_fraction=1-first['valid_tokens']/(first['valid_blocks']*7))
        profile(f'dfly-{stage_name}-npu','dfly','upstream_two_stage',stage_name,'npu',shape,
                len(stage_rows),len(stage_rows),resumes[0]['source_hashes'],['raw/resume-5/results.json'],
                shape_scope='first observed resumed update of this stage', requested_anchors=256,
                continuous_stage_updates=6, freshly_instrumented_updates=len(stage_rows))
    p = copy.deepcopy(profiles[-1]); p.update(id='dfly-stage-boundary-npu',stage='stage_boundary_resume',samples=12,updates=12,
                                             evidence=['raw/resume-5/results.json','raw/resume-6/results.json','raw/recipes-dfly/continuous_checkpoint_metadata.json'],
                                             resume_steps=[5,6],horizon=12)
    profiles.append(p)
    lines += ['DFly 固定 horizon12、split6，连续历史有6个 LK/D-PACE update 和6个 CE+e2e-TV update。step5/6 保存点恢复到12后，model、FP32 master、optimizer、scheduler、RNG 和完整 history 均与连续运行相等。原始连续运行在完成 checkpoint 后遇到主机内存压力，重载验收改用 CPU mmap 与逐个独立进程，未改变 horizon 或训练数学。', '',
              '新鲜 observer 记录的 stage1 梯度/teacher 证据覆盖 step6 一个 update，stage2 覆盖 step7–12 六个 update；不把这一点写成六个新鲜 stage1 observer 样本。连续 checkpoint 的12步历史以 CPU mmap 只读提取，见 [连续元数据](raw/recipes-dfly/continuous_checkpoint_metadata.json)、[step5恢复](raw/resume-5/results.json)、[step6恢复](raw/resume-6/results.json)。', '']

    parity = read(raw/'compaction-v2/parity.json'); require_pass(parity,'A parity')
    require(len(parity['cases']) == 8 and all(c['passed'] for c in parity['cases']), 'A eight parity cases')
    npu_parity = dict(total_cases=len(parity['cases']),
                      strict_bitwise_cases=sum(c['strict_bitwise_contract_pass'] is True for c in parity['cases']),
                      bf16_envelope_cases=sum(c['bf16_envelope_pass'] is True for c in parity['cases']))
    require(npu_parity['bf16_envelope_cases'] == npu_parity['total_cases'], 'A BF16 parity gate')
    batches = read(raw/'compaction-v2/batches.json')
    pairs = []
    phase_sides = {'off':[], 'on':[]}
    lines += ['## A：DFly stage2 compaction（NPU）', '',
              f"数值检查：既有H1 BF16门槛{npu_parity['bf16_envelope_cases']}/{npu_parity['total_cases']}通过；严格逐位一致{npu_parity['strict_bitwise_cases']}/{npu_parity['total_cases']}。其余{npu_parity['total_cases']-npu_parity['strict_bitwise_cases']}个batch存在非逐位一致梯度，虽在冻结BF16门槛内，不能报告为全部bitwise parity。", '',
              '正式 microbatch8、requested anchors256、block8；固定8个自然 batch，各batch每样本分配12–256个anchor槽位，个体有效anchor数由keep决定，不能将分配宽度写成每个样本的有效数量。真实context442–825，batch allocated query rows768–16384，kept rows600–8432，完整逐batch计数见batches.json。按mask/长度预选含低无效比例和高无效比例batch；无效allocated row比例4.07%–48.54%。同checkpoint、FP32 master、optimizer、scheduler、RNG与anchor plan；只改变compaction。objective progress75起、horizon100，整个窗口不跨stage。', '',
              '| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |', '|---|---:|---:|---:|---:|---:|---:|']
    for i in range(3):
        off,on=[read(raw/f'compaction-v2/pair{i}_{v}.json') for v in ('off','on')]
        require(off.get('compaction') is False and on.get('compaction') is True, f'A pair{i} compaction switch mismatch')
        for name,leg in (('off',off),('on',on)):
            require_geometry(leg.get('model_geometry'), f'A pair{i} {name}', include_block=False)
            require([step['measured'] for step in leg['steps']] == [False]*5+[True]*10, f'A pair{i} {name} warmup/measurement order')
        require(off['initial_state_hashes']==on['initial_state_hashes'],f'A pair{i} initial state')
        require(off['binding']==on['binding']==parity['binding'],f'A pair{i} binding')
        a,b=[[s for s in leg['steps'] if s['measured']] for leg in (off,on)]
        phase_sides['off'].extend(a); phase_sides['on'].extend(b)
        require(len(a)==len(b)==10 and len(off['steps'])==len(on['steps'])==15,'A measurement window')
        require(all(x['batch']==y['batch'] and x['valid_tokens']==y['valid_tokens'] for x,y in zip(a,b)),'A same work')
        tm=[statistics.mean(s['wall_seconds'] for s in leg) for leg in (a,b)]
        hb=[leg['peak_hbm_bytes']/2**30 for leg in (off,on)]
        pairs.append(dict(pair=i,off_seconds=tm[0],on_seconds=tm[1],time_ratio=tm[0]/tm[1],off_gib=hb[0],on_gib=hb[1]))
        lines.append(f'| {i} | {tm[0]*1000:.3f} | {tm[1]*1000:.3f} | {tm[0]/tm[1]:.3f} | {hb[0]:.3f} | {hb[1]:.3f} | {100*(1-hb[1]/hb[0]):.2f}% |')
    phase_rows = phase_work('npu', phase_sides)
    case=parity['cases'][0]; batch=batches[0]
    profile('dfly-compaction-npu','dfly','upstream_two_stage','e2e_tv_compaction_pair','npu',
            dict(context=batch['context'],anchors=batch['actual_anchors'],query_rows=batch['query_rows_per_sample'],chunks=math.ceil(batch['query_rows_per_sample']/128),valid_supervision_rows=case['off']['valid_tokens'],padding_fraction=batch['padding_fraction']),
            len(set(i for b in batches for i in b['indices'])),60,{'binding_source_sha256':parity['binding']['source']},
            ['raw/compaction-v2/parity.json']+[f'raw/compaction-v2/pair{i}_{v}.json' for i in range(3) for v in ('off','on')],
            shape_scope='first parity batch; supervised rows are batch total; complete 8-batch shape plan in raw',allocation_scope='anchors and query_rows are allocated per-sample padded widths; individual valid anchors are defined by keep',microbatch=8,warmup_updates=30,measured_updates=60,matched_gpu_performance='PENDING')
    lines += ['', '时间比大于1表示 NPU 此工作负载缩短 update 时间；这不是跨硬件绝对速度排名，也不是已证明 GPU/NPU 都加速。每个 leg 的前5步 warmup 不进入均值。峰值为 allocator 实测 allocated HBM，未用 memory_plan 估算替代。', '',
              '初次 [parity失败](raw/compaction/parity.json) 保留：两个 batch 的 legacy loss_weight 报告标量因零填充布局改变 reduction 顺序而不严格相等。修复只令该标量在两侧按相同非零权重顺序归约；recipe loss/梯度数学未变，容差未放宽。完整8-batch gate和全部6个性能leg在新源码 binding 下重跑，见 [最终parity](raw/compaction-v2/parity.json)。CPU严格检查与既有 H1 BF16 envelope 分开记录；通过 BF16 envelope 不等同逐位相等。', '',
              'GPU 同配置 compaction 对照 PENDING：当前没有独占窗口，同正式最坏batch的CUDA限额容量资格检查尚未运行。NPU约33GiB峰值不能证明CUDA在22GiB限额下不可运行，两个后端的kernel和显存行为不同。保持C优先，准备相同64条features后再资格检查；若限额内可运行，并在有其他任务时完成受保护的3对实验，也只能记为有竞争环境下的描述性诊断，不能称干净的GPU性能基准。现有功能回归不能替代A。', '']

    lines += ['## B：attention 公平三路径与负面结果', '',
              '下表从每个 raw cell 的3个 measured full updates 重算；每cell先5个 pilot/warmup。FP32-master AdamW，BF16 forward/backward。dense comparator 是 harness 的完整 query SDPA，逐层构造 mask；不是 stock HF 的一次 dense-mask 构造路径。checkpoint范围仅 mask构造+SDPA，不包括MLP/projection/loss；KV head展开在checkpoint外。', '',
              '| data | requested C/A | actual C/A | path | chunks | ms/update | peak GiB | OOM |', '|---|---|---|---|---:|---:|---:|---|']
    memory_cells=[]
    for data in ('natural','shape'):
        for label,(context,anchors) in zip(('small','medium','large'),((1024,64),(4096,256),(4096,512))):
            group=[]
            for mode in ('dense','dense_checkpoint','chunked_checkpoint'):
                filename=f'{data}-{context}-{anchors}-{mode}.json'; d=read(raw/'memory'/filename)
                require_pass(d,filename); measured=d['measured_updates']; require(len(measured)==3,filename+' window')
                r=d['rows']; tm=statistics.mean(s['seconds'] for s in measured); peak=max(s['peak_hbm_bytes'] for s in measured)/2**30
                chunks=measured_attention_chunks(d, mode, filename)
                require(d['context'] == context and d['anchors'] == anchors, filename+' requested shape mismatch')
                require(r['allocated_query_rows'] == r['allocated_anchors']*8, filename+' allocated query rows mismatch')
                if group:
                    require(d['source_hashes'] == group_source, filename+' paired source mismatch')
                    require(r == group_rows, filename+' paired work mismatch')
                else:
                    group_source, group_rows = d['source_hashes'], r
                item=dict(data=data,requested_context=context,requested_anchors=anchors,mode=mode,mean_seconds=tm,peak_gib=peak,rows=r,chunks=chunks)
                group.append(item); memory_cells.append(item)
                lines.append(f'| {data} | {context}/{anchors} | {r["actual_context_length"]}/{r["allocated_anchors"]} | {mode} | {chunks} | {tm*1000:.3f} | {peak:.3f} | no |')
            d=read(raw/'memory'/f'{data}-{context}-{anchors}-chunked_checkpoint.json'); r=d['rows']
            for mode in ('dense_checkpoint','chunked_checkpoint'):
                corr=read(raw/'memory'/f'correctness-{data}-{mode}.json');require(corr['pass'] is True,'B small correctness')
            p=profile(f'dflash-attention-{label}-npu'+('-shape-only' if data=='shape' else ''),'dflash','default_decayed_ce','attention_'+label,'npu',
                      dict(context=r['actual_context_length'],anchors=r['allocated_anchors'],query_rows=r['allocated_query_rows'],chunks=group[-1]['chunks'],valid_supervision_rows=r['effective_supervision_rows'],padding_fraction=r['context_padding_fraction']),
                      1,9,d['source_hashes'],[f'raw/memory/{data}-{context}-{anchors}-{mode}.json' for mode in ('dense','dense_checkpoint','chunked_checkpoint')],
                      requested_context=context,allocated_context=context,requested_anchors=anchors,microbatch=1,warmup_updates=15,shape_scope='actual context; padding_fraction is context padding; updates sums measured updates of three paths',correctness_scope='small shape only, BF16 H1 output/loss/all-gradient comparison')
            if data=='shape': p['data_kind']='shape_only_random_tensors_with_real_frozen_target_head'
    lines += ['', '小形状的4组数值对照均通过：natural和shape-only各比较dense_checkpoint、chunked_checkpoint与dense参考的output、loss及全部梯度，沿用H1的rtol=0.02、atol=0.02。中/大形状记录的是完整update与资源可运行性，没有逐格重新做全梯度parity；不能把18个cell全部标为大形状数值等价。所有18个最终cell均可运行，没有 NPU OOM；因此不能声称 dense OOM而chunked可运行，也不计算不存在的容量 speedup。chunked在本表更慢（slower），通过重计算与多次dispatch换取部分大形状的显存下降；小形状峰值没有改善。', '',
              '自然数据只选中同一个最长825-token完整样本：4096是零填充bucket，实际anchors受答案长度限制为249，requested512没有达到。不能把这几行写成4096真实上下文/512有效anchors训练；真正4096/512只能由单独shape-only行支持。NPU1专用于B，但NPU0 recipe并行工作共享主机CPU/RAM，所以时间为有仪器开销的共享主机测量。', '',
              '最初BF16直接AdamW的预演结果保留在 `raw/memory/preflight_BF16optimizer_not_final/`，不参与此表；最终18cell均改用production FP32 master重跑。feature-layout首次配置失败与SSH中断见 [harness事件](raw/memory/harness_incidents.json)，不隐藏为PASS。', '']

    b_total = sum(read(raw/'memory'/'{data}-{requested_context}-{requested_anchors}-{mode}.json'.format(**x))['total_seconds'] for x in memory_cells)
    cuda=read(cuda_directory/'regression_summary.json');require_pass(cuda,'CUDA regression')
    lines += ['## CUDA：固定base/head功能回归', '',
              f'base `{cuda["base_revision"]}`；head基线 `{cuda["head_base_revision"]}` 加每方法dirty source hash。4方法在同一隔离环境、相同3个自然样本、相同初始化/配置上各执行3updates，初始/最终权重、loss/全部梯度与probe均完全相同。', '',
              '| method | actual contexts | actual query rows | valid supervision | result |', '|---|---|---|---|---|']
    for row in cuda['rows']:
        require_pass(row,'CUDA '+row['method']);require(all(row['checks'].values()),'CUDA checks')
        q=[shapes[0][1] for shapes in row['query_shapes']]; c=[x[1] for x in row['input_shapes']]
        lines.append(f'| {row["method"]} | {c} | {q} | {row["valid_tokens"]} | exact PASS |')
        method=row['method']; objective={'dflash':'default_decayed_ce','dspark':'upstream_dspark','dfly':'upstream_two_stage','domino':'default_domino'}[method]
        stage={'dflash':'flex_regression','dspark':'verifier_kl_confidence','dfly':'e2e_tv','domino':'gru_regression'}[method]
        slots=8 if method in ('dspark','domino') else 7
        profile(method+'-cuda-regression',method,objective,stage,'cuda',dict(context=c[-1],anchors=q[-1]//8,query_rows=q[-1],chunks=1,valid_supervision_rows=int(row['valid_tokens'][-1]),padding_fraction=1-row['valid_tokens'][-1]/(q[-1]//8*slots)),3,3,
                {'head_tree_sha256':row['head_source_tree_sha256'],'base_tree_sha256':cuda['source_tree_sha256']['base']},
                ['../evidence/2026-10-21/cuda_regression/regression_summary.json'],shape_scope='last of three samples; actual full series in raw',requested_anchors=256,base_revision=cuda['base_revision'],microbatch=1)
    lines += ['', 'requested anchors256，但实际仅4/8/17；不能称正式anchors256负载回归。A6000有其他任务，限额环境只支持功能对比，不作吞吐结论。隔离venv使用transformers5.12.1，原活动环境5.11.0未升级；环境检查证实未安装torch_npu。最初gradient probe整数索引修复前失败保留在CUDA原始目录，不是模型算子失败。', '',
              '## C、失败与资源边界', '',
              'NPU已用具有接受能力的R3 checkpoint做10步stage2 weight-only continuation，再导出portable权重；使用新optimizer，不称跨后端bitwise恢复，亦不称从头训练。CUDA导出完整验收及64条自然held-out接受结果仍需独立结果完成后更新。训练loss里的e2e_accept_len不是实际解码接受率。', '',
              f'已完成子项wall：NPU A最终全流程 {read(raw/"compaction-v2/result.json")["elapsed_seconds"]:.2f}s；B最终cell合计 {b_total:.2f}s；CUDA base/head功能回归含pilot预留窗口 {cuda["gpu_reservation_wall_seconds_including_pilot_failure"]:.2f}s。这些不是整个项目的总NPU-h/GPU-h，未包含所有准备、失败、排队和其他进程，不能冒充完整预算核算。', '',
              '## 执行路径与复现', '',
              '28个可控CPU proposal fixtures覆盖sequential/parallel两种控制流中的零接受、部分接受、整块接受、EOS、尾块长度截断与KV提交；修复让accepted/proposed计数只覆盖实际发出的后缀。测试源码见 [test_acceptance_path_fixtures.py](../../../tests/test_acceptance_path_fixtures.py)，这些fixture不证明自然图像drafter质量或NPU并行serving。', '',
              '独立团队agent完成代码审查，范围、发现、修复与56个定向测试见 [CODE_REVIEW.md](CODE_REVIEW.md)。这是同团队独立代码审查，不是外部跨模型科学评审。', '',
              '`reset_attention_runtime_stats()` 划定测量窗口；`attention_runtime_report().measurement_window`累计完成的attention-core调用及fused/fallback chunks。记录发生在output projection之前，因此随后projection异常时也可能已计数；不是整个attention module成功完成次数。mixed路径显式标记；内部chunk checkpoint重计算不计作新forward，完整外层checkpoint重放可能算一次完成调用，因此不能把这些计数当optimizer updates。strict禁止fallback，profile允许且保留原因。', '',
              '`recipe_compile_tv_kl` 的 requested/effective/reason 标记的是device gate；`device_gate_not_execution_evidence` 不证明实际进入编译kernel。NPU gate禁用compile；现有CUDA compiled-KL结果不能充当NPU优化。memory_plan始终只是分析估算，表格HBM来自设备allocator实测。', '',
              '复算已附小文件（无需设备）：', '',
              '```bash', 'python scripts/npu/summarize_audit.py --check', 'python scripts/npu/validate_support_matrix.py', '```', '',
              '重新执行硬件（需要原始features/checkpoints与相同环境；并非本生成命令自动执行）：', '',
              '```bash', 'python scripts/npu/audit_recipes.py --device npu:0 --bundle CORPUS_OR_BUNDLE --config-dir configs --out NEW_RECIPE_DIR',
              'python scripts/npu/audit_compaction.py --help', 'python scripts/npu/audit_attention.py --help', 'python scripts/npu/audit_cuda.py --help', '```', '',
              '命令参数、固定batch/shape、首次5步耗时估算及初始状态hash保留在各raw目录和协议文件；重跑应写新目录，不覆盖已发生的失败记录。', '']
    local_tests = read(raw/'local_validation.json')
    ledger = read(raw/'resource_ledger.json')
    integrity = read(raw/'continuation/supplemental_integrity.json')
    require_pass(integrity, 'C supplemental integrity')
    require(integrity['final_state_finite'] and integrity['all_recorded_gradient_norms_finite'] and integrity['all_recorded_losses_finite'] and integrity['original_summary_weight_digest_matches'], 'C finite/digest gate')
    require(integrity['final_weight_sha256'] == read(raw/'continuation/result.json')['final_weight_sha256'], 'C original digest binding')
    lines += ['## 补充完整性检查与账本', '',
              'C 原始运行脚本没有在结束前显式检查全部最终权重有限性。保留原始结果与源码hash，另外对同一已完成产物做posthoc CPU mmap检查：10步持久化loss/gradient norm全部有限，全部最终模型张量有限，weight digest与原始summary一致。该补充不是重新训练，也不把后续修复过的harness冒充原执行源码，见 [supplemental integrity](raw/continuation/supplemental_integrity.json)。', '',
              '| NPU0进程 | start epoch | end epoch | wall秒 | exit |', '|---|---:|---:|---:|---:|']
    for row in ledger['rows']:
        require(row['end_epoch'] - row['start_epoch'] == row['reserved_wall_seconds'], 'resource epoch accounting')
        lines.append(f"| {row['run']} | {row['start_epoch']} | {row['end_epoch']} | {row['reserved_wall_seconds']} | {row['exitcode']} |")
    seconds = sum(row['reserved_wall_seconds'] for row in ledger['rows'])
    lines += ['', f'上述NPU0可核对进程wall合计{seconds}s（{seconds/3600:.4f}设备小时的预留wall），包含两次exit137和初次A失败；不是kernel忙碌时间。B最终18cell另计508.07s，旧BF16预演和capture/shared-forward缺完整进程起止，不能声称预算完整闭合。并行不同设备需按设备求和，不能仅取整体wall。', '',
              f'本地完整CPU套件{local_tests["passed"]} passed / {local_tests["skipped"]} skipped（JUnit摘要与文件hash见raw/local_validation.json）。这些验证不能替代NPU/CUDA硬件证据；本生成器另有失败关闭测试。', '']
    failed_profiles = []
    shared_path = cuda_directory/'C_shared_forward_comparison.json'
    if shared_path.exists():
        shared = read(shared_path)
        relative = '../evidence/2026-10-21/cuda_regression/C_shared_forward_comparison.json'
        failed_profiles.append(shared_forward_failure(shared, relative))
        require(all(shared['identity_checks'].values()), 'C failed numerical comparison identities must match')
        require(shared['tolerance']['rtol'] == shared['tolerance']['atol'] == 0.02, 'C frozen BF16 tolerance changed')
        lines = [('| C：NPU continuation / export → CUDA | FAILED：冻结BF16跨后端forward数值门槛未通过；NPU训练/导出已完成，64条CUDA实际使用验收独立记录 |'
                  if line.startswith('| C：') else line) for line in lines]
        lines += ['## C：BF16跨后端数值门槛失败（必须保留）', '',
                  'NPU/CUDA使用相同权重、输入、dtype、实际student logits dtype、源码与anchor计划，所有身份核对通过。相同loss不能替代全部hidden/logits逐元素检查。冻结的既有H1 BF16门槛为rtol=0.02、atol=0.02，以下三项均失败：', '',
                  '| tensor | 违反门槛元素 / 总元素 | max abs | mean abs |', '|---|---:|---:|---:|']
        for name in ('draft_hidden', 'post_correction_hidden', 'student_logits'):
            check = shared['checks'][name]
            require(check['passed'] is False and check['violating_elements'] > 0, 'C numerical failure evidence')
            lines.append(f"| {name} | {check['violating_elements']} / {check['total_elements']} | {check['max_abs_error']:.6g} | {check['mean_abs_error']:.8g} |")
        loss = shared['checks']['loss']
        lines += ['', f"loss在两端均为{loss['npu']:.10g}，但combined C profile仍为FAILED，不能宣称完整跨后端数值等价。原始记录见 [BF16 shared-forward comparison]({relative})。", '',
                  '后续FP32仅可作为独立诊断，不能替换这个BF16失败、改变容差或改写原始结果。portable格式与64条真实接受的可用性可以独立成功，但不能覆盖此失败；结果未返回前，该可用性部分仍为PENDING。', '']
    usability_path = cuda_directory/'C_export64.json'
    if usability_path.exists():
        usage = read(usability_path)
        require(usage['verification'] == 'sequential' and usage['verify_sdpa_backend'] == 'parity'
                and usage['verify_sdpa_backend_switch_enabled'] is False, 'C verification protocol')
        require(usage['torch_npu_imported'] is False and usage['torch_npu_installed'] is False, 'C torch_npu independence')
        require(usage['format_outputs_equal'] is True and usage['format_acceptance_equal'] is True, 'C format equality')
        require(read(cuda_directory/'C_export64.guard.json')['returncode'] == 0, 'C process exit')
        samples = [json.loads(line) for line in (cuda_directory/'C_export64.samples.jsonl').read_text().splitlines() if line]
        require(len(samples) == 128 and {s['format'] for s in samples} == {'native','portable'}, 'C complete two-format sample set')
        by_format = {name:[s for s in samples if s['format']==name] for name in ('native','portable')}
        heldout_ids = read(raw/'corpus/selection_manifest.json')['heldout_ids']
        keys = ('new_tokens','accepted_tokens','draft_tokens','verify_steps','target_forward_calls')
        for name, cases in by_format.items():
            summary = usage['results'][name]
            require(len(cases) == summary['num_samples'] == 64 and len({s['id'] for s in cases}) == 64, 'C sample coverage')
            require([s['id'] for s in cases] == heldout_ids, 'C fixed heldout identity')
            require(summary['token_exact'] is True and all(s['equivalent'] is True for s in cases), 'C token exactness')
            for key in keys:
                require(sum(s[key] for s in cases) == summary[key], 'C aggregate '+key)
            require(all(len(s['tokens']) == s['new_tokens'] <= 64 for s in cases), 'C emitted token lengths')
            require(summary['accepted_tokens'] > 0, 'C nonzero acceptance')
            require(summary['speculative_rounds'] == summary['verify_steps'], 'C rounds')
            require(math.isclose(summary['al_progress_tokens_per_round'],summary['new_tokens']/summary['speculative_rounds']), 'C tau progress ratio')
            require(math.isclose(summary['al_draft_tokens_per_round'],summary['accepted_tokens']/summary['speculative_rounds']), 'C A accepted ratio')
        for native, portable in zip(by_format['native'],by_format['portable']):
            require(all(native[key] == portable[key] for key in (*keys,'id','tokens','acceptance_events')), 'C per-sample format behavior')
        exported = read(raw/'continuation/export/config.json')
        checksums = read(raw/'continuation/export/checksums.json')
        for filename in ('config.json','draft_config.json','target_contract.json','algorithm.json'):
            require(sha(raw/'continuation/export'/filename) == checksums[filename], 'C exported metadata hash '+filename)
        geometry = exported['drafter']['draft_config']
        require(all(geometry[key] == value for key,value in GEOMETRY.items()), 'C export geometry')
        require(geometry['dtype'] == 'bfloat16', 'C exported dtype')
        require(usage['target_fingerprint'] == exported['identity']['model_fingerprint']
                and usage['processor_fingerprint'] == exported['identity']['processor_fingerprint'], 'C target/processor identity')
        require(usage['train_recipe'] == exported['train_recipe'], 'C exported recipe identity')
        transfer = read(cuda_directory/'C_transfer_verification.json')
        require(transfer['matches_original_npu_bytes'] is True
                and transfer['native_sha256'] == integrity['original_artifacts_sha256']['ckpt_final.pt']
                and transfer['export_weights_sha256'] == checksums['model.safetensors'], 'C transferred weight bytes')
        score = usage['results']['portable']
        decode_shape = dict(max_new_tokens=64, speculative_rounds=int(score['speculative_rounds']),
                            new_tokens=int(score['new_tokens']), draft_tokens=int(score['draft_tokens']),
                            accepted_tokens=int(score['accepted_tokens']))
        p = profile('dfly-portable-usability-cuda','dfly','upstream_two_stage','portable_format_and_nonzero_acceptance',
                    'cuda',decode_shape,64,0,read(raw/'continuation/result.json')['source_hashes'],
                    ['../evidence/2026-10-21/cuda_regression/C_export64.json','../evidence/2026-10-21/cuda_regression/C_export64.samples.jsonl',
                     '../evidence/2026-10-21/cuda_regression/C_transfer_verification.json','raw/continuation/export/checksums.json'],
                    source_hashes_scope='NPU continuation source; CUDA evaluation behavior and protocol pinned by raw artifacts',
                    tau_progress_tokens_per_round=score['al_progress_tokens_per_round'],
                    A_accepted_tokens_per_round=score['al_draft_tokens_per_round'],
                    cross_backend_bf16_numerical_gate='FAILED',export_weights_sha256=transfer['export_weights_sha256'])
        p.update(mode='inference',data_kind='64_fixed_natural_image_heldout_prompts',verification_mode='sequential_generate_parity_switch_false')
        evaluator = cuda_directory/'C_evaluator/audit_cuda.py'
        require(evaluator.is_file(), 'C exact evaluator snapshot missing')
        p['source_hashes']['CUDA_evaluation_snapshot/audit_cuda.py'] = sha(evaluator)
        p['source_hashes_scope'] = 'NPU continuation source plus correct64 CUDA evaluator snapshot; executor confirmed evaluator unchanged since launch'
        p['evidence'].append('../evidence/2026-10-21/cuda_regression/C_evaluator/audit_cuda.py')
        lines = [line.replace('CUDA导出完整验收及64条自然held-out接受结果仍需独立结果完成后更新。',
                              'CUDA上64条实际使用验收独立通过，但BF16跨后端数值门槛失败，故combined C不通过。')
                 .replace('结果未返回前，该可用性部分仍为PENDING。','64条实际使用现已独立通过，见下表；combined C仍为FAILED。') for line in lines]
        checklist_index = next(i for i,line in enumerate(lines) if line.startswith('| C：'))
        lines.insert(checklist_index+1, f"| C：portable格式及CUDA实际使用（独立子项） | PASS：native/portable各64/64 token-exact、{decode_shape['accepted_tokens']}实际接受；不覆盖上面的BF16数值FAILED |")
        lines += ['## C：portable格式与64条CUDA实际使用独立通过', '',
                  '相同NPU continuation权重的native checkpoint与portable export在无torch_npu的CUDA环境加载。传输后的native文件SHA256和重建safetensors完整字节SHA256均匹配NPU原件；export config/target/processor/recipe身份校验通过。正式3层模型几何从SHA256验证后的export metadata读取，而非从测试名称推断。', '',
                  '| format | samples token-exact | accepted | rounds | emitted tokens | proposed draft tokens | τ=progress/round | A=accepted/round |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in ('native','portable'):
            score=usage['results'][name]
            lines.append(f"| {name} | 64/64 | {int(score['accepted_tokens'])} | {int(score['speculative_rounds'])} | {score['new_tokens']} | {int(score['draft_tokens'])} | {score['al_progress_tokens_per_round']:.10f} | {score['al_draft_tokens_per_round']:.10f} |")
        lines += ['', 'τ严格采用原始字段al_progress_tokens_per_round，A采用al_draft_tokens_per_round。两种格式逐样本token序列、接受事件和计数一致，显示真实非零接受路径；不据此声称质量排名或推理加速。每条最多生成64 tokens，verification=sequential，生产generate-parity配置开关为False。', '',
                  '首次运行漏设该False开关，错误使用另一SDPA backend；116条部分结果标为INVALID_PROTOCOL并归档，不计入本表，也不作为checkpoint缺陷证据。修复harness协议后按原64个held-out ID从头重跑，没有换样本或放宽token-exact要求。combined C的BF16跨后端数值FAILED仍保留，不能被格式可用性PASS覆盖。', '']
    fp32_path = cuda_directory/'C_shared_forward_fp32_comparison.json'
    if fp32_path.exists():
        fp32 = read(fp32_path)
        require_pass(fp32,'C FP32 diagnostic')
        require(all(fp32['identity_checks'].values()), 'C FP32 identities')
        require(all(fp32['metadata'][backend][field] == 'torch.float32'
                    for backend in ('npu','cuda') for field in ('dtype','student_logits_dtype')), 'C FP32 actual dtype')
        require(all(check['passed'] is True for check in fp32['checks'].values()), 'C FP32 numerical checks')
        lines += ['## C：FP32补充诊断', '',
                  '另行将相同权重转为FP32进行shared-forward诊断；两端实际hidden/student logits dtype均记录为torch.float32，权重、输入、源码和anchor身份一致。该结果不替代BF16主门槛，不修改其失败记录或容差。原始诊断见 [FP32 comparison](../evidence/2026-10-21/cuda_regression/C_shared_forward_fp32_comparison.json)。诊断通过与低精度执行路径差异一致，但不足以单独证明具体误差根因。', '',
                  '| tensor | max abs error | relative L2 | 违规元素 |', '|---|---:|---:|---:|']
        for name in ('draft_hidden','post_correction_hidden','student_logits'):
            check=fp32['checks'][name]
            lines.append(f"| {name} | {check['max_abs_error']:.8g} | {check['relative_l2_error']:.8g} | {check['violating_elements']} |")
        lines += ['', f"FP32 loss差值为{fp32['checks']['loss']['abs_error']:.8g}。此诊断沿用原comparison的0.02/0.02门槛，同时完整报告实际误差；不把它包装成另一个预注册的FP32严格门槛。", '']
    qualification_dir = cuda_directory.parent/'cuda_A'
    qualifications = []
    pending_profiles = []
    if (qualification_dir/'qualification25.json').exists():
        plan_check = read(qualification_dir/'plan_verification.json')
        require_pass(plan_check,'CUDA A plan verification')
        require(plan_check['all_plan_fields_exact'] is True
                and plan_check['plan_tree_digest'] == parity['binding']['plan']
                and plan_check['worst_plan'] == batches[plan_check['worst_batch']], 'CUDA A frozen plan changed')
        require(read(qualification_dir/'batches.json') == batches, 'CUDA A frozen plan batch list')
        lines = [('| A：匹配 GPU 性能对照 | PENDING：GPU1共享22/25GiB限额资格检查均OOM；正式配对性能结果尚未附齐，不能外推整卡容量 |'
                  if line.startswith('| A：匹配 GPU') else line) for line in lines]
        lines = [('GPU A此前缺少独占窗口；随后同正式最坏batch在GPU1共享22/25GiB限额下均OOM，详见后面的资格检查。NPU峰值不用于推断CUDA容量。另一个设备变为可用后才允许新的独占窗口尝试；未附正式结果前GPU性能仍为PENDING。'
                  if line.startswith('GPU 同配置 compaction 对照 PENDING：') else line) for line in lines]
        lines += ['## A：GPU1共享限额资格检查（保留失败）', '',
                  '没有根据NPU峰值猜测CUDA是否能运行，而是保持同一正式microbatch8、context825、anchors256、block8及固定最坏batch1进行真实OFF完整update资格检查。CUDA plan全部字段与NPU原计划一致，tree digest与NPU binding相同。只改变资源限额；没有缩batch、模型、context、anchors或更换数据。', '',
                  '| allocator cap GiB | outcome | guarded wall s | guard最低采样空闲MiB |', '|---|---|---:|---:|']
        for filename, cap in (('qualification',22),('qualification25',25)):
            attempt = read(qualification_dir/(filename+'.json'))
            guard = read(qualification_dir/(filename+'.guard.json'))
            require(attempt['status'] == 'FAILED' and attempt['oom'] is True and attempt['allocator_cap_gib'] == cap, 'CUDA A bounded OOM evidence')
            require(guard['returncode'] != 0, 'CUDA A failed qualification exit')
            qualifications.append(dict(allocator_cap_gib=cap,oom=True,status='FAILED',
                                       wall_seconds=guard['wall_seconds'],scope=attempt['scope'],error=attempt['error']))
            free=min(point[1] for point in guard['free_MiB_samples'])
            lines.append(f"| {cap} | OOM under cap | {guard['wall_seconds']:.6f} | {free} |")
        lines += ['', 'guard列仅是轮询时观测到的最低空闲值，不是连续测量的瞬时最低值；22GiB运行的快速分配峰值发生在轮询之间，以下OOM报文提供了更低的当时空闲值。这是共享设备上的显式allocator限额OOM，不是整张A6000物理容量OOM，也不是CUDA必然无法运行正式配置的证明。22GiB失败时PyTorch报物理空闲8.81GiB，25GiB失败时报5.91GiB；另一个进程约占16.69GiB。两次都在CE分配594MiB时触发限额，没有完整update时间或配对性能结果可报告。后续若另一张设备出现独占窗口，应作为独立资源条件记录，不改写这两次失败。', '']
        pending_profiles.append(dict(id='dfly-compaction-cuda',method='dfly',objective_profile='upstream_two_stage',stage='e2e_tv_compaction_pair',backend='cuda',mode='offline',status='PENDING',
                                     reason='Formal matched performance not attached; shared GPU1 qualification OOM under 22/25GiB caps is not whole-device capacity failure.',
                                     evidence=['../evidence/2026-10-21/cuda_A/qualification.json','../evidence/2026-10-21/cuda_A/qualification25.json']))
    cuda_compaction = None
    full_dir = qualification_dir/'gpu2_expand'
    if (full_dir/'result.json').exists():
        cuda_compaction = completed_cuda_compaction(full_dir,raw/'compaction-v2')
        guard=read(full_dir/'guard.json');preflight=read(full_dir/'preflight.json')
        require(guard.get('reason') is None and guard['status']=='FINISHED', 'CUDA A full guard completion')
        require(preflight['preflight_apps']==[] and preflight['preflight_free_MiB']>=45*1024, 'CUDA A exclusive preflight')
        require(guard['allocator_environment']==preflight['allocator_environment']
                and guard['allocator_environment']['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
                and guard['allocator_cap_gib']==40, 'CUDA A allocator amendment')
        cfg=read(full_dir/'config.json'); npu_cfg=read(raw/'compaction-v2/config.json')
        require(cfg['train']['micro_batch_size']==8 and cfg['train']['draft_dtype']=='bfloat16', 'CUDA A dtype/microbatch')
        require(cfg['algorithm']==npu_cfg['algorithm'], 'CUDA A recipe/config identity')
        for key in ('num_hidden_layers','block_size','num_anchors','attention_backend','attention_chunk_size'):
            require(cfg['drafter'][key]==npu_cfg['drafter'][key], 'CUDA A matched drafter '+key)
        provenance=read(full_dir/'source_provenance.json')
        digest=hashlib.sha256()
        for key,value in sorted(provenance['source_hashes'].items()):
            digest.update(key.encode());digest.update(repr(value).encode())
        require(digest.hexdigest()==cuda_compaction['binding']['source'], 'CUDA A recorded source digest')
        materialization=read(qualification_dir/'bundle64.materialization.json')
        require(materialization['materialized_sha256']==cuda_compaction['binding']['bundle']
                and materialization['sample_ids']==[r['id'] for r in read(raw/'corpus/selection_manifest.json')['train']], 'CUDA A complete original corpus')
        p=copy.deepcopy(next(p for p in profiles if p['id']=='dfly-compaction-npu'))
        p.update(id='dfly-compaction-cuda',backend='cuda',source_hashes=provenance['source_hashes'],
                 allocator_environment=guard['allocator_environment'],allocator_cap_gib=40,
                 execution_scope=guard['scope'],strict_bitwise_cases=cuda_compaction['strict_bitwise_cases'],bf16_envelope_cases=8,
                 evidence=['../evidence/2026-10-21/cuda_A/gpu2_expand/'+name for name in
                           ['result.json','parity.json','guard.json','preflight.json','source_provenance.json']+
                           [f'pair{i}_{side}.json' for i in range(3) for side in ('off','on')]])
        p.pop('matched_gpu_performance',None)
        profiles.append(p)
        next(p for p in profiles if p['id']=='dfly-compaction-npu')['matched_gpu_performance']='PASS_separate_backend_ratio_only'
        pending_profiles=[p for p in pending_profiles if p['id']!='dfly-compaction-cuda']
        phase_rows.extend(cuda_compaction['phase_work'])
        lines=[('| A：匹配 GPU 性能对照 | PASS：另一张空闲GPU2、相同原始workload，6个完整leg与guard退出0；allocator环境修订和此前OOM独立保留 |'
                if line.startswith('| A：匹配 GPU') else line.replace('也不是已证明 GPU/NPU 都加速。','GPU同workload结果另表报告，不能据绝对时间比较芯片优劣。')
                .replace('另一个设备变为可用后才允许新的独占窗口尝试；未附正式结果前GPU性能仍为PENDING。',
                         '随后另一个设备GPU2出现空闲窗口，完整结果已完成，见GPU2配对表；不改写此前GPU1限额失败。')) for line in lines]
        initial_failure=read(qualification_dir/'gpu2_full/result.json')
        require(initial_failure['status']=='FAILED', 'CUDA A initial allocator failure preserved')
        lines += ['## A：CUDA GPU2独占计算窗口的完整配对结果', '',
                  f"GPU2 preflight无其他compute进程、空闲{preflight['preflight_free_MiB']}MiB，持有GPU UUID advisory lock并监视外部进程；计算设备观测为独占，主机CPU仍共享。最终guard退出0、wall {guard['wall_seconds']:.6f}s；全部6个leg均5warmup+10measured，原始NPU checkpoint权重、8-batch plan、每步有效监督/allocated/kept rows匹配。", '',
                  f"CUDA既有H1 BF16门槛8/8通过；CUDA严格逐位一致{cuda_compaction['strict_bitwise_cases']}/8，其余{8-cuda_compaction['strict_bitwise_cases']}个batch存在非逐位一致梯度。NPU严格检查为4/8，不能把两端都写成bitwise parity。", '',
                  '| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |', '|---|---:|---:|---:|---:|---:|---:|']
        for row in cuda_compaction['pairs']:
            lines.append(f"| {row['pair']} | {row['off_seconds']*1000:.3f} | {row['on_seconds']*1000:.3f} | {row['time_ratio']:.3f} | {row['off_gib']:.3f} | {row['on_gib']:.3f} | {100*(1-row['on_gib']/row['off_gib']):.2f}% |")
        lines += ['', '在这个固定诊断workload上，NPU和CUDA分别有compaction相对收益；只比较各设备内部off/on比值，不以两端绝对时间排名硬件，也不外推到其他shape或完整online训练。', '',
                  'GPU2第一次default allocator完整尝试在parity和pilot通过后、首个OFF benchmark leg backward中OOM：请求4.06GiB时live allocated29.13GiB、reserved但未用7.18GiB，仍有物理空闲10.70GiB，显式cap40GiB。原始失败保留在gpu2_full。缓存碎片是根据这些读数提出的解释，未采memory snapshot，不能断言机制已被证明。', '',
                  '随后仅做一次新进程、新输出目录的allocator环境修订：两臂均设置PYTORCH_ALLOC_CONF=expandable_segments:True，保持40GiB cap、数据、几何、seed、计划与全部5+10窗口；没有每step empty_cache。该修订在任何完整性能leg之前声明，最终全部leg在同一环境下重跑。', '',
                  '该选项在[已安装PyTorch 2.11官方文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)中为experimental、默认False；它可能改善变化分配大小导致的碎片。这里它是执行环境条件，不是VelaDraft算法创新，也不用于解释NPU kernel的收益来源。', '']
    ledger_path = cuda_directory.parent/'cuda_gpu_ledger.json'
    if ledger_path.exists():
        gpu_ledger=read(ledger_path)
        used=0.0
        for entry in gpu_ledger['entries']:
            guard_path=ledger_path.parent/entry['artifact']
            require(sha(guard_path)==entry['sha256'], 'GPU ledger artifact digest')
            guard=read(guard_path)
            require(guard['wall_seconds']==entry['wall_seconds'] and guard['returncode']==entry['returncode'], 'GPU ledger guard values')
            used += entry['wall_seconds']
        require(math.isclose(used,gpu_ledger['used_seconds'],abs_tol=1e-8), 'GPU ledger sum')
        lines += ['## CUDA累计资源账本', '',
                  f"所有已附串行guard命令累计{used:.6f}s（{used/3600:.6f} GPU-h），含CPU加载、错误backend终止、BF16/FP32诊断、初始probe失败、22/25GiB资格OOM、GPU2默认allocator失败及完整重试；不是kernel busy time。预算{gpu_ledger['budget_seconds']}s，已登记余量{gpu_ledger['remaining_seconds']:.6f}s。原始逐项账本见 [cuda_gpu_ledger.json](../evidence/2026-10-21/cuda_gpu_ledger.json)。", '']
    lines += ['## A：阶段开销与实际投影工作', '',
              '每个backend/开关行合并3个leg的30个measured updates。阶段时间为每update均值；optimizer阶段含FP32 master梯度转移、clip、step、scheduler及权重copyback，不是单个optimizer kernel。投影rows为这30步的实际hook计数之和，来自raw steps而不是mask形状估算；forward student/teacher投影与包含backward重计算的全部投影分列。', '',
              '| backend | compaction | forward ms | backward ms | optimizer ms | student rows forward | teacher rows forward | all rows incl. recomputation |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in phase_rows:
        lines.append(f"| {row['backend']} | {row['compaction']} | {row['forward_ms']:.3f} | {row['backward_ms']:.3f} | {row['update_ms']:.3f} | {row['student_projected_rows_forward']} | {row['teacher_projected_rows_forward']} | {row['projected_rows_including_backward_recomputation']} |")
    lines.append('')
    manifest={}
    for base in (raw,cuda_directory,qualification_dir):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                manifest[str(path.relative_to(ROOT/'docs/npu'))]=sha(path)
    if ledger_path.exists():
        manifest[str(ledger_path.relative_to(ROOT/'docs/npu'))]=sha(ledger_path)
    return dict(schema='veladraft-audit-derived-evidence-v1',profile_records=profiles,failed_profile_records=failed_profiles,pending_profile_records=pending_profiles,
                cuda_capacity_qualifications=qualifications,cuda_compaction=cuda_compaction,npu_compaction_parity=npu_parity,raw_artifact_sha256=manifest,
                compaction_pairs=pairs,compaction_phase_work=phase_rows,attention_cells=memory_cells), '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    directory=ROOT/'docs/npu/audit-2026-10-21'
    evidence,report=generate(directory,ROOT/'docs/npu/evidence/2026-10-21/cuda_regression')
    serialized=json.dumps(evidence,indent=2,ensure_ascii=False)+'\n'
    artifact=directory/'profiles.json'
    if args.check:
        require(artifact.read_text()==serialized,'generated profiles.json stale')
        require((directory/'REPORT.md').read_text()==report,'generated REPORT.md stale')
    else:
        artifact.write_text(serialized);(directory/'REPORT.md').write_text(report)
        matrix_path=ROOT/'docs/npu/support_matrix.json';matrix=read(matrix_path)
        promoted={p['id']:p for p in evidence['profile_records']}
        old={p['id']:p for p in matrix['audit_profiles']}
        for id,p in promoted.items():
            old[id]={**p,'artifact':dict(path='audit-2026-10-21/profiles.json',sha256=sha(artifact),record_id=id)}
        for p in evidence['failed_profile_records']:
            old[p['id']] = p
        for p in evidence['pending_profile_records']:
            old[p['id']] = p
        matrix['audit_profiles']=list(old.values())
        matrix['audit_profile_schema']['shape_fields_by_mode'] = {
            'offline_or_online': ['context','anchors','query_rows','chunks','valid_supervision_rows','padding_fraction'],
            'inference': ['max_new_tokens','speculative_rounds','new_tokens','draft_tokens','accepted_tokens'],
            'inference_count_rule': 'positive limit/rounds/new_tokens; nonnegative draft/accepted; accepted <= min(draft,new_tokens); no invented training supervision or prompt context'
        }
        if 'CUDA_FUNCTIONAL_REGRESSION_PASSED' not in matrix['legend']['evidence']:
            matrix['legend']['evidence'].append('CUDA_FUNCTIONAL_REGRESSION_PASSED')
        for backend in matrix['backends']:
            if backend['name']=='cuda':
                backend.update(status='CUDA_FUNCTIONAL_REGRESSION_PASSED',reason='Four formal-geometry methods, three natural microbatch1 updates each, reduced actual anchors; exact base/head functional comparison, not performance.',evidence=['docs/npu/evidence/2026-10-21/cuda_regression/regression_summary.json'])
        matrix_path.write_text(json.dumps(matrix,indent=2,ensure_ascii=False)+'\n')
    from scripts.npu.validate_support_matrix import validate_matrix
    errors=validate_matrix(ROOT/'docs/npu/support_matrix.json')
    require(not errors,'matrix invalid: '+str(errors))
    matrix=read(ROOT/'docs/npu/support_matrix.json')
    require(all(backend['status'] in matrix['legend']['evidence'] for backend in matrix['backends'] if backend['name']=='cuda'), 'CUDA backend status missing from evidence legend')
    print(json.dumps(dict(status='VALID',profiles=len(evidence['profile_records']),raw_artifacts=len(evidence['raw_artifact_sha256']))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
