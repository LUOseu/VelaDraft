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
    require(continuous['step'] == 12 and continuous['recipe_stage'] == 1, 'DFly continuous horizon')
    for report in resumes:
        require_pass(report, 'DFly resume')
        result = report['results'][0]
        require_pass(result, 'DFly resume row')
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
    batches = read(raw/'compaction-v2/batches.json')
    pairs = []
    lines += ['## A：DFly stage2 compaction（NPU）', '',
              '正式 microbatch8、anchors256、block8；固定8个自然 batch，实际每样本 anchors12–256，真实 context442–825。按 mask/长度预选含低无效比例和高无效比例 batch；无效 allocated row 比例4.07%–48.54%。同 checkpoint、FP32 master、optimizer、scheduler、RNG 与 anchor plan；只改变 compaction。objective progress75起、horizon100，整个窗口不跨 stage。', '',
              '| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |', '|---|---:|---:|---:|---:|---:|---:|']
    for i in range(3):
        off,on=[read(raw/f'compaction-v2/pair{i}_{v}.json') for v in ('off','on')]
        require(off['initial_state_hashes']==on['initial_state_hashes'],f'A pair{i} initial state')
        require(off['binding']==on['binding']==parity['binding'],f'A pair{i} binding')
        a,b=[[s for s in leg['steps'] if s['measured']] for leg in (off,on)]
        require(len(a)==len(b)==10 and len(off['steps'])==len(on['steps'])==15,'A measurement window')
        require(all(x['batch']==y['batch'] and x['valid_tokens']==y['valid_tokens'] for x,y in zip(a,b)),'A same work')
        tm=[statistics.mean(s['wall_seconds'] for s in leg) for leg in (a,b)]
        hb=[leg['peak_hbm_bytes']/2**30 for leg in (off,on)]
        pairs.append(dict(pair=i,off_seconds=tm[0],on_seconds=tm[1],time_ratio=tm[0]/tm[1],off_gib=hb[0],on_gib=hb[1]))
        lines.append(f'| {i} | {tm[0]*1000:.3f} | {tm[1]*1000:.3f} | {tm[0]/tm[1]:.3f} | {hb[0]:.3f} | {hb[1]:.3f} | {100*(1-hb[1]/hb[0]):.2f}% |')
    case=parity['cases'][0]; batch=batches[0]
    profile('dfly-compaction-npu','dfly','upstream_two_stage','e2e_tv_compaction_pair','npu',
            dict(context=batch['context'],anchors=batch['actual_anchors'],query_rows=batch['query_rows_per_sample'],chunks=math.ceil(batch['query_rows_per_sample']/128),valid_supervision_rows=case['off']['valid_tokens'],padding_fraction=batch['padding_fraction']),
            len(set(i for b in batches for i in b['indices'])),60,{'binding_source_sha256':parity['binding']['source']},
            ['raw/compaction-v2/parity.json']+[f'raw/compaction-v2/pair{i}_{v}.json' for i in range(3) for v in ('off','on')],
            shape_scope='first parity batch; supervised rows are batch total; complete 8-batch shape plan in raw',microbatch=8,warmup_updates=30,measured_updates=60,matched_gpu_performance='PENDING')
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
                chunks=measured[-1]['attention']['memory_plan']['chunks'] if mode=='chunked_checkpoint' else 1
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
    lines += ['', '所有18个最终cell均可运行，没有 NPU OOM；因此不能声称 dense OOM而chunked可运行，也不计算不存在的容量 speedup。chunked在本表更慢（slower），通过重计算与多次dispatch换取部分大形状的显存下降；小形状峰值没有改善。', '',
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
              f'已有可计算wall：A最终全流程 {read(raw/"compaction-v2/result.json")["elapsed_seconds"]:.2f}s；B最终cell合计 {b_total:.2f}s；CUDA含pilot预留窗口 {cuda["gpu_reservation_wall_seconds_including_pilot_failure"]:.2f}s。这些不是整个项目的总NPU-h/GPU-h，未包含所有准备、失败、排队和其他进程，不能冒充完整预算核算。', '',
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
    manifest={}
    for base in (raw,cuda_directory):
        for path in sorted(base.rglob('*')):
            if path.is_file():
                manifest[str(path.relative_to(ROOT/'docs/npu'))]=sha(path)
    return dict(schema='veladraft-audit-derived-evidence-v1',profile_records=profiles,raw_artifact_sha256=manifest,
                compaction_pairs=pairs,attention_cells=memory_cells), '\n'.join(lines)


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
        matrix['audit_profiles']=list(old.values())
        for backend in matrix['backends']:
            if backend['name']=='cuda':
                backend.update(status='CUDA_FUNCTIONAL_REGRESSION_PASSED',reason='Four formal-geometry methods, three natural microbatch1 updates each, reduced actual anchors; exact base/head functional comparison, not performance.',evidence=['docs/npu/evidence/2026-10-21/cuda_regression/regression_summary.json'])
        matrix_path.write_text(json.dumps(matrix,indent=2,ensure_ascii=False)+'\n')
    from scripts.npu.validate_support_matrix import validate_matrix
    errors=validate_matrix(ROOT/'docs/npu/support_matrix.json')
    require(not errors,'matrix invalid: '+str(errors))
    print(json.dumps(dict(status='VALID',profiles=len(evidence['profile_records']),raw_artifacts=len(evidence['raw_artifact_sha256']))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
