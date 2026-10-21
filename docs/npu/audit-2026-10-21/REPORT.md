# PR30 审稿补强：可重算证据报告（2026-10-21）

本报告区分正式模型几何、实际训练工作量与证据适用范围。NPU recipe、DFly compaction、attention 三路径以及 CUDA base/head 功能回归已有结果；跨后端匹配性能对照与 CUDA 上完整导出验收仍按实际状态列出。

所有训练几何为 3 层、hidden 2048、FFN 11008、16 Q / 2 KV heads、head_dim 128、block8、BF16；诊断 recipe / CUDA / attention 使用 microbatch1，A compaction 使用正式 microbatch8。自然数据来自预先固定的 64 条 capture-v2 子集，未把合成张量或生成几何图片混作自然数据。

## 请求与证据对照

| 请求 | 状态与边界 |
|---|---|
| 正式 DSpark / DFly objective 与阶段 | PASS：真实 teacher 分布、有效监督、专属 head 梯度；短程诊断，不是质量训练 |
| DFly stage 边界恢复 | PASS：固定 horizon12，在已提交 step5/6 重新加载，完整最终状态逐项相等 |
| A：NPU compaction 同工作负载对照 | PASS：8 个固定 batch 数值检查，3 对、每 leg 5 warmup +10 measured |
| A：匹配 GPU 性能对照 | PENDING：当前无独占窗口，同正式最坏batch的CUDA限额容量资格检查尚未运行；CUDA功能回归不代替本实验 |
| B：attention 实测 | PASS：自然短上下文与 shape-only 分列；未发现 OOM 容量边界，不能称容量胜出 |
| CUDA base/head 回归 | PASS：4 方法、各 3 update；同种子初始化、同数据与配置，非吞吐对比 |
| 可控proposal接受路径 | 28个CPU fixtures：零/部分/整块接受、EOS/尾块截断、KV commit；仅控制流正确性，不证明模型质量 |
| C：NPU continuation / export → CUDA | NPU continuation/export 已完成；CUDA forward 与64 held-out真实接受由独立原始结果确认，未完成前保持 PENDING |

## 来源与复算边界

部署基线为 `4625ef676cd77b37a7fa9ed1f0e9f3864ee5f5ca`，但硬件运行含未提交修改，不能把该 commit 当作已运行源码的完整身份。NPU 部署没有 `.git`，原始 JSON 中的 `source_hashes` 和 A binding.source 才标识实际源码；CUDA 每方法记录对应 dirty-tree SHA256。不同时间点源码哈希不同，报告保留这些区别。

`profiles.json` 同时固定所有小型 raw artifact 的 SHA256；矩阵 PASS 行逐字段绑定对应 `profile_records`。本生成器重算时间均值、比值、行数和声明一致性，不重新执行硬件，也不能仅凭 JSON 重建梯度张量。大型 checkpoint / features / 输出梯度文件留在执行主机，未加入仓库。

## 正式配方及恢复

DSpark 执行 10 updates，loss 13.5 → 6.84375；实际每步有效监督 15–607 rows。teacher 流每步实际投影，KL 与 analytic confidence BCE 系数均为1，CE 为0。Markov/confidence head 非零梯度见 [原始记录](raw/recipes/dspark.json)。

DFly 固定 horizon12、split6，连续历史有6个 LK/D-PACE update 和6个 CE+e2e-TV update。step5/6 保存点恢复到12后，model、FP32 master、optimizer、scheduler、RNG 和完整 history 均与连续运行相等。原始连续运行在完成 checkpoint 后遇到主机内存压力，重载验收改用 CPU mmap 与逐个独立进程，未改变 horizon 或训练数学。

新鲜 observer 记录的 stage1 梯度/teacher 证据覆盖 step6 一个 update，stage2 覆盖 step7–12 六个 update；不把这一点写成六个新鲜 stage1 observer 样本。连续 checkpoint 的12步历史以 CPU mmap 只读提取，见 [连续元数据](raw/recipes-dfly/continuous_checkpoint_metadata.json)、[step5恢复](raw/resume-5/results.json)、[step6恢复](raw/resume-6/results.json)。

## A：DFly stage2 compaction（NPU）

正式 microbatch8、anchors256、block8；固定8个自然 batch，实际每样本 anchors12–256，真实 context442–825。按 mask/长度预选含低无效比例和高无效比例 batch；无效 allocated row 比例4.07%–48.54%。同 checkpoint、FP32 master、optimizer、scheduler、RNG 与 anchor plan；只改变 compaction。objective progress75起、horizon100，整个窗口不跨 stage。

| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 703.263 | 622.747 | 1.129 | 33.261 | 24.003 | 27.84% |
| 1 | 702.445 | 623.525 | 1.127 | 33.263 | 24.005 | 27.83% |
| 2 | 703.079 | 623.382 | 1.128 | 33.263 | 24.005 | 27.83% |

时间比大于1表示 NPU 此工作负载缩短 update 时间；这不是跨硬件绝对速度排名，也不是已证明 GPU/NPU 都加速。每个 leg 的前5步 warmup 不进入均值。峰值为 allocator 实测 allocated HBM，未用 memory_plan 估算替代。

初次 [parity失败](raw/compaction/parity.json) 保留：两个 batch 的 legacy loss_weight 报告标量因零填充布局改变 reduction 顺序而不严格相等。修复只令该标量在两侧按相同非零权重顺序归约；recipe loss/梯度数学未变，容差未放宽。完整8-batch gate和全部6个性能leg在新源码 binding 下重跑，见 [最终parity](raw/compaction-v2/parity.json)。CPU严格检查与既有 H1 BF16 envelope 分开记录；通过 BF16 envelope 不等同逐位相等。

GPU 同配置 compaction 对照 PENDING：当前没有独占窗口，同正式最坏batch的CUDA限额容量资格检查尚未运行。NPU约33GiB峰值不能证明CUDA在22GiB限额下不可运行，两个后端的kernel和显存行为不同。保持C优先，准备相同64条features后再资格检查；若限额内可运行，并在有其他任务时完成受保护的3对实验，也只能记为有竞争环境下的描述性诊断，不能称干净的GPU性能基准。现有功能回归不能替代A。

## B：attention 公平三路径与负面结果

下表从每个 raw cell 的3个 measured full updates 重算；每cell先5个 pilot/warmup。FP32-master AdamW，BF16 forward/backward。dense comparator 是 harness 的完整 query SDPA，逐层构造 mask；不是 stock HF 的一次 dense-mask 构造路径。checkpoint范围仅 mask构造+SDPA，不包括MLP/projection/loss；KV head展开在checkpoint外。

| data | requested C/A | actual C/A | path | chunks | ms/update | peak GiB | OOM |
|---|---|---|---|---:|---:|---:|---|
| natural | 1024/64 | 825/64 | dense | 1 | 54.605 | 5.606 | no |
| natural | 1024/64 | 825/64 | dense_checkpoint | 1 | 68.893 | 5.610 | no |
| natural | 1024/64 | 825/64 | chunked_checkpoint | 4 | 112.048 | 5.610 | no |
| natural | 4096/256 | 825/249 | dense | 1 | 152.430 | 8.862 | no |
| natural | 4096/256 | 825/249 | dense_checkpoint | 1 | 182.445 | 7.137 | no |
| natural | 4096/256 | 825/249 | chunked_checkpoint | 16 | 297.617 | 6.537 | no |
| natural | 4096/512 | 825/249 | dense | 1 | 152.429 | 8.862 | no |
| natural | 4096/512 | 825/249 | dense_checkpoint | 1 | 183.176 | 7.137 | no |
| natural | 4096/512 | 825/249 | chunked_checkpoint | 16 | 306.382 | 6.537 | no |
| shape | 1024/64 | 1024/64 | dense | 1 | 56.416 | 5.606 | no |
| shape | 1024/64 | 1024/64 | dense_checkpoint | 1 | 68.824 | 5.610 | no |
| shape | 1024/64 | 1024/64 | chunked_checkpoint | 4 | 112.232 | 5.610 | no |
| shape | 4096/256 | 4096/256 | dense | 1 | 127.582 | 9.016 | no |
| shape | 4096/256 | 4096/256 | dense_checkpoint | 1 | 152.134 | 7.238 | no |
| shape | 4096/256 | 4096/256 | chunked_checkpoint | 16 | 309.306 | 6.607 | no |
| shape | 4096/512 | 4096/512 | dense | 1 | 268.821 | 15.764 | no |
| shape | 4096/512 | 4096/512 | dense_checkpoint | 1 | 333.635 | 11.687 | no |
| shape | 4096/512 | 4096/512 | chunked_checkpoint | 32 | 550.485 | 9.085 | no |

所有18个最终cell均可运行，没有 NPU OOM；因此不能声称 dense OOM而chunked可运行，也不计算不存在的容量 speedup。chunked在本表更慢（slower），通过重计算与多次dispatch换取部分大形状的显存下降；小形状峰值没有改善。

自然数据只选中同一个最长825-token完整样本：4096是零填充bucket，实际anchors受答案长度限制为249，requested512没有达到。不能把这几行写成4096真实上下文/512有效anchors训练；真正4096/512只能由单独shape-only行支持。NPU1专用于B，但NPU0 recipe并行工作共享主机CPU/RAM，所以时间为有仪器开销的共享主机测量。

最初BF16直接AdamW的预演结果保留在 `raw/memory/preflight_BF16optimizer_not_final/`，不参与此表；最终18cell均改用production FP32 master重跑。feature-layout首次配置失败与SSH中断见 [harness事件](raw/memory/harness_incidents.json)，不隐藏为PASS。

## CUDA：固定base/head功能回归

base `8cacc9db4cee10de4a7c7aedfe3b3d461b8d9716`；head基线 `4625ef676cd77b37a7fa9ed1f0e9f3864ee5f5ca` 加每方法dirty source hash。4方法在同一隔离环境、相同3个自然样本、相同初始化/配置上各执行3updates，初始/最终权重、loss/全部梯度与probe均完全相同。

| method | actual contexts | actual query rows | valid supervision | result |
|---|---|---|---|---|
| dflash | [297, 311, 320] | [32, 64, 136] | [28.0, 56.0, 119.0] | exact PASS |
| dspark | [297, 311, 320] | [32, 64, 136] | [31.0, 63.0, 135.0] | exact PASS |
| dfly | [297, 311, 320] | [32, 64, 136] | [28.0, 56.0, 119.0] | exact PASS |
| domino | [297, 311, 320] | [32, 64, 136] | [31.0, 63.0, 135.0] | exact PASS |

requested anchors256，但实际仅4/8/17；不能称正式anchors256负载回归。A6000有其他任务，限额环境只支持功能对比，不作吞吐结论。隔离venv使用transformers5.12.1，原活动环境5.11.0未升级；环境检查证实未安装torch_npu。最初gradient probe整数索引修复前失败保留在CUDA原始目录，不是模型算子失败。

## C、失败与资源边界

NPU已用具有接受能力的R3 checkpoint做10步stage2 weight-only continuation，再导出portable权重；使用新optimizer，不称跨后端bitwise恢复，亦不称从头训练。CUDA导出完整验收及64条自然held-out接受结果仍需独立结果完成后更新。训练loss里的e2e_accept_len不是实际解码接受率。

已有可计算wall：A最终全流程 312.43s；B最终cell合计 508.07s；CUDA含pilot预留窗口 562.70s。这些不是整个项目的总NPU-h/GPU-h，未包含所有准备、失败、排队和其他进程，不能冒充完整预算核算。

## 执行路径与复现

28个可控CPU proposal fixtures覆盖sequential/parallel两种控制流中的零接受、部分接受、整块接受、EOS、尾块长度截断与KV提交；修复让accepted/proposed计数只覆盖实际发出的后缀。测试源码见 [test_acceptance_path_fixtures.py](../../../tests/test_acceptance_path_fixtures.py)，这些fixture不证明自然图像drafter质量或NPU并行serving。

独立团队agent完成代码审查，范围、发现、修复与56个定向测试见 [CODE_REVIEW.md](CODE_REVIEW.md)。这是同团队独立代码审查，不是外部跨模型科学评审。

`reset_attention_runtime_stats()` 划定测量窗口；`attention_runtime_report().measurement_window`累计完成的attention-core调用及fused/fallback chunks。记录发生在output projection之前，因此随后projection异常时也可能已计数；不是整个attention module成功完成次数。mixed路径显式标记；内部chunk checkpoint重计算不计作新forward，完整外层checkpoint重放可能算一次完成调用，因此不能把这些计数当optimizer updates。strict禁止fallback，profile允许且保留原因。

`recipe_compile_tv_kl` 的 requested/effective/reason 标记的是device gate；`device_gate_not_execution_evidence` 不证明实际进入编译kernel。NPU gate禁用compile；现有CUDA compiled-KL结果不能充当NPU优化。memory_plan始终只是分析估算，表格HBM来自设备allocator实测。

复算已附小文件（无需设备）：

```bash
python scripts/npu/summarize_audit.py --check
python scripts/npu/validate_support_matrix.py
```

重新执行硬件（需要原始features/checkpoints与相同环境；并非本生成命令自动执行）：

```bash
python scripts/npu/audit_recipes.py --device npu:0 --bundle CORPUS_OR_BUNDLE --config-dir configs --out NEW_RECIPE_DIR
python scripts/npu/audit_compaction.py --help
python scripts/npu/audit_attention.py --help
python scripts/npu/audit_cuda.py --help
```

命令参数、固定batch/shape、首次5步耗时估算及初始状态hash保留在各raw目录和协议文件；重跑应写新目录，不覆盖已发生的失败记录。

## 补充完整性检查与账本

C 原始运行脚本没有在结束前显式检查全部最终权重有限性。保留原始结果与源码hash，另外对同一已完成产物做posthoc CPU mmap检查：10步持久化loss/gradient norm全部有限，全部最终模型张量有限，weight digest与原始summary一致。该补充不是重新训练，也不把后续修复过的harness冒充原执行源码，见 [supplemental integrity](raw/continuation/supplemental_integrity.json)。

| NPU0进程 | start epoch | end epoch | wall秒 | exit |
|---|---:|---:|---:|---:|
| recipes | 1792574023 | 1792574111 | 88 | 137 |
| recipes-dfly | 1792575245 | 1792575326 | 81 | 137 |
| compaction | 1792575326 | 1792575404 | 78 | 1 |
| resume-5 | 1792576107 | 1792576158 | 51 | 0 |
| resume-6 | 1792576158 | 1792576210 | 52 | 0 |
| compaction-v2 | 1792576210 | 1792576536 | 326 | 0 |
| continuation | 1792576536 | 1792576583 | 47 | 0 |

上述NPU0可核对进程wall合计723s（0.2008设备小时的预留wall），包含两次exit137和初次A失败；不是kernel忙碌时间。B最终18cell另计508.07s，旧BF16预演和capture/shared-forward缺完整进程起止，不能声称预算完整闭合。并行不同设备需按设备求和，不能仅取整体wall。

本地完整CPU套件835 passed / 6 skipped（JUnit摘要与文件hash见raw/local_validation.json）。这些验证不能替代NPU/CUDA硬件证据；本生成器另有失败关闭测试。
