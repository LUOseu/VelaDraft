# PR30 审稿补强：可重算证据报告（2026-10-21）

本报告区分正式模型几何、实际训练工作量与证据适用范围。NPU recipe、DFly compaction、attention 三路径以及 CUDA base/head 功能回归已有结果；跨后端匹配性能对照与 CUDA 上完整导出验收仍按实际状态列出。

所有训练几何为 3 层、hidden 2048、FFN 11008、16 Q / 2 KV heads、head_dim 128、block8、BF16；诊断 recipe / CUDA / attention 使用 microbatch1，A compaction 使用正式 microbatch8。自然数据来自预先固定的 64 条 capture-v2 子集，未把合成张量或生成几何图片混作自然数据。

## 请求与证据对照

| 请求 | 状态与边界 |
|---|---|
| 正式 DSpark / DFly objective 与阶段 | PASS：真实 teacher 分布、有效监督、专属 head 梯度；短程诊断，不是质量训练 |
| DFly stage 边界恢复 | PASS：固定 horizon12，在已提交 step5/6 重新加载，完整最终状态逐项相等 |
| A：NPU compaction 同工作负载对照 | PASS：8 个固定 batch 数值检查，3 对、每 leg 5 warmup +10 measured |
| A：匹配 GPU 性能对照 | PASS：另一张空闲GPU2、相同原始workload，6个完整leg与guard退出0；allocator环境修订和此前OOM独立保留 |
| B：attention 实测 | PASS：自然短上下文与 shape-only 分列；未发现 OOM 容量边界，不能称容量胜出 |
| CUDA base/head 回归 | PASS：4 方法、各 3 update；同种子初始化、同数据与配置，非吞吐对比 |
| 可控proposal接受路径 | 28个CPU fixtures：零/部分/整块接受、EOS/尾块截断、KV commit；仅控制流正确性，不证明模型质量 |
| C：NPU continuation / export → CUDA | FAILED：冻结BF16跨后端forward数值门槛未通过；NPU训练/导出已完成，64条CUDA实际使用验收独立记录 |
| C：portable格式及CUDA实际使用（独立子项） | PASS：native/portable各64/64 token-exact、1931实际接受；不覆盖上面的BF16数值FAILED |

## 来源与复算边界

部署基线为 `4625ef676cd77b37a7fa9ed1f0e9f3864ee5f5ca`，但硬件运行含未提交修改，不能把该 commit 当作已运行源码的完整身份。NPU 部署没有 `.git`，原始 JSON 中的 `source_hashes` 和 A binding.source 才标识实际源码；CUDA 每方法记录对应 dirty-tree SHA256。不同时间点源码哈希不同，报告保留这些区别。

`profiles.json` 同时固定所有小型 raw artifact 的 SHA256；矩阵 PASS 行逐字段绑定对应 `profile_records`。本生成器重算时间均值、比值、行数和声明一致性，不重新执行硬件，也不能仅凭 JSON 重建梯度张量。大型 checkpoint / features / 输出梯度文件留在执行主机，未加入仓库。

## 正式配方及恢复

DSpark 执行 10 updates，loss 13.5 → 6.84375；实际每步有效监督 15–607 rows。teacher 流每步实际投影，KL 与 analytic confidence BCE 系数均为1，CE 为0。Markov/confidence head 非零梯度见 [原始记录](raw/recipes/dspark.json)。

DFly 固定 horizon12、split6，连续历史有6个 LK/D-PACE update 和6个 CE+e2e-TV update。step5/6 保存点恢复到12后，model、FP32 master、optimizer、scheduler、RNG 和完整 history 均与连续运行相等。原始连续运行在完成 checkpoint 后遇到主机内存压力，重载验收改用 CPU mmap 与逐个独立进程，未改变 horizon 或训练数学。

新鲜 observer 记录的 stage1 梯度/teacher 证据覆盖 step6 一个 update，stage2 覆盖 step7–12 六个 update；不把这一点写成六个新鲜 stage1 observer 样本。连续 checkpoint 的12步历史以 CPU mmap 只读提取，见 [连续元数据](raw/recipes-dfly/continuous_checkpoint_metadata.json)、[step5恢复](raw/resume-5/results.json)、[step6恢复](raw/resume-6/results.json)。

## A：DFly stage2 compaction（NPU）

数值检查：既有H1 BF16门槛8/8通过；严格逐位一致4/8。其余4个batch存在非逐位一致梯度，虽在冻结BF16门槛内，不能报告为全部bitwise parity。

正式 microbatch8、requested anchors256、block8；固定8个自然 batch，各batch每样本分配12–256个anchor槽位，个体有效anchor数由keep决定，不能将分配宽度写成每个样本的有效数量。真实context442–825，batch allocated query rows768–16384，kept rows600–8432，完整逐batch计数见batches.json。按mask/长度预选含低无效比例和高无效比例batch；无效allocated row比例4.07%–48.54%。同checkpoint、FP32 master、optimizer、scheduler、RNG与anchor plan；只改变compaction。objective progress75起、horizon100，整个窗口不跨stage。

| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 703.263 | 622.747 | 1.129 | 33.261 | 24.003 | 27.84% |
| 1 | 702.445 | 623.525 | 1.127 | 33.263 | 24.005 | 27.83% |
| 2 | 703.079 | 623.382 | 1.128 | 33.263 | 24.005 | 27.83% |

时间比大于1表示 NPU 此工作负载缩短 update 时间；这不是跨硬件绝对速度排名，GPU同workload结果另表报告，不能据绝对时间比较芯片优劣。每个 leg 的前5步 warmup 不进入均值。峰值为 allocator 实测 allocated HBM，未用 memory_plan 估算替代。

初次 [parity失败](raw/compaction/parity.json) 保留：两个 batch 的 legacy loss_weight 报告标量因零填充布局改变 reduction 顺序而不严格相等。修复只令该标量在两侧按相同非零权重顺序归约；recipe loss/梯度数学未变，容差未放宽。完整8-batch gate和全部6个性能leg在新源码 binding 下重跑，见 [最终parity](raw/compaction-v2/parity.json)。CPU严格检查与既有 H1 BF16 envelope 分开记录；通过 BF16 envelope 不等同逐位相等。

GPU A此前缺少独占窗口；随后同正式最坏batch在GPU1共享22/25GiB限额下均OOM，详见后面的资格检查。NPU峰值不用于推断CUDA容量。随后另一个设备GPU2出现空闲窗口，完整结果已完成，见GPU2配对表；不改写此前GPU1限额失败。

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

小形状的4组数值对照均通过：natural和shape-only各比较dense_checkpoint、chunked_checkpoint与dense参考的output、loss及全部梯度，沿用H1的rtol=0.02、atol=0.02。中/大形状记录的是完整update与资源可运行性，没有逐格重新做全梯度parity；不能把18个cell全部标为大形状数值等价。所有18个最终cell均可运行，没有 NPU OOM；因此不能声称 dense OOM而chunked可运行，也不计算不存在的容量 speedup。chunked在本表更慢（slower），通过重计算与多次dispatch换取部分大形状的显存下降；小形状峰值没有改善。

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

NPU已用具有接受能力的R3 checkpoint做10步stage2 weight-only continuation，再导出portable权重；使用新optimizer，不称跨后端bitwise恢复，亦不称从头训练。CUDA上64条实际使用验收独立通过，但BF16跨后端数值门槛失败，故combined C不通过。训练loss里的e2e_accept_len不是实际解码接受率。

已完成子项wall：NPU A最终全流程 312.43s；B最终cell合计 508.07s；CUDA base/head功能回归含pilot预留窗口 562.70s。这些不是整个项目的总NPU-h/GPU-h，未包含所有准备、失败、排队和其他进程，不能冒充完整预算核算。

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

本地完整CPU套件873 passed / 6 skipped（JUnit摘要与文件hash见raw/local_validation.json）。这些验证不能替代NPU/CUDA硬件证据；本生成器另有失败关闭测试。

## C：BF16跨后端数值门槛失败（必须保留）

NPU/CUDA使用相同权重、输入、dtype、实际student logits dtype、源码与anchor计划，所有身份核对通过。相同loss不能替代全部hidden/logits逐元素检查。冻结的既有H1 BF16门槛为rtol=0.02、atol=0.02，以下三项均失败：

| tensor | 违反门槛元素 / 总元素 | max abs | mean abs |
|---|---:|---:|---:|
| draft_hidden | 5 / 65536 | 0.125 | 0.0036107085 |
| post_correction_hidden | 93 / 65536 | 0.25 | 0.0059103854 |
| student_logits | 5356 / 4861952 | 0.125 | 0.0082938252 |

loss在两端均为0.2353515625，但combined C profile仍为FAILED，不能宣称完整跨后端数值等价。原始记录见 [BF16 shared-forward comparison](../evidence/2026-10-21/cuda_regression/C_shared_forward_comparison.json)。

后续FP32仅可作为独立诊断，不能替换这个BF16失败、改变容差或改写原始结果。portable格式与64条真实接受的可用性可以独立成功，但不能覆盖此失败；64条实际使用现已独立通过，见下表；combined C仍为FAILED。

## C：portable格式与64条CUDA实际使用独立通过

相同NPU continuation权重的native checkpoint与portable export在无torch_npu的CUDA环境加载。传输后的native文件SHA256和重建safetensors完整字节SHA256均匹配NPU原件；export config/target/processor/recipe身份校验通过。正式3层模型几何从SHA256验证后的export metadata读取，而非从测试名称推断。

| format | samples token-exact | accepted | rounds | emitted tokens | proposed draft tokens | τ=progress/round | A=accepted/round |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 64/64 | 1931 | 997 | 2936 | 6507 | 2.9448345035 | 1.9368104313 |
| portable | 64/64 | 1931 | 997 | 2936 | 6507 | 2.9448345035 | 1.9368104313 |

τ严格采用原始字段al_progress_tokens_per_round，A采用al_draft_tokens_per_round。两种格式逐样本token序列、接受事件和计数一致，显示真实非零接受路径；不据此声称质量排名或推理加速。每条最多生成64 tokens，verification=sequential，生产generate-parity配置开关为False。

首次运行漏设该False开关，错误使用另一SDPA backend；116条部分结果标为INVALID_PROTOCOL并归档，不计入本表，也不作为checkpoint缺陷证据。修复harness协议后按原64个held-out ID从头重跑，没有换样本或放宽token-exact要求。combined C的BF16跨后端数值FAILED仍保留，不能被格式可用性PASS覆盖。

## C：FP32补充诊断

另行将相同权重转为FP32进行shared-forward诊断；两端实际hidden/student logits dtype均记录为torch.float32，权重、输入、源码和anchor身份一致。该结果不替代BF16主门槛，不修改其失败记录或容差。原始诊断见 [FP32 comparison](../evidence/2026-10-21/cuda_regression/C_shared_forward_fp32_comparison.json)。诊断通过与低精度执行路径差异一致，但不足以单独证明具体误差根因。

| tensor | max abs error | relative L2 | 违规元素 |
|---|---:|---:|---:|
| draft_hidden | 1.1920929e-05 | 8.4409942e-07 | 0 |
| post_correction_hidden | 3.4332275e-05 | 6.205201e-07 | 0 |
| student_logits | 2.4795532e-05 | 5.4469837e-07 | 0 |

FP32 loss差值为7.8976154e-07。此诊断沿用原comparison的0.02/0.02门槛，同时完整报告实际误差；不把它包装成另一个预注册的FP32严格门槛。

## A：GPU1共享限额资格检查（保留失败）

没有根据NPU峰值猜测CUDA是否能运行，而是保持同一正式microbatch8、context825、anchors256、block8及固定最坏batch1进行真实OFF完整update资格检查。CUDA plan全部字段与NPU原计划一致，tree digest与NPU binding相同。只改变资源限额；没有缩batch、模型、context、anchors或更换数据。

| allocator cap GiB | outcome | guarded wall s | guard最低采样空闲MiB |
|---|---|---:|---:|
| 22 | OOM under cap | 34.070126 | 28798 |
| 25 | OOM under cap | 27.481875 | 6052 |

guard列仅是轮询时观测到的最低空闲值，不是连续测量的瞬时最低值；22GiB运行的快速分配峰值发生在轮询之间，以下OOM报文提供了更低的当时空闲值。这是共享设备上的显式allocator限额OOM，不是整张A6000物理容量OOM，也不是CUDA必然无法运行正式配置的证明。22GiB失败时PyTorch报物理空闲8.81GiB，25GiB失败时报5.91GiB；另一个进程约占16.69GiB。两次都在CE分配594MiB时触发限额，没有完整update时间或配对性能结果可报告。后续若另一张设备出现独占窗口，应作为独立资源条件记录，不改写这两次失败。

## A：CUDA GPU2独占计算窗口的完整配对结果

GPU2 preflight无其他compute进程、空闲48522MiB，持有GPU UUID advisory lock并监视外部进程；计算设备观测为独占，主机CPU仍共享。最终guard退出0、wall 318.377050s；全部6个leg均5warmup+10measured，原始NPU checkpoint权重、8-batch plan、每步有效监督/allocated/kept rows匹配。

CUDA既有H1 BF16门槛8/8通过；CUDA严格逐位一致6/8，其余2个batch存在非逐位一致梯度。NPU严格检查为4/8，不能把两端都写成bitwise parity。

| pair | off ms/update | on ms/update | off/on 时间比 | off GiB | on GiB | HBM减少 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 999.488 | 812.756 | 1.230 | 33.186 | 23.638 | 28.77% |
| 1 | 995.473 | 810.652 | 1.228 | 33.186 | 23.638 | 28.77% |
| 2 | 995.871 | 812.199 | 1.226 | 33.186 | 23.638 | 28.77% |

在这个固定诊断workload上，NPU和CUDA分别有compaction相对收益；只比较各设备内部off/on比值，不以两端绝对时间排名硬件，也不外推到其他shape或完整online训练。

GPU2第一次default allocator完整尝试在parity和pilot通过后、首个OFF benchmark leg backward中OOM：请求4.06GiB时live allocated29.13GiB、reserved但未用7.18GiB，仍有物理空闲10.70GiB，显式cap40GiB。原始失败保留在gpu2_full。缓存碎片是根据这些读数提出的解释，未采memory snapshot，不能断言机制已被证明。

随后仅做一次新进程、新输出目录的allocator环境修订：两臂均设置PYTORCH_ALLOC_CONF=expandable_segments:True，保持40GiB cap、数据、几何、seed、计划与全部5+10窗口；没有每step empty_cache。该修订在任何完整性能leg之前声明，最终全部leg在同一环境下重跑。

该选项在[已安装PyTorch 2.11官方文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)中为experimental、默认False；它可能改善变化分配大小导致的碎片。这里它是执行环境条件，不是VelaDraft算法创新，也不用于解释NPU kernel的收益来源。

## CUDA累计资源账本

所有已附串行guard命令累计2123.615311s（0.589893 GPU-h），含CPU加载、错误backend终止、BF16/FP32诊断、初始probe失败、22/25GiB资格OOM、GPU2默认allocator失败及完整重试；不是kernel busy time。预算3600s，已登记余量1476.384689s。原始逐项账本见 [cuda_gpu_ledger.json](../evidence/2026-10-21/cuda_gpu_ledger.json)。

## A：阶段开销与实际投影工作

每个backend/开关行合并3个leg的30个measured updates。阶段时间为每update均值；optimizer阶段含FP32 master梯度转移、clip、step、scheduler及权重copyback，不是单个optimizer kernel。投影rows为这30步的实际hook计数之和，来自raw steps而不是mask形状估算；forward student/teacher投影与包含backward重计算的全部投影分列。

| backend | compaction | forward ms | backward ms | optimizer ms | student rows forward | teacher rows forward | all rows incl. recomputation |
|---|---|---:|---:|---:|---:|---:|---:|
| npu | off | 193.549 | 487.487 | 21.894 | 196800 | 172200 | 541200 |
| npu | on | 185.259 | 416.032 | 21.926 | 156072 | 136563 | 429198 |
| cuda | off | 265.185 | 692.033 | 39.726 | 196800 | 172200 | 541200 |
| cuda | on | 225.809 | 546.279 | 39.781 | 156072 | 136563 | 429198 |
