# Qwen3.6-35B-A3B：A100 上七种 SD 方法全面评测

本文件是交给执行 agent 的任务说明。目标是在 A100 80GB 上，使用可复现的 vLLM 端到端口径，完成原生 AR 与七种方法的多模态评测，并在运行期间持续把真实进度和结果更新到 Hugging Face。

## 1. 范围和优先级

- Target：`Qwen/Qwen3.6-35B-A3B`，BF16，完整权重和完整目标词表。
- 方法：`dflash`、`dspark`、`domino`、`jetspec`、`dflash2`、`dflare`、`dfly`。每种方法必须加载自己的真实最终 checkpoint 和全部对应模块。
- 对照：同一机器、同一 GPU、同一 target、同一输入和计时边界下的原生 vLLM AR。
- 用户明确不要求评测 ViSpec 方法；不要把 ViSpec、Medusa、EAGLE 或 MTP 当成七方法之一，也不要把 MTP 的加速套用到这七种方法。
- 唯一正式数据范围是 [MMSpec](https://github.com/killthefullmoon/MMSpec) 官方完整 `dataset/MMSpec/test`；不运行之前 ViSpec 论文的 benchmark。模型、方法、硬件和推理框架不同，结果不能写成对 MMSpec 原文数值的直接复现。
- 不重新训练。可在独立开发集选择运行参数；保留原 checkpoint、导出权重和所有原始结果。

先完成七方法和主要数据集的真实评测；数据准备、正式计时、质量评分和上传可以形成队列。短检查必须自动接入正式队列，不能长时间停留在冒烟阶段。单个数据集或评分器不可用时记录原因，并继续其他可运行项。

## 2. 当前代码状态与交接门槛

代码仓库：<https://github.com/LUOseu/VelaDraft>。

七方法 vLLM 适配和 A800 MoE 配置已经通过 [PR #39](https://github.com/LUOseu/VelaDraft/pull/39) 合入 `main`；核对基线为 `4df140e61e2fee7a07921afa209bd9a8b5fde6b5`。关键代码：

| 用途 | 路径 |
| --- | --- |
| 七方法原生适配与各方法模块 | `research/qwen36_native_vllm/` |
| vLLM 0.32 target、位置适配与 worker | `research/qwen36_v032/` |
| 已验证的按 query 宽度选择 MoE 分组 | `research/qwen36_grouped_e2e_20261102/dispatch.py` |
| 七方法导出、校验和已有开发集证据 | `research/qwen36_seven_20261102/` |
| 请求计时和 GPU 资源工具 | `vlm_dflash/eval/native_runtime.py`、`vlm_dflash/eval/native_resources.py` |
| 原生评测与已有质量评分实现 | `vlm_dflash/eval/`、`docs/benchmark.md` |

七个公开 checkpoint 仓库均可通过 `khazic/veladraft-qwen36-400k-{method}` 查找。仓库内 `research/qwen36_seven_20261102/checkpoint_manifest.json` 已记录七个 `ckpt/ckpt_final.pt` 的 revision、大小和 SHA-256；以它作为检查起点，并核对实际使用的文件。目标模型此前使用的 revision 是 `995ad96eacd98c81ed38be0c5b274b04031597b0`，本次实际 revision 必须在运行前复核并冻结。

| 方法 | 需要保留并核对的机制 |
| --- | --- |
| DFlash | 并行 block 草稿与 target 特征 / 位置 |
| DSpark | 前驱 token 条件下的 Markov 修正；记录实际启用的 head |
| Domino | prefix GRU 状态与修正 |
| JetSpec | 本仓库已验证的单路径、block-causal 语义；不可称为已运行上游树实现 |
| DFlash2 | block 内卷积、unary 和 top-16 候选选择器 |
| DFlare | 分层特征融合及独立 context K/V |
| DFly | 特征融合及前驱条件 hidden 修正 |

已合并的适配代码是起点，不代表完整新数据集入口已经可用。现有部分研究脚本绑定 A800、本机绝对路径和 12 个开发 session；不得把它们直接称作本任务的完整数据集 runner。`vlm_dflash/eval/native.py` 的旧版本约束也不能默认等同于本轮 vLLM 0.32 运行路径。

**借卡同学开始正式运行之前，执行 agent 必须：**

1. 在独立分支实现或接好通用数据输入、七方法运行器、结果汇总、断点续跑和 HF 发布器。优先复用仓库与公开评测框架的现成算子、数据处理和评分器。
2. 将所有执行所需的代码、锁定依赖、配置、测试及启动脚本提交到 GitHub PR；检查干净 checkout 的依赖闭包。禁止依赖未提交的本机文件、`/root/autodl-tmp` 私有目录结构或临时手工复制的代码。
3. 在 PR 中记录代码 commit、版本、测试结果和未验证项。优先合入 `main` 后交接；若 PR 尚未合入，提供公开 PR URL、明确的 commit SHA 和可 checkout 的分支，不能让对方默认拉取 `main` 后缺文件。
4. 只有 A100 实际完成模型加载、七方法短检查、数据读取、HF 写入和自动续跑后，才能声称 A100 入口可运行。没有 A100 测试时标为“代码已提交，硬件验收待完成”，不能伪造“已跑通”。

`scripts/run_qwen36_a100_benchmarks.sh` 是本任务要求交付并在 PR 中验证的**最终入口合同**，不是基线 `4df140e` 已提供的脚本。执行 agent 必须先实现它，再使用下面的命令；不要向借卡同学提供一个尚不存在的命令。

现在可以用下面一条命令获取已合并的七方法代码和本交接文件；这只是检出交接材料，不会启动 GPU 评测：

```bash
bash -lc 'git clone https://github.com/LUOseu/VelaDraft.git VelaDraft && cd VelaDraft && git fetch origin refs/pull/40/head && git checkout --detach FETCH_HEAD && cat research/qwen36_a100_seven_eval/AGENTS.md'
```

完整评测脚本实现、测试并进入可获取的 PR 后，执行 agent 使用对应的已审查 commit 和以下最终入口：

```bash
git clone https://github.com/LUOseu/VelaDraft.git
cd VelaDraft
# 将 <REVIEWED_COMMIT> 替换为已合并 main 或公开 PR 的实际 commit SHA。
git checkout --detach <REVIEWED_COMMIT>
test -f scripts/run_qwen36_a100_benchmarks.sh
bash scripts/run_qwen36_a100_benchmarks.sh --config configs/eval/qwen36_a100_seven.yaml --suite mmspec-full --publish --resume
```

入口需支持 `--help`、`--preflight`、`--suite`、`--config`、`--publish`、`--resume`；从空 checkout 按文档完成依赖安装、数据准备和模型下载后，一条 `bash` 命令即可开始或安全续跑。若前置数据或凭据需要人工申请，`--preflight` 要清晰列出，而不是在运行数小时后才失败。命令与配置示例必须由实际 CLI 测试验证。

## 3. MMSpec 全量数据

正式评测使用 [MMSpec 官方仓库](https://github.com/killthefullmoon/MMSpec) 的 `dataset/MMSpec/test`，不是 `testmini`；其论文报告 600 条样本、六类各 100 条。固定仓库 commit，获取完整 `mmspec.jsonl` 和 `images/`，核验文件、图片、类别计数、ID 与 turns。若实际下载数量或定义与论文不一致，记录差异并检查来源，不擅自补样或将子集称为全量。

| MMSpec 类别 | 论文报告样本数 | 数据来源 |
| --- | ---: | --- |
| General VQA | 100 | GQA |
| Text VQA | 100 | TextVQA |
| Image Captioning | 100 | COCO |
| Chart VQA | 100 | CharXiv |
| Complex Reasoning | 100 | MMMU-Pro |
| Multi-turn Conversation | 100 | ConvBench、MM-MT-Bench |

保留官方 prompt、图像和多轮顺序；先检查官方数据和评测代码，固定多轮历史构造、停止规则、生成长度及计时口径，报告任何适配差异。多轮计时和质量分别说明使用固定历史还是各方法自身生成历史；不能混算。按 sample 和 turn 记录覆盖、失败原因，AR 和七方法均完整跑完六类。`testmini` 只允许用于冒烟，不作为正式表；`caption_marker_test` 不代替正式 `test`。不额外运行此前 ViSpec 论文的数据集或其他 caption 扩展。

记录官方数据 commit、各图片 hash、样本 ID、类别、turn、prompt 模板和输入映射。核查官方样本是否附有可评分答案和评分脚本；能复现的质量指标按原方法报告，缺少参考答案或评分器标记 `pending`，保存生成结果供后续评分，不将生成内容或代理指标冒充官方分数。

## 4. A100 配置与方法优化

用户确认 GPU 为 **A100 80GB**。首选单卡 TP1、batch1，保持完整 BF16；先实测 target、drafter、KV/state 和最大上下文是否安全放入。如果显存不够，停止并报告实际峰值，再统一评估多卡；不能悄悄对某个方法量化、CPU offload 或减少视觉输入。

启用 native vLLM、CUDA Graph、target full-attention 的 FlashAttention 和兼容的 draft attention；目标的 Gated DeltaNet / recurrent state 使用其原生实现，不强行替换为 FlashAttention。保持线性 block verification 和 causal/MRoPE 语义，不打开树形验证。拒绝后检查 target 的 KV 与 GDN 状态回滚；至少包含真实接受和拒绝案例。使用现有官方算子和最小适配，不重写 CUDA/Triton 内核。

此前 A800 TP1 的七方法在 K=3、固定 512 token、12 个开发 session / 48 次请求上均有正点估计：vLLM 0.32、V2 model runner、BF16、FlashAttention、CUDA GDN，query 宽度 3–8 使用现有 MoE 分组并配合已测的全局 tile 配置。它属于**已测起点**，不是 A100 或本数据集的正式结果。不要把训练面板的 AL 当成推理接受长度，也不要把 MTP 的词表裁剪结果记到七方法上。

每个方法需要在独立开发集重新比较 `K=1..7` 中该方法真实支持的值，必要时连同可用的 GDN、MoE 分组和 CUDA Graph 配置一起调优。每个方法使用相同的搜索预算和开发数据；冻结方法级配置后再跑测试集。记录方法、K、draft block 约束、target/draft TP、实际 attention、GDN、MoE config 和 fallback。对不支持的 K 记录原因，不伪造有效测量。

保留 AR 的高效配置，并在适用时提供相同的公共后端优化；主结果同时列出原生默认 AR 和测得更快的有效 AR，避免通过改变 AR 制造加速。A800 的 MoE 配置文件以 GPU 名称匹配，不能改名冒充 A100 最优；在 A100 上重新验证或调参，保存实际选择值。历史 MMSpec 小集曾用于调参；核对其图像/样本与本轮测试的重叠，无法排除时如实披露，不能声称完全独立 held-out。

若有多张 A100，可在资源充足时使用独立 GPU lane 加速；每个 AR/SD 配对使用相同硬件布局。限制 CPU 线程与磁盘读取，避免其他 lane 的模型加载或训练干扰计时。采集实际 GPU UUID、拓扑、驱动、显存、进程和干扰日志。必要时同步预热或缩小并发；不终止其他人的任务。

## 5. 冻结实验协议和正确性

主口径为 `temperature=0` 的 greedy，自然停止，`max_new_tokens=1024`，batch1、关闭 prefix caching 和多模态处理缓存；每项至少三次技术重复。具体 `max_model_len`、像素范围和视频帧策略在预检后统一冻结，AR 和 SD 一致。另做固定 512 token 对照：`ignore_eos=true` 且清空显式 stop token，真实生成满 512 个 token；标记可能包含 EOS 后内容，不用它计算自然回答质量。

如需温度 1 扩展，先完成七方法的 greedy 主表，再验证每个方法在目标温度下的 native sampling / rejection 是否正确；仅对通过验证的方法扩展温度 1，并记录多个独立 seed。不可用的组合显式标 `blocked`，不得借用 greedy 数据填表。不同采样路径的随机数消耗不同，不要求同 seed 的 AR/SD 逐 token 相同。

计时从图像读取和 prompt 构造开始，经过 vision encoder、prefill、decode 和 detokenization，结束于请求完成；加载模型、编译和预热不计入。记录原始单请求单调时钟、输出 token 数、结束原因、错误、GPU 环境和源代码 hash。SD 的特征捕获、草稿推理、验证、重算及适配开销均计入 SD。

保留 native target verification。理论上 proposal 的限制不改变目标模型验证规则；实际 BF16 并行和逐 token 执行可能有输出差异。不要把逐 token 完全一致作为速度或质量评测的阻塞门槛，也不要将所有差异自动归因于浮点误差。记录 AR 与 SD 输出一致率、不同样本及其任务分数；如发现 SD 提交不符合其自身 target verification，先修复，相关结果不得入表。

若使用多轮输入，冻结并公开同一历史构造方法。固定 AR 历史用于速度可比时明确标注；真实多轮交互质量另行测量，不能混作同一种口径。

## 6. 指标、质量与有效性

每个数据集、方法、配置和温度至少输出：

1. 请求级 E2E 加速比：`sum(AR_wall_seconds) / sum(SD_wall_seconds)`；报告 AR/SD 原始总时间、每请求延迟分布、样本数与失败数。
2. 每 token 耗时对照：分别计算 `sum(time)/sum(output_tokens)`，再计算 AR/SD 比值。参考 MMSpec 官方论文和代码核实其指标定义，按实际实现单独复算并标注差异。两种口径不能混称或挑较大值充当同一个指标。
3. 原生接受统计：接受的 draft token 数、提议数、验证轮数及每轮平均接受长度；额外输出包含 bonus token 的有效产出时须标注定义。不能把训练 AL、总输出 token/轮数或缺失值冒充原生接受长度。
4. 任务质量：AR 与每种 SD 的官方分数、差值、有效配对数及失败记录；按数据集保留原始尺度和评分版本。Caption 仅在参考标注及对应评分器可用时报告适用指标，说明限制。
5. 输出一致率、输出长度、EOS/length/timeout 比例，以及资源干扰和方法有效性状态。速度、质量和 token-exact 分开报告。

对相同样本做配对 bootstrap，按原始图像或 session 聚类；相关问题、turn、重复和 seed 保留在同一簇。报告 95% CI、聚类单位、样本数、重采样次数和随机种子。多数据集同时提供逐数据集结果、明确标记的宏平均与全量请求池化结果；不要把逐请求 speedup 的简单平均当成主表。

MMSpec 质量评估若需外部 judge / 密钥，先准备可复现的答案与评分输入；无凭据时质量项标 `pending` 并说明，不能填 0 或宣称已评分。持续完成已授权的速度任务和可用评分，不因 token 差异分析或单一评分器阻塞全部任务。只有真实运行、覆盖与资源检查通过的行才能进入正式表；失败重试保留旧记录，不能用历史开发集数字补缺。

## 7. 持续更新 Hugging Face `main`

统一发布到模型仓库：<https://huggingface.co/khazic/veladraft-qwen36-400k-dflash> 的 **`main`**，所有七方法共用一个新目录：

```text
eval/qwen36_a100_seven_v1/
  runs/<run_id>/manifest.json
  runs/<run_id>/status.json
  runs/<run_id>/raw/<dataset>/<method>/<shard>.jsonl
  runs/<run_id>/scores/<dataset>/<method>.json
  runs/<run_id>/summary.json
  INDEX.json
```

启动时生成唯一 `run_id`，并在第一次上传结果前将 target、七个 checkpoint 与数据集的 **实际 revision / hash** 固定到 manifest。由于向 DFlash 仓库的 `main` 上传结果本身会产生新 commit，后续模型下载必须仍使用开始时冻结的 checkpoint revision；不能跟随不断变化的 `main`。

正式计时开始前检查 HF 登录与目标仓库写权限；缺权限时让负责人解决，避免长时间运行后才发现无法持续共享。不要把 token、密码、私有 SSH 地址或密钥写入代码、日志、HF 或 Git。模型和数据缓存、导出、结果均放在服务器磁盘，**不使用用户本机模型缓存**。

由单个发布器在每个完成 shard 后或最长五分钟更新一次状态；长请求期间可上传轻量心跳。完成的原始 shard 不覆盖，`status.json` / `summary.json` 标明 `running`、`partial`、`complete`、`failed` 或 `blocked` 及实际覆盖数。使用 HF commit 的 `parent_commit` / 冲突重读重试，限制上传路径只在本任务目录；其他人的 checkpoint、训练日志和已有 `eval/qwen36_native_vllm` 数据不得覆盖或删除。网络中断时持久化上传队列、指数退避并继续本地可执行工作；恢复后按内容 hash 补传并读回核验。

评分、上传和诊断单独安排；不能让 CPU 大量压缩、模型加载或另一个 GPU 任务干扰正式计时。连续状态更新不得把计划任务、冒烟结果或未通过检查的样本写成正式测量。

## 8. 执行顺序与最终交付

1. 读取本说明及仓库现有适配；在干净分支补齐可移植入口、数据映射、评分和发布器，运行 CPU 单测、CLI / resume / 上传冲突测试并提交 PR。
2. A100 预检：确认驱动、CUDA / PyTorch / vLLM 版本兼容、完整 BF16 权重、七个 checkpoint、显存与磁盘、HF 写权限。缺少依赖时记录并解决；未经环境所有者授权不升级宿主机驱动。
3. 每方法完成一次真实加载和少量功能检查，包含验证拒绝和回滚；检查成功后自动开始独立开发集调参，冻结七个配置与数据 hash。
4. 运行 MMSpec 完整 `test`：AR 和七方法的自然停止计时、固定 512 对照，完成可运行的质量评分；分 shard 持续上传 `main`。
5. 在全量 MMSpec 完成后，仅按需要处理已验证的温度 1 扩展；缺失项列出原因。不要为了补表降低原模型配置或虚构结果。
6. 从原始日志独立重算样本覆盖、时间之和、指标、配对 CI 和质量差值，核对权重、方法模块、GPU 资源及 HF 上传 hash；写出最终汇总 CSV/JSON、失败清单和复现命令。

交付时给出：GitHub PR 与合入状态、可拉取的 commit、真实存在的一键命令、HF `main` 结果目录、实际表格、原始文件及覆盖/失败项。若某项尚未实现或 A100 尚未验证，准确列为未完成，不能把计划中的脚本或 A800 数据说成 A100 正式结果。
