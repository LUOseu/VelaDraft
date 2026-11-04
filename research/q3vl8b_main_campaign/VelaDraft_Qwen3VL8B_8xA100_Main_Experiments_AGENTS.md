# AGENTS.md | VelaDraft 的 Qwen3-VL-8B 主实验

2026-11-04 · 一节点 8 × NVIDIA A100 80 GB · 执行目标：产出论文主表、逐样本原始结果和可复现命令。面向已经持有对应数据与模型的服务器 agent。

## 代码仓库、当前状态与执行环境

**代码位置与状态。** 代码库为 [LUOseu/VelaDraft 的实验准备分支](https://github.com/LUOseu/VelaDraft/tree/codex/q3vl8b-main-campaign-20261104)，上游代码基线是 `56fde8c563222ec3196a053cafb6f6e213bd97ac`；本次准备包在 `research/q3vl8b_main_campaign/`。克隆该分支后记录实际 `git rev-parse HEAD`，若合并了新补丁，同时冻结 commit、dirty diff、依赖版本和验证结果。服务器上现存的准备副本是基线归档加准备包，不是 Git worktree；正式运行请使用上述 Git 分支的新 worktree。

**当前尚未完成正式实验代码。** `prepare_configs.py` 仅生成以原始 8B 配方为依据的四个 B 实验候选配置（默认两个方法各 ref/opt，一个 seed），标记为 `DRAFT_NOT_LAUNCH_READY`；`monitor.sh` 仅导出和同步部分进度，不会启动训练。真实 200K 数据及 dev/test 路径和血缘、同 seed 配对初态和 anchor 流、四臂调度与精确恢复、A/B/C/D 采集与重算尚待实现及端到端验证。不得把旧 seed 42 的四臂调度器或草稿配置当成已就绪的本次完整矩阵。当前准备节点是 4×RTX 3090 24 GB，只用于基础验证；正式 A100 硬件、训练成本和部署性能必须在目标 8×A100 80 GB 节点上实测。已有测试范围见本目录 `VALIDATION.md`。

**代码入口。**

- `research/q3vl8b_main_campaign/prepare_configs.py`：由 `original_configs/dflash2.json`、`dfly.json` 派生一个 seed（默认 17）× 两个已选方法（DFlash2、DFly）× ref/opt 共四份草稿；`original_configs/PROVENANCE.json` 记录公开源版本与 hash。原始路径仅供溯源，须按实际节点重绑数据、target 和 dev/test。
- `research/q3vl8b_main_campaign/progress.py`：从 `work/<run>/train/` 的指定训练、验证和计时记录导出到 `raw/updates/<run>/`。当前不自动导出任意 stdout、模型、optimizer、图片或其他全部原始资料；A/C/D 的采集脚本必须另行实现。
- `research/q3vl8b_main_campaign/monitor.sh` 与 `hf_sync_safe.py`：前者循环调用进度导出和同步，默认 300 秒一次；后者按白名单增量上传，记录 `.hf_sync_state.json` 的成功时间、文件 hash 和 commit，失败后重试，不删除远端其他文件。同步状态不是实验完成凭证。
- `scripts/q3vl8b_trainopt/launch_run.sh` 是已有的单臂训练入口；`launch_campaign.sh`、`make_configs.py`、`jobs.py`、`scheduler.py` 是旧实验接口和实现参考，其现有 seed 42 配置与调度行为不能直接当作本次四臂正式实现。`vlm_dflash/target/sample_capture.py` 负责图文特征捕获，`research/qwen3vl8b_native_vllm/` 提供 native adapter 与质量评测相关入口；具体能力以节点代码和测试为准。

**Bash 环境示例。** 以下路径是占位符，必须由节点负责人替换成实际可读的原始数据和模型；新建唯一运行目录，勿覆盖现有实验。训练器的 `launch_run.sh` 需要节点本地 `TRAINOPT_ENV`，该文件应设置 `VELA_CORPUS`（含 `data/` 和 `regen/`）、`VELA_TARGET`、`VELA_LOCAL`、`HF_CACHE_HOME`、`TP`；如使用旧 `launch_campaign.sh configs`，还须提供 `VELA_OLD_TASK` 和 `VELA_DEVTEST`。用 `export` 定义这些变量，`TP` 为训练 Python 可执行文件路径。训练入口会使用 `HF_HUB_OFFLINE=1`，因此所需模型和数据应事先准备好。**旧 `launch_campaign.sh configs` 会把 `TRAINOPT_ENV` 复制到运行目录 `scripts/node_env.sh`，不得把它或凭据上传。**

```bash
# 若尚未取得代码，在新目录检出独立分支；不要覆盖已有实验目录。
git clone --branch codex/q3vl8b-main-campaign-20261104 \
  https://github.com/LUOseu/VelaDraft.git VelaDraft-q3vl8b-main
cd VelaDraft-q3vl8b-main
git rev-parse HEAD

# 在克隆的实验分支根目录；这些占位路径必须先替换。
export CAMPAIGN=/path/to/new-unique-campaign
export TRAINOPT_ENV=/secure/path/node_trainopt_env.sh
mkdir -p "$CAMPAIGN"

# 基础接口检查，不会开始训练。
python -m vlm_dflash train --help
python -m vlm_dflash native-eval --help

# 只生成候选配置；要求输出目录不存在或为空，不启动训练。
python research/q3vl8b_main_campaign/prepare_configs.py \
  --dflash2 research/q3vl8b_main_campaign/original_configs/dflash2.json \
  --dfly research/q3vl8b_main_campaign/original_configs/dfly.json \
  --out "$CAMPAIGN/configs"

# 完成下文所有输入、数值一致性和 sanity 门控后，才能考虑调用已有单臂接口：
# bash scripts/q3vl8b_trainopt/launch_run.sh CONFIG_JSON RUN_DIR GPU_UUID
# 中断恢复须走训练器自身的 --resume 精确校验；不能只复制模型权重。
```

这里的候选配置没有自动绑定 dev/test 或完成配对初始化验证。程序保留 25k/50k 中间快照设置；75k 是最终端点，必须核对真实最终 checkpoint、导出和完成凭证，不能依赖 `state_step0075000.pt` 必定存在。草稿生成成功不等于可训练或已满足 75k 更新。

## 运行进度与 Hugging Face 归档

归档到公开 model 仓库 [khazic/veladraft-qwen36-400k-dflash](https://huggingface.co/khazic/veladraft-qwen36-400k-dflash) 的独立前缀 `training_system_eval/qwen3vl8b_main/<campaign-name>/`，其中 `<campaign-name>` 是运行目录最后一级目录名；命名要唯一。当前代码的上传白名单是 `manifests/`、`configs/`、`raw/`、`checks/`、`tables/` 中允许的文本文件及 `STATUS.md`、`REPORT.md`、`run_manifest.json`、`jobs.json`。上传前验证公开发布权限、脱敏内容和目标仓库写权限。模型权重、optimizer、图片、节点环境脚本、软链接不会由此同步器上传；模型导出如需另行归档，必须另设受控流程并记录 hash。

同步器需要安装 `huggingface_hub` 的 Python、节点自身的 Hugging Face 写入登录，以及 `VELA_SANITIZE` 指向的节点脱敏 JSON。登录时在目标机器交互完成，不要把 token 写进文档、命令行、运行目录或提交。脱敏 JSON 在运行目录外；格式为 `{"replace": [["待替换的正则", "替换文本"]], "leak": "仍禁止上传的非空正则"}`。应涵盖该节点的内部路径、主机及敏感信息；通用过滤不能替代对实际数据的审查。现有准备节点曾未检测到 Hugging Face token，实际写入权限和长期同步必须重新验证。

```bash
# 使用拥有 huggingface_hub 且已完成交互式登录的 Python；指定节点专用规则。
export MONITOR_PYTHON=/path/to/python-with-huggingface-hub
export VELA_SANITIZE=/secure/path/node_sanitize_rules.json
export MONITOR_INTERVAL=300

# 首次先检查身份和手工导出；whoami 成功不代表一定有仓库写权限。
"$MONITOR_PYTHON" -c 'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])'
"$MONITOR_PYTHON" research/q3vl8b_main_campaign/progress.py "$CAMPAIGN"
"$MONITOR_PYTHON" research/q3vl8b_main_campaign/hf_sync_safe.py "$CAMPAIGN" --once

# 上一步核对实际提交、远端文件和脱敏结果后，才启用持续监控。
# 该命令前台运行；可由节点的 screen / tmux / 作业系统托管。
bash research/q3vl8b_main_campaign/monitor.sh "$CAMPAIGN"
```

每阶段检查 `.hf_sync_state.json` 的 `last_success_utc`、`last_commit` 和 `blocked`，并核对 Hub 页面；同步进程失败不应中断本地训练，但须保留错误、重试和补传。当前同步器不会上传 provenance 目录及未列入扩展名的文件，不能声称已自动归档全部证据；需要上传的新增证据应通过明确实现及验证的白名单路径提交。同步成功或存在状态文件均不代表某个实验完成。

## 论文要回答什么

文本 LLM 上的 block-diffusion drafter 能用一次小模型 backbone 前向提出多个词，VLM 可以用同样的机制减少串行生成轮数。迁移到图文训练后，同一张图的读取、空间对齐和反复编码都要付费，一两个答案词却只能提供很少的起草位置。VelaDraft 保留图文空间状态、跨轮复用冻结 teacher，并按各个 loss 的依赖范围组织预测块。用下列四张表把问题、解决办法和实际收益连起来。

| 主表和唯一职责 | 主对照 | 读者要看到的量 |
|---|---|---|
| A　图像改变了什么 | 同一题的图像证据 / 文字证据，同一 8B target；自然图文 / 自然文本作外部参照 | 实际视觉位置、可训练 anchor、跨阈值比例，以及图像条件下每个有效监督位置的成本 |
| B　平台解决了多少训练成本 | 同一 drafter、全图 teacher、训练目标及数据下的可运行直接移植参考路径 / VelaDraft | 单卡完整更新时间、有效监督位置/s、峰值显存、全程 GPU-h，以及同预算最终质量 |
| C　训练记录里哪部分图像值得保存 | 成熟 drafter、同一题、同一视觉行预算，交换题目证据区与无关区 | 可提议事实 token 的配对命中差、实际接受量、每条记录字节与提取成本 |
| D　训练出来的 drafter 值不值得部署 | 同卡同请求 native target-only AR / 七种 parallel drafter | 六项视觉任务的整请求加速、毫秒/请求、官方质量、实际接受草稿数/轮 |

A、B、D 是平台主张的核心。C 的按问题选区方法依赖视觉证据，和固定 quarter 的旧适配假设分开。四表各按开发时登记的维度出全结果。最终写论文时，每表用一条带数字的结论开头。现有图保持原样。

## 先锁版本、预算与输入

- 代码基线 [LUOseu/VelaDraft@56fde8c](https://github.com/LUOseu/VelaDraft/tree/56fde8c563222ec3196a053cafb6f6e213bd97ac)。从其独立 worktree 工作；服务器另有已验证的空间重放和 native 补丁时，把它们的 SHA、diff 与测试一起冻结。主要入口见文末。确认排到的是八张 A100 80 GB；每个主运行独占一张，TP1。按物理 GPU UUID 编排。计算和 I/O 竞争单独校准。
- Target 用同一 revision 的 Qwen3-VL-8B-Instruct、BF16、同一 processor/tokenizer、像素上下限和 chat template。保存环境、Flash/attention 实际后端、图像预处理、seed、target/adapter/导出权重 hash。参考与优化都处理真图。DFlash2 与 DFly 是训练主方法，DFlash 是第三个机制对照；七种方法全部进入部署表。
- 成熟探针和七方法部署表锁定同一批**原版 8B 权重**。按方法逐一核验原始 run 的真实 step、配置、导出和权重，包含 DSpark 的实际来源。仓库快照未带齐所有权重。125k CONT 单独登记，可作外推，不能补进原版权重的一行。旧 1k pilot 和已经看过的 240 个合成场景限用于开发。新确认集不复用旧 16 臂方案的留出集。
- 训练主数据为原 200K 图文清单，冻结其图像/源文档 ID、再生成答案、样本顺序规则与原有 dev/test。旧配对计时与终点见[训练证据](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflash2/tree/main/training_system_eval/20261029T210530Z_58534f4)。已有终点按实际评估样本数登记。新机制开发、独立确认和 native 质量测试分别建清单，按图像及文档/场景家族隔离、与训练集查重。锁清单和主指标后才开确认集。
- 第一份交付为 `run_manifest.json` 和 `jobs.json`，写明各实验的 seeds、图像清单 hash、数据血缘、方法×条件、卡号、预计卡时和结果目录。新建运行目录，保留现有 CONT 和其他作业。先用 20 请求和 200 次完整更新校准实际预算；所有计时附样本数和设备。

## A | 用同一道题定位多模态代价

**自然面板。** 新取 600 图，图表 200（ChartQA）、文字密集 200（TextVQA/DocVQA 各 100）、自然图 200（COCO/GQA 各 100）。另取 600 条自然文本短答，文本和视觉均用同一 8B target。报告自然工作量，并在共同的问题/答案长度区间给分层结果。阈值沿用有 hash 的旧开发定义；若其 tokenizer 或 cap 不兼容，用另取的开发集固定一次。记录每例文本长度 T、视觉位置 P、上下文 C、可选与实抽 anchor A、有效标签数及回答 token 数；分别数 T≤c<T+P 且 A≤a。旧 3B 的 20K 统计单列。

**同证据配对。** 另锁至少 240 个独立问题（120 张图表、120 件可核对转写的文档，每源至少一条配对）。一侧给原始图片，另一侧给人工/数据源核验的完整数据表或版面转写；两侧使用相同问题、相同有监督答案和 teacher 强制前缀。转写按固定规则生成，排除直接泄露标签的字段，保留解题所需行列或阅读顺序。抽查 40 对并记匹配率。涉及空间而无法忠实转写的题留在自然面板。分列 teacher 在两种呈现上的正确率；主成本比较保留全部有效配对。另做 text-length 匹配分层和相同 C、A 的张量重放，把输入形状与视觉前端成本拆开。

**三种结果分别回答不同问题，不得合并为一个模态结论。** 图像和文字转写即使语义一致，其 token 长度、信息结构、teacher 表示与计算路径也可能不同。① 自然图像 / 自然文本工作负载描述各自真实分布，不是同题因果对照；② 同证据配对比较同一题在真实图像与经核验的文字呈现下的结果，保留各自真实 token 数、teacher 表示及 teacher 正确性，不能声称逐 token 或内部表示相同；③ 相同 \(C,A\) 的张量重放用于隔离指定上下文长度、anchor 数和执行路径下的形状/训练计算成本，记录实际重放张量及其来源、是否包含视觉前端，并排除未测阶段的成本。三者分列样本、单位、分母、图表和结论；重放不能代替真实端到端图文成本，配对语义一致也不意味着前端成本或上下文构成一致。文字长度匹配分层另列覆盖和排除数，不能静默改变主配对集。

**直接移植基线 × 输入模态。** DFlash、DFlash2、DFly 分别完成相同矩阵，DFly 固定 stage 并标明其 loss。对同一底层 VLM 和答案，跑「图/文字 × 直接移植参考/VelaDraft」四格。参考路径必须正常用视觉 encoder、正确 M-RoPE 和真实标签，只使用普通密集执行与逐次 teacher 提取。优化路径按 B 表解耦开关。登记同模态的绝对节省，以及两种模态上节省的配对差；回答按视觉工作量会增加多少收益。固定同一逻辑 batch 和 anchor；每格都测完整更新 wall、teacher 前端、head rows 和每个有效监督位置成本。四格主对照全部关闭缓存以隔离更新成本。以 (图像参考秒 − 图像优化秒) − (文字参考秒 − 文字优化秒) 估计图像呈现带来的额外节省，固定监督量，按源图/文档成对重采样。长文本但 A 少的样本也进入统计。

## B | 配对训练直到收敛，并拆出每项收益

**本节点的 seed 与方法范围。** 本实验 B 只使用一个预先冻结的训练 seed（默认 17），固定选择前期效果较好的两个方法 DFlash2 与 DFly，共四个训练终点。这里的“一个 seed”是训练随机性设置，不是把同一条路径重复计作多个 seed。

**主比较。** DFlash2、DFly 各做 Reference / VelaDraft 两条生产训练路径；每种方法内部保持模型、全图特征、原生 loss、训练数据、学习率、optimizer、初始权重、随机数据/anchor 流和有效 batch 相同。参考保留方法正确的 selector 或前缀依赖；为接收图像所需的 adapter 两边共享。按配置执行每种方法实际支持的优化，DFlash 的联合分组另测。将这个基线标为“同配方的直接移植参考实现”。独立框架（如 SpecForge）只有在真实 8B 图文输入、相同输出语义和数据路径通过验证后才有单独的具名一行。

**规模。** 每臂从初始化训练 200K × 3 epochs，75,000 optimizer updates，原 8B 正式配方的有效 batch 8、BF16、序列上限 4,096、block size 8。以正式解析配置锁定 anchor 上限及 DFly 阶段切换步数。不要套用 256 行 reduced 配置。实验 B 使用一个预先冻结的新 seed，默认 **seed 17**；固定跑前期结果较好的 DFlash2、DFly 两个方法的 Reference / VelaDraft 四条路径。一个 seed 下同方法的 ref/opt 必须共享初始化、数据顺序、optimizer、有效 batch 和 anchor 流；不能把单 seed 结果写成跨 seed 稳健性结论。现有历史 seed 仅作单列历史复验，不增加本次 B 的训练重复数。本节点的全程 wall 和 GPU-h 始终从新运行取得。保存 0/25k/50k/75k 的训练及验证轨迹、最终导出、实际成功更新及中断恢复位置。

**训练质量与训练效率分别报告。** 每方法、每 seed、每路径至少单列：training loss trajectory（实际成功更新步数及记录频率）、固定 dev objective 曲线与终点、最终 checkpoint 在预先冻结的共同任务/样本上的自由解码实际接受草稿 token/验证轮、全程实际训练 wall、活跃 GPU-hours 和 peak GPU memory；保留最终 test objective、各 seed 数值和 ref/opt 配对差。明确 wall 是否含预处理、验证、checkpoint、恢复及排队，GPU-hours 按实测设备占用时段并注明统计方法。test objective 不能单独充当训练质量或接受长度的替代指标，质量与成本的变化也不能合成一个无定义的结论。

**接受长度的预注册非劣检验。** 在查看确认集结果之前，冻结同任务/请求/权重评估协议、配对单位、统计方法和显著性水平，保留有时间戳的注册记录。主量明确为实际自由解码中每次真实验证轮接受的草稿 token 数；target 首词、校正与 bonus 不算草稿接受数。对同方法同 seed，预注册相对差 \((\mu_{opt}-\mu_{ref})/\mu_{ref}\)，非劣界固定为 **−1%（−0.01，相对参考值）**，单侧显著性水平 0.05；只有按预注册方法得到的单侧 95% 下置信界严格高于 −0.01 才能声称非劣。参考值为零、覆盖不足或区间无法可靠估计时，报告不可判定。结果出来后不得修改界值、分母、分析集或单侧方向来获得通过；偏离须透明记录并另列为探索性分析。报告该单一 seed 的接受量、配对相对差与区间，并区分按图/会话重采样的不确定性；单 seed 不能估计跨 seed 训练随机性，也不能写成跨 seed 稳健性结论。若后续增加第二个 seed，必须在查看结果前登记为新的实验条件，不能事后补充以改变非劣结论。

**解释收益的开关。** 同一卡的相同快照、AdamW 状态、方法阶段、逻辑 batch 及 anchor 流上交替执行，三个独立窗口，AB/BA 轮换。在线窗口 50 次预热 + 500 次完整计时更新；固定特征重放 20 + 200 次。断开 sampler/profiler 重新计总 wall。逐层、逐项输出以下对照。

| 成本 | 必做对照 | 记录 |
|---|---|---|
| 重复编码图文 teacher | 新取 1,024 图 × 6 遍；cache OFF / ON（容纳实测工作集）/ 受限 byte budget；匹配访问顺序、整批训练更新和冷启动 | 实际 ViT 样本/批调用次数、语言主干、命中、RAM、I/O、完整训练阶段秒；独立 profiler 秒另列 |
| 长 C、少 A 的组批 | DFlash 相同逻辑 batch，原序等大小拆组 / C-only / C×A；同物理组大小及同一次 optimizer 更新；覆盖自然负载和尾部 | 最大可跑 batch、补齐行、有效行、峰值 GiB、更新秒。DFlash 的结果不平移到其他配方 |
| 目标依赖决定投影粒度 | DFlash2 dense→共用 teacher projection→pack + saved KL；DFly 依正式 stage 分组对照 dense→完整有效块→适用的 fused loss | 真实投影行、KL/CE/TV、梯度及一次更新一致性、秒、峰值；对 row-local 与 prefix-coupled 分列 |

除缓存专门实验外，在线计时两侧关闭 feature cache，与现有 8B 配对表衔接。逐步核对密集/稀疏布局的 loss、梯度、AdamW 后参数（BF16 容差先由重复参考运行标定）。缓存以完整样本及 target/processor/layer 版本为键；同图不同问题保留各自的融合状态。主质量给同方法同 seed 的终点 dev/test objective、各任务接受量和配对差；若要写“接受长度保持”，预注册相对每轮接受量 −1% 单侧非劣界，报告各 seed 及区间。固定权重内按图像配对重采样，训练随机性按独立 seed 对报告；图像和 token 数均不增加训练重复数。

## C | 用正确的图像区域检验视觉语言交互

**主诊断无需重训。** 在 A 的 600 张新图上用成熟 DFlash2 做主行，DFlash、DFly 复核。每个问题保留自然答案和 EOS。对三方法固定同样的 full-image target；先在至少 24 条请求上将 full 参考路径与已验证运行时的实际提议、首次拒绝和接受前缀对齐。覆盖奇偶网格、不同长宽/批 padding、多图（在支持的适配器上）和短轨迹。保存坐标、span、teacher 层顺序与首处数值分歧。冻结真实提议位置；target 自己提交的首词、校正和 bonus 分开记。

**局部证据是主比较。** 另取独立 240 个基础编辑场景（图表读数与类别各 120；原图/相关编辑/无关编辑），保留编辑框。再在独立实图上做不少于 100 张 ChartQA 和 100 张 TextVQA/DocVQA 的证据区域核对，可直接取 A 的独立确认图、按图聚类。保留所有文本行和原始 (t,h,w)。同一真实图的四个视图为 Full、均匀 quarter、固定底图 B + 答案证据区、同一 B + 匹配的无关区；后三者逐图恰好 K 行。B 避开两个候选区域；补入行数、形状和版面尽量匹配。编辑图用自身重新捕获的 teacher 状态。报告目标可解释、可对齐、且确由 drafter 提议的分母，以及全体请求覆盖。主要对照为证据区减无关区的事实位置 top-1 命中百分点和胜/负/平图数，经 10,000 次按源图或模板家族配对重采样给 95% 区间，并单列实图的区间；辅以相关编辑正确翻转率、无关编辑稳定性和自由解码的实际接受量。保留没有七个后续 token 的图表答案。

**测清信息来源与代价。** 在全 600 图上至少比较 Full、均匀 quarter、去显式视觉行（文本仍已读图）、同任务同网格的视觉行替换。替换仅作用于 drafter，固定当前图的文本与目标轨迹；匹配不到 donor 的覆盖另列。记录特征 GB、核验过的槽命中、接受草稿数/实际验证轮。分任务、答案长度和内容词报告。识别 OCR/数值使用冻结的标注匹配规则；词串出现只用于分层。原生日志的“已接受”与固定前缀的“一步预测正确”分别计量。

**固定轨迹命中与自由解码接受必须分列。** 固定 teacher 轨迹、固定前缀、真实 drafter 提议位置上的一步 top-1 命中率，是条件预测指标，分母是符合条件的提议位置；自由解码实际接受草稿数/验证轮，是运行时指标，分子和分母取自真实提议、验证及接受日志。两者记录各自覆盖率、样本量、过滤原因与单位，不把 teacher 首词、纠正或 bonus 混入草稿接受量。论文中必须同时展示两类结果及其边界：top-1 提升本身不能证明真实接受长度、请求时延或加速提升；若两者方向不一致，分别如实报告，不用一种指标代替另一种。

**条件性方法。** 先用独立开发集检查 full 正常、可提议事实位置覆盖充分，并在看确认集前冻结开关。若相关区在开发集有可复现的 top-1 收益，在一次完整 teacher prefill 中仅用问题词与视觉行的注意力选 K 行，层/聚合在开发集锁定；禁止使用答案、后续 hidden 或标注框。把自动选择、均匀网格、分格随机采样在相同 K、实存字节、选择开销下放进同一张确认表。若要主张“这种记录更适合训练”，从同一个成熟 DFlash2 权重做 3 个配对续训 seed ×「自动选区 / 均匀网格」，每臂 10k updates，冻结数据、优化器初始化方式、曝光和独立末次测试。此项另列预算；机制复核无论结果如何都交。开发无信号时，不开这六个续训任务。保留原 1k/16 臂计划的决策记录。

## D | 在同一 A100 上测训练后实际请求

- 主性能面板用 MMSpec 六个任务，每项 100 个会话，共 600 sessions。所有方法共享图像、提示和固定的多轮历史，历史用本次 8B target 生成。同一 GPU 上做各条请求的 AR/SD 配对，8 卡按 session 分片，每片跑 AR 和所有方法并轮换顺序。统一 native vLLM 版本、TP1、batch 1、BF16 greedy、Flash 配置、像素、自然 EOS、生成上限 1,024，3 次重复。使用经复验的原版 8B 推测长度设置，记录真实 K（DSpark 的 anchor 约定单列）。关闭跨请求 prefix / 图像缓存，模型预热从主计时剔除，图像读取到答案输出的整个请求进入 wall。旧 A6000 和 35B/A800 表保留各自硬件标签。
- 主表按任务计算 ΣAR wall / ΣSD wall，另有中位、p95、输出长度、TTFT、实际接受草稿/轮、接受/提出、验证轮数、完整会话数。完整序列一致率、首分歧和共同前缀覆盖配合官方任务质量。逐真实会话聚类，重复和多轮一起重采样给 95% CI。功耗或通信训练指标不作这张表的目标。
- 质量面板沿用可复算的原生评分器，TextVQA、DocVQA、ChartQA 各 256 条，冻结源版本、gold、去重和输出正规化；分别给 soft VQA accuracy、ANLS、relaxed accuracy 和 AR 配对差。按样本返回原始答案及停止原因。另在固定任务子集上用 target-only AR、同一个成熟 DFlash2 的 K=1/K=3/原 K，拆 prefill/draft/verify、按输出长度分层，说明收益来自哪些串行轮数。若有对该 target 和 runner 确已兼容的自回归 drafter，可另列；K=1 不作为另一种算法的替身。
- 用**B 表新训练的最终导出**闭环：对 B 实验的 4 个新终点均跑相同的 120 个 MMSpec 会话（各 20）和配对质量面板，报告同 seed 两种训练路径的真实部署差。成熟七方法的 600 会话表用于外部生态比较，不能冒充新终点。每方法首批 8–12 例核对 target/adapter hash、提议来源、active-KV 和实际 backend，再进入整表。

## 排卡与交付

| 阶段 | GPU 0–1 | GPU 2–3 | GPU 4–5 | GPU 6–7 |
|---|---|---|---|---|
| 首批 | A 的数据与四格校准 | B 的一致性、窗口和显存 | C 的轨迹/区域对齐 | D 的新节点 adapter、AR 校准 |
| 主训练 | DFlash2 seed 17：ref / opt | DFly seed 17：ref / opt | 预留：验证/重排 | 预留：验证/重排 |
| 训练前后空档及结束后 | B 的独占窗口/缓存 | C 的确认集 | D 的逐会话切片 | D 的其余切片和新终点 |

GPU 对是两个单卡作业。每个短计时对在同一物理卡上完成；每个阶段结束做释放与重排。CPU 准备转写、gold、hash、子集及评分可以与主训练并行。不得用其他卡的 AR 时间作分母。按旧实测在线约 2.55–3.18 s/update，8 × 75k 的纯更新约 **425–530 A100 GPU-h**，单轮并行训练约 **53–67 h**。初始给 A/C/缓存/原生性能及质量额外预留 **100–200 GPU-h**；正式预算按 20 条请求的输出长度和计时外推（600 × 8 路径 × 3 重复 = 14,400 次会话执行，未含新终点）。数据就绪、独占整节点时估计 **4–6 天**，数据转写或 adapter 移植另排。可选 6 × 10k 续训以 replay 1.26–1.53 s/update 估约 **21–26 GPU-h**，另计选区提取和评测。记录分配、实际活跃 GPU-h、墙钟与重试，容量不足先保四张主表。

**执行入口。** 使用 [训练配置与 scheduler](https://github.com/LUOseu/VelaDraft/tree/56fde8c563222ec3196a053cafb6f6e213bd97ac/scripts/q3vl8b_trainopt)、[图文特征捕获](https://github.com/LUOseu/VelaDraft/blob/56fde8c563222ec3196a053cafb6f6e213bd97ac/vlm_dflash/target/sample_capture.py)、[native 8B adapter 与官方质量评分入口](https://github.com/LUOseu/VelaDraft/tree/56fde8c563222ec3196a053cafb6f6e213bd97ac/research/qwen3vl8b_native_vllm)。先读节点上的真实命令，不把 research 快照的旧绝对路径照搬。以下仅为已有单臂和旧配置接口；当前尚未实现本任务的完整四表矩阵、选区与报告脚本，不得作为一键正式执行入口：

```bash
# 在新 worktree 根目录执行。TRAINOPT_ENV 与 CAMPAIGN 由节点原有环境和新运行根目录指定。
: "${TRAINOPT_ENV:?set node environment file}" "${CAMPAIGN:?set a new run directory}"
python -m vlm_dflash train --help
python -m vlm_dflash native-eval --help
# 旧 configs 仅用于参考：其 seed 42 配置不是本次 8 臂，且会复制节点环境文件。
# 派生并核对本次 seeds/输入/开关、完成所有门控后，已有单臂接口：
# bash scripts/q3vl8b_trainopt/launch_run.sh CONFIG_JSON RUN_DIR GPU_ID
```

为新矩阵交付一个支持 dry-run、状态查询、断点恢复、按 job hash 跳过已完成工作的 dispatcher。先打印八卡映射与预算，再执行。每阶段写 `STATUS.md` 与机器可读状态。运行目录交付 `manifests/`、`configs/`、`raw/{workload,updates,views,requests}/`、`checks/`、`tables/{A_workload,B_training,C_visual,D_native}.{csv,tex}`、`REPORT.md` 和一条从原始行重建四表的命令。报告按“结论、最强数字、适用设置、对应原始文件”写。上传实验结果时沿用项目既有 HF 归档约定，使用独立前缀；论文表只填有原始记录、配置和完成凭证的格子。
