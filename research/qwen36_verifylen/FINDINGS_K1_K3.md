# Qwen3.6 验证开销：K=1–3 初步排查

数据：2026-11-01，A6000 GPU 1/2（跨 NUMA）、TP2、BF16、DFlash、vLLM 0.22.1。三个固定初始上下文、各重复三次、每次最多生成 128 tokens；等权平均三个上下文的每步慢 rank CUDA stream span。启用诊断钩子，非正式无扰动端到端测试。

| K | 验证位置 | Target forward ms | 相对 AR | Execute + sampling ms | 接受草稿/轮 | 完整宽度验证步 |
|---:|---:|---:|---:|---:|---:|---:|
| AR | 1 | 6.398 | 1.000 | 7.418 | — | — |
| 1 | 2 | 9.010 | 1.408 | 11.445 | 0.747 | 600 |
| 2 | 3 | 10.871 | 1.699 | 13.281 | 1.204 | 480 |
| 3 | 4 | 12.291 | 1.921 | 14.895 | 1.329 | 453 |

K1 最终报告序列化失败，9 条原始请求及双 rank spans 已完整保存并在零外来进程采样条件下恢复，标记 `recovered_complete`，保留原始字节、错误日志和恢复凭据。K2、K3 正常完成。K2/K3 仍属于尚未结束的独占租约；截至 21:34 的资源采样未见 foreign PID/error，不宣称连续独占。

K1、K2、K3 的 9 次请求各自均输出 1044 个 tokens；完整宽度 target forward 的慢 rank span 分别相加约 5419.94、5227.96、5566.70 ms。它们是计时区间之和，**不是端到端墙钟**；不同 K 的后续 token 内容可能不同。K3 比 K2 每轮多接受约 0.125 个草稿 token，但完整宽度验证次数只减少 27 次且单步更贵，因此在该小样本中，K3 的 target-forward span 之和高于 K2。不能据此确定正式集最优 K。

## 已确认的代码路径和实测范围

1. 两边启动均显示 FlashAttention 2、FULL decode graph 配置及 Triton MoE；GDN prefill 日志选择 Triton/FLA。prefill 选择日志不能单独证明验证阶段的 kernel 调用。
2. 执行机上的 [`qwen_gdn_linear_attn.py`](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) 与 [`fused_sigmoid_gating.py`](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py) 与 vLLM v0.22.1 上游文件 SHA-256 完全一致。默认 packed recurrent decode 打开时，普通非 speculative decode 有专门快路径；存在 speculative metadata 时走多位置状态更新分支。后者 kernel 内对 T 位置循环并更新 recurrent state，其他如 head 维度仍可并行。这不等于逐 token 启动多次 target forward，也不表示无法优化。实际 kernel 占比尚缺 trace。
3. K1/K2/K3 的 proposal-total inclusive stream span 分别约 1.633/1.894/1.931 ms；target-forward 分别约 9.010/10.871/12.291 ms。它们是不同有钩子的区间，不代表端到端占比，也不能把 parent 和 child span 相加。
4. 同一口径下 context K/V 包装区间约 0.092/0.095/0.094 ms stream；draft LM head 约 0.827/1.041/1.076 ms。前者主机区间仍有开销，且相关计时可能含同步和 stream 间隙；不能据此认定所有 context 处理都不重要。K1 实测 context K/V 输入形状均为 `(2, 2048)`，并非直接证据表明每轮重算全部历史。
5. vLLM 自动调整 GDN 状态页对齐：attention KV page 在 K1 为 1056、K2/K3 为 1072。这与 draft length 是不同概念。当前测试反映改变 K 后原生运行时的整体反应，不是仅更改单个 kernel 的纯长度 A/B。

## 推论与下一步

当前证据支持「验证路径成本和接受长度之间存在权衡」，并指出官方 GDN 多位置递推是需实测的共性候选；**不支持把整体差距断定为 VelaDraft adapter bug，也不支持认定七个 checkpoint 都没训好**。K4–7 继续运行。需独立 kernel-level trace，区分 GDN、其他 target 层、TP 通信与输出 head；然后在目标 A100 环境做未挂钩的配对端到端 K 对照，A6000 跨 NUMA 结果不能直接替代 A100 主表。
