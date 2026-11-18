# Qwen3.6 验证开销排查：K=1、2 的初步结果

2026-11-01。范围：A6000 物理 GPU 1、2（跨 NUMA）、TP2、BF16、DFlash、native vLLM 0.22.1。3 个固定初始上下文、每个重复 3 次、最多输出 128 tokens；单步为两 rank 中较慢者的 CUDA stream span，按 3 个上下文等权平均。带诊断钩子，不是无扰动端到端测速。

| 条件 | 实际验证位置 | Target forward | 相对 AR | 执行加采样/拒绝 |
|---|---:|---:|---:|---:|
| AR | 1 | 6.398 ms | 1.000 | 7.418 ms |
| K=1 | 2 | 9.010 ms | 1.408 | 11.445 ms |
| K=2 | 3 | 10.871 ms | 1.699 | 13.281 ms |

其中执行加采样不含 draft 和调用之间的间隔。K=1 最终 JSON 序列化失败，原始 9 条请求、双 rank 数据及资源 epoch 经检查后恢复，状态为 `recovered_complete`，原始字节与恢复凭据均保留。K=2 正常完成。

对于相同 9 次诊断请求，K1/K2 都输出 1044 个 token；完整宽度的 target forward 分别记录 600/480 步。逐步较慢 rank 的 target-forward span 加和约 5419.94/5227.96 ms，K2 约少 3.5%。这只是诊断区间的加和，不是端到端耗时或加速比；不同 K 的生成 token 可能分叉，不能声称内容一致。

官方计数器：K1 有 594 rounds、444 accepted、594 proposed，即 0.7475 accepted/round；K2 有 471 rounds、567 accepted、942 proposed，即 1.2038 accepted/round。不得把 `1+accepted/round` 当作实测每轮提交 token。

## 共用路径线索

1. 实际启动日志显示 AR 与 SD 均使用 FlashAttention 2、FULL decode graph 配置及 Triton MoE；GDN 的启动日志确认 prefill 使用 Triton/FLA，但这条日志本身不证明验证阶段 kernel 的调用情况。
2. 执行机上的 `qwen_gdn_linear_attn.py` 和 `fused_sigmoid_gating.py` 与 vLLM v0.22.1 上游字节哈希一致。源码中，启用默认 packed decode 时，普通非 speculative decode 可走 `_forward_core_decode_non_spec`，而 speculative metadata 会走另一分支，调用多位置状态更新。该更新 kernel 的时间维含 `for i_t in range(0, T)`，在每个时间位置更新并保存 recurrent state；heads 和其他数据维度仍可并行。这是实际源码可见的实现，不是自写 kernel，也不等于 target 逐 token 发起多次 forward。尚未获取内核 trace，不能断言该分支是总瓶颈或给出其耗时份额。
3. 在 K1 的 600 个完整宽度步骤中，rank0 的 proposal context K/V 包装区间平均约 0.092 ms stream、0.655 ms host；draft backbone 约 0.572 ms stream；draft LM head 约 0.826 ms stream；完整 proposal 约 1.638 ms stream、2.452 ms host。这些是带事件的区间，父子跨度包含关系会重叠，不可相加或视为精确独立 kernel 时间。后续步骤的 context K/V 输入观察到 `(2, 2048)`，不支持“每一步重算整个历史”的笼统说法。

## 后续判据

- 完成 K3–7，观察 target forward 的增量与真实完整宽度验证步数；当前只有两个 K，不能认定是线性或选最优 K。
- 用 GPU kernel trace 在单独诊断运行中区分 GDN speculative 更新、target 其他层、TP 通信、head；不修改主曲线的配置或把 trace 开销当成 E2E 数字。
- 比较 A100 时重新在 A100 同机实测：A6000 的跨 NUMA TP2 布局不能直接外推到 A100。
- 无需仅凭此结果重新训练，也不能凭现有证据断言 adapter 一定无错。状态回滚与输出质量仍需独立证据。

源码：[GDN 实现](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)、[speculative 递推 kernel](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py)。
