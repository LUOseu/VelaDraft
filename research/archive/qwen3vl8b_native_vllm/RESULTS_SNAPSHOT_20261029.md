> Snapshot captured on 2026-10-29; later server results are not reflected. The raw evidence remains on the experiment server and is not included in this repository.

# Qwen3-VL-8B A6000 端到端实验结果

更新于 2026-10-29T15:56:25.716286+08:00；来源：服务器原始结果。

## 模型清单

| 方法 | HF 修订 | 2026-10-29 Hugging Face 历史快照（非当前根目录状态） |
|---|---|---|
| [dflash](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflash/tree/37a5eb63b12d851fe6114f3aeb87dbc7174dec3d) | `37a5eb63b1` | 快照含 final |
| [dfly](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dfly/tree/24331eae8ac5dbe0f12390aa32c92f4e7efec5ab) | `24331eae8a` | 快照含 final |
| [dflash2](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflash2/tree/ef0288357fedc8313119169b31027e7cd5aaa747) | `ef0288357f` | 快照含 final |
| [dspark](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dspark/tree/5ff8a27d6354d684bf2f068845445abfe9eaebff) | `5ff8a27d63` | 截至 10-29 10:51 快照未见 final |
| [dflare](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-dflare/tree/cbddc83e0aa38aff652102f52d0bfa6d08cc0eee) | `cbddc83e0a` | 快照含 final |
| [domino](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-domino/tree/e85aad00bb83e1fbdcbf89e6f7c156e4ef944439) | `e85aad00bb` | 快照含 final |
| [jetspec](https://huggingface.co/khazic/veladraft-qwen3vl8b-200k-jetspec/tree/0163faebb15cbcc73c77c07230d06f82b9166f44) | `0163faebb1` | 快照含 final |

该表仅记录固定的 HF 修订，不断言当前根目录的下载、校验或任务状态。上传 final 不代表理论收敛；快照未上传 final 也不代表训练一定未完成。

## 速度与一致性

| 方法 | 端到端 speedup | 95% CI（session bootstrap） | AR 秒 | SD 秒 | 输出 token 完全一致 | 样本 | AL | 状态 |
|---|---:|---|---:|---:|---|---|---|---|
| dflash | 2.127× | [1.989, 2.314] | 5062.91 | 2379.89 | 59/435 | 120 session / 145 turn × 3 | 未记录 | 已完成、独立复算通过 |
| dfly | 1.606× | [1.519, 1.710] | 4865.21 | 3029.07 | 63/435 | 120 session / 145 turn × 3 | 未记录 | 已完成、独立复算通过 |
| dflash2 | 2.280× | [2.153, 2.426] | 4940.66 | 2167.07 | 58/435 | 120 session / 145 turn × 3 | 未记录 | 已完成、独立复算通过 |
| dflare | — | — | — | — | — | 计划 120 session / 145 turn × 3 | — | 队列运行中；旧失败记录保留 |
| domino | — | — | — | — | — | 计划 120 session / 145 turn × 3 | — | 队列运行中；旧失败记录保留 |
| jetspec | — | — | — | — | — | 计划 120 session / 145 turn × 3 | — | 队列运行中；旧失败记录保留 |

旧 DFly 速度结果因采样发现其他进程使用同卡而判无效，未列入表。† AL 来自单独的 145 turn、启用 vLLM 官方计数器的补充运行；按 1 + accepted_draft_tokens / num_drafts 计算，包含惯例 bonus token，并非主测速运行的计数。未记录表示实际计数尚不可用，不使用训练日志替代。

## 质量：真实标注，完整输出评分

| 方法 | 数据集 | AR 分数 | SD 分数 | SD−AR（百分点） | 质量 token 一致 | 样本 | 状态 |
|---|---|---:|---:|---:|---|---:|---|
| dflash | textvqa | 80.742 | 80.742 | +0.000 | 256/256 | 256 | 完成 |
| dflash | docvqa | 95.264 | 95.264 | +0.000 | 254/256 | 256 | 完成 |
| dflash | chartqa | 85.547 | 85.547 | +0.000 | 256/256 | 256 | 完成 |
| dfly | textvqa | 80.742 | 80.078 | -0.664 | 254/256 | 256 | 完成 |
| dfly | docvqa | 95.264 | 95.264 | +0.000 | 256/256 | 256 | 完成 |
| dfly | chartqa | 85.547 | 85.547 | +0.000 | 256/256 | 256 | 完成 |
| dflash2 | textvqa | 80.742 | 80.742 | +0.000 | 256/256 | 256 | 完成 |
| dflash2 | docvqa | 95.264 | 95.264 | +0.000 | 256/256 | 256 | 完成 |
| dflash2 | chartqa | 85.547 | 85.547 | +0.000 | 255/256 | 256 | 完成 |
| dflare | textvqa | — | — | — | — | 256（计划） | 等待 |
| dflare | docvqa | — | — | — | — | 256（计划） | 等待 |
| dflare | chartqa | — | — | — | — | 256（计划） | 等待 |
| domino | textvqa | — | — | — | — | 256（计划） | 等待 |
| domino | docvqa | — | — | — | — | 256（计划） | 等待 |
| domino | chartqa | — | — | — | — | 256（计划） | 等待 |
| jetspec | textvqa | 80.742 | 80.742 | +0.000 | 256/256 | 256 | 完成 |
| jetspec | docvqa | 95.264 | 95.264 | +0.000 | 254/256 | 256 | 完成 |
| jetspec | chartqa | 85.547 | 85.547 | +0.000 | 256/256 | 256 | 完成 |

DFlash/DFly 的质量配对在 GPU3 完成；DFlash2 的质量 AR 与 SD 在同一 GPU 上独立配对。分数相同不能证明严格逐 token 无损或统计等价；仅声称标准 target verification 的理论性质，并单独报告实测差异。

## 口径与限制

A6000、Qwen3-VL-8B BF16、vLLM 0.22.1、linear verification、S7/B8、batch 1；恢复运行使用已验证 SHA-256 和 inode/大小/mtime/ctime 复用机制，文件变动会重新哈希；复用依据见 `TARGET_SHA_CACHE_SEED_V1.json` 和 `DRAFT_SHA_CACHE_SEED_V1.json`。速度为 Σ AR 请求时间 / Σ SD 请求时间，排除启动和完整预热、包含图像读入/预处理、prefill、decode 和输出。95% CI 对 120 个 session 做 2000 次 bootstrap，不覆盖多卡、多机器或独立运行的变异。共享主机 CPU/I/O 可能影响速度；质量各 256 题非完整 benchmark，训练与测评重叠未审计。

原始证据：`dflash_formal_audit_v1.json`、`dfly_8b_formal_dynamic_audit_v11.json`、`dflash2_8b_formal_dynamic_audit_v14.json`、`quality_campaign_v1/QUALITY_PAIRED_V1.json`、`quality_dflash2_dynamic_v17/QUALITY_PAIRED_V1.json`、`dflare_8b_formal_dynamic_audit_v11.json`、`quality_dflare_campaign_v1/QUALITY_PAIRED_V1.json`，均位于 `/home/lyh/qwen3vl8b_eval_20261027/`；缺失文件表示尚未完成。
