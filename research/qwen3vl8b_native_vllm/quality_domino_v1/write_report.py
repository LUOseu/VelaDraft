"""Render only completed paired gold evidence; no inferred equivalence."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);data=json.loads((root/'QUALITY_PAIRED_V1.json').read_text())
if data.get('status')!='complete' or data.get('physical_calls')!=2340:raise ValueError('Full paired evidence required')
lines=['# Qwen3-VL-8B AR 与 DFlash / DFly 真实标注质量比较','',
 '每个系统：TextVQA、DocVQA、ChartQA 各256题，单次 greedy 执行；每任务另有4次预热。三系统共2340个物理请求。主表对完整解码字符串计分；旧首行提取规则作为辅助口径。','',
 '| 方法 | 数据集 | AR 分数 | SD 分数 | SD−AR（百分点） | 配对95% CI | token 完全一致 | 改变分数的题数 | 达长度上限 AR/SD |',
 '|---|---|---:|---:|---:|---|---:|---:|---|']
for method in ('dflash','dfly'):
 for task in ('textvqa','docvqa','chartqa'):
  item=data['methods'][method]['tasks'][task];p=item['full'];ci=p['paired_cluster_percentile_CI95']
  cap_ar=item['ar_finish_reasons'].get('length',0);cap_sd=item['sd_finish_reasons'].get('length',0)
  lines.append(f"| {method} | {task} | {p['ar_score']:.4f} | {p['sd_score']:.4f} | {p['delta_percentage_points']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {item['exact_token_outputs']}/256 | {item['changed_full_scores']} | {cap_ar}/{cap_sd} |")
lines+=['','辅助口径：先按固定旧规则提取首个非空行并去除 Answer 标签/包装后，再使用相同真实 gold 和评分函数；这项口径在本次质量运行前确定。','',
 '| 方法 | 数据集 | AR 分数 | SD 分数 | SD−AR（百分点） | 配对95% CI |','|---|---|---:|---:|---:|---|']
for method in ('dflash','dfly'):
 for task in ('textvqa','docvqa','chartqa'):
  p=data['methods'][method]['tasks'][task]['extracted'];ci=p['paired_cluster_percentile_CI95']
  lines.append(f"| {method} | {task} | {p['ar_score']:.4f} | {p['sd_score']:.4f} | {p['delta_percentage_points']:+.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
lines+=['','配对 CI 按图像/页面组做2000次 bootstrap，seed42；它不衡量独立运行/显卡/训练种子的方差，也未对多个方法和数据集比较作多重检验校正。区间包含0不证明等价或非劣。256题子集不是完整官方测试集，亦不直接评估 MMSpec 长回答质量。',
 '得分比较是经验质量证据；理论无损需要另行依据标准验证算法和实际实现审查，分数相同不能证明逐 token 一致，也不能证明数学定理。8B草稿训练集与本评测集的重叠尚未审计。各臂物理GPU、prompt IDs、原图SHA、gold与评分复算通过；资源监控是抽样，不代表整机隔离。',
 '','详细逐题变化及原始文件哈希见 [QUALITY_PAIRED_V1.json](QUALITY_PAIRED_V1.json)；每臂 `predictions.jsonl` 包含真实 gold、完整输出、token、prompt、图像SHA和得分。']
with (root/'QUALITY_REPORT_V1.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('Wrote quality report')
