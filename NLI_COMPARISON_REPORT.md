# 新旧 NLI 实验简报

日期：2026-09-21。结论：新 NLI 明显改善，但矛盾误报仍多；本次同时改变模型和输入长度，不能将收益单独归因于某一项。

## 1. 模型与任务

| 项目 | 旧 NLI | 新 NLI |
| --- | --- | --- |
| 模型名称 | `cross-encoder/nli-deberta-v3-small` | `tasksource/ModernBERT-large-nli` |
| 云端 Job | 39082 | 39156 |
| 输入长度上限 | 512 tokens | 2048 tokens |

输入：拼接的证据 document ＋一句 claim。输出：entailment / contradiction / neutral 三类概率，按阈值生成 supported / conflict / unsupported。当前蕴含、矛盾阈值均为0.5，未达阈值回退 unsupported。

评价对象为同一份 **50条 train 回答拆出的414条句级 pair**；金标签分布为 supported 352、conflict 2、unsupported 60。它们不是414个独立问题，也不是人工裁决的原子事实。

## 2. 已跑出的 pair 结果

| 方法 | Accuracy | 三分类 Macro-F1 | 非支持二分类 F1 | 拒判数 |
| --- | ---: | ---: | ---: | ---: |
| 旧 NLI | 0.2899 | 0.2015 | 0.3024 | 0 |
| 新 NLI | 0.7150 | 0.4651 | 0.5221 | 0 |
| MiniCheck | 0.7585 | 0.4507 | 0.5243 | 0 |
| MiniCheck＋旧 NLI | 0.2585 | 0.2721 | 0.5427 | 217 |
| MiniCheck＋新 NLI | 0.6329 | 0.4844 | 0.6163 | 84 |

全量评价保留拒判，拒判计入真实类别 FN；二分类以 conflict 或 unsupported 为正类。MiniCheck 本身不能区分这两类，三分类结果仅为结构受限的参考。F1 必须结合覆盖率看，不能用融合的高 F1 掩盖拒判。

新 NLI 的 conflict：Precision **2.33%**、Recall **50%**、F1 **0.0444**。预测43条 conflict，仅1条正确，另有33条 supported、9条 unsupported 被误判。真实 conflict 仅2条，结论不稳定。

## 3. 三条真实输入输出样例

以下证据仅展示原文相关摘录，实际模型输入为完整拼接 document，再按长度上限截断；不是只输入这些摘录。概率顺序为蕴含／矛盾／中立。

| pair_id | 证据摘录 | 输入 claim | 金标签 | 旧输出 → 新输出 | 新模型概率 |
| --- | --- | --- | --- | --- | --- |
| `11957_c03` | “Cut the potatoes into French fries (see photos in post).” | Cut the potatoes into French fries | supported | conflict → supported | 0.9128 / 0.0138 / 0.0734 |
| `11876_c05` | “Double cream has a higher fat content than single.”；“Because of the higher fat content it will bind with the flour ion the pasta to produce a thicker sauce.” | Single cream also has a higher binding ability due to its higher fat content | conflict | unsupported → conflict | 0.0282 / 0.9380 / 0.0337 |
| `15268_c01` | “The body stops this blood loss through a complex clotting process called hemostasis.” | Sure | supported | unsupported → conflict | 0.0313 / 0.5851 / 0.3835 |

解读：前两例展示改善；第三例暴露切分问题——“Sure”是话语回应，不是可验证事实，却进入了事实判断。其 supported 金标签来自“无幻觉 span 重叠”的投影，不能简单把这一错误全部归咎于模型。

## 4. 新 NLI 的 response 级结果

以下以“回答含幻觉”为正类，和上面的 pair 三分类不是同一评价单位。

| 数据 | Precision | Recall | F1 | Accuracy | span micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train500 | 0.3679 | 0.9595 | 0.5318 | 0.5000 | 0.3197 |
| test200 | 0.2105 | 0.9412 | 0.3441 | 0.3900 | 0.2247 |

test200 检出32/34条含幻觉回答，但误报120条正常回答，仍明显偏向高召回、低精度。单条与50条 response 也已跑通；500/200尚不能称为对应规模的 pair 三分类结果。

## 5. 状态与下一步

- 运行、414-pair 对齐、融合和指标复算已通过。统一复核阈值0.7后，MiniCheck／新NLI／融合需复核127／120／205条；复核不等于拒判，不修改预测。
- 下一步：在开发集补跑 ModernBERT 512 对照；检查输入截断与 conflict 误报；处理非事实句。不要根据 test200 反复调参。
- 新模型内容指纹：`local-sha256:b169618dac2341b4143ad9bea64708518df5889b77e83474141521cd8e760bd1`；旧模型 revision：`fa2804872c3b4bd748f38c0185cc85775361e735`。

来源：[旧NLI逐条结果](outputs/nli_results_50-39082.jsonl)、[旧方法评价](outputs/review_v1/metrics_summary.csv)、[新NLI逐条结果](outputs/modernbert-all-2048-39156/nli_pairs_50.jsonl)、[统一pair评价](outputs/modernbert_unified_v1/classification_report.txt)、[train500指标](outputs/modernbert-all-2048-39156/response_train500_metrics.json)、[test200指标](outputs/modernbert-all-2048-39156/response_test200_metrics.json)。评价规则见[统一协议](docs/pair_evaluation_protocol.md)。
