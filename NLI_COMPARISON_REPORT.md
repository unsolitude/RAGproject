# 新旧 NLI 实验简报

日期：2026-09-21（补充 Job39205）。结论：**ModernBERT 在相同512长度预算下已明显优于旧 NLI；2048在当前开发子集未带来收益，略低于512。** 矛盾误报仍多，不能据此认定更长上下文普遍无效，也不能证明旧模型的主要问题就是截断。

## 1. 模型与任务

| 项目 | 旧 NLI | 新 NLI |
| --- | --- | --- |
| 模型名称 | `cross-encoder/nli-deberta-v3-small` | `tasksource/ModernBERT-large-nli` |
| 云端 Job | 39082 | 39205（512）；39156（2048） |
| 输入长度上限 | 512 tokens | 512 / 2048 tokens |

输入：拼接的证据 document ＋一句 claim。输出：entailment / contradiction / neutral 三类概率，按阈值生成 supported / conflict / unsupported。当前蕴含、矛盾阈值均为0.5，未达阈值回退 unsupported。

评价对象为同一份 **50条 train 回答拆出的414条句级 pair**；金标签分布为 supported 352、conflict 2、unsupported 60。它们不是414个独立问题，也不是人工裁决的原子事实。

## 2. 已跑出的 pair 结果

| 方法 | Accuracy | 三分类 Macro-F1 | 非支持二分类 F1 | 拒判数 |
| --- | ---: | ---: | ---: | ---: |
| 旧 NLI 512 | 0.2899 | 0.2015 | 0.3024 | 0 |
| ModernBERT 512 | **0.7198** | **0.4684** | **0.5268** | 0 |
| ModernBERT 2048 | 0.7150 | 0.4651 | 0.5221 | 0 |
| MiniCheck | 0.7585 | 0.4507 | 0.5243 | 0 |
| MiniCheck＋旧 NLI | 0.2585 | 0.2721 | 0.5427 | 217 |
| MiniCheck＋ModernBERT 2048 | 0.6329 | 0.4844 | 0.6163 | 84 |

本次没有运行 MiniCheck＋ModernBERT 512 融合，不补造该组结果。

### 两模型、三种配置的逐类比较

P/R/F1分别为精确率、召回率、F1；均在同一414条pair上计算。

| 配置 | supported P/R/F1 | conflict P/R/F1 | unsupported P/R/F1 |
| --- | --- | --- | --- |
| 旧 NLI 512 | 1.0000 / 0.1875 / 0.3158 | 0 / 0 / 0 | 0.1720 / 0.9000 / 0.2888 |
| ModernBERT 512 | 0.9881 / 0.7074 / 0.8245 | 0.0233 / 0.5000 / 0.0444 | 0.4034 / 0.8000 / 0.5363 |
| ModernBERT 2048 | 0.9880 / 0.7017 / 0.8206 | 0.0233 / 0.5000 / 0.0444 | 0.3967 / 0.8000 / 0.5304 |

相对旧模型，ModernBERT 512的Accuracy提高43.00个百分点，Macro-F1提高0.2669。同模型512→2048，412/414条预测相同；`14385_c05`、`17435_c02` 两条gold supported从supported变为unsupported，Accuracy下降0.48个百分点。不同模型tokenizer不同，相同token预算不代表读取了完全相同的文本。

全量评价保留拒判，拒判计入真实类别 FN；二分类以 conflict 或 unsupported 为正类。MiniCheck 本身不能区分这两类，三分类结果仅为结构受限的参考。F1 必须结合覆盖率看，不能用融合的高 F1 掩盖拒判。

新 NLI 的两个长度配置均预测43条 conflict，仅1条正确，另有33条 supported、9条 unsupported 被误判。真实 conflict 仅2条，结论不稳定。

## 3. 三条真实输入输出样例

以下证据仅展示原文相关摘录，实际模型输入为完整拼接 document，再按长度上限截断；不是只输入这些摘录。三例的新模型512/2048预测标签一致；表中概率保留2048结果，顺序为蕴含／矛盾／中立。

| pair_id | 证据摘录 | 输入 claim | 金标签 | 旧输出 → 新输出 | 新模型概率 |
| --- | --- | --- | --- | --- | --- |
| `11957_c03` | “Cut the potatoes into French fries (see photos in post).” | Cut the potatoes into French fries | supported | conflict → supported | 0.9128 / 0.0138 / 0.0734 |
| `11876_c05` | “Double cream has a higher fat content than single.”；“Because of the higher fat content it will bind with the flour ion the pasta to produce a thicker sauce.” | Single cream also has a higher binding ability due to its higher fat content | conflict | unsupported → conflict | 0.0282 / 0.9380 / 0.0337 |
| `15268_c01` | “The body stops this blood loss through a complex clotting process called hemostasis.” | Sure | supported | unsupported → conflict | 0.0313 / 0.5851 / 0.3835 |

解读：前两例展示改善；第三例暴露切分问题——“Sure”是话语回应，不是可验证事实，却进入了事实判断。其 supported 金标签来自“无幻觉 span 重叠”的投影，不能简单把这一错误全部归咎于模型。

## 4. 新 NLI 的 response 级结果

以下以“回答含幻觉”为正类，和上面的 pair 三分类不是同一评价单位。

| 数据 / ModernBERT配置 | Precision | Recall | F1 | Accuracy | span micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| response50 / 512 | 0.4000 | 1.0000 | 0.5714 | 0.5200 | 0.4739 |
| response50 / 2048 | 0.4000 | 1.0000 | 0.5714 | 0.5200 | 0.4739 |
| train500 / 512 | 0.3679 | 0.9595 | 0.5318 | 0.5000 | 0.3197 |
| train500 / 2048 | 0.3679 | 0.9595 | 0.5318 | 0.5000 | 0.3197 |
| test200 / 2048（历史） | 0.2105 | 0.9412 | 0.3441 | 0.3900 | 0.2247 |

512没有重跑test200。response模式逐段判断，pair模式判断拼接document；回答级汇总指标一致不代表所有概率相同，也不能替代pair层面的长度对照。

test200 检出32/34条含幻觉回答，但误报120条正常回答，仍明显偏向高召回、低精度。单条与50条 response 也已跑通；500/200尚不能称为对应规模的 pair 三分类结果。

## 5. 状态与下一步

补充真实token审计：414条pair完整输入长度200–542（中位数330）；仅14条（3.38%）超过512，0条超过2048。512的116条错误中113条发生在未截断输入上，故截断不是当前ModernBERT错误的主要解释。旧DeBERTa tokenizer尚未审计，不能外推。详见[截断审计](docs/nli_truncation_audit.md)。

- 运行、414-pair 对齐、融合和指标复算已通过。统一复核阈值0.7后，MiniCheck／新NLI／融合需复核127／120／205条；复核不等于拒判，不修改预测。
- ModernBERT 512已完整完成：manifest为complete，9条命令全部返回0，单条/50条/train500输出数及指标已复算，414-pair验证通过。与2048的模型指纹及四份开发输入哈希一致，batch4、复核阈值0.7一致。pair报告仅有约1e-16级浮点尾数差异，不影响指标。
- ModernBERT 512需复核121条、2048需复核120条；两者均无拒判。旧NLI原始复核阈值不同，不直接拿复核数量对比。
- 下一步：优先做token截断审计、conflict误报和非事实句排查；旧模型全证据分块B组仍未完成。当前512可作为后续开发参考配置，不宣称普遍优于2048。不要根据test200反复调参。
- 新模型内容指纹：`local-sha256:b169618dac2341b4143ad9bea64708518df5889b77e83474141521cd8e760bd1`；旧模型 revision：`fa2804872c3b4bd748f38c0185cc85775361e735`。

来源：[旧NLI逐条结果](outputs/nli_results_50-39082.jsonl)、[旧方法评价](outputs/review_v1/metrics_summary.csv)、[新NLI逐条结果](outputs/modernbert-all-2048-39156/nli_pairs_50.jsonl)、[统一pair评价](outputs/modernbert_unified_v1/classification_report.txt)、[train500指标](outputs/modernbert-all-2048-39156/response_train500_metrics.json)、[test200指标](outputs/modernbert-all-2048-39156/response_test200_metrics.json)。评价规则见[统一协议](docs/pair_evaluation_protocol.md)。

新增来源：[512运行清单](outputs/modernbert-dev-512-39205/suite_manifest.json)、[512 pair评价](outputs/modernbert-dev-512-39205/pair_metrics/classification_report.txt)、[512逐条结果](outputs/modernbert-dev-512-39205/nli_pairs_50.jsonl)、[512 train500指标](outputs/modernbert-dev-512-39205/response_train500_metrics.json)。
