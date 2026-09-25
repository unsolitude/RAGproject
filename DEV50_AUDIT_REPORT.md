# dev50 助手审定开发集（v1）

## 结论与使用范围

已完成 `dev50` 的开发用途裁决并冻结评价 ID。50 条 RAGTruth QA train 回答产生的 414 条原始句级 pair，纳入 234 条可独立核验的三分类 claim；180 条排除均有逐项决定与理由。**本开发集没有待用户人工审计项**，可以作为 Full / Random / BM25 / BGE 同 ID、同 gold 的对照基础。

裁决来源是此前 `claims_audit_v7.jsonl` 的逐条来源审查、`eval_v1/dev50` 候选决定，以及本轮对指代、原文损坏和命题确定性的复查。状态为 `assistant_adjudicated_dev_only`，不是独立研究者签核的人工金标，也不是论文主测试集。没有修改 RAGTruth 原始文件、旧 pair、历史预测或候选版。

| 项目 | 数量 |
| --- | ---: |
| 原始 pair / 审定决定 | 414 / 414 |
| 纳入 | 234 |
| 纳入标签：supported / conflict / unsupported | 191 / 2 / 41 |
| 非 claim | 79 |
| 原 `needs_adjudication`，本版排除归档 | 55 |
| 来源范围依赖 | 30 |
| 上下文指代片段 | 14 |
| 原文损坏 / 非确定性命题 | 1 / 1 |

相比先前暂定的 238 条，又排除了：`12639_c21`（`withdrausted` 损坏）、`13275_c03`（`it` 可指 HDMI 端口或适配器）、`16477_c01`（未明确任何具体群体与经历）、`16760_c06`（`This`、`they` 依赖上文）。不在旧 `pair_id` 下改写 claim；55 条多事实或缺上下文样本保留拆分建议，今后如重切需另建 atomic 版本。

纳入项中有 7 条相对原 span 投影标签发生审定改标（具体见 `dev50_decisions.jsonl`）：`13030_c06` 为 supported→conflict；`13030_c09`、`15015_c08`、`15268_c13`、`17621_c07`、`17696_c07` 为 supported→unsupported；`17621_c10` 为 unsupported→supported。金标以冻结文件为准，不从旧预测文件读回。

## 文件与复现

- [`dev50_original_pairs.jsonl`](data/ragtruth/eval_dev50_assistant_v1/dev50_original_pairs.jsonl)：414 条未改动原始 pair，供历史结果严格对齐。
- [`dev50_decisions.jsonl`](data/ragtruth/eval_dev50_assistant_v1/dev50_decisions.jsonl)：414 条纳入、排除、原因、证据与来源信息；`assistant_reviewed=true`、`researcher_verified=false`。
- [`dev50_pairs.jsonl`](data/ragtruth/eval_dev50_assistant_v1/dev50_pairs.jsonl)：234 条冻结评价 pair。
- [`dev50_manifest.json`](data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json)：ID、类别数、排除数、SHA-256 与来源哈希。
- [`freeze_dev50_assistant.py`](scripts/freeze_dev50_assistant.py)：可重建脚本；拒绝覆盖已有版本。

已有预测按原始 414 条严格核验后，在内存中筛出冻结 ID、应用审定 gold 并重算。结果位于 `outputs/lesson3/eval_dev50_assistant_v1/metrics_summary.json`，未改动旧预测。与先前暂定版比较：

| 方法 | 暂定 238 条 Accuracy / Macro-F1 | 冻结 234 条 Accuracy / Macro-F1 | 冻结版拒判 |
| --- | ---: | ---: | ---: |
| MiniCheck | 0.9286 / 0.5991 | 0.9316 / 0.6006 | 0 |
| ModernBERT NLI | 0.8866 / 0.6301 | 0.8932 / 0.6335 | 0 |
| 融合 | 0.8319 / 0.6865 | 0.8376 / 0.6905 | 33 |

这些是**开发集回算**，不能解释为新模型训练后的提升。conflict 仅 2 条，不应凭其 F1 判断该类性能。融合存在拒判，完整集 Accuracy 将拒判计为错误；不要只报告“已裁决子集”的条件分数。

## 后续实验的硬约束

使用同一份 `dev50_pairs.jsonl` 冻结 claim、`pair_id` 和 gold。Full、Random、BM25、BGE 仅改变交给 Judge 的证据，并分别保存实际 evidence、预测、配置、随机种子与运行 manifest。`test200` 和冲突挑战集仍是未审定候选集，不能沿用这里的“已审定”状态；若将来用于正式论文评价，必须另建独立审计版。`--require-final` 的研究者签核门槛保留，明确拒绝这份助手审定开发集。
