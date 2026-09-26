# 项目目录与归档说明

本说明区分**当前研究输入**、**可复现历史结果**和**已归档中间稿**。最后更新于 2026-09-26：已归档 5 条证据选择冒烟运行及四份过时 Markdown 快照；没有删除 RAGTruth 原始数据、模型权重、pair、预测或日志。逐项映射和恢复说明见[归档索引](archive/README.md)。

## `data/ragtruth/`

| 位置 | 用途 | 处理原则 |
| --- | --- | --- |
| `raw/` | 官方 `source_info.jsonl`、`response.jsonl` | 本地保留；被 `.gitignore` 忽略；从官方数据重建时仍必需 |
| `processed/qa_one.jsonl`、`doc_claim_pairs.jsonl` | 单样本/3-pair 冒烟和 README 教程 | 保留；脚本仍引用 |
| `processed/sample_50.jsonl`、`doc_claim_pairs_50.jsonl` | 50-response / 414-pair 历史开发集与模型输入 | 保留；v7 审计和旧预测严格对齐所必需 |
| `processed/qa_train_500_seed2026.jsonl`、`qa_test_200_seed42.jsonl` | 固定抽样的 train/test 回答 | 保留；对照实验和审计扩样来源 |
| `eval_dev50_assistant_v1/` | 当前开发主线：234 条助手审定 pair、414 条逐项裁决及 manifest | 冻结使用；无需用户再审计 dev50，但不称独立人工金标 |
| `eval_v1/` | 当前**暂定**审计版：`dev50`、`test200`、`conflict_challenge` | 保留全部 original pairs、decisions、pairs、manifest；不能只留 `*_pairs.jsonl` |
| `eval_v1_signed/` | 研究者逐条确认后才会生成 | 当前不存在；不可把 `eval_v1/` 改名冒充签核版 |

`eval_v1/` 看似有重复数据，是为了分别保留**原始 pair**、**逐条裁决**、**暂定评价输入**和校验哈希。删除其中任一类会破坏旧结果对齐、审核或最终导出。签核步骤见 [EVAL_V1_GUIDE.md](EVAL_V1_GUIDE.md)。

## `outputs/`

| 位置 | 内容与状态 |
| --- | --- |
| 根目录的 MiniCheck/NLI 结果、manifest、`.log` | 历史 GPU 运行原件；报告仍引用，保留原路径 |
| `modernbert-all-2048-39156/`、`modernbert-dev-512-39205/` | 新 NLI 的 2048/512 对照与运行摘要；保留 |
| `modernbert_unified_v1/` | 当前统一复核与融合评价；保留 |
| `review_v1/` | 旧复核规则修正结果；项目情况文档与 NLI 比较报告仍引用，保留 |
| `lesson3/audit/` | 当前 `claims_audit_v7.jsonl`、manifest、审核报告；保留 |
| `lesson3/eval_v1_dev/` | 审计口径下已重算的开发集指标；保留，但仍属暂定结果 |
| `lesson3/eval_dev50_assistant_v1/` | 234 条冻结开发 pair 的 MiniCheck/NLI/融合重算指标；当前开发主线 |
| `evidence_runs/dev50-39484/` | 最新完整四组证据选择实验：234 条 pair、实际输入、选择记录、预测、指标和配对变化 |
| `lesson3/order_runs/dev50-39484-prepared/` | 七组固定证据集合的顺序对照输入，234 条/组；仅 prepared，尚无新增推理结果 |
| `lesson3/cases/dev50-39484/` | 五个已核对错例的原始全文、候选块、概率、结果行号及来源哈希 |
| `archived/smoke_runs/dev50-39483/` | 已通过的 5 条四组冒烟实验；不参与完整开发集指标 |
| `archived/` | 较早的 response 实验、旧审计稿和临时检查；可恢复，不参与当前主线 |
| `logs/` | 环境记录；保留 |

`outputs/` 整体被 `.gitignore` 忽略。上传代码到 GitHub 不会自动携带审计 v7、GPU 预测和日志；如需在云端复现，应单独安全同步这些文件，或重跑生成。

## 代码与说明

- `src/`：数据准备、切分、Judge、融合及统一评价程序；保留。
- `scripts/`：SLURM 和可复现检查/数据集构建入口；保留 `build_audited_eval.py`（候选三集）与 `freeze_dev50_assistant.py`（当前冻结开发集）。
- `tests/`：回归测试；保留。
- `models/`：本地权重与缓存，已忽略；不在此次整理范围。
- 根目录的 `README.md`、`CLAIM_AUDIT_GUIDE.md`、`EVAL_V1_GUIDE.md`、`NLI_COMPARISON_REPORT.md`：当前操作与结果说明；保留。
- `archive/docs/2026-09-26/`：第二节任务清单、9 月 19 日状态简报及两篇旧学习快照；内容保持原样，文中的相对路径按原位置理解。
- `docs/learning/07_metrics_and_current_results.md` 与 `09_progress_and_research_roadmap.md`：当前简短入口，链接到旧教材与最新实验，避免旧阶段结论冒充当前进度。

仍需保留 `qa_one.jsonl`、`doc_claim_pairs.jsonl`、`scripts/slurm/test_minicheck.slurm` 及 `tests/`：它们是可复用的教程输入和测试入口，不是已完成的冒烟输出。两个 ModernBERT 历史 suite 内的单样本结果与整次运行清单有关，也保留原位置，不拆散成不完整的运行记录。

此次归档的旧审计稿位于 `outputs/archived/lesson3_audit_pre_v7/`，临时冒烟报告位于 `outputs/archived/temporary_checks/eval_v1_dev_smoke/`。详细恢复说明见 `outputs/archived/README.md`。
