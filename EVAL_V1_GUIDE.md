# RAGTruth 审计版评价集：开发集与候选测试集

## 当前开发主线：`dev50` 助手审定版

为赶上开发实验，`data/ragtruth/eval_dev50_assistant_v1/` 已冻结：414 条原始 pair 中纳入 234 条（supported 191 / conflict 2 / unsupported 41），其余 180 条都有明确排除决定。它继承 v7 的逐条来源审查，另排除 4 条可核验性问题：`12639_c21` 原文损坏、`13275_c03` 条件指代不明、`16477_c01` 命题不确定、`16760_c06` 缺少独立指代。55 条原 `needs_adjudication` 仍逐条归档排除，**没有被硬填三分类**。

这份数据的状态是 `assistant_adjudicated_dev_only`：`dev50` 的开发实验无需用户继续人工审计或签核，可直接用于固定 ID 的 Full/Random/BM25/BGE 对照及阈值开发。它**不是独立人工金标**，也不能代替未审计的 `test200` 或冲突挑战集。两条 conflict 不足以得出稳定的该类效果结论。

在项目根目录重算已有 414 条历史预测时，运行：

```powershell
python -B src/evaluate_results.py --data-format pair `
  --input-pairs data/ragtruth/eval_dev50_assistant_v1/dev50_original_pairs.jsonl `
  --audit-manifest data/ragtruth/eval_dev50_assistant_v1/dev50_manifest.json `
  --input outputs/modernbert_unified_v1/minicheck.jsonl outputs/modernbert_unified_v1/nli.jsonl outputs/modernbert_unified_v1/merged.jsonl `
  --output-dir outputs/lesson3/eval_dev50_assistant_v1
```

先严格验证整份旧预测与原始 414 条 pair，再筛选冻结 ID 并在内存中应用审定 gold；不会改写历史预测。结果已保存于 `outputs/lesson3/eval_dev50_assistant_v1/`。新证据选择实验须固定 `dev50_pairs.jsonl` 的 ID、claim 和 gold，只改变传给 Judge 的 evidence；每种 evidence 策略的实际输入与预测须各留一个版本化文件。冻结集可由 `python -B scripts/freeze_dev50_assistant.py` 从候选集重建，但脚本拒绝覆盖已存在目录。

## 当前状态（请先读）

`data/ragtruth/eval_v1/` 是此前生成的**可运行候选集**，不是当前冻结的开发主线，更不是最终人工金标。测试集和冲突挑战集目前只有 RAGTruth span 投影标签及机械筛查建议，尚未逐条做来源级裁决。所有记录的 `researcher_verified` 都是 `false`。如需在论文中称为独立人工审计金标，仍须真实的研究者核对、签核和重新导出；这个门槛不适用于本轮 `dev50` 助手审定版的开发使用。

| 子集 | 原始 pair | 暂定纳入 | 暂定标签分布（supported / conflict / unsupported） |
| --- | ---: | ---: | ---: |
| `dev50`：50 条 QA train 回答 | 414 | 238 | 193 / 2 / 43 |
| `test200`：seed 42 的随机 200 条 QA test 回答 | 1370 | 1031 | 959 / 8 / 64 |
| `conflict_challenge`：全部 26 条带 Conflict span 的合格 QA test 回答 | 213 | 134 | 106 / 22 / 6 |

挑战集与随机测试集有 7 条 response 重叠。**分别报告指标，不能把两个子集的结果相加或当作独立样本合并。**数量在研究者复核后可能改变，论文版以签核后的 manifest 为准。

## 目录与裁决口径

每个子目录都有：

- `*_original_pairs.jsonl`：原句级 pair，用于校验历史预测；不能改动。
- `*_decisions.jsonl`：逐条处置与签核工作文件。`action=include` 才进入当前暂定评价集；`exclude_*` 为排除项。保留原标签、建议标签、理由及部分建议拆分。
- `*_pairs.jsonl`：由决定文件物化出的当前暂定评价集；保留原 `pair_id`、claim、document，但 `gold_label` 和 `gold_projection_version` 采用审计口径。
- `*_manifest.json`：哈希、ID 清单、类别支持数与排除原因统计。

开发集的 55 条 `needs_adjudication` 已逐条记为 `exclude_unresolved_v1`，不硬填三分类；79 条非事实句排除。另将 30 条实质依赖指定/所选 passage 的 claim 作为 `exclude_source_relative` 单列，12 条明显缺少独立指代的片段作为 `exclude_context_fragment`。这些集合互不重叠；其余 238 条是暂定开发集。`Based on the given passages, X` 若 X 本身为独立事实，没有因套话而自动排除。

测试集和挑战集中的机械筛查**只是优先审查提示**：`screening_flags` 可提示切分、span 或指代问题；所有暂定 `include` 的 `proposed_gold_label` 仍是自动投影，不代表来源支持性已经确认。请同时检查自动排除项，防止漏收。

## 将候选集升级为独立人工金标时如何签核（可选；本轮 dev50 不需要）

1. 在项目根目录打开三份 `*_decisions.jsonl` 和对应的 `*_original_pairs.jsonl`，按 `pair_id` 对照完整 `query`、`response`、`claim`、`document`、原始 span。先判断是否独立可核验，再判断来源是否完全支持、明确冲突或仅缺证据。
2. 对每行确认或修正 `action`、`proposed_gold_label`、`reason`、`evidence_ids`、`evidence_quote`。若有多事实、缺上下文或来源范围依赖，不要为了保留数量强行给单一三分类；使用对应 `exclude_*`，并把建议拆分保留在 `suggested_claims`。修改后的排除及重新纳入都要写理由。
3. **每条**记录（包括排除项）确认后，将 `researcher_verified` 设为 `true`，填写真实的 `researcher_name` 和 `verified_at`（`YYYY-MM-DD`）。不能批量把未经阅读的行改为已确认。测试与挑战重复的 `pair_id` 应有相同决定和签核信息。
4. 保持 JSONL 一行一对象、UTF-8 编码。签核前另存备份；不要修改历史模型结果或原始 pair 文件。

签核完成后，在项目根目录运行：

```powershell
python -B scripts/build_audited_eval.py --finalize
```

命令会核对原始文件哈希、每条决定及签核、测试/挑战重叠项，再**新建** `data/ragtruth/eval_v1_signed/`。缺任一签核或目标目录已存在就报错，不覆盖旧版本。签核后仍要人工检查最终分布，尤其 conflict 例数很少的开发集。

## 历史候选版结果如何重算（非当前开发主线）

历史预测文件不需修改。以下命令先按原始 414 条 pair 严格核对，再仅用审计版开发集的 238 条暂定样本重算：

```powershell
python -B src/evaluate_results.py --data-format pair `
  --input-pairs data/ragtruth/eval_v1/dev50/dev50_original_pairs.jsonl `
  --audit-manifest data/ragtruth/eval_v1/dev50/dev50_manifest.json `
  --input outputs/modernbert_unified_v1/minicheck.jsonl outputs/modernbert_unified_v1/nli.jsonl outputs/modernbert_unified_v1/merged.jsonl `
  --output-dir outputs/lesson3/eval_v1_dev
```

结果中的 `audit_status=provisional_codex_assisted` 以及 policy 均标明其**不是最终人工金标指标**。待签核版导出后，将 `--input-pairs` 与 `--audit-manifest` 改为 `eval_v1_signed/dev50` 中对应文件，并增加 `--require-final`。如果预测文件的 claim、document、原 gold 或切分版本与原始输入不一致，评价器会拒绝运行。

测试集和挑战集目前**没有** MiniCheck/NLI 全量预测；需要在云端对各自 `*_pairs.jsonl` 分别运行 Judge，不能把开发集预测复制过去。新预测直接用相同版本的 `*_pairs.jsonl` 作为评价输入。进行 Full/Random/BM25/BGE 对照时，各方法须固定同一批 ID 和 gold，只改变 Judge 接收的证据；不要在 test 或 challenge 上调阈值。

## 复现与限制

生成脚本为 `scripts/build_audited_eval.py`，默认拒绝覆盖 `eval_v1`。全部原始 QA test 中有 875 条合格回答、26 条含 Conflict span；随机 200 条中有 7 条与挑战集重叠。样本筛选依赖原始标注而非模型预测，挑战集是有意富集冲突的诊断集，不能代表总体类别比例。

评价器报告各类 support、P/R/F1、混淆矩阵、拒判数量，以及 conflict recall 的 Wilson 95% 区间。该区间仅为 pair 层面的描述性区间，**未处理同一 response 内多个 pair 的相关性**；样本少时应如实说明，不将开发集的 conflict F1 当作稳定论文结论。
