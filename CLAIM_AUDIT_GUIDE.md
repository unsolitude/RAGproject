# 第三节：RAGTruth claim 人工审计指南

本指南用于审计现有的 50 条 QA 回答、414 条句级 document–claim pair。目标是确认哪些文本可以作为核验对象，以及自动投影的三分类标签是否能由原回答和完整来源证据支持。人工记录与旧实验数据并存，后续证据选择实验使用冻结后的同一批有效 claim。

## 1. 文件在哪里

以项目根目录 `D:\coding_app\codes\RAGproject` 为基准：

| 用途 | 项目内路径 | 操作 |
| --- | --- | --- |
| 本轮直接审计的 414 条 pair | `data/ragtruth/processed/doc_claim_pairs_50.jsonl` | 阅读；每行一条 pair，含问题、完整回答、claim、完整来源、原始 span 和自动 gold |
| 同批 50 条完整回答 | `data/ragtruth/processed/sample_50.jsonl` | 需要回看 response 级上下文时使用 |
| RAGTruth 原始回答与标注 | `data/ragtruth/raw/response.jsonl` | 只有怀疑转换或 span 偏移时追溯 |
| RAGTruth 原始问题与来源 | `data/ragtruth/raw/source_info.jsonl` | 只有怀疑来源拼接时追溯 |
| 当前辅助审计底稿 | `outputs/lesson3/audit/claims_audit_v7.jsonl` | 阅读现有审计决定；不要写回 pair 输入 |
| 当前投影和评价规则 | `docs/label_mapping.md`、`docs/pair_evaluation_protocol.md` | 核对现有规则及其限制 |

`data/ragtruth/processed/doc_claim_pairs.jsonl` 是另一个仅含 3 条的测试文件，本轮审计应使用带 `_50` 的文件。`pair_id` 是审计记录与原始 pair 对齐的主键；`qid` 是回答编号，`source_id` 是来源编号。完整 `response`、`document`、`claim_start`、`claim_end` 都已在 pair 行中。

## 2. 保存与版本管理

先在 VS Code 中打开项目根目录。用 PowerShell 创建工作目录：

```powershell
New-Item -ItemType Directory -Force -Path 'outputs/lesson3/audit' | Out-Null
```

现有底稿是 `claims_audit_v7.jsonl`；此前的草稿版位于 `outputs/archived/lesson3_audit_pre_v7/`。如继续人工复核，请另存新版本，不要覆盖 v7 或直接修改 `doc_claim_pairs_50.jsonl`、历史模型输出。VS Code 保存编码使用 UTF-8；一行放一个 JSON 对象，行与行之间不加逗号，也不要在外面包 `[]`。定期保存，并保留带日期的备份副本。

仓库的 `.gitignore` 忽略整个 `outputs/`，因此这里的人工记录**不会随普通 `git add .` 上传**。如果审计结果需要随论文或云端代码同步，先检查是否包含不宜公开的原始回答或来源文本，再选择一种方式：

- 私下同步完整工作文件，仍保存在 `outputs/lesson3/audit/`；
- 完成审核后，另制仅含 `pair_id`、审核决定、必要理由和版本号的最小审计文件，放在版本管理目录中，并明确它与本地完整记录的对应关系。

每次冻结一版时记录文件 SHA-256、审核日期、审核人、输入 pair 文件哈希和 `label_version`。冻结后如需改动，应生成新版本并说明变更原因。

## 3. 在 VS Code 中逐条审计

1. 打开 `doc_claim_pairs_50.jsonl` 与 `claims_audit.jsonl`，右键标签页选择分栏显示；按 `Alt+Z` 打开自动换行。
2. 用 `Ctrl+F` 搜索 `pair_id`，一次处理一条。先看 `query`、完整 `response`、`claim` 与 `claim_start/claim_end`，确认切分保留了主体、否定、比较方向和条件。
3. 看完整 `document`，查找能支持或反驳 claim 的原句及其 `document_ids`；不能只看关键词是否重合。
4. 看 `gold_label`、`gold_label_raw`、`gold_spans`、`gold_overlap_details` 和 `review_reasons`，判断自动投影是否适合这个句级 claim。RAGTruth span 标在**回答**中，不是来源证据的位置。
5. 在审计文件中追加一行，记录决定、最关键的来源摘录、理由及审核人。第一轮不打开 NLI、MiniCheck 或融合预测结果，避免被预测带偏。

### 先审计资格，再审计关系

资格字段 `eligibility` 建议使用：

| 值 | 使用情形 | 后续处理 |
| --- | --- | --- |
| `eligible` | 独立可核验的事实或具体操作建议 | 进入关系标签审核 |
| `nonclaim` | `Sure`、致谢、标题等没有可核验命题的文本 | 排除出 claim 级指标，单独计数 |
| `needs_context` | 代词、比较对象或条件不完整，需要原回答邻句 | 回看并人工决定如何保留语境 |
| `needs_split` | 一句中含多个可独立核验事实 | 记录建议拆分，待新 pair 版本处理 |
| `mixed` | 同一 claim 不同部分具有不同证据关系 | 记录组成部分与理由，二次复核 |

祈使句也可能是有效核验对象，例如 `Cut the potatoes into French fries` 有明确动作与对象。`Sure` 则没有可判定的事实内容；本轮有 `14075_c01`、`14385_c01`、`15268_c01` 三条此类明显候选。

对于 `eligible` 样本，`audited_gold_label` 使用 `supported`、`conflict`、`unsupported`：

- `supported`：完整来源足以支持 claim 的全部重要内容；
- `conflict`：来源明确反驳 claim 的对象、关系、数值、时间或条件；
- `unsupported`：来源既不足以支持，也没有明确反驳。只在提供的来源范围内判断，不把未知事实当成错误事实。

`nonclaim` 用 `not_applicable`。对于 `needs_context`、`needs_split`、`mixed`，尚不能稳定给出单一三分类标签时，使用 `needs_adjudication`，在二次审查前不纳入冻结的有效三分类集合。不要为了凑齐三分类而把不同事实压成一个标签。`case_review` 是模型融合的拒判状态，不是人工 gold 标签。

## 4. 建议记录格式

以下是一条**已知的非事实候选**示例；请本人核对原回答后填写真实审核人和日期。建议每条都记录 `pair_id`、原标签、资格、人工关系、决定、依据及版本。`evidence_ids` 对应原 pair 的证据编号；找不到明确支持或反驳材料时用空数组，并写明检查范围。

```json
{"pair_id":"15268_c01","source_id":"15465","claim_text":"Sure","original_gold_label":"supported","eligibility":"nonclaim","audited_gold_label":"not_applicable","decision":"exclude_from_claim_metrics","evidence_ids":[],"evidence_quote":null,"review_reason":"Acknowledgement without a verifiable proposition; checked in the full response.","suggested_claims":[],"auditor":"填写姓名","audited_at":"填写日期","audit_status":"reviewed","label_version":"lesson3_claim_audit_v1"}
```

对有效 claim，如果保留自动标签，`decision` 可写 `keep`；如人工改动，写 `relabel` 并保留 `original_gold_label`。需拆分可写 `review_and_split`，把拆分建议放入 `suggested_claims`。未完成审核时，`audit_status` 用 `pending`，不应把待审项计入已完成人工标签。

`11876_c05` 是值得优先复核的比较关系案例：claim 把较高脂肪含量和结合能力归于 single cream，而来源中的关键比较涉及 double cream。审核时应记录来源段落及比较方向，再决定是否维持原 `conflict` 标签。`11957_c03` 的烹饪操作建议可用于检查“祈使句不等于 nonclaim”。

## 5. 审计顺序与验收

本轮原始分布为 `supported=352`、`conflict=2`、`unsupported=60`，现有数据复核标记 39 条。优先审计现有复核项、全部 62 条非 supported、三条 `Sure` 及疑似复合 claim；这些集合去重后为 **70 条**。随后审阅其余 344 条，特别检查是否还有未被标记的非事实句和复合句。最后二次复核全部 conflict、改标签、nonclaim、mixed 和需拆分样本，并抽查普通 supported 样本。

完成时应满足：414 个原始 `pair_id` 各有一条审计记录，无重复；所有最终纳入评价的记录均是 `eligible` 且有明确三分类标签；排除和待裁决项单独计数；每条改动可追溯至原标签、来源摘录及审核理由。新旧数据版本分别报告，旧结果仍可按原投影 gold 复现。即使人工审计完成，来源证据锚点和新 gold 也需要明确版本，不能直接复用旧评价文件当作新指标。

### 基本校验命令（项目根目录，PowerShell）

```powershell
$auditPath = 'outputs/lesson3/audit/claims_audit_v7.jsonl'
$rows = @(Get-Content -LiteralPath $auditPath -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$rows.Count
$rows | Group-Object pair_id | Where-Object Count -gt 1 | Select-Object Name,Count
Get-FileHash -Algorithm SHA256 -LiteralPath $auditPath
```

第一项最终应为 414，第二项应无输出。检查缺失 ID 时，继续运行：

```powershell
$pairPath = 'data/ragtruth/processed/doc_claim_pairs_50.jsonl'
$pairs = @(Get-Content -LiteralPath $pairPath -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
Compare-Object @($pairs.pair_id | Sort-Object) @($rows.pair_id | Sort-Object)
```

`Compare-Object` 无输出才表示 ID 集合一致；仍需人工检查每条的资格、标签、证据与理由。审计文件在输出目录中由你逐条填写，本指南和这些校验命令本身不会替你生成或裁决人工标签。
