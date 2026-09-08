# RAGTruth span 到 claim 标签的映射规范

## 1. 适用范围

本规范用于把 response 级 RAGTruth 幻觉字符区间投影为 sentence-level claim 金标签。实现位于 `src/split_claims.py`，规则版本为 `ragtruth_span_overlap_v1`，输出 pair schema 为 `ragtruth_doc_claim_pair_v3`。

所有字符区间均使用 Python 风格的半开区间 `[start, end)`。投影前必须确认 span 未超出回答长度；若原 span 带有 `text`，该文本必须与 `answer[start:end]` 完全一致，否则停止转换并报错。

## 2. 三分类映射

只要 claim 与 gold span 至少重叠一个字符，该 span 就参与标签投影：

| RAGTruth 原始标签 | claim 金标签 | 默认复核 |
| --- | --- | --- |
| 无重叠 span | `supported` | 否 |
| `Evident Conflict` | `conflict` | 否 |
| `Subtle Conflict` | `conflict` | 是 |
| `Evident Baseless Info` | `unsupported` | 否 |
| `Subtle Baseless Info` | `unsupported` | 是 |

当同一 claim 同时命中 conflict 与 baseless span 时，确定性标签使用 `conflict`，并添加 `mixed_gold_types` 复核原因。这个优先级仅用于生成可计算的三分类金标签；原始类型始终保存在 `gold_label_raw`，不会被覆盖或丢弃。

## 3. 重叠度与复核规则

默认阈值如下，可通过 `split_claims.py` 的命令行参数调整：

| 复核原因 | 默认触发条件 |
| --- | --- |
| `subtle_gold_span` | 至少命中一个 `Subtle` span |
| `mixed_gold_types` | 同时命中 conflict 与 baseless 类型 |
| `multiple_gold_spans` | 同一 claim 命中两个或更多 span |
| `low_claim_overlap` | 任一 span 与 claim 的重叠字符数 / claim 长度 `< 0.20` |
| `span_split_across_claims` | 任一 span 在当前 claim 内的字符数 / span 长度 `< 0.50` |
| `boundary_only_overlap` | 重叠位于 claim 起点或终点，且重叠不超过 3 个字符 |

复核标记不会改变自动投影出的 `gold_label`。这保证全部 pair 都可参与可复现实验，同时可以通过 `review_flag=true` 筛出边界样本供人工核查。

## 4. 审计字段

每个 pair 保存：

- `gold_label`：三分类投影结果；
- `gold_label_raw`：命中的 RAGTruth 原始类型集合；
- `gold_spans`：命中的完整原始 span 对象；
- `gold_projection_version`：投影规则版本；
- `gold_overlap_details`：每个 span 的原区间、交集区间、交集字符数、claim 覆盖率和 span 覆盖率；
- `gold_overlap_char_count`：所有交集区间合并后的字符数，避免重叠 span 重复计数；
- `gold_claim_coverage_ratio`：合并交集字符数除以 claim 长度；
- `review_flag` 与 `review_reasons`：切分和投影阶段合并后的复核信息。

## 5. 可复现命令

```bash
python src/split_claims.py \
  --input data/ragtruth/processed/qa_train_500_seed2026.jsonl \
  --output data/ragtruth/processed/doc_claim_pairs_train_500.jsonl \
  --low-claim-overlap-ratio 0.20 \
  --low-span-coverage-ratio 0.50 \
  --boundary-overlap-chars 3
```

正式实验必须记录脚本汇总中的投影版本和阈值。若更改阈值或规则，应生成新结果文件，不应覆盖旧实验产物。
