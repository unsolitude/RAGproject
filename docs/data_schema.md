# 项目数据字段规范

## 1. 兼容原则

项目写出文件时使用固定规范字段，读取外部或课件格式时接受有限别名。若规范字段与别名同时存在但内容不一致，程序直接报错，避免静默使用错误数据。

## 2. Response 级输入

Schema 版本：`ragtruth_response_v1`

| 规范字段 | 可读取别名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `qid` | 无 | string | RAGTruth response ID |
| `query` | 无 | string | 问题或任务输入 |
| `contexts` | `context` | list[object] | 证据列表，规范元素为 `{id, text}`；别名可使用字符串列表 |
| `answer` | `response` | string | 数据集中已有的模型回答 |
| `gold_spans` | `span_label` | list[object] | 原始 RAGTruth `[start, end)` 幻觉标注 |
| `gold_hallucinated` | 可由 `gold_spans` 推导 | boolean | 回答是否包含至少一个人工幻觉 span |

`prepare_ragtruth.py` 始终写出规范字段。兼容别名只用于读取，不作为新的默认输出。

## 3. Document-claim pair

Schema 版本：`ragtruth_doc_claim_pair_v2`

每行代表一个可独立核验的 pair。关键字段如下：

| 字段 | 说明 |
| --- | --- |
| `pair_id` | `<qid>_<claim_id>`，例如 `11858_c02` |
| `claim_id` | 回答内稳定的顺序编号 |
| `document` | 带 evidence 标记的拼接证据 |
| `document_ids` | document 中包含的原始证据 ID |
| `response` | 完整原回答，用于回溯 |
| `claim` | 当前句级事实陈述 |
| `claim_start/claim_end` | claim 在原回答中的 `[start, end)` 字符位置 |
| `claim_token_count` | 无模型依赖的英文单词、数字和中文字符计数 |
| `split_features` | 编号列表或项目符号等切分来源 |
| `gold_label` | `supported`、`conflict` 或 `unsupported` |
| `gold_label_raw` | 与 claim 重叠的原始 RAGTruth 标签类型 |
| `gold_spans` | 与 claim 重叠的原始标注对象 |
| `review_flag/review_reasons` | 需要后续人工复核的标记及原因 |

pair 输入只保存 `gold_label`，不预填模型预测。

## 4. Baseline 结果

Schema 版本：`ragtruth_baseline_result_v2`

回答级结果继续使用嵌套 `claims`。每个 claim 写出：

- `pred_label`：规范预测字段；
- `nli_scores` 或 `minicheck_scores`：Judge 专属分数；
- `evidence_id/evidence_text`：预测依据。

新结果不再重复写入旧 `label` 字段。评价器只在读取历史结果时回退到 `label`。

## 5. 标签命名

项目统一使用小写标签：

```text
supported
conflict
unsupported
```

`gold_label` 表示人工标注投影结果，`pred_label` 表示模型预测。不得使用无前缀的 `label` 同时表达二者。
