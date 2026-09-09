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

Schema 版本：`ragtruth_doc_claim_pair_v3`

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
| `gold_projection_version` | span 到 claim 的标签投影规则版本 |
| `gold_overlap_details` | 每个命中 span 的交集位置、字符数和双向覆盖率 |
| `gold_overlap_char_count` | 合并重叠区间后被 gold span 覆盖的 claim 字符数 |
| `gold_claim_coverage_ratio` | gold 覆盖字符数占 claim 长度的比例 |
| `review_flag/review_reasons` | 需要后续人工复核的标记及原因 |

pair 输入只保存 `gold_label`，不预填模型预测。
具体标签优先级、低重叠阈值和复核原因见 `docs/label_mapping.md`。

## 4. Baseline 结果

Schema 版本：`ragtruth_baseline_result_v2`

回答级结果继续使用嵌套 `claims`。每个 claim 写出：

- `pred_label`：规范预测字段；
- `nli_scores` 或 `minicheck_scores`：Judge 专属分数；
- `evidence_id/evidence_text`：预测依据。

新结果不再重复写入旧 `label` 字段。评价器只在读取历史结果时回退到 `label`。

## 5. MiniCheck pair 结果

Schema 版本：`ragtruth_minicheck_pair_result_v1`

pair 模式每行对应一个输入 pair，并保持输入顺序。主要字段：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 本次运行的唯一标识 |
| `pair_id/qid/claim_id` | 与 pair 输入一致的追踪字段 |
| `document/claim` | MiniCheck 实际接收的文本 |
| `gold_label/gold_label_raw` | 输入中的人工标签及原始 RAGTruth 类型 |
| `pred_label` | `supported` 或 `unsupported` |
| `prediction` | MiniCheck 二值输出，支持为 1，否则为 0 |
| `score` | `[0,1]` 范围的支持概率 |
| `minicheck_scores` | 阈值、选中 chunk 和各 chunk 支持概率 |
| `model_name/model_path/threshold` | 模型和判定配置 |
| `document_len/claim_len` | 字符长度，单位见 `length_unit` |
| `latency_ms` | 当前 batch 墙钟耗时除以实际 batch size |
| `latency_measurement` | 固定为 `batch_wall_clock_amortized`，避免误解为逐条独立推理耗时 |
| `batch_id/batch_size` | 该结果对应的批次信息 |

同名 `.manifest.json` 保存精确运行总耗时、每个 batch 的总耗时、模型配置、输入输出数量、标签分布、输入 SHA-256 和切分版本。SHA-256 计算前统一换行为 LF，确保 Windows 与 Linux 得到相同值。验证器会拒绝与当前 pair 文件哈希或切分版本不一致的旧结果。逐 pair 行不会把整个 batch 耗时重复记作单条耗时。

## 6. NLI pair 结果

Schema 版本：`ragtruth_nli_pair_result_v1`

NLI pair 模式与 MiniCheck 使用相同的 `document`、`claim` 和输入顺序，每行额外保存：

| 字段 | 说明 |
| --- | --- |
| `pred_label` | 项目三分类：`supported`、`conflict` 或 `unsupported` |
| `nli_scores` | 原生 `entailment`、`contradiction`、`neutral` 三类概率 |
| `top_label/top_score` | 三类概率中的最高类别及其概率 |
| `model_name/model_revision` | 固定的 Hugging Face 模型版本 |
| `entailment_threshold/contradiction_threshold` | 从 NLI 类别映射到项目标签的阈值 |
| `review_flag/review_reasons` | 继承数据复核标记；最高概率低于复核阈值时追加 `low_nli_confidence` |
| `latency_ms/batch_id/batch_size` | 批次摊销耗时及批次信息 |

NLI manifest 还保存设备、GPU 名称、PyTorch 峰值显存、模型 revision、最大输入长度、三类阈值、低置信度阈值、逐批耗时和总耗时。峰值显存为当前推理进程的 `torch.cuda.max_memory_allocated`，不是整张显卡或其他进程的占用。

## 7. 标签命名

项目统一使用小写标签：

```text
supported
conflict
unsupported
```

`gold_label` 表示人工标注投影结果，`pred_label` 表示模型预测。不得使用无前缀的 `label` 同时表达二者。
