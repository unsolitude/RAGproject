# Evidence-Grounded RAG Verifier Baseline

这是依据《RAG_Lesson1》搭建的第一版可复现基线。目前使用 RAGTruth 的真实 QA 样本验证完整检测闭环：

`query + provided contexts + existing answer → 句级支持性判断 → 回答级/span级指标`

项目研究目标是回答：**生成答案中的每个陈述，是否能由检索证据支持？** 输出标签为 `supported`、`conflict` 或 `unsupported`。

## 快速开始

本版本仅使用 Python 标准库，无需下载模型或配置 API。原始文件应位于 `data/ragtruth/raw/`。

```powershell
python src/prepare_ragtruth.py --source-info data/ragtruth/raw/source_info.jsonl --responses data/ragtruth/raw/response.jsonl --output data/ragtruth/processed/qa_one.jsonl --response-id 11858
python src/run_baseline.py --input data/ragtruth/processed/qa_one.jsonl --output outputs/ragtruth_one_result.jsonl
```

运行后会生成：

- `data/ragtruth/processed/qa_one.jsonl`：统一格式的真实样本；
- `outputs/ragtruth_one_result.jsonl`：句级预测、证据对齐、字符区间和金标准对照。

## 数据格式

每行一个 JSON 对象；字段与第一节课的统一格式对应。

```json
{
  "qid": "11858",
  "query": "butcher shop phone number",
  "contexts": [
    {"id": "passage_1", "text": "Butcher Shop - Hayward ..."}
  ],
  "answer": "Based on the given passages ...",
  "gold_spans": [{"start": 102, "end": 214, "label_type": "Evident Baseless Info"}],
  "gold_hallucinated": true
}
```

## 当前 baseline 与后续实验

当前实现是使用 RAGTruth 官方 contexts 和已有 answer 的规则化 Support Judge。它是透明的检测基线，后续使用 NLI/LLM Judge 替换。

后续按课件逐步替换模块即可：

1. 将 `retrieve()` 替换为 BGE/FAISS，并加入 cross-encoder 重排；
2. 用 API 或开源指令模型替换 `generate_answer()`；
3. 用 NLI 模型和 LLM Judge 替换 `classify_claim()`；
4. 从 RAGTruth QA 固定测试子集中抽样，再扩展至全量；
5. 为每条答案保留 `claim → evidence → label`，再开展消融与误差分析。

## 论文实验对应

| 课件模块 | 当前实现 | 下一步可比较版本 |
| --- | --- | --- |
| Retrieval | BM25 | BGE / FAISS / reranker |
| Evidence Filter | 词项重叠阈值 | NLI/LLM relevance judge |
| Generator | 抽取式保守生成 | API / 开源 LLM |
| Support Verifier | 规则化三分类 | NLI judge / LLM judge |
| Evaluation | F1、Evidence Recall、Faithfulness、Hallucination Rate | 加 Macro-F1、Citation Precision |

## 跑通一条 RAGTruth 样本

先把官方的 `source_info.jsonl` 与 `response.jsonl` 按 `source_id` 关联，并提取一条带人工幻觉标注的 QA 样本：

```powershell
python src/prepare_ragtruth.py --source-info data/ragtruth/raw/source_info.jsonl --responses data/ragtruth/raw/response.jsonl --output data/ragtruth/processed/qa_one.jsonl --response-id 11858
```

然后使用数据集已有回答进行证据支持性验证：

```powershell
python src/run_baseline.py --input data/ragtruth/processed/qa_one.jsonl --output outputs/ragtruth_one_result.jsonl
```

RAGTruth 验证模式会保留官方提供给生成模型的全部 passages，不重新生成答案。输出同时包含句级预测、字符区间、回答级正确性及 span-level Precision/Recall/F1。

## 固定抽取 200 条 QA 测试样本

使用官方 `test` 划分、只保留 `quality=good`，并用固定随机种子抽样。不要使用 `--require-hallucination`，否则会人为改变正负样本比例并扭曲 Precision。

```powershell
python src/prepare_ragtruth.py --source-info data/ragtruth/raw/source_info.jsonl --responses data/ragtruth/raw/response.jsonl --output data/ragtruth/processed/qa_test_200_seed42.jsonl --task-type QA --quality good --split test --sample-size 200 --seed 42
python src/run_baseline.py --input data/ragtruth/processed/qa_test_200_seed42.jsonl --output outputs/ragtruth_qa_test_200_predictions.jsonl
python src/evaluate_results.py --input outputs/ragtruth_qa_test_200_predictions.jsonl --output outputs/ragtruth_qa_test_200_metrics.json
```

汇总文件包含 TP/FP/FN/TN、回答级 Precision/Recall/F1/Accuracy、字符级 micro Precision/Recall/F1、四种人工错误类型的 span 数量、预测标签分布以及各生成模型的分组表现。
