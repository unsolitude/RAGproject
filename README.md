# Evidence-Grounded RAG Verifier

面向检索增强生成（Retrieval-Augmented Generation, RAG）的证据忠实性与幻觉检测研究项目。

本仓库当前以 [RAGTruth](https://github.com/ParticleMedia/RAGTruth) 为基准数据集，建立一个透明、零外部依赖的规则基线。系统将模型回答切分为句级陈述，与给定检索上下文逐条对齐，并输出 `supported`、`conflict` 或 `unsupported` 标签，同时计算回答级与字符级 span 指标。

> 当前状态：已跑通单样本和固定随机种子抽取的 200 条 QA 测试子集。现有规则方法仅用于建立可复现下限，不代表最终研究方法。

## 研究目标

本项目关注以下问题：

1. RAG 回答中的事实陈述是否得到检索上下文支持？
2. 系统能否区分无证据内容、证据冲突和可信陈述？
3. 系统能否定位回答中的具体幻觉片段，而不只判断整条回答？
4. 证据过滤、答案切分和支持性判断分别带来多大改进？

当前 RAGTruth 验证流程为：

```text
query + provided contexts + existing answer
                    ↓
            sentence splitting
                    ↓
       evidence-support verification
                    ↓
 claim labels + predicted spans + aggregate metrics
```

## 功能

- 将 RAGTruth 的 `source_info.jsonl` 与 `response.jsonl` 按 `source_id` 合并；
- 筛选 QA、Summary 或 Data2txt 任务及官方 train/test 划分；
- 按指定响应 ID 提取单样本，或使用固定种子随机抽样；
- 将 QA passages 解析为独立候选上下文；
- 对已有回答进行句级证据支持性判断；
- 输出预测幻觉区间，并与人工字符区间比较；
- 汇总混淆矩阵、Precision、Recall、F1、Accuracy 和错误类型分布；
- 按原始生成模型统计分组表现。

## 项目结构

```text
RAGproject/
├── data/
│   └── ragtruth/
│       ├── raw/                  # RAGTruth 官方原始文件（不提交）
│       └── processed/            # 转换后的统一 JSONL 数据
├── outputs/                      # 逐样本预测与指标汇总（不提交）
├── src/
│   ├── prepare_ragtruth.py       # 数据关联、筛选和随机抽样
│   ├── run_baseline.py           # 规则化证据支持性基线
│   ├── evaluate_results.py       # 批量指标与分布统计
│   └── verification/
│       └── nli_judge.py          # Transformer NLI Support Judge
├── requirements-nli.txt          # 固定版本的 NLI 运行依赖
├── .gitignore
└── README.md
```

## 环境要求

- Python 3.10 或更高版本；
- Rule Judge 只使用 Python 标准库；NLI Judge 依赖 PyTorch、Transformers 和 SentencePiece；
- Windows PowerShell、macOS 和 Linux 均可运行，命令示例以 PowerShell 为主。

检查环境：

```powershell
python --version
```

创建隔离环境并安装 NLI 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-nli.txt
```

## 数据准备

从 RAGTruth 官方仓库取得以下文件，并放入 `data/ragtruth/raw/`：

```text
data/ragtruth/raw/source_info.jsonl
data/ragtruth/raw/response.jsonl
```

原始数据不会提交到本仓库。两个文件的作用分别是：

| 文件 | 主要内容 |
| --- | --- |
| `source_info.jsonl` | 任务类型、问题、检索上下文和原始 prompt |
| `response.jsonl` | 模型回答、生成模型、数据划分、质量状态和人工幻觉 span |

一条实验样本以 `response` 为单位；同一 `source_id` 通常对应多个模型回答。

## 快速开始

### 1. 跑通一个真实样本

从项目根目录执行：

```powershell
python src/prepare_ragtruth.py `
  --source-info data/ragtruth/raw/source_info.jsonl `
  --responses data/ragtruth/raw/response.jsonl `
  --output data/ragtruth/processed/qa_one.jsonl `
  --response-id 11858

python src/run_baseline.py `
  --input data/ragtruth/processed/qa_one.jsonl `
  --output outputs/ragtruth_one_result.jsonl

python src/evaluate_results.py `
  --input outputs/ragtruth_one_result.jsonl
```

该样本包含一段人工标注的 `Evident Baseless Info`。当前基线能够在回答级识别该样本含有幻觉，并定位相应句子。

### 2. 复现 200 条 QA 测试子集

下面的命令使用官方 test 划分、`quality=good`、固定随机种子 42。不要添加 `--require-hallucination`，否则会改变自然正负比例并影响 Precision。

```powershell
python src/prepare_ragtruth.py `
  --source-info data/ragtruth/raw/source_info.jsonl `
  --responses data/ragtruth/raw/response.jsonl `
  --output data/ragtruth/processed/qa_test_200_seed42.jsonl `
  --task-type QA `
  --quality good `
  --split test `
  --sample-size 200 `
  --seed 42

python src/run_baseline.py `
  --input data/ragtruth/processed/qa_test_200_seed42.jsonl `
  --output outputs/ragtruth_qa_test_200_predictions.jsonl

python src/evaluate_results.py `
  --input outputs/ragtruth_qa_test_200_predictions.jsonl `
  --output outputs/ragtruth_qa_test_200_metrics.json
```

### 3. 创建开发集

测试数据不应参与规则、阈值或模型选择。后续开发应从 train 划分另行抽样：

```powershell
python src/prepare_ragtruth.py `
  --source-info data/ragtruth/raw/source_info.jsonl `
  --responses data/ragtruth/raw/response.jsonl `
  --output data/ragtruth/processed/qa_train_500_seed2026.jsonl `
  --task-type QA `
  --quality good `
  --split train `
  --sample-size 500 `
  --seed 2026
```

在该开发集上确定阈值和方法后，再对固定测试子集或完整 test 划分进行一次最终评价。

## 统一数据格式

转换后的文件为 JSONL，每行是一条模型回答：

```json
{
  "qid": "11858",
  "source_id": "14292",
  "task_type": "QA",
  "split": "train",
  "generator_model": "mistral-7B-instruct",
  "query": "butcher shop phone number",
  "contexts": [
    {
      "id": "passage_1",
      "text": "Butcher Shop - Hayward ..."
    }
  ],
  "answer": "Based on the given passages ...",
  "gold_spans": [
    {
      "start": 102,
      "end": 214,
      "text": "However, none of them ...",
      "label_type": "Evident Baseless Info"
    }
  ],
  "gold_hallucinated": true
}
```

## 评价指标

### 回答级指标

只要一条回答包含至少一个人工幻觉 span，其金标准回答标签就是 `hallucinated`。

- **Precision**：预测为幻觉的回答中，实际含幻觉的比例；
- **Recall**：实际含幻觉的回答中，被系统检测出的比例；
- **F1**：Precision 与 Recall 的调和平均；
- **Accuracy**：回答级判断正确的比例。

### Span 级指标

将预测区间与人工标注的 `[start, end)` 字符区间进行比较，汇总 micro Precision、Recall 和 F1。字符级 micro 指标按所有样本的预测字符数、金标准字符数和重叠字符数统一计算，不对单样本 F1 直接取平均。

### 错误类型分布

汇总器统计 RAGTruth 人工标签中的：

- `Evident Baseless Info`
- `Subtle Baseless Info`
- `Evident Conflict`
- `Subtle Conflict`

某个随机子集中缺少一种类型时，该类型不会出现在输出字典中。

## 当前基线结果

固定的 `QA / quality=good / test / n=200 / seed=42` 子集包含 34 条幻觉回答和 166 条忠实回答。

| 层级 | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 回答级 | 0.2092 | 0.9412 | 0.3422 | 0.3850 |
| Span 字符级 micro | 0.0771 | 0.5138 | 0.1341 | — |

回答级混淆矩阵为 `TP=32, FP=121, FN=2, TN=45`。结果表明该规则基线召回率较高，但误报明显；后续工作的首要目标是通过 NLI 或 LLM Judge 提升 Precision。

> 该测试子集的结果已经公开查看，不应再用于阈值调整。它只作为初始基线快照，不替代完整 test 集上的最终报告。

## NLI Support Judge

NLI 模式将每条 passage 作为 `premise`、答案句作为 `hypothesis`，计算 entailment、neutral 和 contradiction 概率。默认使用 `cross-encoder/nli-deberta-v3-small`：

```powershell
.\.venv\Scripts\python.exe -B src/run_baseline.py `
  --input data/ragtruth/processed/qa_train_500_seed2026.jsonl `
  --output outputs/ragtruth_qa_train_500_nli_predictions.jsonl `
  --judge nli `
  --nli-model cross-encoder/nli-deberta-v3-small `
  --nli-revision fa2804872c3b4bd748f38c0185cc85775361e735 `
  --model-cache-dir models/huggingface `
  --entailment-threshold 0.5 `
  --contradiction-threshold 0.5

.\.venv\Scripts\python.exe -B src/evaluate_results.py `
  --input outputs/ragtruth_qa_train_500_nli_predictions.jsonl `
  --output outputs/ragtruth_qa_train_500_nli_metrics.json
```

模型会缓存在 `models/`，该目录不会提交。阈值必须只根据 train 派生开发集选择。

在 `QA / quality=good / train / n=500 / seed=2026` 开发集上的默认阈值结果：

| Judge | Response Precision | Response Recall | Response F1 | Response Accuracy | Span Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule | 0.3413 | 0.8649 | 0.4895 | 0.4660 | 0.2229 |
| NLI (`0.5/0.5`) | 0.3157 | 0.9662 | 0.4759 | 0.3700 | 0.2219 |

默认 NLI Judge 进一步提高了 Recall，但产生更多假阳性，尚未在 F1 上超过规则基线。一次只改变 entailment 阈值的初步扫描显示，使用 NLI argmax 判定（阈值 `0.0`）时开发集 Response F1 可达到 `0.5028`；该结果仍需通过正式阈值搜索脚本固化，并在冻结测试集上验证。

## 命令行参数

查看每个程序的完整参数：

```powershell
python src/prepare_ragtruth.py --help
python src/run_baseline.py --help
python src/evaluate_results.py --help
```

常用数据筛选参数：

| 参数 | 说明 |
| --- | --- |
| `--task-type` | `QA`、`Summary`、`Data2txt` 或 `all` |
| `--quality` | 默认仅保留 `good`；使用 `all` 保留全部 |
| `--split` | `train`、`test` 或 `all` |
| `--response-id` | 精确提取一条指定回答 |
| `--sample-size` | 过滤后随机抽取 N 条 |
| `--seed` | 随机种子，默认 42 |
| `--require-hallucination` | 只保留含人工幻觉标注的回答，仅用于案例分析 |

## 当前方法与局限

当前项目提供 Rule 与 NLI 两种 Support Verifier。Rule Judge 使用词项覆盖、否定极性和保守拒答规则；NLI Judge 使用预训练 DeBERTa 对 evidence-claim 文本对进行三分类。两者仍存在明显限制：

- 词项重叠不等于语义蕴含；
- 简单否定规则容易产生大量假阳性；
- 通用 NLI 标签尚不能直接对应四种 RAGTruth 幻觉类型；
- 句子切分不能完整表达 atomic facts；
- 当前使用 RAGTruth 已提供的 contexts 与 answers，不评价端到端检索和重新生成能力。

因此，本基线的作用是提供可复现下限和统一实验接口，而不是作为最终方法。

## 路线图

- [x] 关联 RAGTruth source 与 response 文件
- [x] 单样本检测闭环
- [x] 固定随机种子批量抽样
- [x] 回答级与字符级 span 评价
- [x] 错误类型及生成模型分组统计
- [x] 构建独立 train 开发集
- [x] 加入 NLI Support Judge
- [ ] 固化 NLI 阈值搜索并选择开发集最优配置
- [ ] 加入 LLM Judge 基线
- [ ] 实现答案 atomic-fact 切分
- [ ] 实现证据过滤与消融实验
- [ ] 扩展到 RAGTruth 全任务和完整测试集

## 可复现性约定

- 保留 RAGTruth 官方 `train/test` 划分；
- 记录抽样条件和随机种子；
- 只在 train 派生的开发集上选择阈值和模型；
- 保存逐样本预测，不能只保留汇总数字；
- 后续接入 LLM 时记录提供商、模型 ID、prompt 版本、生成参数和调用日期；
- RAGTruth 的人工 span 只对应数据集中已有回答，不能直接评价重新生成的新回答。

## 引用

若在研究中使用 RAGTruth，请引用原论文：

```bibtex
@inproceedings{niu2024ragtruth,
  title     = {RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models},
  author    = {Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and Shum, KaShun and Zhong, Randy and Song, Juntong and Zhang, Tong},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2024}
}
```

## 许可

RAGTruth 数据及标注遵循其上游仓库声明的许可；使用或再分发前请检查上游 `LICENSE`。本仓库当前尚未声明独立的软件许可证。
