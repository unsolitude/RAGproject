# 第二节课件与当前项目的差异及新增任务

> 对照来源：`D:/学习资料/课题相关/lesson2.pptx`（62 页）  
> 对照项目：当前 `RAGproject` 仓库  
> 整理日期：2026-09-08

## 1. 结论摘要

当前仓库已经具备 RAGTruth 数据转换、句级切分、Rule Judge、NLI Judge、MiniCheck Judge、回答级与字符 span 级评价，以及 SLURM 冒烟测试入口。它可以把 RAGTruth 中已有的回答作为整体样本进行幻觉检测。

第二节课件要求的 baseline 以 **claim 配对** 为实验基本单位，目标链路是：

```text
RAGTruth 原始数据
    ↓
统一 response 样本
    ↓
独立 doc_claim_pairs 数据集
    ↓
MiniCheck 结果 + NLI 结果
    ↓
组合三分类与人工复核
    ↓
claim 级指标、成本和案例文件
```

因此，当前项目不需要推倒重写，但需要在现有回答级流程中间增加一个稳定的 claim 配对层，并扩展结果格式、混合判断器和 claim 级评价。课件中的部分命令参数只是设计示例，与当前脚本的真实 CLI 不一致，也需要统一。

---

## 2. 当前项目实际状态

### 2.1 已有数据

| 文件 | 数量 | 当前用途 |
| --- | ---: | --- |
| `data/ragtruth/processed/qa_one.jsonl` | 1 | 单样本调试 |
| `data/ragtruth/processed/qa_train_500_seed2026.jsonl` | 500 | train 派生开发集与阈值选择 |
| `data/ragtruth/processed/qa_test_200_seed42.jsonl` | 200 | 已查看结果的固定测试子集 |

### 2.2 已有程序

| 文件 | 当前职责 |
| --- | --- |
| `src/prepare_ragtruth.py` | 关联 `source_info.jsonl` 与 `response.jsonl`，筛选任务、质量和划分，解析 passages，固定种子抽样 |
| `src/run_baseline.py` | 在运行时切分回答，调用 Rule、NLI 或 MiniCheck Judge，生成回答级预测和字符区间 |
| `src/verification/nli_judge.py` | 对每条 evidence 与 claim 计算 entailment、contradiction、neutral 概率 |
| `src/verification/minicheck_judge.py` | 从本地目录加载 Bespoke-MiniCheck-7B，批量计算 claim 支持概率 |
| `src/evaluate_results.py` | 计算回答级二分类指标、字符 span micro 指标、人工错误类型数量和生成模型分组结果 |
| `scripts/slurm/test_minicheck.slurm` | 在一张 GPU 上运行 MiniCheck 单样本冒烟测试 |
| `scripts/slurm/download_minicheck.slurm` | 通过 Hugging Face 镜像下载 MiniCheck 权重 |

### 2.3 已有结果

500 条 train 开发集上已经生成 Rule 与 NLI 结果。当前最好的一次探索性 NLI 配置取得回答级 F1 `0.5028`，但正式阈值搜索工具尚未固化。MiniCheck 代码与 CPU 接口测试已经完成，真实 A6000 指标仍待集群恢复后运行。

---

## 3. 课件与当前项目的主要差异

### 3.1 数据模式缺少独立 claim 配对层

> 实施状态（2026-09-08）：已完成 `src/split_claims.py`、公共 `sentence_v2` 切分逻辑和单样本 `doc_claim_pairs.jsonl`。后续 50 条数据可直接复用该命令生成。

**课件要求（第 7、11、23、39、60 页）**

- `processed.jsonl` 保存统一 response 样本；
- `split_claims.py` 将回答展开为 `doc_claim_pairs.jsonl`；
- 每一行对应一个 `document + claim`；
- 每条记录包含唯一 `pair_id`、`claim_id`、`gold_label` 和来源信息。

**当前实现**

- `prepare_ragtruth.py` 只生成回答级样本；
- `run_baseline.py` 在内存中临时切分 answer；
- claim 结果嵌套在回答结果的 `claims` 数组中；
- 没有可复用的 `doc_claim_pairs.jsonl`。

**建议修改：P0**

新增 `src/split_claims.py`，把 claim 切分从模型运行中解耦。保留当前回答级文件，同时生成：

```json
{
  "pair_id": "11858_c01",
  "qid": "11858",
  "claim_id": "c01",
  "query": "...",
  "document": "...",
  "document_ids": ["passage_1", "passage_2", "passage_3"],
  "claim": "...",
  "claim_start": 0,
  "claim_end": 42,
  "gold_label": "supported",
  "gold_label_raw": [],
  "review_flag": false,
  "source": "ragtruth",
  "split": "train",
  "generator_model": "..."
}
```

### 3.2 字段命名与课件不一致（已完成）

> 实施状态（2026-09-08）：已完成规范字段与课件别名兼容层，新增 schema 版本和 `docs/data_schema.md`；新 baseline 结果只写 `pred_label`，评价器仍可读取历史 `label`。

**课件要求（第 9、11 页）**

- `context`、`response`、`span_label`、`claim_label`；
- 原始字段与处理字段同时保存；
- `gold_label` 与 `pred_label` 明确分离。

**当前实现**

- 使用 `contexts`、`answer`、`gold_spans`、`gold_hallucinated`；
- 输出使用 `label`，未明确命名为 `pred_label`；
- claim 级金标签尚未生成。

**建议修改：P0**

不必破坏已有字段。将现有回答级 schema 视为 canonical response schema，在 pair schema 中使用课件要求的 `document`、`claim`、`gold_label`、`pred_label`。如需兼容外部代码，可在读取层接受 `answer/response` 和 `contexts/context` 两组别名，但写出格式只保留一种规范。

### 3.3 claim 切分规则不完整（已完成第二节版本）

> 实施状态（2026-09-08）：已升级为 `sentence_v2`，支持换行、编号列表和项目符号切分，保存 `claim_token_count` 与 `split_features`，并对超过可配置阈值的长 claim 和疑似并列事实添加复核原因。atomic fact 切分仍按计划留到第三节。

**课件要求（第 12、44、49、51 页）**

- 处理句号、分号、编号列表和并列事实；
- 长句增加复核标记；
- 第二节使用 sentence claim，后续扩展 atomic fact；
- claim 切分错误可以回到独立脚本修正。

**当前实现**

- `split_claims_with_offsets()` 主要按标点切分；
- 没有编号列表专门规则；
- 不拆分并列事实；
- 没有长度阈值和 `review_flag`；
- 切分逻辑与 `run_baseline.py` 耦合。

**建议修改：P0/P1**

第二节先完成可复现的 sentence-level 脚本，并增加：

- `split_rule_version`；
- `claim_token_count`；
- `review_flag=long_claim/list_parse/ambiguous_overlap`；
- 保留原始字符 `[start, end)`；
- 暂不自动拆分所有 `and`，避免破坏语义，先把多事实长句送入人工复核。

atomic fact 切分放到第三节增强实验，不应混入本轮 baseline。

### 3.4 尚未把 RAGTruth span 投影为 claim 级金标签

**课件要求（第 8、13、14 页）**

将 RAGTruth 的四类 span 映射为：

| RAGTruth 标签 | 项目标签 |
| --- | --- |
| 无幻觉 span | `supported` |
| Evident Conflict | `conflict` |
| Subtle Conflict | `conflict`，并建议复核 |
| Evident Baseless Info | `unsupported` |
| Subtle Baseless Info | `unsupported`，并建议复核 |

**当前实现**

- 只保存原始 `gold_spans`；
- 只构造回答级 `gold_hallucinated`；
- 评价没有 claim 级 `gold_label`。

**建议修改：P0**

在 `split_claims.py` 中按字符区间重叠进行投影，同时保留原始类型：

1. claim 不与任何 gold span 重叠，标为 `supported`；
2. 与 conflict span 重叠，标为 `conflict`；
3. 只与 baseless span 重叠，标为 `unsupported`；
4. 同时覆盖多种标签、重叠比例过低或只覆盖边界时，添加 `review_flag`；
5. 不要丢弃 `gold_label_raw`，否则后续无法分析 Evident/Subtle 差异。

标签规则应单独写入 `docs/label_mapping.md`。

### 3.5 MiniCheck 输入与输出仍是回答级封装

**课件要求（第 22–24、38、41 页）**

- MiniCheck 直接读取 pair 数据；
- 每条输出保存 `pair_id`、`prediction`、`score`；
- 保存 `latency_ms`、`model_path`、`document_len`、`claim_len`；
- 运行 50 条小样本 baseline。

**当前实现**

- `MiniCheckJudge` 已经能批量判断 claim；
- 支持概率保存在嵌套的 `minicheck_scores`；
- 所有 contexts 默认拼接后输入；
- 没有逐 pair 延迟、长度和独立 MiniCheck 结果文件；
- 尚未完成真实 GPU 测试。

**建议修改：P0**

保留 `MiniCheckJudge`，扩展 `run_baseline.py` 支持 `--data-format response|pair`。pair 模式输出独立的 `outputs/minicheck_results.jsonl`，每行一个 pair。运行级耗时放入 run manifest，逐条耗时需要明确计时口径，批量推理时不能简单把整批时间重复记到每个 pair。

### 3.6 NLI 模型与课件建议不同，但不构成实现错误

**课件要求（第 25–31、56 页）**

- NLI 原生三分类；
- 建议 `microsoft/deberta-v3-large-mnli` 或同类模型；
- 保存三类概率、最高分、批大小、显存和耗时；
- 低置信度样本进入复核。

**当前实现**

- 使用 `cross-encoder/nli-deberta-v3-small`；
- 已保存 entailment、contradiction、neutral 概率；
- 已支持阈值和 batch size；
- 没有 `review_flag`、耗时和显存字段；
- NLI 结果仍嵌套在回答结果中。

**建议修改：P0/P1**

- 第二节先保留 small 模型作为低成本 NLI baseline；
- 把 large 模型作为后续模型规模消融，不要直接替换后丢失可比性；
- 新增独立 `outputs/nli_results.jsonl`；
- 保存 `top_label`、`top_score`、三类概率和 `review_flag`；
- 在 run manifest 记录模型 revision、batch size、设备、耗时和峰值显存。

### 3.7 课件要求的 MiniCheck + NLI 混合判断器尚未实现

**课件要求（第 18、32、33、45、48、52 页）**

| MiniCheck | NLI | final label |
| --- | --- | --- |
| supported | entailment | supported |
| unsupported | contradiction | conflict |
| unsupported | neutral | unsupported |
| supported | contradiction | case_review |
| unsupported | entailment | case_review |

**当前实现**

- Rule、NLI、MiniCheck 通过 `--judge` 三选一独立运行；
- 没有结果合并脚本；
- 没有 `final_label` 和模型分歧队列。

**建议修改：P0**

新增 `src/merge_judges.py`：

- 按 `pair_id` 合并两个结果文件；
- 实现课件中的确定性组合表；
- 保留两个模型的原始分数；
- 输出 `outputs/merged_results.jsonl`；
- 分歧时不强行覆盖，写入 `review_flag=true` 和 `review_reason=model_disagreement`。

融合规则需要版本号，例如 `fusion_rule_version=v1`。

### 3.8 评价口径与课件目标不同

**课件要求（第 34–37、41 页）**

- 评价对象为 claim；
- 计算三分类 Accuracy、Macro-F1、各标签 Precision/Recall/F1；
- 输出 3×3 confusion matrix；
- 重点分析 conflict 与 unsupported 混淆；
- 结果写入 `metrics_summary.csv` 和 `classification_report.txt`。

**当前实现**

- 主要计算回答级“是否含幻觉”二分类；
- 计算字符 span micro 指标；
- `predicted_claim_labels` 只是预测数量统计；
- 没有 claim 级三分类混淆矩阵、Macro-F1 和 label-wise F1；
- 指标输出为 JSON，不是 CSV 和文本报告。

**建议修改：P0**

不要删除当前回答级与 span 级指标。在 `evaluate_results.py` 中新增 pair 评价模式，形成三个并列层级：

1. claim 三分类指标，作为第二节主要 baseline 表；
2. response hallucination 二分类指标，保留现有论文价值；
3. 字符 span micro 指标，保留定位能力评价。

新增输出：

- `outputs/metrics_summary.json`：完整机器可读结果；
- `outputs/metrics_summary.csv`：方法对比表；
- `outputs/classification_report.txt`：逐标签报告；
- `outputs/confusion_matrix.csv`：3×3 数值矩阵。

### 3.9 运行成本和日志字段不足

**课件要求（第 21、24、34、38、54、56 页）**

保存 `run_id`、模型版本、数据划分、阈值、总耗时、显存、batch size 和模型路径。

**当前实现**

- SLURM 会生成文本日志；
- 结果记录了部分模型配置；
- 没有结构化 run manifest；
- 没有峰值 GPU 显存和总运行时间；
- 没有环境依赖快照与结果文件绑定。

**建议修改：P0**

每次运行生成 `outputs/logs/<run_id>.json` 或 `<run_id>.md`，至少包含：

```text
run_id, git_commit, command, start_time, end_time, elapsed_seconds,
model_name, model_revision/model_path, threshold, batch_size,
data_file, sample_count, random_seed, GPU name, peak_gpu_memory_mb,
torch_version, transformers_version, vllm_version, output_file
```

### 3.10 证据构造策略与课件不一致

**课件要求（第 47、50、58 页）**

- 第二节 baseline 使用 Top-3 concat；
- 保留 evidence id；
- 记录截断或分块；
- 第三节比较 Top-k、过滤和 reranker。

**当前实现**

- RAGTruth verification 模式直接使用所有 `contexts`；
- BM25 和 `top_k` 只在没有 answer 的 generation 模式启用；
- MiniCheck 会拼接所有 evidence，再由模型内部按 token 分块；
- NLI 对每条 passage 分别打分并取最大值；
- 两个 Judge 实际看到的证据组织方式并不完全一致。

**建议修改：P0/P1**

- 为 pair 构造明确指定 `evidence_policy=top3_concat_v1`；
- 第二节所有 Judge 使用同一份 pair 文件，保证输入一致；
- 保存 `document_ids`、`document_token_count` 和截断信息；
- Top-k、BM25/BGE/reranker 作为第三节消融变量，不要在第二节不同 Judge 之间混用不同证据。

### 3.11 课件命令与当前 CLI 不一致

课件第 40 页为概念命令，不能直接在当前仓库执行：

| 课件参数 | 当前真实参数 |
| --- | --- |
| `prepare_ragtruth.py --input data/raw` | `--source-info ... --responses ...` |
| `--sample_size 50` | `--sample-size 50` |
| `run_baseline.py --data ...` | `--input ...` |
| `evaluate_results.py --pred ... --gold ...` | 当前只有 `--input ...`，gold 嵌在结果中 |
| `metrics_summary.csv` | 当前默认产生 JSON 汇总 |

**建议修改：P0**

以仓库真实 CLI 为准更新项目文档。新增 pair 模式时可以增加课件需要的功能，但应尽量保留旧参数和旧结果格式，避免已有实验无法复现。

### 3.12 环境与模型目录存在复现风险

**当前发现**

- `requirements-nli.txt` 当前固定 `torch==2.14.0`、`transformers==5.16.1`；
- 已验证的云端 MiniCheck 环境此前使用另一组 Torch、Transformers、vLLM 版本；
- `download_minicheck.slurm` 下载到 `models/minicheck` 的 Hugging Face cache；
- `test_minicheck.slurm` 默认读取平铺目录 `models/Bespoke-MiniCheck-7B`。

**建议修改：P0**

- 分离 `requirements-nli.txt` 与 `requirements-minicheck.txt`；
- 以真实通过验证的云端 `pip freeze` 为依据固定 MiniCheck 环境；
- 统一下载脚本与推理脚本的模型路径；
- 模型权重继续由 `.gitignore` 排除，只提交下载说明、版本标识和校验信息。

---

## 4. 课件新增任务清单

### 4.1 第二节必须完成的任务（P0）

#### A. 样本整理

- [ ] 从 train 划分固定种子抽取 50 条 QA 样本，生成 `data/ragtruth/processed/sample_50.jsonl`
- [x] 新增 `src/split_claims.py`
- [x] 生成 `data/ragtruth/processed/doc_claim_pairs.jsonl`（当前为单样本验证产物）
- [x] 为每个 pair 生成 `pair_id` 和 `claim_id`
- [x] 将 RAGTruth span 投影为最小可用的 `gold_label`
- [x] 保存 `gold_label_raw`、字符位置和 evidence id
- [ ] 长句、Subtle 标签和多标签重叠样本增加 `review_flag`
- [ ] 编写 `docs/label_mapping.md`
- [ ] 手工核查至少 5 条，生成 `docs/manual_check.md`

#### B. 两个 baseline 独立运行

- [ ] 在 1 条样本上完成 MiniCheck GPU 冒烟测试
- [ ] 在 50 条 pair 数据上运行 MiniCheck
- [ ] 生成 `outputs/minicheck_results.jsonl`
- [ ] 在相同 pair 数据上运行 NLI
- [ ] 生成 `outputs/nli_results.jsonl`
- [ ] 两份结果都保存模型版本、分数、输入长度和运行标识

#### C. 混合判断器

- [ ] 新增 `src/merge_judges.py`
- [ ] 按 `pair_id` 合并 MiniCheck 与 NLI
- [ ] 实现课件第 32 页组合规则
- [ ] 输出 `final_label`、`review_flag`、`review_reason`
- [ ] 生成 `outputs/merged_results.jsonl`

#### D. claim 级评价

- [ ] 扩展 `src/evaluate_results.py` 的 pair 模式
- [ ] 计算 claim 三分类 Accuracy
- [ ] 计算 Macro-F1
- [ ] 计算 supported/conflict/unsupported 的 Precision、Recall、F1
- [ ] 输出 3×3 confusion matrix
- [ ] 保留现有 response 与 span 指标
- [ ] 生成 `metrics_summary.json/csv` 和 `classification_report.txt`

#### E. 日志与案例

- [ ] 为每次运行创建唯一 `run_id`
- [ ] 记录命令、Git commit、环境、阈值、耗时和峰值显存
- [ ] 生成 `outputs/cases.jsonl`
- [ ] 至少整理 supported、conflict、unsupported 成功案例各 1–2 条
- [ ] 整理 MiniCheck/NLI 分歧和人工判断不一致的误判案例

### 4.2 第二节完成后立即开展的任务（P1）

- [ ] 在 train 开发集上正式搜索 MiniCheck 阈值
- [ ] 固化 NLI 阈值搜索脚本，不再手工修改结果
- [ ] 设计 `0.50–0.70` 低置信度复核区间，但阈值必须由开发集验证
- [ ] 比较 MiniCheck、NLI、MiniCheck+NLI、MiniCheck+rules
- [ ] 分析 conflict 被误判为 unsupported 的比例
- [ ] 按 document 长度、claim 长度、生成模型和 RAGTruth 原始错误类型分组
- [ ] 将实验结果同步整理到论文的 Task Definition、Experiments、Method、Analysis 和 Implementation Details 草稿

### 4.3 第三节入口任务（P2）

- [ ] 比较 Top-k=`3/5/10`
- [ ] 将 BM25 正式接入 verification 证据选择，而不只用于 generation 分支
- [ ] 增加 BGE embedding 检索或 reranker
- [ ] 比较原始 context 与过滤后 context
- [ ] 记录 Evidence Recall 和 Citation Accuracy
- [ ] 实现 atomic fact 与 hybrid claim split
- [ ] 基于 MiniCheck support score、NLI 概率和 score margin 设计可学习或规则型 hybrid score
- [ ] 将 C2D、D2C 和 Error Injection 作为后续训练数据扩展，不纳入当前 baseline

---

## 5. 建议的目标项目结构

```text
RAGproject/
├── data/ragtruth/
│   ├── raw/
│   └── processed/
│       ├── qa_one.jsonl
│       ├── sample_50.jsonl
│       ├── qa_train_500_seed2026.jsonl
│       ├── qa_test_200_seed42.jsonl
│       └── doc_claim_pairs.jsonl
├── docs/
│   ├── label_mapping.md
│   └── manual_check.md
├── models/
│   └── Bespoke-MiniCheck-7B/
├── outputs/
│   ├── minicheck_results.jsonl
│   ├── nli_results.jsonl
│   ├── merged_results.jsonl
│   ├── metrics_summary.json
│   ├── metrics_summary.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── cases.jsonl
│   └── logs/
├── scripts/slurm/
│   ├── download_minicheck.slurm
│   ├── test_minicheck.slurm
│   └── run_minicheck_50.slurm
├── src/
│   ├── prepare_ragtruth.py
│   ├── split_claims.py
│   ├── run_baseline.py
│   ├── merge_judges.py
│   ├── evaluate_results.py
│   └── verification/
│       ├── nli_judge.py
│       └── minicheck_judge.py
├── tests/
│   ├── test_minicheck_judge.py
│   ├── test_split_claims.py
│   ├── test_label_projection.py
│   ├── test_merge_judges.py
│   └── test_evaluate_results.py
├── requirements-nli.txt
├── requirements-minicheck.txt
└── README.md
```

---

## 6. 推荐修改顺序

### 第一步：先固定实验数据接口

实现 `split_claims.py`、claim 级标签投影和 `doc_claim_pairs.jsonl`。在这一步完成前，不建议继续扩展 Judge，因为不同 Judge 需要读取同一份输入才能公平比较。

### 第二步：让两个 Judge 输出相同 schema

MiniCheck 和 NLI 都输出 `pair_id + pred_label + probabilities/scores + metadata`，模型专属字段可以保留在子对象中。

### 第三步：实现组合判断和人工复核队列

先使用课件给出的确定性组合表，保存所有分歧，不要在无记录的情况下覆盖模型结果。

### 第四步：补齐 claim 级评价

生成 Accuracy、Macro-F1、label-wise F1 和三分类混淆矩阵，同时继续保留当前回答级和 span 级指标。

### 第五步：集群恢复后运行 1 条和 50 条

先验证单样本格式和显存，再运行 50 条小闭环。确认结果文件、日志和案例均可生成后，才扩大到 500 条开发集。

### 第六步：冻结配置后再运行 test

所有阈值、证据长度、Top-k、融合规则和 claim 切分版本都只在 train 派生开发集上确定。当前 200 条 test 子集已经查看过结果，不再用于调参。

---

## 7. 第二节结果包验收标准

第二节任务完成时，项目至少应满足以下条件：

- [ ] 50 条 response 样本可以稳定转换成 claim pair；
- [ ] 任意 pair 可以回溯到 qid、原回答字符区间和 evidence id；
- [ ] 人工抽查的 5 条标签投影记录完整；
- [ ] MiniCheck 与 NLI 使用完全相同的 pair 输入；
- [ ] 两个 Judge 的结果可以按 `pair_id` 无损合并；
- [ ] 分歧样本进入 case review，不被静默覆盖；
- [ ] 指标包含 claim 级 Accuracy、Macro-F1、逐类 F1 和混淆矩阵；
- [ ] 现有回答级和字符 span 指标仍可复现；
- [ ] 每次实验都有 run id、完整命令、模型版本、数据版本、耗时和资源记录；
- [ ] 结果包包含课件第 60 页要求的 pair、两个 Judge 结果、指标和案例文件。

完成这些项目后，当前仓库就与第二节课件的 baseline 设计基本对齐，并具备进入第三节证据过滤、Top-k、重排和 atomic fact 实验的稳定接口。
