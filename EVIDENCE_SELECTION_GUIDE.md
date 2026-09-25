# dev50 证据选择实验：云端运行指南

本实验固定 [`dev50_pairs.jsonl`](data/ragtruth/eval_dev50_assistant_v1/dev50_pairs.jsonl) 中的 234 条 claim 与助手审定 gold，固定 ModernBERT-large-nli 判断器。`full_source` 使用完整来源作参考；`random_budget`、`bm25_budget`、`bge_budget` 共用来源内候选池，最多选 3 块，ModernBERT 的 **document + claim 总输入不超过 512 tokens**。模型本身的最大输入长度仍设为 2048。Full 不属于等预算对照。

## 运行前准备

1. 将本仓库最新代码和 `data/ragtruth/eval_dev50_assistant_v1/`、`data/ragtruth/processed/sample_50.jsonl` 同步到云端。`models/` 被 `.gitignore` 忽略，模型须单独放置。
2. 确认云端存在完整的 `models/ModernBERT-large-nli/` 和 `models/bge-reranker-v2-m3/`，各含 `config.json`、tokenizer 文件及 `.safetensors` 权重。BGE 模型是 [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)，可下载到本地再上传；脚本只读取本地模型，不会在计算作业中联网下载。
3. 现有 `ragtruth-modernbert` 虚拟环境需能导入 `torch` 与 `transformers`。BGE 使用官方文档支持的 Transformers 交叉编码器推理方式，不强制安装 FlagEmbedding。若想隔离依赖，可设置 `BGE_VENV_PATH` 指向另一已装好 CUDA PyTorch 与 Transformers 的环境。
4. 提交前确保云端已有 `outputs/logs/`。只在登录节点执行 `sbatch`，GPU 模型加载与推理都由 SLURM 作业完成。

先提交 5 条冒烟作业：

```bash
cd /home/kangzj/RAGproject
mkdir -p outputs/logs
LIMIT=5 sbatch scripts/slurm/run_evidence_dev50.slurm
```

检查日志末尾是否出现 `Completed evidence experiment`，并查看该 Job ID 对应目录中的 `metrics_summary.json`。冒烟结果在 manifest 中标为 `smoke_prefix`，**不得当作 234 条实验指标**。通过后提交完整作业：

```bash
cd /home/kangzj/RAGproject
sbatch scripts/slurm/run_evidence_dev50.slurm
```

默认输出位于 `outputs/lesson3/evidence_runs/dev50-<JobID>/`，每次提交均新建目录，脚本拒绝覆盖。可通过环境变量 `NLI_VENV_PATH`、`BGE_VENV_PATH`、`NLI_MODEL_PATH`、`BGE_MODEL_PATH`、`BATCH_SIZE` 改配置；不要在四组之间改 NLI 模型或阈值。

## 输出与核查

| 文件 | 内容 |
| --- | --- |
| `candidate_pool.jsonl` | 每条 pair 的同源候选块、chunk ID、来源内字符位置、文本及 token 数 |
| `<variant>_pairs.jsonl` | **实际**送给 ModernBERT 的 document–claim 输入 |
| `<variant>_selection.jsonl` | 所选 ID、各候选原始排序分数、实际 token、截断标记 |
| `<variant>_results.jsonl`、`<variant>_run_manifest.json` | 三分类预测、概率、复核标记、耗时及模型配置 |
| `bge_scores.jsonl`、`bge_scores.manifest.json` | BGE 原始交叉编码器得分与本地模型指纹；得分不是支持概率 |
| `experiment_manifest.json` | 冻结数据与各中间文件哈希、随机种子、预算及状态 |
| `metrics_summary.json`、`.csv`、`paired_changes.jsonl` | 四组指标和逐 pair 的改对／改错记录 |

日志若提示缺模型、缺 CUDA、输入哈希不符或重复输出，应修正原因后以**新 Job ID**重提，不改旧结果。`review_flag` 仅表示需要复核；当前 NLI 每条仍有三分类预测，因此 Coverage 为 100%，复核率单列。局部证据遗漏时不改完整来源 gold。Conflict 仅有 2 条，单列其 support，勿把该类召回解读为稳定结论。

耗时列分别报告排序/预算选择、NLI 推理和两者之和，均为逐 pair 观测；不包含一次性的模型加载和候选池构建。Full-source 的选择耗时记为 0，不能将这列当作完整端到端服务延迟。

本数据是**助手审定开发集**，用于选择预算和分析机制；不是独立人工金标或正式测试集。位置压力测试、语义噪声、Oracle 和 `k=1/5` 扫描应在本轮真实结果出来后另建实验版本。
