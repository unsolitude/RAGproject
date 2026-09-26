# dev50：k=1/3/5 与关键证据保留诊断

## 1. 本轮回答什么问题

固定同一份 234 条助手审定 dev50、原始候选块及排序分数，检查“多选一些证据是否改善判别，以及改善是否伴随已标注关键材料被保留”。不改 claim、gold 或模型，也不把开发集称为独立人工金标。

前置顺序实验 Job 39496 已完成：BM25 与 BGE 的原文顺序 Macro-F1 分别为 0.6262、0.6316，高于排名顺序的 0.6028、0.5994；Random 则从 0.5264 降至 0.5205。因此本轮统一原文顺序，作为开发阶段暂定协议，不声称顺序对所有方法都有益。

| 固定项 | 本轮设置 |
| --- | --- |
| 数据 | `data/ragtruth/eval_dev50_assistant_v1/dev50_pairs.jsonl`，234 条：191 supported / 2 conflict / 41 unsupported |
| 判别器 | `tasksource/ModernBERT-large-nli`，与 Job 39484 的本地权重/分词器指纹一致 |
| 分类和复核 | 沿用 entailment/contradiction 阈值 0.5、review 阈值 0.7；review 不是拒判 |
| 方法 | Random、BM25、BGE，各 k=1/3/5，共 9 组 |
| 参照 | 另跑 Full source、Full chunked source，共 11 组、2574 条预测 |
| 候选与排名 | 完整复用 Job 39484；Random 同一固定 seed，BGE 不重新加载或推理 |
| 拼接 | 按分数取前 min(k, 候选数) 块，再按原文位置排列 |
| 输入上限 | **2048 总 token 安全上限，超出直接报错，不静默截断** |

重要变化：本轮取消前三种方法的 512-token 选择预算。否则 k=5 容易变成“仍只放得下 3 块”，无法清楚观察块数变化。这是**不同 k、不同实际 token 数的对照**，不是等预算实验；原先 512-token 四组结果继续保留。当前历史 k=3 均为实际 top-3，脚本会核对该桥接条件，保证新 k=3 与旧顺序实验选择集合一致。

234 条中有 155 条候选池不足 5 块（107 条只有 3 块、48 条只有 4 块）。k=5 不补空块、不重复填充，而记录 `actual_k` 和 `candidate_shortfall`。两种 Full 也不是纯顺序对照：分块标题和重叠可能改变文本。

## 2. 关键证据是什么、如何统计

标注文件：[`configs/dev50_key_evidence_v1.json`](configs/dev50_key_evidence_v1.json)。这是**助手标注、已看过预测、目的性选取的诊断子集**，不是 234 条全量证据金标，也不能据此估计总体证据召回率。

| pair_id | 检查材料 | gold |
| --- | --- | --- |
| 11876_c01 | single/double cream 脂肪含量比较 | supported |
| 11876_c05 | 脂肪比较及使酱汁变稠的机制 | conflict |
| 14244_c05 | dry snorkel 浮动阀门关闭及阻水机制 | supported |
| 14959_c02 | bagel 浸水、抖掉水及不覆盖加热的操作 | supported |
| 17621_c10 | 第一层漆干后再涂后续层 | supported |
| 13030_c06 | 来源对吉他和钢琴和弦音符数量的比较 | conflict |

六条 claim 共 8 个必需文本单元。标的是**来源文档中的文字及字符位置**，不是回答中的幻觉 span。源文档出现重复或等价证据时，一个单元允许多个替代位置（OR）；一条 claim 的全部单元均保留才算“完整标注证据保留”（AND）。选中块的字符区间取并集，忽略块边界的纯空白间隙，不忽略缺失的实词。所有引文必须精确匹配冻结来源，否则停止。

两个指标：

- `key_unit_retention`：保留的标注单元数 / 此次实际覆盖的标注单元数；完整运行分母为 8。
- `complete_annotated_evidence_rate`：所有标注单元均保留的 claim 数 / 此次标注 claim 数；完整运行分母为 6。

这些标注只在离线诊断中使用，不传入排序查询、模型输入或金标签。它们描述文本覆盖，不自动证明逻辑充分性；没保留标注位置也不排除其他等价证据。两条 conflict 标的是来源中的反驳材料。未给 unsupported 强造正证据。

本地已完成的无 GPU 覆盖检查如下（不是新模型效果）：

| 方法 | k=1 单元保留 | k=3 单元保留 | k=5 单元保留 |
| --- | --- | --- | --- |
| Random | 5/8 | 6/8 | 7/8 |
| BM25 | 6/8 | 6/8 | 8/8 |
| BGE | 7/8 | 7/8 | 8/8 |

预测完成后再交叉查看“标注保留/缺失 × 判对/判错”。不能仅凭这四格把所有错误归因于选择器或 NLI；它们用于定位进一步阅读的案例。

## 3. 云端提交

先同步本轮代码、`configs/` 和测试。无需安装新依赖，也无需再运行 BGE。必须保留 Job 39484 **完整目录**，包括原始四组预测、输入、选择记录、候选池和 manifest；仅有 CSV 不够。冻结 dev50 文件必须与旧运行一致。脚本自动使用 ModernBERT 环境的 Python，不需要先激活环境。

在云端 SSH 终端（不是本地 PowerShell）运行：

```bash
cd /home/kangzj/RAGproject
mkdir -p outputs/logs
export SOURCE_RUN=/home/kangzj/RAGproject/outputs/lesson3/evidence_runs/dev50-39484
test -f "$SOURCE_RUN/experiment_manifest.json"
sbatch --export=ALL,LIMIT=5 scripts/slurm/run_k_sweep_dev50.slurm
```

若服务器旧结果实际位于 `outputs/evidence_runs/dev50-39484`，只修改 `SOURCE_RUN`，不要复制或改写旧 manifest。查看任务（将 JOBID 替换为提交返回值）：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -f outputs/logs/k-dev50-JOBID.log
```

五条冒烟全部成功后提交完整实验，显式清空 LIMIT：

```bash
sbatch --export=ALL,LIMIT= scripts/slurm/run_k_sweep_dev50.slurm
```

输出为 `outputs/lesson3/k_runs/dev50-JOBID/`，每个 Job 独立目录，不覆盖历史文件。冒烟是固定前 5 条，各组只预测 5 条，诊断分母可能不足 6，不能当完整开发集结果。

如显存不足，用新 Job 降低 batch：

```bash
sbatch --export=ALL,LIMIT=,BATCH_SIZE=1 scripts/slurm/run_k_sweep_dev50.slurm
```

不手动设置 CUDA_VISIBLE_DEVICES，由 SLURM 分配 GPU。日志目录必须在提交前存在。哈希失败应恢复正确原文件，不能删除校验或手动改哈希。部分预测已写入的目录拒绝继续覆盖：修复问题后提交新 Job，保留失败日志定位原因。

## 4. 输出、验证与下载

| 文件 | 用途 |
| --- | --- |
| `k_manifest.json` | 冻结来源、模型指纹、k 协议、实际状态和输入/预测哈希 |
| `*_pairs.jsonl`、`*_selection.jsonl` | 每组真实输入、完整排名、选中块和实际 k |
| `token_counts.json` | 真实分词器测得的 document + claim + 特殊 token 总长度 |
| `*_results.jsonl`、`*_run_manifest.json` | 11 组实际推理及各自运行摘要 |
| `annotation_spec.json`、`resolved_annotations.json` | 标注副本、精确来源字符位置与来源文本哈希 |
| `evidence_retention.jsonl`、`evidence_retention_summary.json` | 不依赖预测的文本保留统计 |
| `k_metrics.json`、`k_metrics.csv` | 分类指标、逐类 support/F1/recall、token、复核率、耗时和诊断指标 |
| `k_changes.jsonl` | 同方法不同 k、同 k 的 BM25/BGE 及两种 Full 的逐 ID 改对/改错 |
| `evidence_retention_with_predictions.jsonl` | 标注保留情况与预测概率、对错的联表 |

完整验收：11 组各 234 条、同一 ID/claim/gold；总预测 2574 条；诊断联表 66 条；安全分词检查通过；不截断；Coverage=100%，复核率单报；status=`predicted` 且评价文件存在。`k_manifest` 的状态描述推理完成，评价状态另见 `k_metrics.json`。排名成本复用旧缓存，耗时仅用于本轮 NLI，不作为检索端到端成本比较。

可单独检查下载后的目录（下面使用本地目录示例；将 JOBID 替换为实际值）：

```powershell
python -B scripts/run_k_sweep.py validate --source-run outputs/evidence_runs/dev50-39484 --output-dir outputs/lesson3/k_runs/dev50-JOBID
```

`validate` 检查哈希、冻结数据、选择和标注一致性；提交脚本中的 `evaluate` 还会严格校验预测/manifest 并生成评价，不覆盖已有报告。请下载整份新 Job 目录及 `outputs/logs/k-dev50-JOBID.log`；不要只下载指标表。

无 GPU 本地准备命令（输出目录须不存在）：

```powershell
python -B scripts/run_k_sweep.py prepare --source-run outputs/evidence_runs/dev50-39484 --output-dir outputs/lesson3/k_runs/local-check --limit 5
python -B -m unittest discover -s tests
```

已准备的完整输入位于 `outputs/lesson3/k_runs/dev50-39484-prepared/`，不含伪造预测。实际 token 验证留到云端加载原模型分词器后执行。

## 5. 如何解释下一批结果

先看 k=3 是否复现前次原文顺序结果，再看 k=1→3→5 的 Macro-F1、actual_k、token 数、改对/改错 ID；对六个诊断案例结合原始全文判断是否找回限定词、因果或反驳材料。k 增大保留率应单调不降，但模型准确率未必如此。

目前限制：单个 Random seed；conflict 仅 2 条；六案例选择并非随机或盲审；k=5 候选池短缺严重；k 与实际文本量共同变化。不能据此得出统计稳定的最优 k，也不应用测试集调参。若趋势值得继续验证，后续独立扩充证据标注、多 seed 与候选较丰富样本分层，再推进主测试集。
