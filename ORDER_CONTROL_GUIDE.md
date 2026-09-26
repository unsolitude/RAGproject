# dev50 证据顺序对照

## 目的与当前交付

固定 Job 39484 中每个方法已选中的 chunk 集合、chunk 文本、标识、claim、gold、ModernBERT 权重及阈值，唯一干预是拼接顺序。不重新运行 BM25/BGE，不重新选择证据，也不补齐遗漏片段。

已在本地生成 234 条 × 7 组实际 pair 输入，目录为 `outputs/lesson3/order_runs/dev50-39484-prepared/`。状态仅为 `prepared`，**尚无新增模型预测或顺序实验指标**。原始结果与冻结数据未修改。token 上限必须由云端真实 tokenizer 在推理前核验，未核验时不得评价。

| 组别 | 输入 | 作用 |
| --- | --- | --- |
| `random_budget_rank` / `random_budget_source` | 同一组随机选中块；原排序 / 来源顺序 | 纯顺序配对 |
| `bm25_budget_rank` / `bm25_budget_source` | 同一组 BM25 选中块；原排序 / 来源顺序 | 纯顺序配对 |
| `bge_budget_rank` / `bge_budget_source` | 同一组 BGE 选中块；原排序 / 来源顺序 | 纯顺序配对 |
| `full_chunked_source` | 所有候选块按来源顺序拼接 | 分块呈现参考；含标题变化及可能的重叠重复，不是纯顺序实验 |

来源顺序指原 document 中的 passage 顺序，再按 chunk 起止位置排序，**不是按相关性分数，也不是字符串排序**。需改变顺序的 pair 数：Random 191、BM25 178、BGE 188；其余原本已按来源顺序，用作相同输入稳定性检查。

三组 budget 的总输入均须不超过 512 tokens；完整分块参考须不超过 2048。超限时报错，不能悄悄截断或删除 chunk 来“凑预算”。NLI 本身 max_length=2048，阈值沿用 0.5/0.5，复核阈值 0.7。

## 云端运行

同步代码后，在云端项目根目录操作。脚本直接使用 `ragtruth-modernbert/bin/python`，无需手动激活环境，也不需要 BGE 环境。

```bash
cd /home/kangzj/RAGproject
mkdir -p outputs/logs
sbatch --export=ALL,LIMIT=5 scripts/slurm/run_order_control_dev50.slurm
```

默认历史结果位置为 `/home/kangzj/RAGproject/outputs/lesson3/evidence_runs/dev50-39484/`。若你的云端与本地一样放在 `outputs/evidence_runs/`，改为：

```bash
sbatch --export=ALL,LIMIT=5,SOURCE_RUN=/home/kangzj/RAGproject/outputs/evidence_runs/dev50-39484 scripts/slurm/run_order_control_dev50.slurm
```

冒烟通过后提交完整作业，**显式清空 LIMIT**，避免继承 shell 中残留变量：

```bash
sbatch --export=ALL,LIMIT= scripts/slurm/run_order_control_dev50.slurm
```

如果使用自定义 `SOURCE_RUN`，完整作业同样携带该设置。每次生成新 Job ID 目录，不覆盖任何历史预测。部分作业失败时保留目录诊断，修复后提交新作业。

默认输出：`outputs/lesson3/order_runs/dev50-<JOBID>/`。日志：`outputs/logs/order-dev50-<JOBID>.log`。请把整份输出目录和日志下载到本地，保留历史 `dev50-39484/` 以便核验。

## 结果验收

- `order_manifest.json`：完整作业应为 `predicted`、234 条，`token_validation=passed_real_tokenizer`，七组模型 revision 一致。
- `order_metrics.json`：七组完整指标、各方法 rank→source 的翻转率、改对/改错数、复核率、实际 token。
- `order_changes.jsonl`：每条 pair 的前后标签和概率，以及输入顺序是否真正发生改变。
- `historical_rank_drift`：原排序重跑与历史预测的差异。若出现较多差异，先检查运行环境/数值稳定性。
- `identical_input_flips`：未换顺序却翻转的数量，应优先排查，不能归因于顺序。

主要比较必须是**同一方法、同一次作业内**的 rank 与 source。不同方法之间还可能改变证据集合；完整来源与完整分块还会改变标题、重复内容。conflict 仅 2 条，按描述性结果报告；此实验不是正式开头/中间/末尾关键证据压力测试。

## 本地只读检查与准备

```powershell
python -B scripts/run_order_control.py prepare --source-run outputs/evidence_runs/dev50-39484 --output-dir outputs/lesson3/order_runs/dev50-new-prepared
python -m unittest tests.test_order_control -v
```

输出目录需不存在。准备阶段不加载模型，不需要 PyTorch；完整推理需云端环境。程序核对来源 manifest、文件哈希、冻结 gold、chunk 原文位置、选中集合及预测与输入的一致性，不允许从有缺失记录或被修改过的历史实验开始。
