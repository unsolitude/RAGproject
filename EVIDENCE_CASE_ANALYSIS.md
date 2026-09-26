# dev50 五个可回溯案例

分析时间：2026-09-26。仅使用真实 Job 39484 的既有预测，未生成新 GPU 结果，未修改冻结 gold。案例按机制挑选，不代表总体错误率。

原始结果：[完整实验](outputs/evidence_runs/dev50-39484/experiment_manifest.json)。机器可读全文、chunk 位置、结果行号和三类概率见[案例包](outputs/lesson3/cases/dev50-39484/cases.jsonl)。下表的 S/C/U 分别为 supported/conflict/unsupported。

| pair_id | 冻结 gold | Full | Random | BM25 | BGE | 主要观察 |
| --- | --- | --- | --- | --- | --- | --- |
| 11876_c01 | S | U | S | S | U | 同三块不同顺序，跨过支持阈值 |
| 14244_c05 | S | S | U | S | U | 首块相同，仅两块背景顺序不同 |
| 14959_c02 | S | S | S | U | S | 两方法均保留关键指令，不能归因于缺证 |
| 17621_c10 | S | S | S | U | S | BM25 漏证，BGE 选回明确支持句 |
| 13030_c06 | C | U | U | U | U | 跨句对比缺失；完整来源仍未识别冲突 |

## 1. 乳脂含量：相同证据集合的排列敏感性

**Claim**：`Based on the given passages, we can say that the primary difference between single cream and double cream in Britain is the butterfat content`

来源明确写有 single cream 为 10–12% butterfat、double cream 为 48%，并写明 `Double cream has a higher fat content than single.`。两种排序均选中全部三块，文本与 chunk 标识相同。

- BM25 顺序：`passage_3_c001 → passage_1_c001 → passage_2_c001`；entailment=0.521557，预测 S。
- BGE 顺序：`passage_1_c001 → passage_2_c001 → passage_3_c001`；entailment=0.337089，预测 U。

**判断**：这是顺序敏感性的明确观察，不能解释成 BM25 找到了 BGE 没找到的证据。两者支持概率跨过 0.5 阈值。Full→分块还改变了 Evidence 标识，不能直接作为纯顺序因果对照；需要新作业内重跑验证稳定性。

## 2. 呼吸管：关键句都在，背景顺序改变输出

**Claim**：`This design is meant to keep the snorkel dry even when underwater`

passage 3 描述浮子闭合并 `seals it which prevents water from filling the tube.`。passage 1 谈潜艇发动机，passage 2 是面罩套装推荐，均为背景。

- BM25：`p3_c001 → p1_c001 → p2_c001`；entailment=0.530068，预测 S。
- BGE：`p3_c001 → p2_c001 → p1_c001`；entailment=0.181282，预测 U。

这里 p1/p2/p3 是对应 `passage_1/2/3` 的简写。两组不仅选中集合相同，关键证据还都放在第一块。

**判断**：不能简单归为“关键证据被埋在中间”，背景排列也可能影响模型。另一个限制是 `This design` 仍有指代性：本轮保持冻结 gold 不改，但这个案例也提醒我们不要把所有不稳定性都归咎于模型结构。

## 3. 贝果加热：证据在场，仍然误判

**Claim**：`Another option is to dip the bagels in water, shake off the excess, and heat them uncovered in the oven for 10 to 15 minutes`

关键原句：`Dip bagels in water, shake off, then heat uncovered for 10 to 15 minutes.`

- BM25 选 p2_c001、p2_c002、p1_c002，entailment=0.437199，预测 U。
- BGE 选 p1_c001、p2_c001、p2_c002，entailment=0.696195，预测 S。
- 关键句在两组输入中均存在，并非 BM25 完全没找到支持材料。

**判断**：只能将它归为“证据存在但判别失败”。候选块有重复指令和分块重叠，可能造成上下文敏感性；但排序与内容同时变化，尚不能证明具体原因是噪声、重复或位置。先做固定集合的顺序对照，再做去重复等独立消融。

## 4. 多层涂漆：可定位的证据遗漏

**Claim**：`Apply multiple coats as needed`

来源 passage 3 明确提到 `second and, if need be, third coat`。

- BM25 选 p1_c001、p1_c002、p2_c001：视频介绍、许可信息和打磨说明，缺少涂层次数；entailment=0.023622，预测 U。
- BGE 选 p3_c001、p3_c002、p2_c001：保留第二/第三层涂漆说明；entailment=0.547810，预测 S。

**判断**：这是有实证支持的 `evidence_missing` 案例。BM25 在局部输入上输出 U 可以理解，但整个“选择＋核验”任务仍按完整来源 gold=S 计错，不能为了局部缺证改 gold。BGE 在该例的收益与关键句被保留一致；不是所有 BGE 改对案例都能这样解释。

## 5. 吉他与钢琴：分块切断反证，完整输入仍错

**Claim**：`Simply subtract those notes that cannot be played together to create a piano-friendly version of the chord`

passage 1 描述吉他因弦排列需要删音，随后说 `The piano has all five.`。冻结的助手审定 gold=C，理由是建议把吉他的删音限制错误转向钢琴。

- BM25/BGE 选中了 p1_c001，但该块在吉他只能同时奏四个不同音的句子处结束，未保留钢琴有五个音的对比句。
- Random 未选 passage 1，亦缺少核心对比。
- Full 包含完整反证，contradiction 仍只有 0.213891，预测 U；BM25 为 0.224321，BGE 为 0.154056，也均未达到 0.5。

**判断**：同时存在两层问题：预算选择没有保留完整跨句对比；Full 即使读到对比也未判冲突。不能说“选对证据就一定解决”。该 claim 的 `those notes` 有上下文依赖，C 也是助手裁决而非独立人工终审；本报告不把这一个难例当作稳定矛盾检测能力的证据。

## 已完成与尚待验证

这五条已逐条检查来源、实际选择和概率。新增顺序对照的输入与脚本已准备，见[运行指南](ORDER_CONTROL_GUIDE.md)；**新顺序对照的指标尚未产生**。暂时可以说“存在顺序敏感和缺证案例”，不能提前说“恢复原文顺序提升了整体分数”。
