# 历史文档归档

2026-09-26 整理。以下文件保留原始内容，移动前后 SHA-256 一致；其中的“当前进度”、路径和命令均代表当时版本，不应直接作为当前运行指南。

| 原位置 | 归档位置 | 归档原因 |
| --- | --- | --- |
| 根目录 `LESSON2_PROJECT_GAP_AND_TASKS.md` | [第二节任务快照](docs/2026-09-26/LESSON2_PROJECT_GAP_AND_TASKS.md) | 前期差距与实施记录，当前已进入第三节实验 |
| `docs/PROJECT_STATUS_2026-09-19.md` | [9 月 19 日状态](docs/2026-09-26/docs/PROJECT_STATUS_2026-09-19.md) | 旧 NLI、414 条原始 pair 的历史状态 |
| `docs/learning/07_metrics_and_current_results.md` | [旧指标教材](docs/2026-09-26/docs/learning/07_metrics_and_current_results.md) | 保留公式与历史算例，数值不代表当前冻结开发集 |
| `docs/learning/09_progress_and_research_roadmap.md` | [旧研究路线](docs/2026-09-26/docs/learning/09_progress_and_research_roadmap.md) | 其中部分“未完成”任务现已完成 |

当前入口为 [README](../README.md)、[项目目录](../PROJECT_DIRECTORY.md)、[dev50 审计报告](../DEV50_AUDIT_REPORT.md)和[证据选择运行指南](../EVIDENCE_SELECTION_GUIDE.md)。两篇学习文档原位置保留新的简短导航，不丢失学习目录入口。

## 冒烟输出

`outputs/evidence_runs/dev50-39483/` 已整体移至 `outputs/archived/smoke_runs/dev50-39483/`，23 个文件均未改写。该运行仅有 5 条 pair，完整的 234 条实验保留在 `outputs/evidence_runs/dev50-39484/`。

运行 manifest 中的云端路径是历史来源信息，不随本地归档修改。`outputs/` 仍被 Git 忽略，需要单独同步；本目录中的文档可以纳入 Git。

## 恢复原则

按上表反向移动即可恢复。恢复前确认目标不存在；学习入口已有新版时先另外备份，不覆盖。不要为了匹配当前位置而修改历史清单哈希。原始数据、模型、冻结 pair、正式实验、单元测试和可复用的冒烟脚本均未删除或移动。
