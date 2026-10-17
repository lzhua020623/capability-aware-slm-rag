# E 历史归档

这里保存 v1 至 v6r1 的文档、实验设计与早期进度。当前工作基础是 v7，下一轮 v8 尚未实现。当前说明见 [项目进度](../../PROJECT_STATUS_AND_DIRECTION.md)、[E 交接](../../ANSWER_REPAIR_HANDOFF.md) 和 [v7 结果](../../ANSWER_REPAIR_RESULTS_20261014_V7.md)。

## 文档索引

| 文件 | 用途 |
|---|---|
| [ANSWER_REPAIR_RESULTS_20261013.md](ANSWER_REPAIR_RESULTS_20261013.md) | v1 至 v4、Reanswer 对照及 Controller 联调的真实结果 |
| [ANSWER_REPAIR_RESULTS_20261014.md](ANSWER_REPAIR_RESULTS_20261014.md) | v5、v6r1 结果与 128 条辅助审核、关键人工确认 |
| [ANSWER_REPAIR_REVIEW_20261013.md](ANSWER_REPAIR_REVIEW_20261013.md) | 37 条开发候选的选择、辅助审核与来源 |
| [ANSWER_REPAIR_V5.md](ANSWER_REPAIR_V5.md) | 审核后开发集与同模型二次核验的冻结协议 |
| [ANSWER_REPAIR_V6.md](ANSWER_REPAIR_V6.md) | NQ 原文抽取与 FEVER 独立 MNLI 的冻结协议 |
| [STRATEGY_HISTORY.md](STRATEGY_HISTORY.md) | 早期 JSON 格式、引用匹配、候选生成和策略演化 |
| [GPU_RUNS_V1_V6R1.md](GPU_RUNS_V1_V6R1.md) | 历史 Colab、Kaggle 执行、包哈希和 Controller 联调说明 |
| [PROGRESS_UPDATES_20261013_14.md](PROGRESS_UPDATES_20261013_14.md) | 阶段进度、路由试跑、当时结论和数据限制 |
| [HANDOFF_20261014.md](HANDOFF_20261014.md) | 整理前的交接快照，保留旧检查数、接口示例和后续补充 |
| [archive_index.json](archive_index.json) | 本次迁移的旧路径、新路径和前后 SHA-256 |

人工确认口径仍保留在原路径 [ANSWER_REPAIR_CONFIRMATION_20261014.md](../../ANSWER_REPAIR_CONFIRMATION_20261014.md)，以保持人工来源记录的引用稳定。人工记录和 gold 不迁入历史目录。

## Notebook 索引

历史 Notebook 位于 [notebooks/history/answer_repair](../../../notebooks/history/answer_repair/)。移动只改变目录，其内容与移动前完全相同。

| 版本 | Notebook |
|---|---|
| v1 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab.ipynb) |
| v2 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v2.ipynb) |
| v3 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v3.ipynb) |
| v4 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v4.ipynb) |
| v5 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v5_reviewed.ipynb) |
| v6 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v6_reviewed.ipynb) |
| v6r1 | [Colab](../../../notebooks/history/answer_repair/answer_repair_colab_v6r1_reviewed.ipynb) / [Kaggle](../../../notebooks/history/answer_repair/answer_repair_kaggle_v6r1_reviewed.ipynb) |

Notebook 负责加载各自冻结包，不能通过换输入包来冒充新策略。历史构建脚本仍保留，默认输出已指向本目录；已有实验包和 Notebook 不需要重新生成。

## 原始实验材料

输入包、下载 ZIP、逐样本预测、运行日志和审核资料仍保留在仓库根目录下的 `artifacts/answer_repair/`，没有搬移或修改冻结字节：

- `v1/`、`v2/`、`v3/`：早期输入包及原始导出。
- `session_20261013/`：v4 与 Controller 联调的组合原始导出。
- `v5_reviewed/`、`v6_reviewed/`、`v6r1_reviewed/`：审核后开发实验及相应冻结材料。
- `review_20261013/`、`handoff_20261014/`：早期审核来源与交付快照。

准确路径及哈希见 [git_archive_manifest.json](../../../artifacts/answer_repair/git_archive_manifest.json)。复现使用输入包中的冻结代码；文档链接和当前构建脚本的路径调整，不会重写旧包。

## 早期 Reference RAG 工程验证

以下均为 20 条 smoke，验证工程连通性，不能代替 500 条正式预实验；FEVER 使用缩小的受控候选池。

| 历史试跑 | 指标 | 检索与错误情况 |
|---|---|---|
| NQ 3B v2 | EM 0.3000，F1 0.3460 | gold 在 Top-5：17/20；检索未命中 3/20；命中却答错 11/20 |
| NQ 7B | EM 0.4000，F1 0.4770 | gold 在 Top-5：17/20；检索未命中 3/20；命中却答错 9/20 |
| FEVER 3B | accuracy 0.8000 | gold 在 Top-5：20/20；命中却答错 4/20；不是全量语料的正式检索结果 |

原始输出仍在 `results/base_rag/`。这些历史模型规模与当前冻结 Qwen 7B 配置不同，应按各自实验口径解释。
