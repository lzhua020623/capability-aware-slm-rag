# NQ Train 初标接入

2026-10-22 从团队 Drive 的 [NQ_15批](https://drive.google.com/drive/u/0/folders/1JWLByrI_hZ1qv70TSD23QjOlHngGBP1X) 接收 15 份 `NQ_C1_000` 至 `014_preliminary_annotations.xlsx`。这批是原 74,097 条的**新增初标层**，没有增加独立样本。

- [逐文件核验与来源说明](import_audit_20261022.json)：SHA-256、全量标签数、既有划分分布、工作簿 Summary 原文，以及与既有人审的分歧。
- 本地原始文件：`data/processed/external/nq_train_preliminary_20261022/`，包含 ZIP、15 个未改写的 XLSX 和下载清单。
- 本地全量标准化层：`data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl`。按 ID 关联，保留原标签、文件哈希、sheet 和行号；不含问题、证据或 gold，`runtime_use=false`。
- 标签映射：`Correct`→`correct`；`Evidence Insufficiency`→`evidence_insufficiency`；`Evidence Accessibility Failure`→`evidence_accessibility`；`Evidence Utilisation Failure`→`evidence_utilisation`。

工作簿声明 73,536 条为自动启发式初标、561 条做过选定样本语义审核；未提供足以核验的逐条审核人及审核状态。因此全部以 preliminary 保存，未导入任何新的 `human_verified=true` 标签。不能把这些标签当作人工 gold、模型准确率或确定的错误分布。

此目录与 `nq_train_returned/` 的人工标注回传流程分开。原 JSONL、人工定案、25 条暂缓和实验冻结包均未修改。75 条既有人审与初标有 36 条分歧，仍以既有人审为准；不重新开启已处理的人审事项。详情见 [接入报告](../../../docs/ANSWER_REPAIR_NQ_PRELIMINARY_20261022.md)。

复现：先从 Drive 下载上述工作簿，恢复本地下载清单及 15 个原始 JSONL，然后执行 `python scripts/import_nq_train_preliminary.py`。任何 ID、冻结字段、哈希、数量或标签异常均拒绝发布。输出相同可重跑，内容不同拒绝覆盖。
