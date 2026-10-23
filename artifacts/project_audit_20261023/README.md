本目录保存 2026-10-23 的跨分支与数据审计。结论及后续设计见 [项目审计与 RSI 计划](../../docs/PROJECT_AUDIT_AND_RSI_PLAN_20261023.md)。

- `manifest.json`：源提交、检查范围、环境、日志与产物 SHA-256。
- `branches.json`：10 个远端分支和 4 个本地分支的提交、文件清单、相对 main 的差异。
- `preliminary_artifact_inventory.json`：跨分支预实验文件的 Git blob 一致性。
- `nq_delivery_audit.json`：74,097 条 NQ Train 交付及标准化初标的结构、来源、重复和划分检查。
- `cross_branch_split_audit.json`：与已发布 Diagnosis NQ 划分的 ID 和规范化问题交叉检查。
- `split_variant_groups.json`：17 组跨划分 Unicode／标点变体，35 个 ID；不含题干或答案，原 split 未修改。
- `*_tests*.log`：原始测试日志，包括初始环境问题及在正确环境中的重跑。最终 435 项通过；不是 GPU 效果实验。
- `audit_data.py`：只读的数据扫描脚本，仅依赖 Python 标准库，需要已接收的本地原始分片、标签层和分区文件。

重跑数据检查时使用新输出目录，不覆盖本审计快照：

```sh
python artifacts/project_audit_20261023/audit_data.py /absolute/path/to/capability-aware-slm-rag /tmp/a-new-data-audit-output
```

本审计没有修改原始数据、知识库、索引、标签、分区或算法。全 NQ corpus 不在配置本地路径，FEVER corpus 为云盘占位；语义正确性、全库完整性、页面／语义近重复隔离以及预训练污染不在已通过检查的声明范围内。
