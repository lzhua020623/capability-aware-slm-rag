# NQ Train 检索资源恢复 · 2026-10-24

**原语料、原索引已取回，ID/offsets 已恢复，64 处独立 BGE 向量比对通过。** 来源仍为团队已有 NQ Train 子集及其配套检索库，没有下载全量官方 NQ，没有重算全部向量。

| 核验项 | 结果 |
|---|---|
| corpus / FAISS 向量数量 | 1,224,119 段；不是问题样本数量 |
| 索引 | IndexFlatIP，768 维，全部向量有限且 L2 归一化 |
| ID 和 offsets | 已生成；原顺序保留，64 处原文定位通过 |
| 抽样向量比对 | 64/64 通过；最小余弦 0.9999998808，最大分量差 3.39×10⁻⁷ |
| BGE 验证快照 | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，CPU；不冒充历史 C1 revision |
| 最新回归 | **681 项通过**，含新增 16 项恢复测试 |
| 原数据冻结检查 | 552 项来源哈希核对通过，样本/保留池未改 |

生成文件在仓库本机的 `data/processed/indexes/`，受 Git 忽略。便于迁移的 [nq_train_sidecars.tar.gz](nq_train_sidecars.tar.gz) 约 6.56 MB，只包含恢复后的三个配套文件；不包含原语料、原索引或模型权重。包内成员及内容 SHA 均已核验。

迁移到新的运行环境时，先取得相同指纹的原 corpus/index，再解压配套包。以下命令拒绝覆盖已有文件：

```bash
mkdir -p data/processed/indexes
tar -xkzf artifacts/controller/nq_index_recovery_v1/nq_train_sidecars.tar.gz -C data/processed/indexes
```

关键记录：

- `recovered_metadata.json`：正式恢复 metadata 的副本，记录输入/产物 SHA、读取模式及验证数值。
- `validation.json`：运行命令、CPU 环境、测试结果、包指纹及仍待完成的条件。
- `resource_discovery.json`：Drive 来源与原文件指纹。
- `crash_diagnosis.json`：最小化的 macOS 崩溃摘要；原始完整系统报告不入库。
- `recovery_mmap.log`：成功的实际恢复日志。
- `recovery.log`：初次合并进程退出 139 的失败日志；崩溃栈位于 libomp。
- `recovery_isolated.log`：应用中断前的尝试，退出状态不明，不记为成功或同因崩溃。
- `structural_diagnostic.log` / `structural_diagnostic_metadata.json`：仅结构恢复的诊断记录，不能当作通过独立向量比对。
- `tests.log` / `targeted_tests.log`：当前版本的全量及专项测试。
- `tests_before_runtime_fix.log`：修正运行方式前的历史回归记录。
- `data_freeze_validation.json`：冻结 T2 产物和来源保持一致；其资源缺口描述仍是原冻结状态。

本次通过只读 mmap 和独立单线程 CPU 编码进程完成恢复，未要求更换 CPU/GPU；原崩溃报告不支持将故障直接归因为内存耗尽。没有运行 Qwen QA 或计算新的准确率。抽样通过不代表全量逐段重新编码核验，也不代表完整知识库经过人工清洗。

FEVER 包内容、NQ Train Controller provider 适配、新 FEVER C1 与正式 GPU 执行验收仍待完成，`full_T2_ready` 保持 false。详情见 [恢复说明](../../../docs/NQ_TRAIN_INDEX_RECOVERY.md)。
