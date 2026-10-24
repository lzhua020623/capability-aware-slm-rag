# FEVER 团队资源验收

- `import_validation.json`：两个原 ZIP 的大小/SHA-256、选定成员的 CRC/哈希验收、全量 5,396,106 条 corpus/ID 顺序与文本检查、offsets 指纹。
- `original_metadata.json`：从原包取出的 601-byte 建库记录，与仓库原 `results/retrieval/fever_bge_ivfpq_meta.json` 字节一致。
- `structure_validation.json`：实际加载 IndexIVFPQ，核对数量、维度、IP、PQ 参数、nprobe、by_residual；三条重建 PQ 向量搜索 Top20 后回读原文。
- `import.log` / `structure.log`：对应实际执行日志。

本机资源位于 `data/processed/controller_resources/fever_v1/`，大型资源和 offsets 不进原始 Git。复现命令见 [接入说明](../../../docs/CONTROLLER_RESOURCE_ADAPTER_V1.md)。两个原 ZIP 保留在本机 Downloads，来源为团队 Drive；需要迁移时复用原包并按哈希核对。

`import_validation.json` 的 remaining 是导入完成当时的记录；其后 structural/search 检查已通过。FEVER 原 BGE revision 仍未记录，本轮不声称做过语义 BGE 兼容验收或问答质量评测。ZIP 的旧 fever_dev 只记指纹，不替换 T2 冻结来源。后续还需 FEVER provider、新 C1 与正式 execution seal。
