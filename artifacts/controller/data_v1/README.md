# Controller T2 数据包 v1

使用说明和限制见 [数据冻结文档](../../../docs/CONTROLLER_DATA_V1.md)。

- NQ 和 FEVER 各 200 开发、200 选型、500 最终评测子集。
- 完整保留池为 NQ 7,564（10.21%）、FEVER 1,886（有效候选的 14.26%）。`evaluation_reserve_ids.json` 中全部 ID 均禁止用于训练/调参，未抽中的记录继续保留。
- `nq/` 是实际 C1 运行时输入；`fever/` 是待生成 C1 的 query-only 任务。
- `offline/` 是独立参考答案/标签，禁止进入 Controller 或模型提示。
- `final_candidate/` 仅最终版本使用，不能用于开发排错。文件名不表示可以在其他用途使用。
- `manifest.json` / `policy.json` / `exposure_audit.json` / `exclusions.jsonl` 保存来源、抽样、曝光及排除理由。
- `asset_inventory.json` 记录缺失/云占位资源，未验收匹配索引和 BGE revision。

生成器拒绝覆盖已有目录。注册入口为 [controller_data_v1.json](../../../configs/splits/controller_data_v1.json)，其中绑定 manifest SHA-256。
