# Controller 2 集成验证 · 2026-10-27

来源与范围见 [交接](../../../docs/CONTROLLER2_INTEGRATION.md) 和 `merge_sources.json`。

- 补检索专项：50 项通过，`retrieval_tests.log`。
- 重排与集成专项：329 项通过，`reranking_tests.log`；包括 6 项新增合并边界回归。
- 初次检查发现的历史包测试来源设置、NLTK/pandas 缺失及修正前日志一并保留；没有放宽历史源码哈希校验。
- `before.json` 记录开工状态及用户文件、运行配置的哈希。
- 完整回归和收工自检在合并提交后运行，最终记录为 `full_tests.log`、`self_check.json` 和 `environment.json`；以最终验证提交为准。

本轮不生成教师数据、不训练、不启用新模型权重、不读取统一 test 的题目或 gold。旧冻结配置/包保持原样，P60 缺失材料和 E unavailable 继续如实登记。
