# Controller integration v1

本目录保存 2026-10-23 `controller` 首轮集成的来源与工程验证，不是新的 benchmark 结果。

- `branch_manifest.json`：10 个远端源提交均已合并；764 个唯一历史数据／结果／模型／划分路径的 Git blob 对比。
- `initial_merge_tests.log`：合并后的首轮 606 项测试，8 个兼容性失败的原始记录。
- `tests.log`：修复并补充集成／评分测试后的完整 635 项通过记录。
- `cpu_test_environment.json`：实际 CPU 回归环境；并非 CUDA 实验环境验收。
- `smoke/`：两个虚构输入 × 五种模式的执行轨迹与清单；假 provider 输出，不计作 NQ／FEVER 质量结果。

方法、复现命令与限制见 [集成交接](../../../docs/CONTROLLER_INTEGRATION.md)。历史 E 人工记录、预测、数据划分均未改写。
