# FEVER provider 验收

`query_replay.json/log`：固定前 10 条已曝光历史 C1 查询，真实固定 BGE 快照、原 IVFPQ(nprobe=16)、lazy offset 读取；10/10 的 Top5 ID 及顺序与原 C1 一致，Top20 前五项也一致。未使用历史预测、gold 或新最终题目判断。

`tests.log`：全仓库 726 项 CPU 回归通过。定向测试覆盖 FEVER 的提示/标签回退/一次检索重答、索引与 dataset 分派、资源哈希链、C1 全 900 条虚构工程样本覆盖、源码/输入漂移与中断保存。`initial_targeted_tests.log` 保留旧 smoke 预期 FEVER unsupported 的一次失败；测试已更新为显式验证新 FEVER 搜索与重答，未跳过。

`protocol_check.json`：原 T1 设计与 68 个参考 Git blob 完整；登记 P3/execution/providers/runner 及相关测试的当前源码差异，不回写冻结协议。`data_check.json`：T2 划分及全部 552 个来源指纹不变。

完整执行状态与源码/文件指纹见 `validation.json`。CPU 验证不代表真实 Qwen GPU 推理已经完成，更不代表 QA 指标提升。
