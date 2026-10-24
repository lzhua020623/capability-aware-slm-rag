# NQ 全量定位与 provider 验收

`validation.json`：对全部 1,224,119 个段落核对 ID、真实 UTF-8 行边界和正文，并实际 text-mode seek 回读 1,224,119 次；验证 corpus/index/ID/offset SHA-256。没有重新编码全部向量，原 64 段独立向量比对信息保留在报告中。

`preflight/manifest.json`：新的显式 NQ Train 资源注册入口在全部 200 条开发输入上的预检查，未加载 Qwen、未生成答案。源码哈希记录的是验收时工作树；commit 字段为变更前父提交，不能单独用它复现此次新增代码。

`retrieval_replay.json`：开发集前 10 题的真实 BGE → 原 FlatIP → lazy offset provider 重放，**10/10 题的 Top5 ID 及顺序与旧 C1 相同**，Top20 前五项也相同；无 QA 指标。`retrieval_replay.log` 保留实际运行日志。

完整源码登记、环境和回归结果见 [resource_adapter_v1](../resource_adapter_v1/README.md)。原 NQ 资源及恢复文件仍位于被 Git 忽略的 `data/processed/`；资源指纹和相对路径在 [注册表](../../../configs/controller/nq_train_resources_v1.json)。
