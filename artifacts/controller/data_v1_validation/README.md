# T2 验证记录 · 2026-10-23

- `tests.log`：完整 CPU 回归，665 项通过；本次新增 17 项。
- `data_validation.json`：已冻结样本、引用、保留池与源文件哈希检查，不计算 EM/F1 或新收益。
- `readiness_check.json`：`--require-full-t2` 按预期返回 2；FEVER 新 C1 和匹配索引资源仍待封存。
- `branch_resource_inventory.json`：11 个远端分支的 C1/资源文件树核查，包括 controller；未跟踪 `.index`/`.npy`。不把被 Git 忽略的资源或档案内部推断为不存在。
- `validation.json`：测试及检查命令、结果与产物哈希。

这不是 GPU 实验结果，不能据此报告 QA 提升、完整知识库干净或新 FEVER C1 已完成。
