# 10 月 24 日 Diagnosis 当前主线接入 · 2026-10-27

来源为 `feature/diagnosis-probe-zheng@138d52aa22e0eee78637e01f981a030d7ad388df`。用户要求后续以它为主；该提交已经是 controller 的祖先，无须重复 merge。本轮新增当前基线登记、P64 到真实增强的适配和严格启动入口；没有运行新的 GPU、训练、重新选阈值或最终评测。

## 分支材料核查

[upstream_material_audit.json](upstream_material_audit.json) 来自 GitHub 指定提交的完整 tarball 和未截断递归树，共 **142 个文件、0 个 Notebook**。另有本地 git grep 交叉核对。[upstream_history_audit.json](upstream_history_audit.json) 覆盖该分支全部 **10 个提交**及每次改动清单。

| 材料 | 分支里是否存在 | 核对结果 |
|---|---|---|
| P60 配置、阈值和特征顺序 | 有 | 78 个位置、75 个唯一名称 |
| 三个 HGB 模型 | 有 | 均为 78 维；本地字节与 GitHub 原始 blob 相同 |
| P64 host、Controller、增强协议 | 有 | 本轮沿用，接上真实增强代码 |
| 原始 78 维在线特征实现 | 未找到 | `features_for` 只在 `src/pipeline/adaptive_rag.py` 的 Protocol 和测试 Stub 中定义 |
| P60 训练／调参／校准／错误分析 ID | 未找到 | 旧 Diagnosis 两份 split 存在，但不能当作三个 HGB 的来源；模型无样本 ID 属性 |
| EXP-P26／EXP-P60 训练与特征实验脚本、Notebook | 未找到 | 全文件树和历史改动清单均未见额外交付 |

这只是已入 Git 的范围，不代表作者本地或 Drive 没有。Drive 连接的三次关键词搜索均报 `ACCESS_TOKEN_SCOPE_INSUFFICIENT`；未发送队友消息。来源及权限状态见 [source_receipt.json](source_receipt.json)。

## 已完成的工程接入

当前登记是 `configs/controller/current_baseline.json`。生产入口为 `src.pipeline.current.load_current_pipeline()`，流程为 C1 → Diagnosis → P60 → Controller → EnhancementExecutor，Probe OFF。三个真实动作使用已有模块和 CallLog；不会调用旧 `.run()` 再做一次旧路由。

E 仍 unavailable：接入该动作不等于通过修复验收，原答保留，E 调用为零。特征必须按 78 个原始位置提供，不补零、不去重、不近似替代。生产成本凭据必须绑定当前输入；模型使用 ID 按原始记录核对完整保留池与 selection。

[readiness.json](readiness.json) 明确列出两项原始交付缺口，检查命令正常以退出码 2 停止，**未声称 P60 真实运行已就绪**，也不回退旧路由。该状态是缺少输入，不是模型效果失败。

本轮新建的适配器测试使用真实模块逻辑、真实冻结 HGB 加载和显式虚构的底层输出，仅验证工程行为，不产生任何 benchmark 准确率。新增 17 项针对性测试通过，完整 982 项回归全部通过（452.020 秒）。本地缓存读取停顿的验证进程已单独停止，换独立缓存后完整执行，未跳过或改写测试。完整记录与前后自检在 [self_check.json](self_check.json)。

下一步需 Diagnosis 负责人补交原始特征生成实现（含所需模型/版本）以及三个 HGB 的使用 ID。接入并核对后，另立新协议和独立验证；已曝光的旧 T5 只能作历史/工程材料。详细交付格式见 [当前主线交接](../../../docs/CONTROLLER_1024_BASELINE.md)。
