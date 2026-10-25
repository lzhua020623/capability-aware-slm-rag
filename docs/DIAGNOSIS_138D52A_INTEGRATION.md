# Diagnosis 更新的兼容集成 · 2026-10-26

来源：`feature/diagnosis-probe-zheng`，提交 `138d52aa22e0eee78637e01f981a030d7ad388df`。本次合并代码框架与模型资产，保留现有默认实验执行器；P60 未启用，未运行新的质量实验。

## 实际接入

- 原 Diagnosis 权重、27 维特征、Probe 提示和 Safe Binary 规则保持不变。
- 新增 `primary_failure_type / active_failures / failure_scores`。活跃类型来自原 Stage-2 条件概率 ≥1/3 的表示规则，实际仍按单一主类型选择动作；没有训练独立多标签模型，不能据此声称多标签识别效果通过。
- 新增 Enhancement 请求/结果协议、注册执行器和可注入的 Adaptive RAG host。未注册模块及异常保留原状态；success 仅表示执行完成。
- P60 选择器、固定配置和三个 HGB 资产保留为显式可选入口，不传入当前 `IntegratedController`。新 host 的 Probe-OFF 不改变默认 T4/T5 的 Probe 路径。

## 兼容与返回检查

原重排 `RefinementConfig / RefinementResult / RefinementStatus` 与新 Enhancement 协议共存，包级保持延迟导出，不引入对检索模块的强制 Diagnosis/Probe 加载。

执行器新增检查：返回 action 必须与请求动作一致；changed_answer 必须与实际字符串变化一致；非成功结果不得发布新的答案或证据。违例记为 error 并保留请求进入状态。新增契约的 latency 必须有限且非负。

官方序列化入口为各类型的 `to_dict()`；新决策包含只读分数字典，不依赖通用 `dataclasses.asdict()`。

## P60 尚未满足的启用条件

1. 提供 EXP-P60 的训练、调参、校准和错误分析 ID，核对 selection/final 及完整保留池曝光。
2. 接入真实在线特征提供器。配置要求按精确顺序提供 78 项特征，名称只有 75 个唯一值；不得擅自去重、补零或使用离线 gold/标签。新增特征及 no_action 前的特征计算也必须计量。
3. 使用真实三条增强的适配器；对齐预算、失败调用、token、上下文和轨迹，而非以 mocks 作为生产模块。
4. 若启用 P60，先登记为独立实验，说明三模型选择、特征和 Probe-OFF 同时变化的归因限制；不与唯一 NQ 阈值候选静默混用。

P60 仍将 Diagnosis 判 correct 的题直接保留，不会自动解决这部分漏判。零 FEVER E 路由和旧模型的类型区分问题仍需实测，不能由接口测试推断已解决。

## 验证范围

合并回归包括旧执行器、r2 和上游新增测试；另从 T4 ZIP 重建 800 条 Adaptive/No Probe 的 Diagnosis 与 Probe 输入，逐字段比较原路由。该回放不重新生成答案、不读取 gold，不是 P60 效果或最终评测。

验收回执见 [合并自检](../artifacts/controller/diagnosis_merge_v1/merge_self_check.json)。后续 r2 包必须引用合并后的固定源码；旧运行包的原始字节和原结果不改。当前任务见 [T1–T9](CONTROLLER_TASKS.md)。
