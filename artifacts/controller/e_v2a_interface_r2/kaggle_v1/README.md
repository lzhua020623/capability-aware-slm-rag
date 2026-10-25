# r2 首次 GPU 结果 · 2026-10-26

[Kaggle Version 1 / 362285592](https://www.kaggle.com/code/yearo111/controller-e-r2-shared-frozen-20261026?scriptVersionId=362285592) 已结束，页面用时 534.1 秒，GPU T4×2。实际模型在 cuda:0，共享 Qwen2.5-7B / NF4，revision `a09a35458c702b33eeacc393d103063234e8bc28`；环境原件见 [environment.json](raw/environment.json)。

**结论：r2 未通过工程验收，按预登记停止。NQ、FEVER 和所有含 E 的组统一为 unavailable；主线不再安排 r3。** 这次没有运行真实开发集、选型集或最终评测，不能报告新的 EM、修复净收益或真实开发合规率。

## 完整性与实际结果

- [outputs.zip](outputs.zip) 为下载原件，118,102 字节；SHA-256：`9cf94fb3e8573213e0a86462690b7291b8801a176423e91e949d501323c2f28d`，与 Kaggle 日志一致。
- [raw/](raw/) 保留全部原字节解包；[audit.json](audit.json) 独立从 ZIP 重算，26 项校验和及冻结输入清单一致，源码为 `e3ddead`。**审计通过表示归档可信，不表示 E 通过。**
- 18 道短题、2 道长题全部运行：10 项通过、1 项结构失败、9 项语义失败、0 项执行失败。
- NQ 共 11 项，通过 7 项；FEVER 共 9 项，通过 3 项，均含各自 1 道长题。这是刻意覆盖边界行为的虚构验收，比例不是 benchmark 准确率。
- 两道长输入为 17,514 和 17,539 tokens，XGrammar 0.2.8 与 prefill256 同时启用；执行和输出契约通过，修复行为均未通过。
- [gate_summary.json](gate_summary.json) 记录 20 次模型调用：58,096 输入 tokens、估计 1,586 输出 tokens、模型耗时合计约 328.3 秒。两道长题各约 99.3／103.0 秒，短题中位约 6.5 秒；这是验收成本，不是开发成本或 AC5 端到端成本。

Notebook 红色失败来自预登记的 exit 3，经外层转换为 RuntimeError；不是 CUDA 崩溃、OOM 或结果 ZIP 丢失。存在结构失败，所以不符合“只有语义失败才运行 400 条只核验开发”的条件；实际开发调用为 0。

## 原始记录说明了什么

| 问题 | 可复核记录 | 解释 |
|---|---|---|
| 正确候选配错证据句 | [NQ repair-1](raw/engineering/v2a-r2-fiction-nq-contradicted_repair-1.json) | 候选 Ilsa Marrow 是正确人名，输出引用 R2S2；该句只说 Kerran Pell 制作画框，人名在 R2S1。连续片段检查拒绝替换、保留原答并标记未解决。这是 1 项结构失败；合法句号 ID 不保证候选存在于该句。 |
| 冲突被当作可以裁决 | [NQ conflict-2](raw/engineering/v2a-r2-fiction-nq-conflict_insufficient-2.json)、[FEVER conflict-1](raw/engineering/v2a-r2-fiction-fever-conflict_insufficient-1.json) | 理由已经指出两证据冲突，仍判 contradicted 并选择一方改答；4 道冲突题均未通过。证据来源检查挡不住这种语义错误。 |
| FEVER 判断对象混淆 | [FEVER supported-2](raw/engineering/v2a-r2-fiction-fever-supported_keep-2.json)、[FEVER repair-1](raw/engineering/v2a-r2-fiction-fever-contradicted_repair-1.json) | 声明被反驳、原标签 REFUTES 正确时，却判原答 contradicted；声明被支持、原标签 REFUTES 错误时，却判 supported 并保留。它有时在判断声明，而非输入标签是否正确。 |
| 缺失事实仍被当作反驳 | [FEVER missing-2](raw/engineering/v2a-r2-fiction-fever-missing_insufficient-2.json) | 没有发现者信息，却改为 REFUTES。NQ 两道缺失事实题均通过；不能据此认定跨数据集问题已经解决。 |
| 长输入下理由与动作不一致 | [NQ long](raw/engineering/v2a-r2-fiction-nq-contradicted_repair-1-long.json)、[FEVER long](raw/engineering/v2a-r2-fiction-fever-contradicted_repair-1-long.json) | NQ 理由说原答被反驳，标签却为 supported；FEVER 判 contradicted 却不提供有证据的反向标签。能执行长输入不等于能正确核验。 |

这次支持的结论是：受约束解码和来源检查可以约束输出并阻断某些无效替换，但仍不能提供可靠的证据语义判断。它不证明扩大数据集、换 GPU 或继续改提示会解决问题；也没有测量 r2 的真实数据收益。实际缺陷是由本轮受控题观察到的，不能外推为整体模型准确率。

## 后续主线

部署决定见 [../deployment_decision.json](../deployment_decision.json)。运行时 unavailable 不调用 E 模型，保留进入 E 的答案并记录核验未完成；不伪造 supported/insufficient 结论。工程不可用告警不能冒充经核验的语义失败标签，用来宣称 AC6 达标。FR12 的 E 修复打通要求仍未验收，必须在报告披露。

按 [T1–T9 清单](../../../../docs/CONTROLLER_TASKS.md) 继续 T5：完成 [17 题人工审核](../../t5_assisted_review_v1/review/review17.html)，用固定 unavailable 模式回放预登记的 NQ 路由网格，至多冻结一个候选，再做一次选型。FEVER 路由不改；不启用 P60，不重训 Diagnosis，不将新的 E 修订混进路由候选。后续选型和最终六组必须记录同一个模式；Always Repair 组届时是不执行 E 的保留策略，不能把该组的零误修写成有效 E 的安全证据。

复核命令（不调用 GPU、不读 gold）：

```bash
python scripts/audit_controller_e_v2a_outputs.py --archive artifacts/controller/e_v2a_interface_r2/kaggle_v1/outputs.zip --release-dir artifacts/controller/e_v2a_interface_r2/gpu_release_v2
```
