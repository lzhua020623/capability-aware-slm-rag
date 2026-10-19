# v8r2 smoke：入口恢复，候选拒收及语义核验仍需修正

Kaggle 第 1 版 `359869429` 已成功完成 8 条 ×4 组的 32 条真实输出。结果 ZIP SHA-256 为 `a1e1d549e5887682be0b554695fb41f412271ffec733e2e303b6fc49489ce387`，共 28 文件。输入包为 `ca318d513f5feab0ff2f4c94acf19888e46ef0021ec454ef6dfd43f166bf58fe`，93 个负载文件已核对。

原版 75 条及其结果保持不变。此次仅用于工程验收，不能估计总体或未见集收益。

## 实际进展

- 完整版的 8 条原答案核验全部有效：5 条判断 supported、3 条 unsupported；去候选核验组也有 8 条有效。原版同一子集全部为 original_check_invalid，入口问题已在真实生成上得到修正。
- 三类阶段都已实际执行：16 次有效 original_check、5 次有效 minimal_repair、3 次有效 candidate_check；其中 1 条完整三阶段路径。
- 33 个 v8 阶段中，9 次由输出边界停止器结束、24 次未由该停止器结束（可能为 EOS 或上限，不能仅据此判断）。实际生成未被事后裁切。
- 但仍有 9 次 minimal_repair 验证拒收，预设工程门槛未通过，因此不扩到 75 条，也不修改门槛以追认此次通过。

## 四组结果

初始冻结 EM 为 1/8；初始人工分组是 4 个正确对照与 4 个 E 目标，不能混用两种口径。

| 组别 | EM 正确数 | 改答 | 调用 | 主要状态 |
|---|---:|---:|---:|---|
| v7 对照 | 1→2 | 5 | 8 | 3 invalid_output、5 repaired |
| 完整 v8r2 | 1→1 | 0 | 12 | 5 original_supported、2 candidate_invalid、1 candidate_rejected |
| 去原答案核验 | 1→0 | 1 | 10 | 1 candidate_unchanged、5 candidate_invalid、1 repaired、1 candidate_rejected |
| 去候选核验 | 1→2 | 1 | 11 | 5 original_supported、2 candidate_invalid、1 repaired |

共实际生成 41 次，不超过 64 次预算。7 条改答记录的人工最终标签保留 pending。EM 不等于语义真值；本轮不能据此声称正提升。

## 两类问题需要分开处理

候选输出拒收共有 9 次，来自 5 个不同的“问题＋候选提示”输出（不同消融会重复相同提示）：

1. **格式省略**：N12 输出三行裸值，N96 第一行只输出 `1` 而非 `RANK: 1`。完整／去原核验／去候选核验各出现这两类，共 6 次。可在新修订中设计明确且无歧义的候选文法兼容，但仍必须逐项做原文定位，不能接受任意散文或从多套响应中挑选。
2. **来源约束拒绝**：去原核验组另有 3 次。N2 的 ANSWER 不在所引经纬度片段内；N8 的答案未在所选引文中定位；N31 的 QUOTE 删除了原文中间内容。这些拒绝体现了来源保护，不能通过模糊匹配、拼接或放宽连续片段要求来消除。

格式拒收与模型提出了不合格证据应分别记录；本轮统一的 candidate_invalid 不够清楚。下一版应明确区分格式、来源和长度等限制的拒绝，前瞻固定新的工程验收定义；本轮既定 false 保留。

入口恢复后也暴露了语义误判：已确认为 E 目标的 N8/N31 被原答案核验放行；N48 提取 Govind Ballabh Pant Sagar 后被候选核验拒绝，而去掉候选核验的组接受了该答案。去原核验组把正确的 inflation 改为 a rise in the price level 并通过自检。这说明同模型的单独六项检查仍可能误放和误拒，不能把结构有效当作语义正确。

下一步先修候选文法与拒绝分类，保留来源防护；再独立检验六项核验对于作品身份、比较关系及术语类型的可靠性。语义规则的修改必须单列为新实验因素，而不是归入格式兼容或事后改写本次结果。

## 保存与审核

- [原始输出](../artifacts/answer_repair/v8r2_smoke8/kaggle_20261019/)
- [冻结 CPU 回放](../artifacts/answer_repair/v8r2_smoke8/release_audit/cpu_replay.json)
- [结果校验](../artifacts/answer_repair/v8r2_smoke8/result_validation.json)
- [7 条改答待审](../artifacts/answer_repair/v8r2_smoke8/review/changed_outputs_pending.jsonl)
- [预设门槛原始结果](../artifacts/answer_repair/v8r2_smoke8/kaggle_20261019/smoke_acceptance.json)

本地隔离副本的全部 146 项 Python 检查通过，包内 106 项便携检查通过；32 条模拟流程和实际 32 条 GPU 输出分别验证，模拟记录不作为研究结果。任何新的修订、提示或模型结果均另存，不覆盖此包。
