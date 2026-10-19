# v8r3 结果：格式修正通过，语义核验仍是主要限制

2026-10-19，Kaggle 第 1 版成功执行 553.1 秒（约 9.2 分钟），固定 8 个开发样本、四组 32 条真实输出，共 43 次模型生成。全部记录通过冻结 CPU 决策和成本回放，工程门槛本地重算与云端一致。**v8 三组格式错误为 0，完整流程开始产生有效修复；这仍不是可靠净增益或默认策略验收。**

## 来源与一致性

- [Kaggle 执行记录](https://www.kaggle.com/code/yearo111/e-v8r3-smoke8-contract-fix-20261019/log?scriptVersionId=359908613)：私有 Notebook，T4 ×2。
- 固定 N1/N2/N8/N12/N15/N31/N48/N96，4 个已接受正确对照、4 个内部 E 目标。均为已使用的开发样本，无盲测或 held-out 声明。
- 模型仍为 Qwen/Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`；NF4、greedy、seed 42、384 token 预算与上一轮相同。所有语义核验均为同一 Qwen，不能称为独立模型验证。
- 运行时只用 C1 问题、原答案及实际检索 Top-5；gold、人工标签仅用于离线评测和审核。与 r2 输入逐字一致。
- r3 冻结包 SHA-256：`72ed761fd8017ebbaf439729ff318caa55c0823acf4b62429c4db8f31310b6f1`。
- 结果 ZIP SHA-256：`766c9c377be6b7cc4123a1225f3b4670ae10348df7db3a703e72d9665d36da45`。
- 两轮共有的 41 次生成，其已解码输出全部逐字一致；r3 新增 2 次 N96 候选核验。说明本轮差异来自明确文法恢复及后续流程进入，没有更换语义提示或运行时参考答案。

## 工程验收

v8 阶段共 35 次：原检查 16 次有效、候选生成 11 次有效、候选核验 5 次有效；另 3 次候选生成因来源约束被拒，均在去原检查组。格式拒绝为 0，未分类拒绝为 0；完整三阶段路径 2 条，事后裁切 0 次，生成总数 43 小于上限 64。云端 111 项便携检查通过。v7 对照仍有 3 条原有 invalid_output，不计入 v8 文法修正效果。

r3 预设工程政策区分格式故障与正确的来源／长度拒绝。它未降低逐字连续原文定位要求，也未把 r2 的旧门槛失败重算为通过。来源拒绝不是成功修复，工程通过也不代表答案正确。没有自动启动 75 条重跑。

## 实际输出

以下 EM 为严格参考答案匹配，每组分母均为 8；“改答数”不是修复成功数。

| 组别 | EM 正确数 | 改答数 | 模型调用数 | 关键状态 |
|---|---:|---:|---:|---|
| v7 对照 | 1→2 | 5 | 8 | 3 invalid_output、5 repaired |
| 完整 v8r3 | 1→1 | 1 | 13 | 5 原支持、1 候选不变、1 候选被拒、1 修复 |
| 去原答案核验 | 1→0 | 2 | 11 | 2 不变、3 来源拒绝、1 候选被拒、2 改答 |
| 去候选核验 | 1→2 | 2 | 11 | 5 原支持、1 不变、2 改答 |

完整流程仅修改 N96：`Hungarian literature` → `the pastry being a finger food : the consumer would pick apart the bread as a monkey would`。引文 R1 直接解释了名称来源，辅助审核建议视为修复，仍待人工确认。离线参考为 `being a finger food: the consumer would pick apart the bread as a monkey would`；候选多了 `the pastry`，严格 EM 仍为 0，因此整体 EM 没增加。不能为这条样本事后修改评分规则。

完整流程四个正确对照最终均保留原答案，但 N12 的原检查仍误判为错，后来候选与原答案相同才保留。四个 E 目标中，N8 和 N31 被误判为已支持而跳过修复；N48 的正确候选 Govind Ballabh Pant Sagar 仍被后续核验错误拒绝。这些是答案与证据对应的语义错误，并非剩余字段格式故障。

## 辅助审核与下一轮

[10 条改答待审](../artifacts/answer_repair/v8r3_smoke8/review/changed_outputs_pending.jsonl) 对应 5 个唯一 ID；[辅助意见](../artifacts/answer_repair/v8r3_smoke8/review/assisted_changed_outputs.jsonl) 建议 N31/N48/N96 为修复、N12 为无必要扩写、N15 为术语退化成现象复述。跨组重复不能当独立样本，辅助意见不覆盖人工结论。所有新最终答案的人审字段仍为 pending。

下一轮先冻结并对照测试“按问题从证据抽取事实，再比较被审答案”的核验方法，重点检验作品身份、比较关系方向和术语类型，保留同样的正确对照、干扰证据与成本统计。它是待验证算法路线，不能预设一定提升。工程已经可测量，应把主要精力转向减少语义误放与误拒；待行为稳定后再运行固定 75 条开发集，并补充自然目标错误、使用共同未调参测试集。当前 8 条只用于工程排错，不足以估计最终系统收益。

## 保存位置

- [原始输出](../artifacts/answer_repair/v8r3_smoke8/kaggle_20261019/)
- [冻结 CPU 回放](../artifacts/answer_repair/v8r3_smoke8/release_audit/cpu_replay.json)
- [结果校验与跨版本一致性](../artifacts/answer_repair/v8r3_smoke8/result_validation.json)
- [云端工程门槛](../artifacts/answer_repair/v8r3_smoke8/kaggle_20261019/smoke_acceptance.json)
