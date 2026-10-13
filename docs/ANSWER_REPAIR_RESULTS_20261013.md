# E 实验结果与交接（2026-10-13）

E 的工程链路、真实 T4 推理和冻结 Controller→E 整合已跑通；**算法有效性尚未通过验收**。当前结果不支持“引用检查加同模型再生成能可靠改善答案”。应保留 v4 作为可复现基线，下一步解决查询匹配与语义支持，再评估 utilisation 样本和误路由风险。

## 实验范围与冻结协议

- C1 来源：`6f08f2fe2dc2dcf39b75177763207682b2890a31`。
- Diagnosis / Probe / Safe Binary / Controller：`7c7863f98f87e892119481a79e23ca6dcebdced0`，本次没有重训或修改阈值。
- 模型：Qwen2.5-7B-Instruct，revision `a09a35458c702b33eeacc393d103063234e8bc28`；NF4、double quant、FP16 compute、greedy、seed 42。
- 硬件：Google Colab Tesla T4。实际检查到 196 个 NF4 线性层，参数在 CUDA；模型加载后的 allocated 显存约 5.56 GB，不是完整生成峰值。T4 足以运行本次量化推理。
- 固定开发样本：FEVER / NQ 各 50 条，从 Diagnosis v1/v2 的训练交集按 seed 42 的 ID 哈希选择；排除两版所有 validation/test ID。选样不用答案正确性、gold 或 diagnosis 标签。10 条 smoke 是各自 50 条的子集，不能合并为更多独立样本。
- E 和 Reanswer 均使用原 C1 Top-5，每条最多一次额外生成，上限 384 new tokens。运行时不读取 gold、C3、70B 或人工 diagnosis；gold 仅用于生成后的离线评分。
- 这是训练样本上的探索性开发评估，不能用来报告 held-out 性能。历史 C1 没有记录模型 revision，不能声称与其环境完全一致。

环境记录：Python 3.13.15、torch 2.11.0+cu130、CUDA 13.0、transformers 5.18.0、accelerate 1.15.0、bitsandbytes 0.50.2。Controller 运行使用 numpy 2.1.3 / scikit-learn 1.6.1。

## Always-E 与 Reanswer 对照

开发策略 `grounded_reanswer` 隐藏初始答案，让模型从 query 和证据产生 answer/citations；程序再比较候选与初始答案，推导 verdict 并检查引用。`reanswer` 使用原 C1 指令重新回答，不产生结构化引用。两者不是独立 verifier。

| 数据集 | C1 正确数 | Grounded Reanswer | 改对 / 改错 | Reanswer 正确数 |
|---|---:|---:|---:|---:|
| FEVER accuracy | 45/50 | 46/50 | 1 / 0 | 45/50 |
| NQ EM | 20/50 | 20/50 | 1 / 1 | 20/50 |

NQ token-F1：C1 `0.548912` → Grounded Reanswer `0.544144`；Reanswer 为 `0.548912`。配对正确性变化的 exact McNemar p 值均为 1.0；这批结果没有提供显著收益证据。

| 运行 | 答案变化 | 无效输出 | 保留 / insufficient / 修复 | 额外调用 | 总生成延迟 |
|---|---:|---:|---|---:|---:|
| FEVER Grounded | 1 | 3 | 15 / 31 / 1 | 50 | 185.446 s |
| NQ Grounded | 8 | 14 | 18 / 10 / 8 | 50 | 221.673 s |
| FEVER Reanswer | 0 | 5 | 有效再回答 45 条 | 50 | 120.010 s |
| NQ Reanswer | 1 | 0 | 有效再回答 50 条 | 50 | 64.137 s |

`insufficient_evidence` 和 `invalid_output` 都回退初始答案；不能将这些记录算作已验证正确。延迟是额外模型生成时间，不包含模型加载、原 C1、完整检索或全系统成本。输出 token 为重新分词估计值。

## 关键案例：有引用仍可能答错

以下是 Codex 对固定开发输出的定性审核，**不是人工金标准，也不是运行时独立语义核验**。原评分和原始输出保持不变。

| ID | 变化与观察 | 对 E 的含义 |
|---|---|---|
| FEVER `219125` | Valencia 从 REFUTES 改成 gold 的 SUPPORTS；引用来自委内瑞拉 `Valencia_Municipality,_Carabobo`，不能支持查询所指西班牙 Valencia。已有 diagnosis 为 insufficiency。 | 指标 +1，不能视为可信的证据支持修复。需要核验实体和完整声明。 |
| NQ `-5748368670037082784` | 问 Thomas 名字的含义，原答案 `twin` 正确；改成中世纪流行及圣经人物的介绍，EM/F1 降为 0。引用合法。 | 查询与答案不匹配的明确误修；文字出现在证据里不等于回答了问题。 |
| NQ `-1814654291537702625` | Christianity 案例从 `love` 扩成 Great Commandment 完整句，EM 0→1；已有 diagnosis 为 correct。 | 可能是参考答案文字对齐，不能直接认定原语义错误被修复。 |
| NQ `-2282501421613384225` | 问 tibia/fibula，候选仍只描述其中一个骨头的位置；EM 未改善。引用合法。 | answer span 检查不能验证问题要求的关系和覆盖范围。 |
| NQ `4305916235259203400` | 问苏联开始/结束年份，原答案 `1922-1991`；冻结 gold 为 `in Eurasia`。候选使用非连续拼接引用被拒绝。 | 单独标记参考答案问题，不修改 gold 或 scorer 获取收益。 |

该 100 条开发样本中已有 diagnosis 标注的 utilisation 只有 5 条。Always-E 没有修复它们：Sugarland Express、thumbs twiddling、Annie 歌手出现 insufficient；Loihi 输出无效；tibia/fibula 改写后仍错。部分原 Top-5 已包含直接答案，例如 Sugarland Express 的导演和 Annie 的歌手，显示候选生成本身也会过度 abstain。样本太少，不能外推总体修复率。

FEVER 无效输出：2 条引用不匹配、1 条缺引用。NQ 无效输出：7 条引用不匹配、7 条 answer 不在引用中。后者包括真实词语差异、拼接以及连字符/标点空格的严格匹配问题；后续可以单独修正规范化规则，但必须发布新版本，不能回改这次结果。引用合法仅表示来源匹配，本次所有记录的 `independent_support_status` 都是 `not_checked`。

## 真实冻结 Controller→E

使用同一 100 条开发样本，实际执行冻结 classifier、选择性 Probe、Safe Binary、Controller，再由 E adapter 消费最终 `ControllerDecision`。Probe 与 E 共享一份量化模型。CPU fixture 契约验证与这次真实 GPU 运行分别保存。

| 数据集 | no_action | retrieval_enhancement | evidence_refinement | generation_repair | Probe 调用 | E 调用 | C1 → 最终正确数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEVER | 31 | 19 | 0 | 0 | 14 | 0 | 45 → 45 |
| NQ | 40 | 7 | 2 | 1 | 13 | 1 | 20 → 20 |

唯一 E 路由是苏联案例，输出无效并回退；已有标注的 5 个 utilisation 均被最终 Controller 路由为 no_action，没有执行 Probe。这里既有 E 的能力问题，也有目标样本未进入 E 的路由问题；不应通过修改上游冻结阈值来掩盖两者。

实际追加 27 次 Probe + 1 次 E。Probe 生成约 `177.722 s`，E 约 `7.569 s`；Probe 输入 token 33,311、输出估计 1,915，E 输入 1,100、输出估计 98。**其他两种补偿动作只记录，没有执行，所以不是完整 adaptive framework 结果。** 这一训练开发 cohort 的路由分布也不能替代上游 held-out Diagnosis 评测。

## 校验与可复现产物

已用冻结归档内的代码在 CPU 逐条重放 v4 的 240 条输出（40 smoke + 200 development），核对输入 ID/顺序、代码和数据指纹、解析/citations、评分、成本和汇总；Controller 的 100 条另外重放实际 classifier、Probe parser、Safe Binary、decision 和 E fallback。汇总浮点误差容差为 `1e-12`。主代码 54 项测试通过。上述校验验证实现与记录一致，不证明语义支持。

完整结果包：[E_answer_repair_20261013_results.zip](../artifacts/answer_repair/session_20261013/E_answer_repair_20261013_results.zip)。原始数据、代码归档、日志、环境及 checksums 已下载到本机。`artifacts/` 被 Git 忽略；代码和文档修改尚未提交或推送。组合包 SHA-256（本地计算）：`cf8dfd42fc949e39bb4dfa66c1307149178565dc5c2ad9b1b126b560b8368e5a`。

| 包 | SHA-256 |
|---|---|
| v4 GPU 代码与输入 | `2a40d4fbaff591b02d116243f92404731cb71ccc5ce8a77092f4fda70dd3156e` |
| 冻结 Controller + v4 E | `60c4241e8a800ecbd8d780a913cbaf137927bbef7f954cadd16bd58ea7f7506c` |
| v4 smoke 导出 | `e961bec18592bf2371a716cf87657c5f4eb83c569b595a3025013d9c52ddb27b` |
| v4 development 导出 | `7a4def101c1ce5d524b941b51ba00e94a05b0b5d4ea471ed601a9fa59627dc72` |
| Controller→E 导出 | `b28533f874786836ee9c92fd4f1a084dda64723068fd030daf38be18fbd36d55` |

可读本地记录：

- [Always-E / Reanswer 输出与校验](../artifacts/answer_repair/v4/colab_20261013_development/)
- [真实 Controller→E 输出与校验](../artifacts/answer_repair/controller_20261013/)
- [Colab / Kaggle 执行方法](ANSWER_REPAIR_COLAB.md) 与 [v4 notebook](../notebooks/answer_repair_colab_v4.ipynb)
- [E 协议](ANSWER_REPAIR.md)、[引用与输出检查](../src/compensation/answer_repair.py)、[模型生成与 prompt](../src/compensation/qwen_repair.py)、[Controller adapter](../src/compensation/controller_adapter.py)

## 下一步与验收条件

1. 冻结本次 v4 及负结果。将“引用来源合法”“回答了查询”“证据支持结论”分开，新增能够拒绝 Thomas / Valencia 这类候选的语义核验。若仍复用同一个 7B，需要通过误修率消融验证；不能称为独立核验。
2. 在尚未用于 prompt 调整的训练内 utilisation 分层 cohort 做 operation-level/oracle-route 评估，同时保留原本正确及误路由样本。分别测 E 的修复能力、误修率、abstain/invalid 和成本；oracle diagnosis 仅作离线消融，不输入部署链路。
3. 把 E 能力与冻结路由覆盖分开报告。当前 5 个目标案例都未进 E，需与 diagnosis 负责人交接这些案例；是否发布上游新版本由其独立实验决定。
4. 汇合三个补偿模块后再测 failure-type × action 矩阵及 Vanilla / Always / Adaptive 对照，并记录检索、特征提取、Probe 和补偿的完整成本。
5. 重新确认最终评测数据未用于各模块选策略。新 `reranking` 分支用 NQ frozen500 比较 keep3/keep5 后选择默认 keep5；这 500 条含 Diagnosis v2 的 65 个 validation、68 个 test ID。该重排结果应作探索性开发证据，不能在同一批上宣称新的自适应系统未见测试收益。既有 Diagnosis 自身冻结测试指标不因此被重算。

只有 E 在独立评测中展示可信的证据支持修复、受控误修和合理成本后，才能宣布算法完成。若仍无净收益，项目可如实报告补偿操作适用边界和路由失配；不需要为了匹配 proposal 而制造正结果。
