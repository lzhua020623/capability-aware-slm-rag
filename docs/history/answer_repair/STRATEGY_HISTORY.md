# E 早期策略开发记录

这是 2026-10-16 整理时从 `docs/ANSWER_REPAIR.md` 提取的历史说明，保留当时的运行状态和结论。当前状态以 [项目进度](../../PROJECT_STATUS_AND_DIRECTION.md)、[v7 结果](../../ANSWER_REPAIR_RESULTS_20261014_V7.md) 和 [当前交接](../../ANSWER_REPAIR_HANDOFF.md) 为准。

当前是可执行的研究原型，工程、本轮开发实验及自动验收已完成。v4–v6r1 真实 T4 输出均已归档；v5 同模型二次复核无冻结指标增益。v6r1 的 NQ 单次原文抽取和 FEVER 独立 MNLI 完成 128 条真实预测、81 项云端检查及全部冻结代码回放。FEVER 审核后 4→7，但 dev50 45→44；NQ dev50 EM 20→7，可靠净收益仍未验收。两种新策略未晋升默认，人工正确性/支持性审核按用户要求暂缓。当前交付与剩余项见 [交接清单](../../ANSWER_REPAIR_HANDOFF.md)，实验结论见 [10 月 14 日报告](ANSWER_REPAIR_RESULTS_20261014.md)。

2026-10-14 晚：128 条辅助审核的 28 个关键案例经独立盲评二读，9 个修复、3 个误修两读一致，待你按 [确认清单](../../ANSWER_REPAIR_CONFIRMATION_20261014.md) 逐号确认。下一版 [v7](../../ANSWER_REPAIR_V7.md) 已冻结并打包：`short_answer_quote`（NQ 短答案 + 引文分离、空白/引号归一化定位、内容词锚点）与 `fever_nli_anchored`（只允许含声明主体实体的窗口投票）。v6r1 轨迹的离线回放显示 FEVER 审核后 4→10、dev50 45→46 且无误修，但规则是看着这些样本设计的，效果以 held-out 为准。

2026-10-14 已完成 [37 条训练内样本的聊天辅助审核](ANSWER_REPAIR_REVIEW_20261013.md)：确认 19 条 utilisation 核心候选、9 条正确对照、9 条暂缓。源标签未改写。基于前两组准备 [v5 的固定 28 条开发实验](ANSWER_REPAIR_V5.md)，新增完整答题 prompt 与可选同模型第二次语义复核；CPU 检查通过，尚未完成真实 GPU 验证。下文的单次调用约束描述 v4 基线；v5 复核策略最多两次调用，成本单独记录。

以下保留版本历史：v1 的 FEVER 10 条真实 smoke 出现 8 条 `invalid_output`、2 条 `insufficient_evidence`，正确数仍为 7/10，无答案变化，随后停止扩大该版本。

首轮原始结果、模型审计及 Codex 离线定性审核保存在 `artifacts/answer_repair/v1/colab_20261013_smoke/`。审核不是人工标注，也不是运行时语义 verifier。独立模型审计验证了实际 Qwen2 架构、196 个 NF4 层、FP16 compute 和 double quant；模型加载后 allocated 显存约 5.56GB，这不是完整推理峰值。Transformers 5.18 不提供该加载对象的 `_commit_hash`/`hf_device_map`，因此 v2 显式固定已观察到的 snapshot `a09a35458c702b33eeacc393d103063234e8bc28`，并核验实际量化层。历史 C1 未记录 revision，不能声称历史环境完全相同。

历史 `v2_contract_last` 保留同一证据、一次调用和 384-token 上限，把决策规则放在输入末尾，给出当前原答案的 keep 模板和完整的 insufficient 空值模板，强调短引用与真实 rank；未放宽解析、标签或引用检查。`repair` 必须改变答案，同答案须使用 `keep`。历史实验包保留，代码变化另建版本包。

v2 已完成 FEVER/NQ 各 10 条及对应 Reanswer 对照，均无正确率增益；Verify 的 invalid 为 FEVER 3/10、NQ 8/10。v3 加入三个决策的虚构示例，要求先找查询所问事实再比较初始答案，在完整 JSON 对象时停止。真实 v3 各 10 条仍无答案变化，invalid 为 FEVER 5/10、NQ 6/10。

v4 保留 `v3_balanced_examples` 的 Verify prompt，修复 token 边界：个别最后一个 BPE token 同时含 JSON 闭括号及多余反引号，停止器记录该实际边界，只将其完整对象交给严格解析器。`raw_output` 保存完整输出，`validation_input` 记录两者有差异时的对象文本；不修改字段、引用或答案。普通解析器仍拒绝任意尾部文本。Reanswer 对照不使用 JSON 停止器。

v4 另有 `--strategy grounded_reanswer`（`v4_candidate_first`）：模型只看 query 和原始 Top-5，生成 answer/citations，不看 initial answer；程序随后根据候选与原答案是否相同推导 keep/repair，再执行同样的引用规则。目的在于检验原答案锚定是否限制修复；仍是同一模型一次调用，不引入参考答案，也不能作为独立语义验证。该策略已运行固定 50 条开发集，结果、原始输出与审核分开保存。

v3 引用按完整词和标点 token 匹配，忽略标点两侧空格，保留每个词和标点。`Paris , not London .` 与 `Paris, not London.` 可匹配；删掉逗号或否定词、换词、改变引用编号、把 York 从 Yorkshire 中截出，均不允许。不同引号/括号写法仍不自动替换。这仅修复格式差异，不提供语义支持判定。
