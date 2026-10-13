# E v5：审核后开发集与同模型语义复核

2026-10-14：37 条聊天辅助审核意见已确认，其中 19 条 utilisation 核心候选、9 条正确对照、9 条暂缓案例。固定开发集只纳入前两组，共 28 条：FEVER 10+4，NQ 9+5。全部来自冻结 C1 的 Diagnosis v1/v2 训练交集，排除两版所有 validation/test ID。样本已被用于人工辅助审核与策略开发，不能作为未见测试集或代表部署总体分布。

本轮已在 Colab T4 完成 112 条真实预测，并通过 CPU 确定性回放。v5 候选与同模型复核没有提高 FEVER/NQ 冻结正确数；具体结果见 `ANSWER_REPAIR_RESULTS_20261014.md`。v5 不晋升默认策略，旧实验包保持原样。

## 实验问题

1. 用“最短但完整的答案”代替过度强调最短片段，能否减少答非所问、漏答实体和误解事件关系？
2. 候选答案通过引用检查后，再由同一 Qwen 7B 检查问题、实体、关系、覆盖范围和证据支持，能否减少误修？其额外成本与拒绝正确修复的代价是多少？

第二步是同模型的第二次调用，不是独立 NLI verifier 或人工裁决。同一个模型仍可能重复犯错。所有输出继续标记 `independent_support_status=not_checked`；模型自称检查通过不算正确性证据。

## 固定对照

四个条件使用完全相同的 28 条 query、初始答案与原顺序 Top-5，Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`、NF4、FP16、double quant、greedy、seed 42。

| strategy | 行为 | 每条最多调用 | 输出 token 上限 |
|---|---|---:|---:|
| `grounded_reanswer` | 保留 v4 候选 prompt 与解析行为 | 1 | 384 |
| `query_grounded` | 完整回答问题的新候选 prompt | 1 | 384 |
| `guarded_reanswer` | 与上一行相同的候选，再加语义复核 | 2 | 384 + 192 |
| `reanswer` | 原 C1 指令直接再次回答 | 1 | 576 |

全套最多 140 次调用、112 条最终预测。候选无效或 abstain 时不调用复核。复核也覆盖与初始答案相同的候选。四个条件不是四批独立样本。576-token Reanswer 只控制总输出预算上限，调用次数、输入长度和实际 token/延迟仍不同，须如实报告成本。`query_grounded` 与 `guarded_reanswer` 用于分离复核的影响；notebook 会检查两组候选原始输出是否一致，有差异时不能把最终差异全归因于复核。

这改变了 v4 单次调用的实验预算，作为显式新版本记录。Controller 默认策略和上游 Diagnosis/Probe 阈值未更改；本轮全部样本执行 E，属于按人工类别选样的 operation-level 开发消融，不是部署路由收益评测。

## 实现

- `src/compensation/answer_repair.py`：新候选 prompt 明确答案类型、完整覆盖、实体、时间和谓词关系。初始答案、人工标签、参考答案不进入候选 prompt。
- `src/compensation/support_check.py`：第二次调用只看 query、完整证据、候选与引用，检查 `query_match/entity_match/relation_match/coverage/evidence_support/conflict_free`。六项均为 `yes` 才接受候选；`no`、`uncertain`、格式错误均保留初始答案。失败不表示初始答案已经被证实正确。
- v5 NQ 的答案片段匹配忽略词语和标点之间的空格，但必须连续匹配所有词和标点。例如 `1922-1991` 可匹配 `1922 - 1991`；不能删除否定、拼接片段或把词的一部分当完整词。v4 的旧行为保留。
- 输出保存候选、复核原始输出和检查项、两阶段 token/延迟及调用总数。模型异常继续抛出，避免伪装为 keep。中断未完成的样本重跑，已完成记录的成本不包括未落盘的失败尝试；运行日志须随结果保存。

这些检查不保证模型具备语义判断能力。CPU 测试只验证拒绝/接受逻辑、格式契约、信息隔离和成本记录，模型质量需要真实输出与人工复评。

## 样本和评分

固定名单见 `configs/splits/answer_repair_reviewed_dev_v1.json`，其中保留纳入/暂缓原因、旧 dev50 重叠、来源 commit/hash、人工结论与冻结指标不一致的 ID。准备器验证审核包内容哈希、人工记录匹配、完整确认状态、训练划分以及 C1 query/answer/证据/参考答案的一致性。

原始 source diagnosis、C1 预测和 gold 均不重写。第 23 条药学院白大褂、第 34 条哥伦比亚号和第 37 条 bias 定义已确认可作正确对照，但冻结 EM 仍判为错误。NQ 人工正确对照共 5 条，旧 EM 只认 2 条；FEVER 正确对照 4 条与原标签指标一致。不能将这些参考答案或措辞差异当作语义修复成功或失败。

结果报告分两层：

1. 原始 FEVER accuracy、NQ EM/F1，按全部样本、核心组、正确对照组分别报告，并统计 invalid、abstain、复核拒绝、调用/token/延迟。
2. 单独生成前后答案成对审核表。按相同标准检查初始与最终答案的正确性和证据支持；所有人工结果栏初始为空。未完成人工复评时，语义修复数、误修率显示 `null`，不会自动把模型复核的 `yes` 当作人工真值。

审核包含未改动答案、keep、abstain、invalid 和复核拒绝，不能只检查看起来改对的结果。人工评分绑定具体结果 SHA-256，防止套用到另一版输出。模板与汇总不会静默覆盖已有文件；复评后的汇总使用新的 `--out-prefix`。

9 条暂缓案例保留在 manifest 中，等待团队统一定义；本轮不把这些 unresolved 标签作为评分真值。已有 19+9 条是聊天辅助确认，不是独立盲评。

## 复现与执行

本地准备和 CPU 检查：

```sh
python3.11 -m unittest discover -s tests -v
python3.11 scripts/prepare_reviewed_answer_repair.py
python3.11 scripts/build_reviewed_answer_repair_notebook.py
```

实验包：`artifacts/answer_repair/v5_reviewed/answer_repair_v5_reviewed_gpu.tar.gz`。
Notebook：`notebooks/answer_repair_colab_v5_reviewed.ipynb`。
在 Colab 上传 notebook 和实验包，选择 T4，依次执行。Kaggle 可导入 notebook，并将包添加为私有输入。无需 clone、重建检索或下载原始数据；首次模型加载仍需下载模型权重。

Notebook 校验归档和各文件哈希，运行环境检查与 CPU 测试，预检八组数据集/策略命令，然后执行整套开发实验并导出 ZIP。不自动挂载 Drive，也不上传仓库或提交代码。已完成结果可在同一环境续跑。

单组命令（在解压后的包根目录）：

```sh
python scripts/run_answer_repair.py --dataset nq --c1 inputs/nq_reviewed_c1.jsonl --all --strategy guarded_reanswer --max-new-tokens 384 --out outputs/nq_guarded_reanswer.jsonl
python scripts/summarize_reviewed_answer_repair.py --manifest manifest.json --c1 inputs/nq_reviewed_c1.jsonl --results outputs/nq_guarded_reanswer.jsonl --out-prefix outputs/nq_guarded_reanswer_reviewed
```

后续用真实结果决定是否保留新候选 prompt 和复核。先看 19 条错误样本的可信修复、9 条正确对照的误修和总成本，再回到冻结 Controller 检查路由覆盖；小样本不用于宣称整体系统有效性。
