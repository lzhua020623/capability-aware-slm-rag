# E 当前交接

截至 2026-10-17，E 以 v7 为基础，下一轮新实现将使用 v8 名称，尚未实现。工程可运行，已有 GPU 实验和离线回放；可靠净收益与人工语义验收仍未完成。交付分支为 `feature/verification-answer-repair`。

## 当前入口

| 内容 | 入口或状态 |
|---|---|
| E 职责、运行时输入与接口 | [ANSWER_REPAIR.md](ANSWER_REPAIR.md) |
| v7 冻结策略与协议 | [ANSWER_REPAIR_V7.md](ANSWER_REPAIR_V7.md) |
| v7 主包与第 4 批结果、失败分析 | [结果报告](ANSWER_REPAIR_RESULTS_20261014_V7.md) |
| 当前人工审核 | [review_all.txt](../artifacts/answer_repair/v7_combined_review/review_all.txt) / [review_all.jsonl](../artifacts/answer_repair/v7_combined_review/review_all.jsonl) |
| 新 NQ Train 数据及首批开发审核 | [接入说明](ANSWER_REPAIR_NQ_TRAIN.md) / [100 条待审索引](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.txt) |
| 新 NQ Train 的代理辅助判断 | [100 条理由与优先候选](ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md)，人工确认仍全部 pending |
| Colab、Kaggle 和离线回放 | [运行说明](ANSWER_REPAIR_COLAB.md) |
| 旧版结果、Notebook 和交接快照 | [历史归档](history/answer_repair/README.md) |
| 原始实验材料与哈希 | [git_archive_manifest.json](../artifacts/answer_repair/git_archive_manifest.json) |

## 已完成与待完成

NQ Train 的 15 个 Drive 分片共 74,097 条，已核对哈希、逐条适配并隔离 gold。E 内部划分为 59,188 开发、7,345 校准、7,564 预留评测，尚非团队共同未见划分；首批 80 筛查候选加 20 EM 匹配对照全部来自开发部分，人工字段待审。112 项测试和现有 E 入口检查通过，没有新 GPU 结果。这解决了新候选池的来源与工程接入，目标错误数量仍需语义审核确认。

新 Train 首批 100 条已逐条完成代理辅助审核，建议 69 可接受、8 utilisation 候选、23 暂缓；其中 80 条 EM 不匹配候选有 49 条建议可接受。引用原文位置、来源哈希和开发划分已校验，人工确认数为 0。main 的来源优先级、原始 gold 和冻结 scorer 保持不变。优先候选为 N6/N29/N31/N48/N49/N74/N81/N96，不能把本批建议计数当作 E 的修复效果。

v7 主包 541 条、第 4 批 300 条已完成一次 Kaggle 实验，并通过各自冻结代码回放，合计 836 个不同数据集/样本 ID。v7 NQ 使用短答案与引文分离；FEVER 使用主体约束的独立 MNLI。两种策略均未晋升默认策略，不能把工程完成写成算法有效性已验收。

v7 统一审核表的 841 条人工字段均为 pending，先审核其中 95 条答案变化记录（93 个唯一 ID）。第 4 批的上游诊断标签为 AI 初标，输出是否正确、证据是否充分及原失败类型需分别确认。

早期 37 条候选的用户确认已形成 19 条核心错误候选、9 条正确对照和 9 条暂缓；v6r1 的 28 个关键案例（含重复共 30 条记录）已由用户确认，去重后为 9 修复、3 误修、2 暂缓。其余 v6r1 记录仍待确认，见 [确认清单](ANSWER_REPAIR_CONFIRMATION_20261014.md) 与 `data/annotations/human/`。

下一轮可先使用已有人工确认开发样本，修正格式契约、引用定位和 FEVER 标签映射；新语义规则定稿前，优先人工复核少量关键误修和争议案例，不需等待全部样本审核。继续验证原答案核验、最小修复和候选核验，并保留未用于选策略的验收数据。

## 对接边界

E 接受 query、C1 Top-5、初始答案和上游最终 `ControllerDecision`，只在 action 为 `generation_repair` 时运行。其他动作返回原答案并且零额外调用。Diagnosis、Probe、Safe Binary 及路由阈值由上游负责人维护，当前 E 分支未合并其全部实现。

直接执行使用 `scripts/run_answer_repair.py`；服务集成使用 `src/compensation/controller_adapter.py`。模型和 verifier 由调用方复用，策略必须显式选择。运行时不读取 gold、人工正确性或 C0/C3 结果。引用来源校验只证明可追溯，不能证明语义蕴含。

## 复现已有结果

原始输入包、输出 ZIP、预测、审核来源及哈希均保留原位置和原字节。回放使用各自输入包中的冻结代码，输出到新的临时目录；不覆盖既有结果，不重新调用 GPU。具体操作见 [运行说明](ANSWER_REPAIR_COLAB.md)。历史交接中的 81/99 项云端检查数属于相应冻结包，不代表当前代码测试数。
