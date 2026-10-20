# E 当前交接

最终组件实验版本为 **v8r3**，本轮已停止调参。2026-10-21 收齐最终75开发的300条、一次性125挑战的250条新增预测，并通过各自冻结代码回放及汇总复算。v7 的 EM 分别为20→23、25→28；完整r3为20→20、25→25，只在开发集改答N96。**r3未晋升默认策略**：几乎不改答使其未见收益为零。最后r4未通过N31／N48预声明回归，按原规则回退r3，不追加r5，也没有按本轮EM选版本。详细数字与限制见 [最终结果报告](ANSWER_REPAIR_RESULTS_20261021_FINAL.md)。

根目录r4推理源码保留用于复现最后候选；复现最终实验必须使用 `final_nq75/` 与 `unseen_nq125/` 中选定r3的冻结包。两包29个推理源码文件与选定源一致。交付分支为 `feature/verification-answer-repair`。

## 当前入口

| 内容 | 入口与状态 |
|---|---|
| 最终数字、失败机制、成本、适用边界 | [最终报告](ANSWER_REPAIR_RESULTS_20261021_FINAL.md) |
| 冻结范围与停止调参规则 | [最终协议](ANSWER_REPAIR_FINAL_PROTOCOL.md) |
| 最终75开发结果与回放 | [final_nq75](../artifacts/answer_repair/final_nq75/README.md)，300条已核对 |
| 一次性125分层挑战结果与回放 | [unseen_nq125](../artifacts/answer_repair/unseen_nq125/README.md)，250条已核对 |
| 最终主策略全部改答 | [F1–F42审核网页](../artifacts/answer_repair/final_review_v1/F42_review.html)／[范围与分层](../artifacts/answer_repair/final_review_v1/README.md)，辅助42/42，人工0/42 |
| 历史v7改答固定抽审 | [H1–H45审核网页](../artifacts/answer_repair/v7_historical_review_v1/H45_review.html)／[范围](../artifacts/answer_repair/v7_historical_review_v1/README.md)，辅助45/45，人工0/45 |
| E职责、运行时输入及接口 | [ANSWER_REPAIR.md](ANSWER_REPAIR.md) |
| v7旧批结果及完整队列 | [v7报告](ANSWER_REPAIR_RESULTS_20261014_V7.md)／[841条历史队列](../artifacts/answer_repair/v7_combined_review/review_all.jsonl) |
| NQ Train数据接入与初始人审 | [数据说明](ANSWER_REPAIR_NQ_TRAIN.md)／[原100条审核记录](ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md) |
| 云端运行与离线回放 | [运行说明](ANSWER_REPAIR_COLAB.md) |
| 旧版结果及交接快照 | [历史归档](history/answer_repair/README.md) |
| 归档指纹 | [git_archive_manifest.json](../artifacts/answer_repair/git_archive_manifest.json) |

## 本轮完成与待完成

组件代码、真实GPU执行、冻结回放、成本记录及代理辅助审核已完成。所有新最终语义判断仍是参考可见的代理意见，不能作为人工金标或最终净收益。F13与F15为同一题的不同策略输出，不能相加成两个独立修复。消融14条额外改答留在完整审计，未混入主审核范围。历史H清单仅覆盖77条held-out改答中的固定45条，不代表841条历史预测都已人工审核。

人工收口优先确认F1–F42，再确认H1–H45；均可按编号在聊天中回复。F4、F19、F21、F22、F35、F36、F40至少一项判断保留null。F32的FIFA／时点口径、F30的degree／diploma口径也保留说明。未经用户确认，不更改pair_review，不改gold，不删除歧义题。最终语义收益按批次和策略分别结算，不用只审改答推算完整语义准确率。

最终75沿用11个初始人审错误目标、64个正确对照；原100条中的25条待定继续隔离。NQ Train 15个分片74,097条已核验接入，内部划分为59,188开发、7,345校准、7,564预留评测。125挑战自动来自E未调参的校准部分，100个EM筛查候选不等于100个utilisation错答。其ID已登记为E已使用，不得继续用于选模型或调规则；团队Diagnosis训练隔离和页面／语义近重复隔离尚未建立。

## 团队端到端交接

待与Zheng确认执行人、GPU机器、共同测试ID、Diagnosis训练排除和测试数据角色。本轮使用 `--all` 的E组件结果不能替代实际Diagnosis→Controller→E端到端结果，也未声称已完成团队协调。下一步在既定数据和冻结策略下检查实际触发的修复、误修及成本，避免因挑题或换策略污染评测。

E接受query、实际C1 Top-5、初始答案及上游最终ControllerDecision；只在action为generation_repair时运行，其他动作保留原答案并零额外调用。模型与verifier由调用方复用，运行时不得读取gold、人工正确性或C0/C3。直接入口为scripts/run_answer_repair.py，服务适配为src/compensation/controller_adapter.py。引用定位证明可追溯，不能证明语义蕴含。

早期v6r1确认结果仍保留于 [确认清单](ANSWER_REPAIR_CONFIRMATION_20261014.md) 和data/annotations/human/，不与本轮人审混算。FEVER仍是历史v7主体约束MNLI策略，本轮最终75／125均为NQ，不能把NQ实验结论直接推广到FEVER。

## 复现与下一轮边界

原始输入包、输出ZIP、预测和日志保持原字节，接收回执记录逐文件SHA；回放不重新调用GPU。接收使用scripts/collect_answer_repair_final.py，合成结果默认拒收。审核并集使用scripts/prepare_answer_repair_final_review.py；代理意见独立存放并验证来源定位，人工字段保持pending。

本轮不追加优化或数据增强。若后续研究更换核验模型、训练或放宽来源约束，应建立独立后续实验和新的未见评测，不继续使用本轮125条挑选方案。当前首先完成用户确认、报告结算和团队端到端执行安排。
