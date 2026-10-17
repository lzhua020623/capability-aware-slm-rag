# E 早期进度与整合记录

这是 2026-10-16 整理时从 `docs/PROJECT_STATUS_AND_DIRECTION.md` 提取的历史说明，保留当时的运行状态和结论。当前状态以 [项目进度](../../PROJECT_STATUS_AND_DIRECTION.md)、[v7 结果](../../ANSWER_REPAIR_RESULTS_20261014_V7.md) 和 [当前交接](../../ANSWER_REPAIR_HANDOFF.md) 为准。

> 2026-10-16：已撤回失败的开发实现及其正式版本入口，当前代码回到 v7 基础；新修改重新使用 v8 名称，尚未实现。失败记录隔离在仓库外。先使用既有人审开发样本开展工程修正，同时优先审核 v7 关键误修/争议案例；不要求全量人审完成后才开发。当前人审总表为 `artifacts/answer_repair/v7_combined_review/review_all.txt`。

> 2026-10-14 夜 E 复核：v7 主包 541 条与第 4 批 300 条均通过冻结代码回放（合计 836 个唯一 ID），[结果及路线复核](../../ANSWER_REPAIR_RESULTS_20261014_V7.md)。FEVER held-out 267 条正确数 232→229（1 修复/4 误修），既有 NLI 判断错误也有主体/实体消歧缺陷。NQ held-out 146 条 EM 71→62；3216 分层 300 条 EM 63→67，其中 utilisation 标签层 0→10/120、correct 标签层 63→54/100。v7 语义人工字段仍全部 pending，代理初读不能写成正式语义误修率；第 4 批上游分层标签全为 AI 初标。两种策略仍不晋升。下一轮候选方向为原答案核验→最小修复→候选核验，需保留原答案可见性和核验器的受控消融；具体实现尚待重新开发验证。剩余 1,287 个 NQ 训练 ID 按现有标签只有 correct 1,164 / insufficiency 123，没有 utilisation/accessibility，不能单独验收目标修复率。需定向补充新的自然错误及正确对照，保留共同未见系统测试；不优先更换主数据集。

> 2026-10-14 晚 E 更新：128 条辅助审核的 28 个关键案例由独立盲评二读，9 个修复、3 个误修完全一致，7 个未决项给出默认建议并已由用户确认（去重后 9 修复 / 3 误修 / 2 暂缓；其余 97 个唯一样本待批量确认），确认清单在 [ANSWER_REPAIR_CONFIRMATION_20261014.md](../../ANSWER_REPAIR_CONFIRMATION_20261014.md)。v7 已冻结打包未运行（[ANSWER_REPAIR_V7.md](../../ANSWER_REPAIR_V7.md)）：NQ 短答案与引文分离，FEVER 只允许含声明主体的窗口投票；用 v6r1 轨迹离线回放，FEVER 审核后 4→10、dev50 45→46、无误修（设计阶段证据）。E 未触碰的共享训练样本已全部冻结为 held-out（FEVER 267、NQ 146），但其中没有 utilisation 标注；含 120 条 utilisation 的 NQ 3216 池 held-out 已冻结 ID，等 Diagnosis 负责人提供 gold。另注意：冻结 Controller 在 PRIMARY test 上从不把 FEVER 路由到 generation_repair，所以 E 的 FEVER 结果只用于 Always-Verify 基线和失败类型 × 操作矩阵，不进入自适应系统的 FEVER 路径。

> 2026-10-14 E 本地更新：37 条聊天辅助初始样本审核已确认，纳入 19 条 utilisation 核心候选 + 9 条正确对照，9 条暂缓。v5 的 112 条真实 T4 输出已归档并回放；同模型复核没有改善结果。v6r1 在 Kaggle 完成 128 条真实预测（123 个唯一 ID）、81 项云端检查及全部冻结代码回放，GPU 会话已释放。FEVER 审核后 4→7，但 dev50 45→44；NQ dev50 EM 20→7，包含长答案与无关片段问题。两种新策略均未晋升默认。工程和本轮开发实验已交接，人工正确性/支持性与团队 held-out 验收仍待完成；新版 42 条改动已优先排序。本次未重新同步远端。见 [当前交接](../../ANSWER_REPAIR_HANDOFF.md) 和 [最新结果](ANSWER_REPAIR_RESULTS_20261014.md)。

> 2026-10-13 更新：已重新 fetch 所有远端分支。Diagnosis / Probe / Controller 有冻结交接；新增 `reranking` 分支有检索侧评估；E 已完成真实 T4 开发对照及冻结 Controller→E 整合。E 工程可运行，但可靠算法收益尚未通过验收，见 [实测报告](ANSWER_REPAIR_RESULTS_20261013.md)。本地仍没有 CUDA。

2026-10-14 辅助审核补充：已审阅新版全部 128 条结果（123 个唯一样本），逐条记录前后正确性、Top-5 支持性、实际引用判断及理由。去重后代理暂判 9 个修复、3 个误修；7 个样本存在未决字段。NQ 的 15 条 EM 退步暂判均为有据扩写，并发现 6 个 EM 漏掉的唯一修复候选和 1 个漏掉的误修。判断使用可见参考及 C1，尚未人工确认；这改变对 NQ EM 下降的解释，不改变冻结指标或默认策略。见最新结果报告中的代理辅助审核部分。

## 10 月 13 日当前交接状态

Diagnosis、Probe、Safe Binary 和 Controller 已有冻结实现及验证/测试产物，负责人交接文件为远端分支的 `docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`。E 消费其最终 `ControllerDecision`，不重训或调整上述模块。已有 FEVER/NQ 各 500 条的 normalized diagnosis 标注，以及 NQ 扩展标注；这属于项目固定实验样本，不是 FEVER/NQ 原始数据集全量的四类 failure 标注。

E 的 `generation_repair` 接口与上游真实 `ControllerDecision` 类已完成 CPU 契约验证：四种 action 只在 E 路由时产生一次调用，Safe Binary 改回 correct 后零调用。代码及协议测试共 54 项通过。另在 T4 实际运行冻结 classifier、27 次 Probe 和 1 次 E；100 条开发样本没有净收益，其他两类补偿尚未在这次整合中执行。

E 的固定开发 cohort 使用训练划分交集，排除 Diagnosis 两版 validation/test。v1–v3 暂无正确率增益；v4 隐藏原答案的带引用候选生成在 FEVER 50 条上 45→46，但新增正确引用了错误实体；NQ 50 条为 20→20，改对 1 条、改错 1 条，F1 略降。相同额外预算的直接 Reanswer 对照无增益。10 条 smoke 包含在 50 条中，不能重复计数。工程接口可交接，算法有效性和整个 adaptive 系统尚未完成。

`reranking` 默认 keep5（只重排），因为 keep3 的 frozen500 检索侧比较丢失了 14 条含答案证据；该分支尚未测生成质量。其 500 条比较包含 Diagnosis v2 的 65 个 validation、68 个 test ID，选择默认策略后，这批数据不宜再作为完整 adaptive framework 未见测试集。这个问题不改写已冻结的 Diagnosis 自身指标。

下一步：冻结 v4 负结果 → 训练内新 utilisation 分层 cohort 与语义核验 → 分别验证 E 能力和冻结路由覆盖 → 汇合三个补偿操作 → 用共同未用于选策略的数据评测质量和完整成本。引用来源合法不代表支持结论，独立 faithfulness/verifier 仍需另行评估。E 的最终完成标准见 [ANSWER_REPAIR.md](../../ANSWER_REPAIR.md)。

初次核对日期：2026-10-11，最近同步：2026-10-13。方向依据为团队的 `ELEC5623_Group12_Proposal_v21.pdf`，实际实验配置以冻结配置及结果文件为准。当前阶段应从 recoverability 预实验转入补偿操作验证及自适应系统评测。E 的当前分工按本次讨论确定为 Verification / Answer Repair；diagnosis 由其他成员负责。Proposal 第 10.2 节的旧人员分工不覆盖这次新分工。
