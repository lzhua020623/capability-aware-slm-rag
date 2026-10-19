# E 当前交接

当前最终组件实验版本已选定为 **v8r3**。最后 r4 的 32 条真实输出通过工程检查及冻结回放，但 N31／N48 回归未通过，仅 N12 通过，依运行前规则回退 r3，不追加 r5，也未用 EM 选版本。最终 75 条四组实验使用 `final_nq75/` 中的 `b5be903d…` 冻结包，29 个推理源码文件与 r3 一致，已提交 Kaggle `360070600`，结果尚未收取。根目录 r4 源码保留用于复现最后候选，最终实验应使用选定冻结包；默认策略晋升仍未验收。见 [r4 结果及选择依据](ANSWER_REPAIR_RESULTS_20261020_V8_R4.md) 与 [最终协议](ANSWER_REPAIR_FINAL_PROTOCOL.md)。

截至 2026-10-19，原始 v8（r1）的四组各 75 条 GPU 结果已收取并通过冻结决策回放；该版完整流程在 75 条原答案核验上全部无效回退，未进入修复。后续 r3 已通过真实模型工程验收，当前重点为语义误放与误拒；原 15 条新改答及新增小批量改答仍待人工确认，可靠净收益未完成验收。交付分支为 `feature/verification-answer-repair`，见 [原始 v8 结果](ANSWER_REPAIR_RESULTS_20261019_V8.md)。

## 当前入口

| 内容 | 入口或状态 |
|---|---|
| E 职责、运行时输入与接口 | [ANSWER_REPAIR.md](ANSWER_REPAIR.md) |
| v8 当前实现与 75 条四组实验 | [ANSWER_REPAIR_V8.md](ANSWER_REPAIR_V8.md) |
| v8 真实结果、阶段失败与下一步 | [结果报告](ANSWER_REPAIR_RESULTS_20261019_V8.md) / [15 条改答待审](../artifacts/answer_repair/v8_nq75/review/changed_outputs_pending.jsonl) |
| v8r2 输出契约修正与 8 条工程试跑 | [修正说明](ANSWER_REPAIR_V8_R2.md) / [独立包及运行状态](../artifacts/answer_repair/v8r2_smoke8/README.md) |
| v8r3 候选文法及拒绝分类 | [方案](ANSWER_REPAIR_V8_R3.md) / [冻结包与验证](../artifacts/answer_repair/v8r3_smoke8/README.md) |
| v8r4 最后一次 FACT 核验与测试兼容重试 | [说明](ANSWER_REPAIR_V8_R4.md) / [当前云端状态](../artifacts/answer_repair/v8r4_smoke8/cloud_status.json) |
| 最终 75 条开发实验 | [冻结 r3 包及运行状态](../artifacts/answer_repair/final_nq75/README.md)，`360070600`，300 条新增输出待收取 |
| 一次性 E-only 125 条挑战 | [固定输入、分层及推理包](../artifacts/answer_repair/unseen_nq125/README.md)，`360075807` 已提交排队，团队范围训练隔离待协调 |
| v7 冻结策略与协议 | [ANSWER_REPAIR_V7.md](ANSWER_REPAIR_V7.md) |
| v7 主包与第 4 批结果、失败分析 | [结果报告](ANSWER_REPAIR_RESULTS_20261014_V7.md) |
| 当前人工审核 | [review_all.txt](../artifacts/answer_repair/v7_combined_review/review_all.txt) / [review_all.jsonl](../artifacts/answer_repair/v7_combined_review/review_all.jsonl) |
| 历史 v7 改答固定抽审 | [45 条清单与分层覆盖](../artifacts/answer_repair/v7_historical_review_v1/README.md)，来自 77 条历史 held-out 改答，人工字段全部待审 |
| 新 NQ Train 数据及首批开发审核 | [接入说明](ANSWER_REPAIR_NQ_TRAIN.md) / [100 条待审索引](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.txt) |
| 新 NQ Train 的代理辅助判断与确认 | [100 条理由与确认记录](ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md)，100 条首轮已过目；75 条定案（11 个 E 目标、64 个正确对照），25 条待定 |
| Colab、Kaggle 和离线回放 | [运行说明](ANSWER_REPAIR_COLAB.md) |
| 旧版结果、Notebook 和交接快照 | [历史归档](history/answer_repair/README.md) |
| 原始实验材料与哈希 | [git_archive_manifest.json](../artifacts/answer_repair/git_archive_manifest.json) |

## 已完成与待完成

NQ Train 的 15 个 Drive 分片共 74,097 条，已核对哈希、逐条适配并隔离 gold。E 内部划分为 59,188 开发、7,345 校准、7,564 预留评测，尚非团队共同未见划分；首批 80 筛查候选加 20 EM 匹配对照全部来自开发部分。本地当前 169 项测试通过；r3 的输出格式工程门槛已通过，语义误放与误拒仍存在。已确认开发样本可用，但候选池的总体目标错误数量仍需语义审核。本次人工收口遵循最终协议：最终 75／125 批中 v7 与最终 v8 的全部改答并集，以及历史 v7 的 45 条固定抽审；各中间版本的旧待审记录继续保留。

新 Train 首批 100 条已完成代理辅助审核，初次建议 69 可接受、8 utilisation 候选、23 暂缓，其中 80 条 EM 不匹配候选有 49 条建议可接受。前六批人工意见已覆盖全部 100 条，首轮过目完成：75 条定案（11 个 E 目标、64 个正确对照）、25 条待定、未过目为 0。N76 按用户的荷兰圣尼古拉斯来源说明与 C1 同实体核对，已标明解释依据；25 条待定包含最初 17 条暂缓、N13/N18/N22/N41/N43 口径分歧及 N58/N61/N83 条件或补充要求。N58 的 21 仅有 R1 中持牌商手枪购买条件支持，未擅自改写参考；N83 补充时间为 R1 的截至 2018 年 7 月。首轮过目完成不等于全部主标签或全量 74,097 条已标注，完整分批历史和原话见审核记录。

当前使用 [75 条开发清单](../configs/splits/answer_repair_nq_train_confirmed_dev_v6.json) 与冻结输入，保留 11 个 E 目标及 64 个正确对照用于开发修复／误修检查。25 条待定继续隔离。四组共 300 条真实输出已通过回放，main 来源、gold 和冻结评分保持原样；有选择的开发集及审核计数不能证明模型净收益。

v7 主包 541 条、第 4 批 300 条已完成一次 Kaggle 实验，并通过各自冻结代码回放，合计 836 个不同数据集/样本 ID。v7 NQ 使用短答案与引文分离；FEVER 使用主体约束的独立 MNLI。两种策略均未晋升默认策略，不能把工程完成写成算法有效性已验收。

v7 统一审核表的 841 条人工字段均为 pending，先审核其中 95 条答案变化记录（93 个唯一 ID）。第 4 批的上游诊断标签为 AI 初标，输出是否正确、证据是否充分及原失败类型需分别确认。

早期 37 条候选的用户确认已形成 19 条核心错误候选、9 条正确对照和 9 条暂缓；v6r1 的 28 个关键案例（含重复共 30 条记录）已由用户确认，去重后为 9 修复、3 误修、2 暂缓。其余 v6r1 记录仍待确认，见 [确认清单](ANSWER_REPAIR_CONFIRMATION_20261014.md) 与 `data/annotations/human/`。

本轮完整 v8 与去候选核验组均因 65 条非九行输出、10 条引文定位失败而全部保留原答案，无法评价其保护与修复作用。v7 对照严格 EM 20→23，14 条改答；去原答案核验组 EM 20→20，1 条改答。下一步先修输出契约、真实模型小批量验收，再用新修订重跑固定四组，并人工审核 15 条变化；不覆盖本轮结果。FEVER 现有规则保持，后续固定策略再与团队确认未用于开发的验收数据。

v8r3 的固定 8 条、四组 32 条 GPU 输出已完成并通过冻结回放，v8 格式错误为 0，工程门槛通过。完整流程只改答 N96，辅助判断为命名缘由修复，仍待人审；EM 1→1，N8/N31 误放和 N48 正确候选误拒仍存在。见 [r3 结果](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md)。下一步单独验证先提取证据事实、再比较被审答案的语义核验方法；未启动 75 条重跑，策略未晋升默认。

## 对接边界

E 接受 query、C1 Top-5、初始答案和上游最终 `ControllerDecision`，只在 action 为 `generation_repair` 时运行。其他动作返回原答案并且零额外调用。Diagnosis、Probe、Safe Binary 及路由阈值由上游负责人维护，当前 E 分支未合并其全部实现。

直接执行使用 `scripts/run_answer_repair.py`；服务集成使用 `src/compensation/controller_adapter.py`。模型和 verifier 由调用方复用，策略必须显式选择。运行时不读取 gold、人工正确性或 C0/C3 结果。引用来源校验只证明可追溯，不能证明语义蕴含。

## 复现已有结果

原始输入包、输出 ZIP、预测、审核来源及哈希均保留原位置和原字节。回放使用各自输入包中的冻结代码，输出到新的临时目录；不覆盖既有结果，不重新调用 GPU。具体操作见 [运行说明](ANSWER_REPAIR_COLAB.md)。历史交接中的 81/99 项云端检查数属于相应冻结包，不代表当前代码测试数。

## 2026-10-19 云端加载修复

v8 第 1 版拿到 T4 后在输入加载阶段失败，没有模型预测。Kaggle 会自动解压 tar.gz，因此当前使用 [transport r1 Notebook](../notebooks/answer_repair_v8_nq75_transport_r1.ipynb)。新加载器固定 checksums 文档的 SHA-256，再逐一验证原冻结包的 85 个文件；已验证真实解压布局并通过 7 项加载与回放测试。原 906f096f 包、75 条输入及四组算法参数均未改变。失败与后续执行状态见 `artifacts/answer_repair/v8_nq75/`。
