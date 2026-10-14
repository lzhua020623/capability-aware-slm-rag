# 项目进度与研究方向

> 2026-10-14 夜 E 更新：v7 已在 Kaggle 跑完一次（[结果](ANSWER_REPAIR_RESULTS_20261014_V7.md)）。FEVER 主体实体约束在开发批次与离线回放一致（4→10、45→46、无误修），held-out 267 条 1 修复 / 4 误修（误修都是 NLI 在实体正确窗口上的误判）；NQ 短答案策略冻结 EM 在 held-out 146 条上 71→62（p=0.0225），放宽解析后更差（71→58），改动里约 5 条真实语义误修。两种策略均不晋升。候选优先的 NQ 设计会改坏正确答案，若做 v8 应回到 verify-first 并用 3216 池剩余 1,287 条未用 ID 做新 held-out。`heldout3216`（300 条，含 120 条 utilisation）也已跑一次：utilisation 层 EM 0→10（语义约 13/120）、accessibility 0→3、correct 层 63→54（7 条假退步、1 条争议、1 条真实误修），整体 63→67（p=0.52）。这是 E 首次在未见数据上证明对 utilisation 失败有修复能力，但两批 held-out 合计 correct 样本真实误修约 5%，在 Controller 误路由（48 条 generation_repair 中 32 条本来正确）下仍不可接受，结论不变。

> 2026-10-14 晚 E 更新：128 条辅助审核的 28 个关键案例由独立盲评二读，9 个修复、3 个误修完全一致，7 个未决项给出默认建议并已由用户确认（去重后 9 修复 / 3 误修 / 2 暂缓；其余 97 个唯一样本待批量确认），确认清单在 [ANSWER_REPAIR_CONFIRMATION_20261014.md](ANSWER_REPAIR_CONFIRMATION_20261014.md)。v7 已冻结打包未运行（[ANSWER_REPAIR_V7.md](ANSWER_REPAIR_V7.md)）：NQ 短答案与引文分离，FEVER 只允许含声明主体的窗口投票；用 v6r1 轨迹离线回放，FEVER 审核后 4→10、dev50 45→46、无误修（设计阶段证据）。E 未触碰的共享训练样本已全部冻结为 held-out（FEVER 267、NQ 146），但其中没有 utilisation 标注；含 120 条 utilisation 的 NQ 3216 池 held-out 已冻结 ID，等 Diagnosis 负责人提供 gold。另注意：冻结 Controller 在 PRIMARY test 上从不把 FEVER 路由到 generation_repair，所以 E 的 FEVER 结果只用于 Always-Verify 基线和失败类型 × 操作矩阵，不进入自适应系统的 FEVER 路径。

> 2026-10-14 E 本地更新：37 条聊天辅助初始样本审核已确认，纳入 19 条 utilisation 核心候选 + 9 条正确对照，9 条暂缓。v5 的 112 条真实 T4 输出已归档并回放；同模型复核没有改善结果。v6r1 在 Kaggle 完成 128 条真实预测（123 个唯一 ID）、81 项云端检查及全部冻结代码回放，GPU 会话已释放。FEVER 审核后 4→7，但 dev50 45→44；NQ dev50 EM 20→7，包含长答案与无关片段问题。两种新策略均未晋升默认。工程和本轮开发实验已交接，人工正确性/支持性与团队 held-out 验收仍待完成；新版 42 条改动已优先排序。本次未重新同步远端。见 [当前交接](ANSWER_REPAIR_HANDOFF.md) 和 [最新结果](ANSWER_REPAIR_RESULTS_20261014.md)。

> 2026-10-13 更新：已重新 fetch 所有远端分支。Diagnosis / Probe / Controller 有冻结交接；新增 `reranking` 分支有检索侧评估；E 已完成真实 T4 开发对照及冻结 Controller→E 整合。E 工程可运行，但可靠算法收益尚未通过验收，见 [实测报告](ANSWER_REPAIR_RESULTS_20261013.md)。本地仍没有 CUDA。

2026-10-14 辅助审核补充：已审阅新版全部 128 条结果（123 个唯一样本），逐条记录前后正确性、Top-5 支持性、实际引用判断及理由。去重后代理暂判 9 个修复、3 个误修；7 个样本存在未决字段。NQ 的 15 条 EM 退步暂判均为有据扩写，并发现 6 个 EM 漏掉的唯一修复候选和 1 个漏掉的误修。判断使用可见参考及 C1，尚未人工确认；这改变对 NQ EM 下降的解释，不改变冻结指标或默认策略。见最新结果报告中的代理辅助审核部分。

## 10 月 13 日当前交接状态

Diagnosis、Probe、Safe Binary 和 Controller 已有冻结实现及验证/测试产物，负责人交接文件为远端分支的 `docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`。E 消费其最终 `ControllerDecision`，不重训或调整上述模块。已有 FEVER/NQ 各 500 条的 normalized diagnosis 标注，以及 NQ 扩展标注；这属于项目固定实验样本，不是 FEVER/NQ 原始数据集全量的四类 failure 标注。

E 的 `generation_repair` 接口与上游真实 `ControllerDecision` 类已完成 CPU 契约验证：四种 action 只在 E 路由时产生一次调用，Safe Binary 改回 correct 后零调用。代码及协议测试共 54 项通过。另在 T4 实际运行冻结 classifier、27 次 Probe 和 1 次 E；100 条开发样本没有净收益，其他两类补偿尚未在这次整合中执行。

E 的固定开发 cohort 使用训练划分交集，排除 Diagnosis 两版 validation/test。v1–v3 暂无正确率增益；v4 隐藏原答案的带引用候选生成在 FEVER 50 条上 45→46，但新增正确引用了错误实体；NQ 50 条为 20→20，改对 1 条、改错 1 条，F1 略降。相同额外预算的直接 Reanswer 对照无增益。10 条 smoke 包含在 50 条中，不能重复计数。工程接口可交接，算法有效性和整个 adaptive 系统尚未完成。

`reranking` 默认 keep5（只重排），因为 keep3 的 frozen500 检索侧比较丢失了 14 条含答案证据；该分支尚未测生成质量。其 500 条比较包含 Diagnosis v2 的 65 个 validation、68 个 test ID，选择默认策略后，这批数据不宜再作为完整 adaptive framework 未见测试集。这个问题不改写已冻结的 Diagnosis 自身指标。

下一步：冻结 v4 负结果 → 训练内新 utilisation 分层 cohort 与语义核验 → 分别验证 E 能力和冻结路由覆盖 → 汇合三个补偿操作 → 用共同未用于选策略的数据评测质量和完整成本。引用来源合法不代表支持结论，独立 faithfulness/verifier 仍需另行评估。E 的最终完成标准见 [ANSWER_REPAIR.md](ANSWER_REPAIR.md)。

初次核对日期：2026-10-11，最近同步：2026-10-13。方向依据为团队的 `ELEC5623_Group12_Proposal_v21.pdf`，实际实验配置以冻结配置及结果文件为准。当前阶段应从 recoverability 预实验转入补偿操作验证及自适应系统评测。E 的当前分工按本次讨论确定为 Verification / Answer Repair；diagnosis 由其他成员负责。Proposal 第 10.2 节的旧人员分工不覆盖这次新分工。

## 项目要解决什么

同一个小模型 RAG 出错，可能是没有拿到证据、拿到证据却被噪声干扰，或者证据清楚但答案仍然错误。项目研究这些原因是否对应不同的有效补偿操作，以及按原因选择操作，能否比固定执行全部操作更省成本，同时保持答案质量与证据支持。

```mermaid
flowchart LR
    Q[问题或声明] --> R[BGE与FAISS检索Top 5]
    R --> A[固定Qwen 7B生成初始答案]
    A --> D[Diagnosis]
    D --> P[不确定时一次Probe]
    D --> C[Controller]
    P --> C
    C --> K[保留初始答案]
    C --> I[Retrieval Enhancement]
    C --> F[Reranking与Filtering]
    C --> E[Verification与Answer Repair]
    I --> O[最终答案与证据及成本记录]
    F --> O
    E --> O
    K --> O
```

Reference RAG 提供共同实验底座。项目的主要贡献需要来自诊断、操作匹配、选择性 Probe，以及质量和成本之间的实验比较。仅增加一条修复 prompt 不足以验证整体研究假设。

## 所有远程分支的实际进度

| 分支 | 核对提交 | 已有产物 | 当前限制 |
|---|---|---|---|
| `main` | `c15977c` | Reference RAG、数据处理、冻结预实验协议 | README 仍写预实验未开始；没有汇合后续成果 |
| `feature/nq-c0-c1` | `bcb248c` | NQ C0/C1 各 500 条、续跑脚本 | 没有 NQ C3 |
| `feature/nq-c3` | `8b03009` | 汇合 NQ C0/C1/C3、recoverability cohorts | 没有 FEVER C1 |
| `feature/fever-c1` | `6ebdfc9` | 全量 FEVER 索引 C1 500 条、检索 checkpoint、FEVER 配对分析；包含 NQ 分支成果 | 没有本次 FEVER C0 runner 及其结果 |
| `feature/F500-c0c3` | `1238a4f` | FEVER C0/C3 各 500 条、完整续跑校验、全量索引 metadata | 本地原分支停在 `064447d`，落后 metadata 提交 |
| `feature/-ce-70B` | `6f08f2f` | NQ C0/C1/C3、FEVER C1/C3、NQ judge 复评及 Llama 70B 对照 | 不包含 FEVER C0；没有完成配置中要求的 Qwen 32B 对照 |
| `feature/diagnosis-probe-zheng` | `7c7863f` | normalized 标注、Diagnosis v2-expanded、Probe、Safe Binary、Controller、冻结划分及测试交接 | utilisation/accessibility 路由精度和覆盖有限；上游只做路由，不执行补偿 |
| `reranking` | `338ff34` | 已汇合冻结上游；证据重排接口、68 项 CPU 测试及 NQ frozen500 keep3/keep5 检索侧比较 | 未测最终答案收益；keep5 为开发结果选定默认 |

原有本地 `main`、`Qihan` 都位于 `c15977c`；原有本地 `feature/F500-c0c3` 位于 `064447d`。现已同步远程引用。导致 fetch 失败的非法重复引用 `origin/HEAD 2` 已移到 `.git/codex-backups/origin-HEAD-2-20261011` 保存，原分支提交未改写。

E 开发与交付分支为 `feature/verification-answer-repair`，以 diagnosis 接口分支为基底。其他分支的实验结果通过 Git ref 读取，不需要复制数据集、检索库或重新跑 C1。实现、冻结实验归档和辅助审核一并纳入版本控制，最新交付以该分支的 Git 记录为准，复现入口见 `ANSWER_REPAIR_HANDOFF.md`。

## 预实验已经确认的事实

10 月 11 日逐条核对了以下数据集每个条件的 500 个唯一 ID，并确认 C0/C1/C3 的 ID 集合一致。当时七个远程分支的 `configs/preliminary.yaml` 和两个 sample manifests 字节完全一致。70B 分支保存的 NQ/FEVER C1/C3 文件与原产出分支字节一致。

| 数据集与指标 | C0 无证据 | C1 原始 Top 5 | C3 gold evidence |
|---|---|---|---|
| NQ EM | 51/500 = 10.2% | 176/500 = 35.2% | 258/500 = 51.6% |
| NQ token F1 | 0.1939 | 0.4899 | 0.6778 |
| FEVER accuracy | 360/500 = 72.0% | 402/500 = 80.4% | 476/500 = 95.2% |

来源：NQ 与 FEVER C1/C3 在 `feature/-ce-70B` 的 `results/preliminary/`；FEVER C0 在 `feature/F500-c0c3` 同一目录。

Recoverable Failure Rate 的冻结定义是 `C1 wrong 且 C3-7B correct / C1 wrong`：

| 口径 | C1 错误 | gold evidence 后恢复 | RFR | C1/C3 均错误 |
|---|---|---|---|---|
| NQ 冻结 EM | 324 | 97 | 29.94% | 227 |
| FEVER 冻结标签 | 98 | 80 | 81.63% | 18 |
| NQ judge 复评 | 192 | 125 | 65.10% | 67 |

NQ judge 复评后的 C1/C3 正确率分别为 61.6%/81.8%。冻结指标要求最短 answer span，而一些 gold 字符串更长或存在 token spacing 差异；这会混入评测误判。Judge 文件有 626 条 C1/C3 verdict，包含 566 条 EM 错误答案和 60 条 EM 正确答案的检查。60/60 一致只验证了这一批已正确答案，不能证明 judge 没有宽松误判。冻结 EM 和 judge 复评必须分别报告，不能把复评结果写成已经完成的人类金标准标注。

更大模型对照实际使用 Llama-3.3-70B：NQ 在 judge 修正后的 67 个 persistent failures 上，EM 恢复 8 个、judge 恢复 15 个；FEVER 在 18 个 persistent failures 上恢复 8 个。统一采用该分支复评口径时合计为 23/85。它支持“这些选中的 7B 失败中有一些可由该 70B 模型恢复”，不能独立证明参数规模是原因，也不能比较两个模型的总体准确率；模型家族、推理环境及评测口径都发生了变化。剩余 62 个案例应称为当前设置下尚未解决，不能据此断言不可修复。

C3 的一次直接生成也不是所有外部补偿策略的严格上界。由上述配对计数可得，NQ 有 15 个、FEVER 有 6 个样本在 C1 正确而 C3 错误；改变提示、约束输出或增加核验可能产生不同结果。

## Diagnosis 与标注状态

10 月 13 日远端已发布上述产物，替代 10 月 11 日“仅接口、未找到标签”的历史结论。`feature/diagnosis-probe-zheng` 中有 `fever_500_labels.jsonl`、`nq_500_labels.jsonl` 和 `nq_3216_labels_normalized.jsonl`，以及训练/验证/测试划分、classifier 参数和冻结评测。它们是项目样本的四类 diagnosis 标注，不是原始 FEVER/NQ benchmark 全量 failure 标注。

Diagnosis v2-expanded 使用 NQ 1889/406/403、FEVER 331/71/71 的 train/validation/test 划分；过滤 `needs_manual_review` 后还有未纳入样本。上游 PRIMARY test 473 条 pipeline accuracy 0.691、macro-F1 0.370、utilisation F1 0.217。E 路由 48 条中只有 9 条标注 utilisation，32 条原本 correct，因此必须评估误路由下的误修风险。详细来源为冻结 `DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md`，这些汇总只用于交接，没有使用测试单条数据调 E。

当前 `gold_evidence_in_top5` 是离线检索命中标记。FEVER 代码按任意 gold page 命中判断，命中页面不能保证所需全部证据句或证据组合存在。NQ 命中 gold passage 也不能证明上下文已经清晰。不能直接用这个布尔值区分 accessibility 与 utilisation，更不能把它输入在线 diagnosis 或 E。

## Proposal 与实际实现需要对齐的地方

| 项目 | Proposal v21 | 仓库实际状态与建议 |
|---|---|---|
| 运行模型 | 3B 候选，视 pilot 调整到 3B–7B | 已冻结 Qwen2.5-7B-Instruct、NF4、greedy；主实验继续保持这一设置 |
| 大模型对照 | 同家族 14B，资源允许时增加 32B | 冻结配置改为要求 32B；实际跑的是 Llama 70B。报告需要如实记录替代及其限制 |
| C2 | 清洗/重排 C1 中已有证据的 accessibility control | 冻结预实验明确排除 C2。需要补充该控制或明确采用人工证据审核等替代标签依据；E 的结果命名为 `E` |
| diagnosis | 三个独立 sigmoid 分数，允许多标签 | 接口只有一个主标签、confidence、needs_probe。主标签可用于 MVP 路由，但不等价于多标签 profile；不能按当前接口宣称满足 FR6 |
| Evidence Utilisation 标签 | C3 下仍错误/unsupported | 当前 runtime 定义是实际 C1 证据清楚充分但回答错误。两种条件有差异；离线 oracle failure 不能直接等同于在线 C1 utilisation |
| 证据支持核验 | 共用轻量 NLI，并审核 unresolved flags | E v6r1 已冻结并实测独立 MNLI，记录完整概率与成本；存在错误实体误判，尚未验收 faithfulness 或接成全系统共用核验组件 |
| Probe 推断 | 响应变化用于更新 failure profile | 响应变了不一定更正确；响应没变不证明能力限制。需要核验证据支持，并对 Probe 的诊断收益做消融 |

建议保留“固定 7B + 原因诊断 + 选择性补偿 + 质量/成本比较”的主线，先把能复现实验的 MVP 做实。若采用单主标签简化，报告应明确这是工程调整；若坚持 proposal 的多标签要求，需要由 diagnosis 负责人扩展 profile 输出及评测，E 本身无需改变。

## 下一阶段需要验证的假设

第一步：冻结标签依据和 ID 划分，检查 NQ judge 翻转样本，补足 accessibility/utilisation 的证据。已有 500+500 预实验样本可作开发材料；若它们用于挑选策略或阈值，最终主结果宜采用剩余未使用样本作为独立评测集。资源限制下复用这些样本时，应报告为探索性结果。

第二步：三个补偿操作分别可运行。E 先在包含原本正确及错误答案的固定开发样本上运行 Always-Verify，统计改对、改错、解析失败与成本；再接 diagnosis 主标签做选择性修复。单独在已知错误集上跑不能测量误修风险，也不能证明总体提升。

第三步：按 diagnosis 分组，每个样本都评测三个操作，形成“失败类型 × 操作”的效果矩阵。只有此时才能验证图中的对应关系。对同一批样本使用配对比较和置信区间；若有多重比较，应在解释显著性时处理相应影响。

第四步：端到端比较 Vanilla RAG、Always-Rerank、Always-Verify、Full Enhancement 和 adaptive framework；另比较无 Probe 与选择性 Probe。记录包括 diagnosis 特征提取在内的全部模型调用、token、延迟，不能只计补偿操作的成本。

E 的交付和执行方法见 [ANSWER_REPAIR.md](ANSWER_REPAIR.md)，最新真实结果见 [10 月 14 日报告](ANSWER_REPAIR_RESULTS_20261014.md)。81 项检查、v6r1 全部 128 条真实 GPU 结果回放及 100 条历史 Controller 契约回放通过；可靠净收益与人工 faithfulness 验收尚未完成。后续先完成成对语义复评，再决定短答案/独立引文与实体约束的下一版，随后由团队开展三个操作的共同 held-out 比较。

## 方法依据

[D2R-RAG](https://proceedings.mlr.press/v318/hashemifar26a.html) 已研究 observable diagnosis 和预算约束下的 repair selection。因此整体贡献要通过小模型失败类型、选择性 evidence Probe 和质量/成本实验具体体现。

[Self-Refine](https://arxiv.org/abs/2303.17651) 提供同一模型反馈和修订的研究参照；[自纠错局限研究](https://arxiv.org/abs/2310.01798) 在其推理设置中发现，缺少外部反馈的自纠错可能无效或造成退化。它们不直接给出本项目 7B 的效果，E 的证据约束及误修率需要实测。
