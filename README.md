# capability-aware-slm-rag

本项目研究小参数模型 RAG 的诊断与动态增强：固定 Reference RAG 产生初答后，由 Diagnosis、P60 Selection 与 Controller 选择是否以及如何增强，再得到 Final Answer。

## 当前默认主线：B + BM25 + 自适应答案选择（2026-10-31）

默认登记为 [quality_adaptive_v1.json](configs/controller/quality_adaptive_v1.json)。
`load_current_pipeline()`、`check_controller_current.py` 和 `run_quality_eval.py` 统一使用它。
配置用途、历史复现和续跑约束见 [配置索引](configs/controller/README.md)。

```text
Query → Reference RAG → Diagnosis / P60 → B Controller
      → 保留或 P3-20 + BM25 混合检索 → 自适应答案选择 → Final Answer
```

| 组件 | 当前默认行为 |
|---|---|
| Reference RAG | matched NQ-Train BGE Top-5 + Qwen2.5-7B NF4 + context_fallback；C1 使用 full prefill |
| Diagnosis / P60 | 78D online provider；Probe OFF；使用 B 在 Validation sweep 上冻结的阈值 |
| B Controller | 选中的三类失败均送到 P3；一级失败概率至少 0.39 的 correct 判断也送到 P3 |
| BM25 下游 | BGE Top-20 与 BM25 Top-20 合并，cross-encoder 选 Top-5；按登记条件重答 |
| 答案选择器 | 同一 Qwen 在 3 / 5 / 8 个检索上下文上作答，复用已有结果；θ=0、τ=0.8 |
| E / 独立重排路线 | 实现保留在仓库；B 的动作映射不触发这两条独立路线；下游仍使用 cross-encoder 重排 |

θ=0 表示每题都进入答案选择阶段，一致性决定何时停止。
本主线已评测 NQ；FEVER 需要独立登记数据边界和评测，不能套用 NQ 的结果。

```bash
python scripts/check_controller_current.py
```

这个检查不启动推理，核对登记、代码及模型指纹和团队数据边界；GPU、检索大文件等运行依赖仍须就位。
实际推理入口：`src.pipeline.current.load_current_pipeline()` → `pipeline.run(query)`。
确认集命令、显式历史配置及跨平台字节要求见 [配置索引](configs/controller/README.md)。

NQ Test 11,114 条的归档判官准确率：团队 RAG 57.50%、B 59.67%、B+BM25 60.62%、
当前组合 62.76%。这是已曝光 Test 的再次评测，详见 [结果和使用次数披露](artifacts/controller/adaptive_selection_eval_v1/README.md)。
接入后的产品代码在确认集 2,440 条上与回放最终答案全部一致，见 [产品等价检查](artifacts/controller/adaptive_product_equivalence_v1/README.md)。
Test 分数来自归档回放；完整新产品代码的等价检查在确认集进行。

团队数据边界见 [数据对齐与边界](docs/TEAM_DATA_BOUNDARIES.md)。大型 index / model / cache 不进 Git。
历史配置、原始结果及其哈希保持原位置，按各自登记复现；切换配置必须使用新的输出目录。

## 历史 Controller 集成（2026-10-23）

`controller` 已汇合当日全部 10 个远端分支的提交历史、功能实现和入库实验结果。
统一入口从冻结 C1 输出开始，执行 Diagnosis → 可选 Probe → Safe Binary → 一条增强，
输出原答、终答、路由、实际调用轨迹、回退原因和增量成本。模型与各模块原有策略保持显式版本；
本次集成不代表准确率提升，也未开始蒸馏、DPO 或 RSI。

集成说明、运行命令、已知限制和下一步见 [Controller 集成交接](docs/CONTROLLER_INTEGRATION.md)，
来源与原始结果一致性见 [分支清单](artifacts/controller/integration_v1/branch_manifest.json)。
P3-20 已补齐 FEVER 检索路径；真实运行需显式提供匹配资源。FEVER 新 900 条 Top5 与 GPU C1 采集包已准备，实际生成状态另行登记。

```bash
python scripts/run_controller_integration.py --smoke --output-dir /tmp/controller-smoke-new
python -m unittest discover -s tests -v
```

Smoke 使用虚构样本和假模型输出，只验证执行链路。真实推理需要 CUDA、匹配 C1 来源的 NQ 索引及元数据。

## 资料入口

当前生产路径以 [配置索引](configs/controller/README.md) 中的 `quality_adaptive_v1` 为准。T1–T9 见 [任务清单](docs/CONTROLLER_TASKS.md)。下表含历史协议与组件记录。

| 需要了解的内容 | 入口 |
|---|---|
| **当前生产链路、默认配置与启动** | [配置索引](configs/controller/README.md) |
| 历史 controller_2 四动作生产基线 | [2026-10-24 基线](docs/CONTROLLER_1024_BASELINE.md) |
| 历史集成、运行、评测与下一步 | [Controller 集成交接](docs/CONTROLLER_INTEGRATION.md) |
| T4 后 E 修订与唯一 NQ 路由候选 | [r2 登记与分支规则](docs/CONTROLLER_E_V2A_R2.md)：18＋2 新验收题已独立复核封存，r2 真实验收未通过，E unavailable；[r1 原始失败](artifacts/controller/e_v2a_interface_r1/README.md)保留，未晋升策略 |
| T1 冻结比较组、预算、评分和停止条件 | [统一实验协议 v1](docs/CONTROLLER_EXPERIMENT_PROTOCOL_V1.md)（设计已冻结；T2/T3 执行门槛待补） |
| T2 样本、完整评测保留池与资源缺口 | [数据冻结 v1](docs/CONTROLLER_DATA_V1.md)：NQ 保留 7,564，FEVER 保留 1,886；C1／检索资源已补齐，新 P60 的来源仍待核对 |
| NQ 原索引配套文件及崩溃核对 | [资源恢复](docs/NQ_TRAIN_INDEX_RECOVERY.md)：ID/offsets 已恢复，64 处向量比对通过；附恢复包与 681 项 CPU 回归记录 |
| NQ 全量定位、Controller Train 接口及 FEVER 资源 | [资源接入](docs/CONTROLLER_RESOURCE_ADAPTER_V1.md)：NQ 全量 1,224,119 段回读，FEVER 全量 5,396,106 段 ID/顺序核对；独立验收入库 |
| FEVER Controller 检索与新 C1 | [采集说明](docs/CONTROLLER_FEVER_C1_V1.md)：10/10 历史检索回放一致，900 条真实 Top5、冻结 GPU 包与 726 项 CPU 回归 |
| 合并前的全分支与数据审计 | [2026-10-23 审计](docs/PROJECT_AUDIT_AND_RSI_PLAN_20261023.md) |
| Diagnosis / Probe / 原路由 | [冻结交接](docs/DIAGNOSIS_PROBE_CONTROLLER_HANDOFF.md) |
| 重排 / 检索增强 | [重排说明](docs/EVIDENCE_REFINEMENT_RERANK.md) / [P3-20](src/compensation/retrieval_enhancement/README.md) |
| 预实验和大模型对照 | [FEVER C0/C3](docs/FEVER_C0_C3.md) / [70B 对照](docs/LARGER_MODEL_CONTROL.md) |
| 项目进度、分支分工与下一轮方向 | [项目进度](docs/PROJECT_STATUS_AND_DIRECTION.md) |
| E 的职责、接口与交接 | [E 说明](docs/ANSWER_REPAIR.md) / [当前交接](docs/ANSWER_REPAIR_HANDOFF.md) |
| 最新完成的 v7 实验 | [冻结协议](docs/ANSWER_REPAIR_V7.md) / [结果及问题分析](docs/ANSWER_REPAIR_RESULTS_20261014_V7.md) |
| v7 统一人工审核 | [可读总表](artifacts/answer_repair/v7_combined_review/review_all.txt) / [结构化记录](artifacts/answer_repair/v7_combined_review/review_all.jsonl) |
| 新 NQ Train 数据与开发审核 | [接入说明](docs/ANSWER_REPAIR_NQ_TRAIN.md) / [100 条审核索引](artifacts/answer_repair/nq_train_dev_v1/review/review_queue.txt) |
| 新 NQ Train 的辅助审核与确认 | [100 条判断及确认记录](docs/ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md)，首轮均已过目：75 条定案（11 个 E 目标、64 个正确对照），25 条待定 |
| 已完成实验的运行和离线回放 | [Colab / Kaggle](docs/ANSWER_REPAIR_COLAB.md) |
| 环境、数据与预实验 | [安装](docs/SETUP.md) / [数据目录](data/README.md) / [预实验协议](docs/PRELIMINARY_EXPERIMENT.md) |
| 旧版文档与 Notebook | [历史归档](docs/history/answer_repair/README.md) |
| 每个文件的用途与保留建议 | [文件清单](docs/REPOSITORY_INVENTORY_20261016.html) |

## E 分支历史进度记录

下面保留原 E 分支的阶段记录，其中“待审核”“未合并”“下一步”等描述以记录当时为准。
本分支当前状态以上方默认主线和 [配置索引](configs/controller/README.md) 为准。

截至 2026-10-19，E v8 的 75 条开发样本四组 GPU 实验已完成，共 300 条输出并通过冻结决策回放；完整 v8 的原答案核验 75 条全部无效回退，未进入修复。原 15 条新改答待人审；后续 r3 的工程修正已通过真实试跑，当前限制转为语义核验，见下方最新结果。见 [v8 方案](docs/ANSWER_REPAIR_V8.md) 与 [实际结果](docs/ANSWER_REPAIR_RESULTS_20261019_V8.md)。v7 历史主包和第 4 批共 841 条结果、836 个唯一数据集/样本 ID，均通过冻结回放。策略均未晋升默认，可靠净收益仍待验收。

v8r3 的固定 8 条、四组 32 条 GPU 输出已完成并通过冻结回放，v8 格式错误为 0，工程门槛通过。完整流程只改答 N96，辅助判断为命名缘由修复，仍待人审；EM 1→1，N8/N31 误放和 N48 正确候选误拒仍存在。见 [r3 结果](docs/ANSWER_REPAIR_RESULTS_20261019_V8_R3.md)。下一步单独验证先提取证据事实、再比较被审答案的语义核验方法；未启动 75 条重跑，策略未晋升默认。

已合入 main 的 NQ Train C1 流水线，并核对 Drive 全部 74,097 条数据，完成 E 格式适配、内部用途划分及首批 100 条审核材料。原始 diagnosis 标签全为 null；75 条定案开发样本的 GPU 结果单独报告。新增 Train 检索库有 1,224,119 passages，其指标与旧 dev 分别报告，详见接入说明。

当前统一审核队列含 95 条答案变化记录，对应 93 个唯一样本，优先审核这些记录；全部 v7 人工字段仍为 pending。已确认的早期开发样本可用于工程开发，无需等待全量审核。第 4 批的诊断分层来自上游 AI 初标，不能视为全量人工标签。

NQ、FEVER 各 500 条 C0/C1/C3 预实验结果已分散保存在实验分支。Diagnosis、Probe、Safe Binary 和 Controller 冻结在 `feature/diagnosis-probe-zheng` 的 `7c7863f`；重排增强在 `reranking` 的 `338ff34`，尚未完成生成质量验证。当前 E 分支未合并这些模块的全部实现，分支记录见项目进度。

## 冻结 Reference RAG（production）

- 生成：Qwen2.5-7B-Instruct，4-bit NF4，确定性生成。
- 检索：BAAI/bge-base-en-v1.5 + matched NQ-Train FAISS，Top-5。
- Prompt：`context_fallback`（见 `configs/controller/reference_rag_v1.json`）。
- 禁止旧 Dev index（`nq_bge_base`）。

历史预实验规模（含旧 NQ validation corpus 等）仍可在预实验协议与 `configs/` 中查阅；**当前 production NQ 检索底座为 matched NQ-Train（约 1,224,119 passages）**。

## 历史与复现

旧版报告、设计和 Notebook 已移入历史目录；原始实验包、预测、人工记录和 [SHA-256 清单](artifacts/answer_repair/git_archive_manifest.json) 保持原位置及原字节。历史包自带冻结代码，复现已有结果时使用该包，不用当前代码重新解释旧输出。

早期 20 条 smoke 只验证工程连通性，不能代替正式 benchmark。其结果与版本记录见 [历史归档](docs/history/answer_repair/README.md)。
