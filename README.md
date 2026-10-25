# capability-aware-slm-rag

本项目研究小参数模型 RAG 的诊断与动态增强：先用固定检索和生成模型产生 C1 答案，再由 Diagnosis、可选 Probe 和 Controller 选择补检索、重排过滤或答案核验与修复。Reference RAG 是共同实验底座。

## Controller 集成分支（2026-10-23）

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

## 当前入口

当前 T1–T9 阶段、r2 分支条件与新增 Diagnosis 分层合并任务统一见 [任务清单](docs/CONTROLLER_TASKS.md)。r2 已实现并完成全局 E 模式对齐，v2 包引用兼容集成提交 `e3ddead`；P60 新路由未启用。

| 需要了解的内容 | 入口 |
|---|---|
| 当前集成、运行、评测与下一步 | [Controller 集成交接](docs/CONTROLLER_INTEGRATION.md) |
| T4 后 E 修订与唯一 NQ 路由候选 | [r2 登记与分支规则](docs/CONTROLLER_E_V2A_R2.md)：18＋2 新验收题已独立复核封存，r2 已实现，GPU 尚未运行；[r1 原始失败](artifacts/controller/e_v2a_interface_r1/README.md)保留，未晋升策略 |
| T1 冻结比较组、预算、评分和停止条件 | [统一实验协议 v1](docs/CONTROLLER_EXPERIMENT_PROTOCOL_V1.md)（设计已冻结；T2/T3 执行门槛待补） |
| T2 样本、完整评测保留池与资源缺口 | [数据冻结 v1](docs/CONTROLLER_DATA_V1.md)：NQ 保留 7,564，FEVER 保留 1,886；新 FEVER C1 和执行验收仍待补齐 |
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
本分支当前状态以 [Controller 集成交接](docs/CONTROLLER_INTEGRATION.md) 为准。

截至 2026-10-19，E v8 的 75 条开发样本四组 GPU 实验已完成，共 300 条输出并通过冻结决策回放；完整 v8 的原答案核验 75 条全部无效回退，未进入修复。原 15 条新改答待人审；后续 r3 的工程修正已通过真实试跑，当前限制转为语义核验，见下方最新结果。见 [v8 方案](docs/ANSWER_REPAIR_V8.md) 与 [实际结果](docs/ANSWER_REPAIR_RESULTS_20261019_V8.md)。v7 历史主包和第 4 批共 841 条结果、836 个唯一数据集/样本 ID，均通过冻结回放。策略均未晋升默认，可靠净收益仍待验收。

v8r3 的固定 8 条、四组 32 条 GPU 输出已完成并通过冻结回放，v8 格式错误为 0，工程门槛通过。完整流程只改答 N96，辅助判断为命名缘由修复，仍待人审；EM 1→1，N8/N31 误放和 N48 正确候选误拒仍存在。见 [r3 结果](docs/ANSWER_REPAIR_RESULTS_20261019_V8_R3.md)。下一步单独验证先提取证据事实、再比较被审答案的语义核验方法；未启动 75 条重跑，策略未晋升默认。

已合入 main 的 NQ Train C1 流水线，并核对 Drive 全部 74,097 条数据，完成 E 格式适配、内部用途划分及首批 100 条审核材料。原始 diagnosis 标签全为 null；75 条定案开发样本的 GPU 结果单独报告。新增 Train 检索库有 1,224,119 passages，其指标与旧 dev 分别报告，详见接入说明。

当前统一审核队列含 95 条答案变化记录，对应 93 个唯一样本，优先审核这些记录；全部 v7 人工字段仍为 pending。已确认的早期开发样本可用于工程开发，无需等待全量审核。第 4 批的诊断分层来自上游 AI 初标，不能视为全量人工标签。

NQ、FEVER 各 500 条 C0/C1/C3 预实验结果已分散保存在实验分支。Diagnosis、Probe、Safe Binary 和 Controller 冻结在 `feature/diagnosis-probe-zheng` 的 `7c7863f`；重排增强在 `reranking` 的 `338ff34`，尚未完成生成质量验证。当前 E 分支未合并这些模块的全部实现，分支记录见项目进度。

## 冻结 Reference RAG

- 生成模型：Qwen2.5-7B-Instruct，4-bit NF4，确定性生成。
- 检索：BAAI/bge-base-en-v1.5 + FAISS，Top-5。
- 基准：Natural Questions 与 FEVER；固定样本、提示和评分见预实验协议及 `configs/`。

| 数据 | 处理规模 |
|---|---:|
| NQ validation / controlled 子集 | 7,830 / 3,216 |
| NQ 检索 corpus | 99,922 passages |
| FEVER train / dev | 145,449 / 19,998 |
| FEVER 处理后 dev | 6,666 SUPPORTS + 6,666 REFUTES |
| FEVER 有有效 gold evidence 的处理样本 | 13,229 / 13,332 |
| FEVER 检索 corpus | 5,396,106 passages |

以上为处理流程的规模，不表示当前机器保存了全部数据，也不表示这些样本全部完成人工诊断。原始数据和索引按需下载或重建；E 可直接使用冻结 C1 输入开发。

## 历史与复现

旧版报告、设计和 Notebook 已移入历史目录；原始实验包、预测、人工记录和 [SHA-256 清单](artifacts/answer_repair/git_archive_manifest.json) 保持原位置及原字节。历史包自带冻结代码，复现已有结果时使用该包，不用当前代码重新解释旧输出。

早期 20 条 smoke 只验证工程连通性，不能代替正式 benchmark。其结果与版本记录见 [历史归档](docs/history/answer_repair/README.md)。
