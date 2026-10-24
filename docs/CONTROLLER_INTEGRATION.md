# Controller 集成交接 · 2026-10-23

2026-10-24 当前执行入口：[T3 执行登记](CONTROLLER_T3_EXECUTION_V1.md)。FEVER 900 条 C1 已完成并验收，两套完整检索资源已到位；Full/No Probe/离线人工类型接口、统一预算与标签适配、严格评测已补齐，773项完整回归通过，[最终发布包与输入预检](../artifacts/controller/t3_v1/README.md)已封存。真实CUDA工程验收V1及本地输出验收已通过，T4 Notebook已绑定凭证生成；检索资源正在私有上传，T4尚未启动，下面逐次状态保持历史含义。

当前有效开发分支为 `controller`。本次汇合当日 10 个远端分支，并实现从冻结 C1 输出到最终答案的统一执行与离线评测入口。此版本用于建立系统基线，未晋升任何 E 策略为已证明有效的默认方案；尚未完成真实 GPU 端到端质量实验，未开始 Qwen 训练或 RSI。

后续实验设计已在 [T1 统一实验协议 v1](CONTROLLER_EXPERIMENT_PROTOCOL_V1.md) 冻结：六个主比较组、两个解释性对照、每组预算、配对评分和一次集中优化限制。T2 样本已冻结、资源验收仍待补齐，T3 仍需补齐 Full、FEVER 检索及对照/验收；本页的“已有”描述仅对应参考实现，不表示所有协议组均可执行。

T1 的完整 CPU 回归为 648 项通过（新增 13 项协议测试），68 个参考文件无漂移，验证归档见 [protocol_v1](../artifacts/controller/protocol_v1/README.md)。下面第 4 节的 635 项是先前集成提交的历史验收数量。

T2 已冻结 [两数据集的样本和完整评测保留池](CONTROLLER_DATA_V1.md)：NQ 7,564、FEVER 1,886 条禁止用于训练/调参；各抽 500 条为本轮最终子集，开发/选型各 200 条。NQ 配对输入已导出，FEVER 仅生成 query-only C1 任务；该冻结提交时匹配索引、BGE revision、新 FEVER C1 待补齐，CPU 回归 665 项通过，未宣称 T2 全部就绪。NQ Train 的 ID/offset/corpus 资源格式也需在 T3 接入。

2026-10-24 已[取回 NQ Train 原语料/索引并恢复配套文件](NQ_TRAIN_INDEX_RECOVERY.md)，64 段独立 BGE 向量比对通过，最新 CPU 回归 681 项通过。所核验快照与历史 C1 revision 的区别保留在恢复记录中；FEVER 和 Controller provider 接入仍按原 T2/T3 待办推进，原冻结样本、保留池及算法提示不改。

随后完成 [NQ 全量定位核对、Train provider 接入和 FEVER 资源导入](CONTROLLER_RESOURCE_ADAPTER_V1.md)。NQ 新增 `--nq-resources` 入口；FEVER 原包及全量 ID/原文顺序已验收，FEVER 动态 provider、新 C1 和其余 T3 组仍待完成。下方旧接口说明与历史验收数字保留；资源接入的新增验证见链接。

最新已完成 [FEVER provider 与新 900 条 C1 采集包](CONTROLLER_FEVER_C1_V1.md)，全仓库 726 项 CPU 回归通过；固定 10 条历史查询 Top5 ID/顺序全部复现。FEVER 动态补检索现可通过 `--fever-resources` 启用，新 C1 实际 GPU 状态单独登记。下方 FEVER unsupported 描述对应初始合并版本；Full/No Probe/人工类型组和正式 seal 仍待 T3。

## 1. 来源与合并

从 E 分支 `5da1fe5` 创建新分支，依次合并：

| 合并源 | 源提交 | 内容 |
|---|---|---|
| `main` | `058f10b` | 已是 E 祖先；Reference RAG、NQ Train 流水线 |
| `reranking` | `338ff34` | 包含 Diagnosis／Probe／Controller `7c7863f`，以及证据重排 |
| `feature/retrieval-enhancement` | `1d97780` | 冻结 NQ P3-20 检索增强 |
| `feature/fever-c1` | `6ebdfc9` | 包含 NQ C0/C1 `bcb248c`、C3 `8b03009`，及 FEVER C1 |
| `feature/F500-c0c3` | `1238a4f` | FEVER C0/C3、索引来源 |
| `feature/-ce-70B` | `6f08f2f` | 70B 对照、离线 judge 与结果 |
| `feature/verification-answer-repair` | `5da1fe5` | E 全部历史代码、实验、审核和交接 |

全部 10 个远端分支均为集成提交的祖先。逐分支比对 `data/`、`results/`、`artifacts/`、`models/`、`configs/splits/` 的 Git blob：原路径下的已入库数据与实验结果均保留原字节，详见 [branch_manifest.json](../artifacts/controller/integration_v1/branch_manifest.json)。未重写原分支历史。

合并冲突处理：

- README 汇合入口，旧进度明确标为历史记录。
- `src/generation/qwen.py` 保留 E 已验证的 revision 固定、显式 NF4 GPU 放置和输出停止逻辑；不启用 70B 分支旧的自动 CPU offload 修改。该分支原代码仍在其提交历史中，`docs/LOCAL_ENV.md` 中旧 offload 环境变量说明不适用于本集成版本。
- 重排分支的包级立即导入会破坏检索模块独立性，改为延迟导出，保留公共类名与函数名。
- FEVER 分支把纯提示函数移到 `src/generation/prompts.py`，三个检索测试改从实际定义处核对，提示字节未改。
- r4 冻结 AST 测试兼容 Python 3.12 的空 `type_params` 字段；原冻结 fixture 与提示哈希不改。

## 2. 新增执行路径

```text
冻结 C1 JSONL / 纯 DiagnosisInput JSONL
  → 白名单适配（不向运行时传 gold 或人工标签）
  → 冻结 Diagnosis → 可选一次 Probe → 原 Safe Binary
  → no_action / retrieval_enhancement / evidence_refinement / generation_repair
  → 统一结果、调用凭证、回退、预算和成本
  → 单独进程加载 gold 做离线评测
```

| 文件 | 职责 |
|---|---|
| `src/controller/execution.py` | 统一执行器、严格输入验证、各模块适配、共享预算、逐调用轨迹 |
| `src/controller/providers.py` | 外部加载并共享一个 Qwen；CPU reranker 和可选 NQ 检索资源 |
| `configs/controller/integration_v1.json` | 明确选择 E 策略和调用／输出预算 |
| `scripts/run_controller_integration.py` | 预检查、CPU smoke、真实运行、逐条保存和运行清单 |
| `scripts/evaluate_controller_integration.py` | 预测之后独立 join gold，计算修复、误修、净收益、成本 |
| `tests/test_controller_integration.py` | 四路执行、Probe 覆盖、失败和预算、gold 隔离、共享模型及 CLI |
| `tests/test_controller_evaluation.py` | 配对样本、缺失／重复引用、修复与误修计分 |

原 `run_controller()` 和各模块公共接口继续保留。本次不改 Diagnosis 特征、分类器参数、Probe 提示、Safe Binary 阈值、P3-20 排序公式、重排默认 keep5 或 E 提示。

### E 的明确版本选择

集成配置使用 NQ `short_answer_quote`（已有 v7 Qwen 短答案与引文策略）、FEVER `verify_repair`（已有 Qwen 核验修复策略）。两者使用同一个 Qwen 模型对象；仅输出格式及停止适配不同。这个组合是**用于集成比较的显式基线**，并不是 v7 FEVER NLI 结果的复现，也不是新的性能提升结论。

历史 NLI 实验与 v9 代码完整保留。v9 的合成契约通过尚不能替代真实质量验收，所以未自动启用。当前源码的 `conservative_repair` 是 **r4**；不能把配置改成这个名字就声称在运行冻结 r3。复现 r3 应使用原冻结包。

### 执行与回退

- `no_action` 不调用增强 provider；若之前触发 Probe，Probe 成本仍计入。
- NQ P3-20 使用原 query Top20、固定 revision 的 BGE reranker raw logits、RRF、Top5，至多一次重答。
- FEVER 的 P3-20 请求返回 `unsupported_dataset`，不调用该路线的 provider，保持 C1；必须在覆盖率中单列。
- 重排保留原默认 keep5；未改变顺序则不重答。
- E 无有效输出时使用原模块的回退。引用／短答案出现于证据，只证明来源匹配，不能当作语义正确。
- provider 异常计入 attempted calls，保存错误与 C1 回退；Probe 异常生成 uncertain 结果，不覆盖原类型。分类器加载／推断异常中止运行并标记失败，不伪造 correct 标签。
- 每题最多 4 次 Qwen 调用（含 Probe）、1 次补检索、1 次 reranker API 调用。实际次数按 provider 调用计，不等于模型内部 batch 数。
- 记录模型看到的完整提示、提示哈希、生成结果、证据、诊断、最终动作和异常；不声称这些轨迹是模型内部推理。
- 未知 token 数保留 `null`，失败调用不记作零成本。计时明确是缓存 C1 之后的增量；全系统成本仍需加共同 C1 成本，加载时间单列。
- 若保持 C1 答案，`final_evidence` 保持 C1 上下文；尝试过的增强上下文另保存在 module trace／调用记录中。

## 3. 运行

使用单独环境。`requirements-controller.txt` 复用合并后团队依赖，并固定 Diagnosis 所用 sklearn；CUDA PyTorch 按 GPU 平台安装。不要与历史实验包的 Transformers 固定版本混装。CPU 回归环境版本见验证归档；它不代表已验证真实 CUDA 加载。

```bash
python -m pip install -r requirements-controller.txt
python scripts/run_controller_integration.py \
  --smoke --output-dir /tmp/controller-smoke-new
python -m unittest discover -s tests -v
```

Smoke 仅有 2 个虚构输入，运行 5 种模式，共 10 条结果；真实冻结分类器和真实路由／增强代码连接假 provider。不是新的 NQ／FEVER benchmark，也不用于调语义策略。

输入可为原标准 C1 导出（含 `condition`、`model_name`、`retrieved_top5_ids` 等），或仅含 `sample_id,dataset,query,evidence,initial_answer` 的运行时 JSONL。每条严格要求有序 Top5、有限检索分数、非空文本和唯一 ID。NQ Train 交付应先转换成此接口；不会直接把初标工作簿传给模型。

先在开发数据上检查，不打开最终测试集：

```bash
python scripts/run_controller_integration.py \
  --input /path/to/frozen_dev_c1.jsonl \
  --preflight --output-dir /path/to/new_preflight
```

CUDA 上正式比较（资源路径必须属于该批 C1 的同一个 corpus/index 版本）：

```bash
python scripts/run_controller_integration.py \
  --input /path/to/frozen_dev_c1.jsonl \
  --nq-index /path/to/matching_nq.index \
  --nq-metadata /path/to/matching_nq_metadata.jsonl \
  --mode c1 --mode adaptive --mode always_retrieve \
  --mode always_rerank --mode always_repair \
  --output-dir /path/to/new_controller_run
```

NQ adaptive／always_retrieve 在真实运行时必须明确给出索引与元数据，不会悄悄用旧 dev 索引处理 Train 样本。结构与选中 ID/text 会校验，资源 SHA-256 进入环境记录；**这些哈希不自动证明索引与原始 C1 的版本相配**，运行者仍需核对交付来源。FEVER-only 不需要 NQ 索引，但检索增强仍未支持 FEVER。

每次要求新输出目录，逐条 flush JSONL。产生 `manifest.json`、`environment.json`、`predictions.jsonl`、`summary.json`；包含 provider 失败或预算超限的运行返回非零退出码。暂不自动恢复半次运行，避免跨代码版本混用结果。

离线评价：gold JSONL 每条包含 `sample_id,dataset,gold_answer_or_label`，NQ 可为答案字符串列表，FEVER 为 SUPPORTS／REFUTES。

```bash
python scripts/evaluate_controller_integration.py \
  --predictions /path/to/new_controller_run/predictions.jsonl \
  --gold /path/to/dev_references.jsonl \
  --output /path/to/new_controller_run/offline_evaluation.json
```

缺失／重复 gold、重复预测、各模式样本不配对或同题 C1 输入不一致均报错。NQ EM/F1 与 FEVER label accuracy 分开；修复数、误修数、净收益、动作覆盖和费用一并报告。NQ EM 不能替代语义人工审核；此计分也不等于证据忠实度验收。

## 4. 验证记录

完整 CPU 回归：**635 项通过**，见 [tests.log](../artifacts/controller/integration_v1/tests.log)；环境见 [cpu_test_environment.json](../artifacts/controller/integration_v1/cpu_test_environment.json)。合并后首轮的 8 个兼容性失败也保留在 `initial_merge_tests.log`，未通过跳过测试处理。

分支来源检查覆盖 764 个唯一入库数据／结果／模型／划分文件路径（不含 data README 与占位文件），与各来源分支的对应 Git blob 均一致。CPU smoke 的运行清单与逐条输出单独归档；这些只用于工程验证，不报告 QA 提升。

## 5. 当前完成范围与下一步

已完成：全分支合并、C1 后处理执行器、两数据集接口、可追溯回退、共享调用预算、配对离线评分、CPU 测试和虚构 smoke。原始问题到 C1 继续使用既有 Reference RAG 运行器，避免此次集成重算或改变冻结 C1。

仍需完成，按优先级：

1. T2 按 T1 协议冻结同一批开发/选型/最终保留数据、配对 gold、样本数、曝光登记和匹配资源；当前尚没有集成后的 QA 收益数字。
2. T3 补充离线人工类型触发、No Probe 和失败类型 × 操作分析，区分路由损失和模块损失。固定动作结果复用，人工标签不进入在线模型提示。
3. T3 补齐 FEVER 检索增强；如资源限制只能做部分范围，必须在看结果前声明，不能把回退算作已实现。
4. T3 按已冻结顺序和预算实现 Full；补齐评分/完整清单验收和执行 seal 后进入 T4 GPU 比较。
5. 根据同批比较仅优化最大的瓶颈，最多一轮预登记候选。保留最终未见测试集；蒸馏、DPO、知识库修改和 RSI 均不作为本次集成的隐含改动。

合并前的全仓库与数据质量结论见 [10 月 23 日审计](PROJECT_AUDIT_AND_RSI_PLAN_20261023.md)。审计中的“尚未集成”描述对应合并前状态；本文件记录其后的工程进度。
