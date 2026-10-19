# E v8：NQ 原答案保护与候选核验

2026-10-19 开发版本，尚未晋升默认。原 75 条四组实验因原检查格式失败，r2 修复入口后候选格式仍未通过。现 [v8r3 真实结果](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md) 已完成 32 条输出、通过冻结回放及工程门槛，完整流程只改答 N96，EM 1→1；语义误放与误拒仍需优化，人审未完成。历史包与结果均保留。

## 本轮要检验什么

v7 能找到真实引文，但真实引文可能属于另一作品、另一事件或另一关系，也可能没有回答问题。直接替换还会损害本来正确的别名或简短答案。本轮检验：在同一模型、同一问题、同一 C1 Top-5 下，先判断原答案是否确有错误，再要求修复候选通过语义核验，能否减少误修并保留有效修复。

运行时只读取 `DiagnosisInput` 的问题、初始答案、检索证据及样本标识。证据是 C1 实际检索的 Top-5，不是 oracle gold evidence。参考答案、人工标签和 EM/F1 只用于选开发队列及离线报告，不进入提示或运行时分支。

## 固定输入与对照

采用 [75 条已定案开发清单](../configs/splits/answer_repair_nq_train_confirmed_dev_v6.json)：11 个内部 E utilisation 目标、64 个正确对照。两条部分回答 N33/N62 也属于目标。25 条未决继续排除；不改 main 来源、C1 回答、gold 或冻结评分。人工确认采用本对话辅助审核，并非盲审或独立标注。

这是有意筛选的 NQ Train 开发队列，不能估计总体错误比例、系统路由收益或未见集净提升。不能把这 75 条用于调优后再称为 held-out。FEVER 本轮保持原有实现，并通过回归测试；本轮不对 FEVER 声称新收益。

| 对照组 | 处理 | 单条生成上限 |
|---|---|---:|
| C1 | 复用原始答案，不重新生成 | 0 |
| `short_answer_quote` | v7 一次抽取短答案和引文 | 1 |
| `conservative_repair` | 原答案核验 → 最小修复 → 候选核验 | 3 |
| `conservative_no_original_gate` | 最小修复 → 候选核验 | 2 |
| `conservative_no_candidate_gate` | 原答案核验 → 最小修复 | 2 |

三组 v8 使用完全相同的修复提示；消融仅移除对应核验。四组各 75 条，共 300 条输出，最多 600 次额外生成。全部使用冻结 Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`、NF4、FP16、greedy、seed 42；每次最多 384 输出 token。v8 的额外计算高于 v7，须并列报告调用、输入/输出 token、延迟，不能声称等预算比较。

实验使用 `--all`，故意同时覆盖错误目标和正确对照，以测试修复／误修。它绕过 Diagnosis/Controller；生产集成仍只在上游最终 action 为 `generation_repair` 时调用 E，其余动作零调用保留原答案。不在 E 新设路由阈值。

## 实际决策规则

1. **原答案核验**：六项分别检查实体／作品／事件、关系及其方向和条件、答案类型、必要完整性、证据支持、冲突。模型只能输出 yes/no/uncertain，加一段可定位引文和简短理由。六项全 yes 保留原答案；有 no 且没有 uncertain 才进入修复；任一 uncertain 或格式／引用错误均保留原答案。
2. **最小修复**：结合问题、完整 Top-5 与原答案，提取最短但足以回答问题的连续证据片段。why/how/definition 可以是从句或句子；避免因 20 词限制把原因截成实体。候选最多 64 词，引文最多 100 词。无法用一个完整片段回答时返回 `INSUFFICIENT`。没有额外示例、样本 ID 特判或人工答案注入。
3. **候选核验**：重新检查候选本身是否回答问题，并复核完整 Top-5。只有六项全 yes 才替换；任一 no/uncertain 或格式／引用错误均回退 C1。去掉此步的组仅为开发消融。

核验由同一个 Qwen 独立调用执行，**不是独立 NLI，也不是独立裁判**。错误可能相关，六项通过也不能当作真实正确性。引用与答案定位是可追溯检查，语义判断仍有误差。

所有阶段采用严格的带标签逐行输出，关闭 JSON 停止器，保留原始生成。引文允许空白和 PTB 引号的显示差异，但必须落在原段的连续、有词边界的窗口内；不会模糊替换实体、删除否定或拼接句段。最终候选采用原文切片。规范化后相同则保留 C1 原始表达，不强制改写有效别名、日期、昵称，也不统一要求完整姓名或额外行政层级。

本版仍受连续抽取限制：跨段综合、多跳推理或无明确可抽取表述时可能不修复。原答案核验过于保守也可能漏掉真实错误；这些必须按阶段失败数、未修复目标和人工复核记录报告。

无证据返回零调用。格式／引用／不确定性属于可记录回退；CUDA、网络、模型载入等后端异常仍停止运行并保留已完成记录，避免把基础设施失败算作模型“保持正确”。同一环境可校验后续跑，更换模型／代码／环境则需独立输出。每条 v8 保存阶段、提示哈希、原始输出、解析、引用偏移和逐次成本，可在 CPU 上重新执行决策并验证最终结果与成本。

## 代码与运行

- 核心实现：[conservative_repair.py](../src/compensation/conservative_repair.py)。
- 策略分发：[answer_repair.py](../src/compensation/answer_repair.py)；Controller 适配：[controller_adapter.py](../src/compensation/controller_adapter.py)。
- 协议：[answer_repair_v8_nq75_v1.json](../configs/answer_repair_v8_nq75_v1.json)。
- 冻结打包：`python scripts/prepare_answer_repair_v8.py`。
- Kaggle/Colab：[answer_repair_v8_nq75_transport_r1.ipynb](../notebooks/answer_repair_v8_nq75_transport_r1.ipynb)，配套 `artifacts/answer_repair/v8_nq75/answer_repair_v8_nq75_gpu.tar.gz`。
- 单组预检：`python scripts/run_answer_repair.py --dataset nq --c1 artifacts/answer_repair/nq_train_dev_v1/confirmed/through_batch06_c1.jsonl --all --strategy conservative_repair --check-only`。
- CPU 回放入口：`repair_from_v8_trace`；断点续跑自动检查记录。

Notebook 按 tar.gz 及每文件 SHA-256 校验冻结代码；安装与 v7 云端记录相同的四项直接依赖版本，保留宿主 CUDA PyTorch 并导出完整环境。四组按顺序运行，不能根据中途答案选择样本或改策略；不同云端环境的随机性和时延比较仍需谨慎解释。

## 输出后如何验收

`scripts/summarize_answer_repair_v8.py` 生成 `comparison.json` 和 `output_review_queue.jsonl`：原始 EM/F1、各组修复／误修、阶段通过／拒绝／无效计数、调用及延迟；另按 11 目标和 64 对照分组。任何新的修复答案都保留 pending 人工字段；先审变化，再看未修复目标，核对问题、初始答案、候选与 Top-5。

旧 C1 人工正确性不自动成为新答案的标签。新输出的人审净收益在审核前为 null。若只有 EM 改善而语义质量下降，应报告下降；若 EM 不匹配但人审认可，保留冻结指标并另报语义结果。

本轮只决定是否值得继续：修复收益要超过对正确答案的损害，且额外计算可接受；不得因少量开发样本的正增益就替换默认。之后固定规则，与团队确认未用于开发的 NQ/FEVER 评测划分，报告完整结果及成本，再决定扩充自然错误、调整核验方式或采用独立 verifier。

## 云端执行修正（不改算法）

第 1 版实际取得 T4，运行 30.5 秒后在读取实验包阶段失败，模型推理未开始、预测为 0。Kaggle 自动解压了上传的归档，原 Notebook 只搜索 tar.gz 文件名。当前使用 transport r1 Notebook，保留原包 SHA-256 `906f096f78e36a4f272cdd863ce2e7383d62acb638c1e480a06c5d3f75daa7a3`。

对于自动解压目录，先比对已固定的 checksums 文档哈希 `5499fcca237c5d8f50187c27bf24f5f6189176dab8d86184b507116420f3997e`，再核对全部 85 个文件，并只复制已核对的内容到运行目录。7 项加载与回放测试和真实冻结包的解压布局验证通过；全仓现为 136 项测试通过。新旧 Notebook、失败记录和加载验证均保留，模型、75 条样本、提示、评分及四组消融未调整。
