# E v7：短答案与引文分离、FEVER 主体实体约束

> **已运行（2026-10-14，Kaggle T4 x2）**：结果见 [ANSWER_REPAIR_RESULTS_20261014_V7.md](ANSWER_REPAIR_RESULTS_20261014_V7.md)。FEVER 开发批次与离线回放一致，held-out 1 修复 / 4 误修；NQ 各批次冻结 EM 净负或持平，held-out 显著为负。两种策略均不晋升；本文是运行前协议，不随结果回改。第 4 批 `heldout3216` 也已运行一次：utilisation 层 EM 0→10、correct 层 63→54，见同一结果报告。

2026-10-14 设计并冻结。v7 是 v6r1 辅助审核后的下一版，修改范围只有两点：NQ 把“答案”和“支持引文”分成两个字段；FEVER 的 NLI 判定只允许提到声明主体实体的窗口投票。模型、量化、greedy、seed、NLI 模型版本与 0.90 阈值全部沿用 v6，不新增模型调用。GPU 运行前的协议即本文；结果另写报告，不回改本文。

## 为什么是这两点

v6r1 的 128 条辅助审核（代理 + 独立盲评两读一致）说明：

- NQ `extractive_reanswer` 的 15 条冻结 EM 退步都是“答案正确但扩成整句”，另有 6 条语义修复 EM 没识别；真正的语义误修只有 1 条（statute）。问题在输出形态，不在选择能力。dev50 有 24/50 无效输出，其中 17 条是空格/引号格式差异导致的“非逐字”拒绝，5 条是 JSON 内部引号没转义。
- FEVER `fever_nli` 的 2 条误修（Men in Black II、Henry Cavill）都由引用错误实体的高分 contradiction 窗口触发；审核后 14 条里 4 条 conflict 也是同名/无关页面的窗口在捣乱（Eva Green 对 The Big Green，Alan Shepard 对 Weinel/Kinnan，Babur 对 Bidar Bakht）。

## 预先固定的策略

### `short_answer_quote`（仅 NQ）

同一 Qwen2.5-7B-Instruct / NF4 / FP16 / greedy / seed 42，最多一次 384-token 生成，不使用 JSON 停止器。模型输出三行：

```
RANK: <证据块编号>
QUOTE: <该块内一段连续原文，≤60 词>
ANSWER: <QUOTE 内最短的回答片段，≤20 词>
```

或单行 `INSUFFICIENT`。程序规则：

1. QUOTE 必须能定位为该块的字符切片；ANSWER 必须能定位为 QUOTE 内的字符切片。匹配忽略空白和引号写法（`` '' 与弯引号归一为 "），保留全部词与标点，切片必须落在词/标点边界上。`Can't` 可匹配语料的 `Ca n't`，`York` 不能匹配 `Yorkshire`，不能删否定词或拼接。
2. 最终答案 = ANSWER 的原文切片，引用 = QUOTE 的原文切片；不返回任何模型自撰文本。
3. ANSWER 与初始答案按冻结 EM 归一化相等时为 `kept`，否则 `repaired`。
4. 词法锚点：问题的内容词（≥3 字母，去停用词）至少三分之一出现在被引用的整段里，否则状态 `unanchored_candidate`，保留初始答案。这是弱规则，只挡完全离题的段落；它是否有用要靠 held-out 上的有/无锚点回放对比，不能假设。
5. 格式、编号、定位、长度任一失败回退初始答案并记 `invalid_output`。

### `fever_nli_anchored`（仅 FEVER）

同一 `microsoft/deberta-large-mnli`（revision `7296194b…`）、同样的分窗、FP32、阈值 0.90、窗口概率全部保存；分类器成本与 v6 完全相同。唯一改动是判定规则：

- 取声明的主体实体块 = 去掉句首冠词/介词后第一串大写开头的词（数字不算，所有格 s 不断开）。例如 "Henry Cavill starred in two Superman films." → `henry cavill`；"The 1974 crime film The Sugarland Express was…" → `sugarland express`；"In 2003, Eva Green debuted in film." → `eva green`。
- 只有主体块每个词都出现在窗口文本里（允许单复数）的窗口才有资格投票；其余窗口的分数记录但忽略。
- 投票规则与 v6 相同：有资格窗口中任一 entailment ≥ 0.90 且无 contradiction ≥ 0.90 → SUPPORTS；反之 → REFUTES；双向或均无 → 保留初始答案。
- 被排除的高分窗口、每个窗口的锚点覆盖、主体块缺失词都写入 `anchor_filter`，回放可复现。

不要求谓语里的属性词出现（如 "Brubaker is a solely Australian film" 的 Australian），因为反驳证据本来就会给出另一个值。主体词在证据里以别名/代词出现时会被保守地排除，这是有意的：排除只会让答案保持不变，不会改错。

### 离线回放（设计阶段证据，非 held-out）

在 v6r1 已保存的窗口概率上只换判定规则（`scripts/replay_fever_nli_anchored.py`，输出在 `artifacts/answer_repair/v7_design/`）：

| 批次 | n | 初始正确 | v6 最终 | v7 最终 | v6 修复/误修 | v7 修复/误修 |
|---|---:|---:|---:|---:|---:|---:|
| FEVER 审核后 14 条 | 14 | 4 | 7 | 10 | 3/0 | 6/0 |
| FEVER 原 dev50 | 50 | 45 | 44 | 46 | 1/2 | 1/0 |

新增的 3 个修复是 Eva Green、Alan Shepard、Babur 三条 conflict 被解开；消除的 2 个误修是 Men in Black II 与 Henry Cavill。这个规则是看着这些样本设计的，所以上述数字只说明规则按设计工作，不能当作效果证据；效果要看下面的 held-out。NQ 的新策略需要真实生成，没有离线回放。

## 固定数据与运行顺序

实验包由 `scripts/prepare_reviewed_answer_repair.py --experiment v7` 生成，manifest 为 `configs/splits/answer_repair_reviewed_v7.json`。

1. `reviewed`：v5/v6 的 28 条审核后训练开发样本（FEVER 14，NQ 14），与 v6r1 成对比较。
2. `dev50`：v1 的固定 dev50 × 2，观察正确答案误修，与 v6r1 成对比较。
3. `heldout`：E 从未看过的全部共享训练样本（Diagnosis v1/v2 训练交集 − dev50 − 37 条审核包）。FEVER 267 条，NQ 146 条。选样不读标签、正确性或参考答案。**只运行一次，不论结果如何都如实报告，不在它上面调任何阈值或提示词。**

第 3 批的离线标签构成（只在这里说明，脚本不读）：FEVER 232 条 correct + 35 条 insufficiency，没有 utilisation 标注；NQ 126 条 correct、13 条 insufficiency、7 条 accessibility，也没有 utilisation 标注，但其中 75 条冻结 EM 为错。所以这一批主要回答“E 在误路由（本来就对）的样本上会不会改错”和“EM 判错但语义正确的答案会不会被改坏”，不能回答修复率。真正含 utilisation 标注、E 又没碰过的 NQ 样本在 Diagnosis v2 训练集的 3216 扩展池里（1587 条，含 120 条 utilisation）。
4. `heldout3216`（仅 NQ，独立实验包）：从上述 1587 条按离线标签分层取 300 条：utilisation 全部 120、accessibility 全部 35、correct 100、insufficiency 45（seed 42 哈希排序，`configs/splits/answer_repair_nq_heldout_3216_v1.json`）。gold 短答案取自 Diagnosis 负责人提供的本地工作簿 `NQ_C1_3216_合并全量.xlsx`（SHA-256 `a82961a6…`，不入库）的 `gold_answer` 列，导出为 `data/annotations/gold/nq_heldout_3216_v1_gold.jsonl`（入库）；工作簿的 question、c1_prediction 与上游入库导出 300/300 一致，`em_reference` 与重算 EM 300/300 一致。冻结 EM 初始正确 63 / 错误 237。实验包 `artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz`（SHA-256 `6f27757935128f3e04e2881e1fb832885b10d45ef55316fa425b889950651bde`），notebook `notebooks/answer_repair_colab_v7_heldout3216.ipynb` / `answer_repair_kaggle_v7_heldout3216.ipynb`。**同样只跑一次**，结果按四个层分别报告；这是按标签分层的 held-out，不代表部署分布，修复率只在 utilisation 层内解释。

成本按 v6 协议分开记：Qwen 调用/生成秒数/输入 token 只计 NQ；NLI forward 批次/分类秒数/输入 token 只计 FEVER。

## 报告口径

- 冻结 FEVER accuracy 与 NQ EM/F1 照旧是主表，分批报告，重叠 ID 不合并。
- 成对语义审核沿用 v6r1 的四项（初始/最终正确性、初始/最终支持性）和 `docs/ANSWER_REPAIR_CONFIRMATION_20261014.md` 里统一的完整性口径；代理建议不等于人工真值。
- 必须同时报告：`unanchored_candidate`、`invalid_output`、`nli_uncertain` 的数量和它们里面原本错误的比例；被锚点排除的高分窗口数；有/无 NQ 锚点规则的回放对比。
- 任一批出现净负收益或 held-out 误修率高于 dev50，都照实写进结果，不得回头改规则再跑同一批。

## 代码

| 文件 | 内容 |
|---|---|
| `src/compensation/anchors.py` | 主体实体块、内容词锚点、覆盖率；无模型 |
| `src/compensation/nli_repair.py` | `repair_from_nli_trace_anchored`；v6 函数原样保留供历史回放 |
| `src/compensation/short_answer_repair.py` | 三行格式、空白/引号归一化定位、短答案校验 |
| `scripts/replay_fever_nli_anchored.py` | 用 v6r1 轨迹离线回放 v7 FEVER 规则 |
| `scripts/run_answer_repair.py` | 注册两个策略；`fever_nli_anchored` 续跑按 v7 规则校验 |
| `tests/test_answer_repair_v7.py` | 锚点、决策、定位、解析、续跑、adapter 边界 |

```sh
python3.11 -m unittest discover -s tests
python3.11 scripts/prepare_reviewed_answer_repair.py --experiment v7 --manifest configs/splits/answer_repair_reviewed_v7.json --out-dir artifacts/answer_repair/v7_reviewed --bundle-name answer_repair_v7_reviewed_gpu
python3.11 scripts/build_reviewed_answer_repair_notebook.py --experiment v7 --bundle artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz --out notebooks/answer_repair_colab_v7_reviewed.ipynb
python3.11 scripts/build_kaggle_answer_repair_notebook.py --notebook notebooks/answer_repair_colab_v7_reviewed.ipynb --bundle artifacts/answer_repair/v7_reviewed/answer_repair_v7_reviewed_gpu.tar.gz --out notebooks/answer_repair_kaggle_v7_reviewed.ipynb
# 第 4 批（需要本地工作簿）
.venv/bin/python scripts/build_nq_heldout_from_diagnosis_inputs.py --workbook "data/annotations/NQ_C1_3216_合并全量.xlsx"
python3.11 scripts/prepare_nq_heldout3216_bundle.py --c1 artifacts/answer_repair/v7_reviewed/inputs_extra/nq_heldout3216_c1.jsonl
python3.11 scripts/build_kaggle_answer_repair_notebook.py --notebook notebooks/answer_repair_colab_v7_heldout3216.ipynb --bundle artifacts/answer_repair/v7_heldout3216/answer_repair_v7_heldout3216_gpu.tar.gz --out notebooks/answer_repair_kaggle_v7_heldout3216.ipynb
```
