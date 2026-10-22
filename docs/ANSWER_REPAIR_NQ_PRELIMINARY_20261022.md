# NQ Train 新初标接入与 E 的使用边界

2026-10-22 已从 [团队 Drive：NQ_15批](https://drive.google.com/drive/u/0/folders/1JWLByrI_hZ1qv70TSD23QjOlHngGBP1X) 接收 15 份 `NQ_C1_000_preliminary_annotations.xlsx` 至 `NQ_C1_014_preliminary_annotations.xlsx`，覆盖原来的全部 **74,097 条 NQ Train C1**。这解决了本地仍只有旧空标签交付副本的问题；不是新增 74,097 道独立题，也不是 FEVER 数据。

完整结果见 [机器审计](../data/annotations/nq_train_preliminary/import_audit_20261022.json)。15 个工作簿、原始 JSONL、标签、划分及人审均按各自来源保存，没有覆盖旧材料；本轮未调用 GPU、未选新评测题、未改提示或策略。

## 一致性与标注来源

逐条核对 sample_id、dataset、query、initial_answer、evidence、annotation_reference。嵌套 JSON 反序列化后比较全部字段和类型；原始 JSONL 的 SHA-256 与 main 交付清单一致。74,097 个 ID 完整、唯一，无新增、遗漏或字段改写；Excel 中全部 ID 为文本，未发生浮点精度截断。原证据排名、passage ID、正文、分数及离线参考均保持一致。

工作簿 Summary 的自述合计为：

| 来源声明 | 条数 |
|---|---:|
| 自动／启发式初标 | 73,536 |
| 选定样本 Top-5 语义审核 | 561 |
| 总计 | 74,097 |

“语义审核”是文件的来源声明，未核验为独立人工审核。工作簿没有提供完整、可验证的逐条审核者与状态；部分说明写了开头若干行，部分只称 selected examples。因此本次不推断逐条人审身份，所有标签以 **preliminary** 导入，`human_verified=false`。这不表示重新认定其中每一条都未被过目，而是没有足够依据把它们升级成人工金标。

各 Summary 明确说明，自动部分采用答案／证据词面、近似匹配及检索信号，不能由 gold 命中或词面重合直接证明正确性、证据充分性或错误因果。Review_Questions 共 37,703 行，含已处理记录，仅用于保留疑点追踪；不能称为尚有 37,703 条未决，也不能认为表外样本已确认正确。

## 初标分布

以下是文件标签的数量，**不是经验证的错误率或 C1 准确率**。

| 初标 | development | calibration | reserved_evaluation | 全部 |
|---|---:|---:|---:|---:|
| correct | 29,344 | 3,722 | 3,733 | 36,799 |
| evidence_insufficiency | 11,788 | 1,436 | 1,526 | 14,750 |
| evidence_accessibility | 14,410 | 1,765 | 1,842 | 18,017 |
| evidence_utilisation | 3,646 | 422 | 463 | 4,531 |
| 总计 | 59,188 | 7,345 | 7,564 | 74,097 |

沿用 10 月 17 日既定划分，不因标签或结果重新分组。这里只统计各用途的标签计数，不使用预留评测题内容挑提示。既有 125 条挑战已使用，不能再被称为新未见样本；团队 Diagnosis 训练排除、Wikipedia 页面与语义近重复隔离仍未确认。

## 与既有人审交叉检查

对照首批 100 条的累计人审记录：75 条已定案中，39 条标签一致、36 条不一致；25 条原暂缓继续暂缓。所有既有人审保留，初标不会覆盖人工结果。以人审标签为行、初标为列：

| 既有人审 | correct | insufficiency | accessibility | utilisation | 合计 |
|---|---:|---:|---:|---:|---:|
| correct | 39 | 1 | 19 | 5 | 64 |
| evidence_utilisation | 1 | 0 | 10 | 0 | 11 |

在这个已审开发子集中，11 个 E 错误有 10 个被初标分到 accessibility、1 个分到 correct，均未被分到 utilisation。若仅按初标 utilisation 取题，会漏掉这些已知错误；也有 5 个已确认正确答案被初标为 utilisation。

这说明初标的类别边界与当前人审存在明显分歧，特别是词面匹配无法确认具体误用证据的原因。**36/75 不能外推为全量标签错误率**：原 100 条是按 EM／参考召回筛选的开发审核批，并非随机抽样或独立盲审。也不能由此宣布 4,531 个候选都不可信或都可修复。后续应先在 development 分层抽审，包括 utilisation、accessibility、correct 及 insufficiency，查清漏分和误分；不能只审“被判为 E”的样本。

## 对接下来的影响

1. 这批标签有用：可以扩大检索、筛选和分层审核范围，不需要从头重标全部 7.4 万条。已有定案样本直接复用。
2. 暂不需要换 NQ 或做数据增强。当前先解决标签可信度、E 输出接口与评测隔离；新增合成题不能替代自然分布的真实评测。
3. 最近的 10 道题是工程 smoke：NQ 5＋FEVER 5，三组生成 30 条预测。它们只验证执行、来源与输出结构，无法衡量总体收益，也不能反复用于挑提示。
4. v9 Version 3 的结构合法率为 11/20，未过事先声明的 18/20 门槛。本次接入不会解除该停止条件；先另立受约束输出接口工程实验，通过后再冻结更大的开发与未见评测。
5. E 核心验证需要可信的自然错误和正确对照，报告修复、误修、无效回退与成本。分层诊断集和自然分布的整集收益应分别报告，不能把定向挑错集当总体准确率。
6. 74,097 条都是 NQ。FEVER 需要单独补充和核验其数据及 diagnosis 标签；不能据此增加 FEVER 的样本数。

## 保存与复现

| 内容 | 路径 |
|---|---|
| 原 ZIP、15 份 XLSX、逐文件下载清单（本地，Git 忽略） | `data/processed/external/nq_train_preliminary_20261022/` |
| 全量标准化初标层（本地，Git 忽略） | `data/processed/answer_repair/nq_train_preliminary_20261022/labels.jsonl` |
| 可入库审计、Summary 原文、分歧 ID 与标签 | `data/annotations/nq_train_preliminary/import_audit_20261022.json` |
| 读取与验证脚本 | `scripts/import_nq_train_preliminary.py` |
| 精度、来源、冻结字段及拒绝覆盖检查 | `tests/test_nq_train_preliminary.py` |

原 ZIP SHA-256：`fb7c0d3e404a24f306cc6f9e11763187e027cb2acff68c063bdfe102e3eb0742`。标准化层 SHA-256：`1c08f65bd0351c66d27dd8ec6400083c5c2b898bcb95fa08c194aa8c3d9d4da1`。每条仅导出离线标签、用途、来源 sheet／行号及哈希，`runtime_use=false`，不含问题、证据和 gold；模型输入仍使用原有白名单。

```sh
python scripts/import_nq_train_preliminary.py
python -m unittest tests.test_nq_train_preliminary -q
```

脚本只读 XLSX，不修改或重新导出原表。先核验全部工作簿、来源分片、冻结字段、ID、标签、Summary 与划分，再发布结果；相同内容允许回放，已有不同结果拒绝覆盖。完整 XLSX／ZIP／全量 JSONL 不纳入 Git，以审计指纹和原 Drive 来源复现。
