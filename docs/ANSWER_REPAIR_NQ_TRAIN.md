# NQ Train 接入 E

**2026-10-22 更新：** 已从 Drive 的 `NQ_15批` 接入覆盖全部 74,097 条的 15 份 XLSX 初标，原问题、C1、Top-5、参考与旧交付逐条一致。初标含 4,531 条 utilisation，但主要为启发式生成，不是全量人工金标；与既有 75 条人审有 36 条分歧，人工结论保持优先。新增数据状态、分布及使用边界见 [初标接入报告](ANSWER_REPAIR_NQ_PRELIMINARY_20261022.md)。下文“空标签交付”描述的是 10 月 17 日原始来源，不表示最新 Drive 没有标签。

2026-10-17 已接入 Google Drive 的全部 15 个分片，共 74,097 条 NQ Train C1 记录。全量已转换为 E 兼容输入，首批 100 条开发审核材料已固定。原始材料保留未确认快照。截至 2026-10-19 第六批，100 条均已有一轮人工意见：75 条定案（11 个内部 E utilisation 目标、64 个正确对照）、25 条待定、未过目为 0。25 条含最初 17 条暂缓、5 条原口径分歧及 N58/N61/N83 的条件／补充要求；全量数据未因此变成人工标注完成。75 条已完成 [v8 四组真实实验及回放](ANSWER_REPAIR_RESULTS_20261019_V8.md)，完整 v8 全部在原答案核验上无效回退；原 15 条新改答待人审。后续 [r3 真实试跑](ANSWER_REPAIR_RESULTS_20261019_V8_R3.md) 已通过工程门槛，完整流程仅修正 N96；下一步优化语义核验，不能将 8 条开发排错样本视为最终收益证据。

初次代理建议为 69 条可接受、8 条 utilisation 候选、23 条暂缓。前六批意见均分别保存，当前状态以累计审核记录为准；条件、完整性与表达分歧和代理语义判断分别保留，不自动纳入 E 核心。N76 以用户提供的同实体来源说明核对，已标明解释依据。可先使用定案的 75 条开发，25 条继续隔离。见 [审核结果、确认记录及下一轮约束](ANSWER_REPAIR_NQ_TRAIN_REVIEW_20261017.md)。main 的原始数据及 gold 作为来源基准保持原样；本地审核与适配为派生产物。

## 来源与检查

上游代码与交付清单来自 `main@058f10baa47083220f95edb39db8f106aa7e85d6`，已合入 E 分支。原始分片来自 [Google Drive 的 nq_train_annotation](https://drive.google.com/drive/u/0/folders/1GRWOjia-PXfYh7YfvC1f3JzdjJqtJNDU)。15 个文件的 SHA-256 均与 [交付清单](../data/annotations/nq_train_delivery/manifest.json) 一致，逐条字段检查无异常；见 [完整审计](../data/annotations/nq_train_delivery/drive_audit_20261017.json)。

| 项目 | 数量或结果 |
|---|---:|
| 官方 NQ Train 中选出的 controlled 样本 | 74,097 |
| 上游检索库 | 1,224,119 passages |
| C1 生成失败 / 空答案 | 0 / 0 |
| 参考 passage 在 Top-5 | 46,633（62.94%） |
| 冻结 EM 匹配 | 28,501（38.46%） |
| EM 不匹配且参考 passage 在 Top-5 | 23,745 |
| 上游非空人工 diagnosis 标签 | 0 |

23,745 是筛查候选数，不能视为 utilisation 标签数。EM=0 不等于语义错误，命中参考 passage 不等于证据已经清楚充分。上游 9,032 个重复组指相同**答案字符串**；问题经大小写和空白归一化后没有重复。与旧 NQ3216、500 条预实验、旧 E 审核/v7 输入的 ID 及该归一化问题均无重合。语义近重复和 Wikipedia 页面分组尚未检查。

Train 检索库大于旧 dev 的 99,922 passages，指标应与旧批次分别报告。交付记载 Qwen2.5-7B-Instruct、NF4、greedy、64 tokens，未导出模型 revision 与完整环境；适配结果明确将源 C1 revision 记为未知。

## E 内部划分

先按归一化问题分组并哈希划分，再计算 EM 和筛查审核候选。哈希不读取答案、gold 或标签，seed=42，80/10/10 是哈希桶比例，实际数量如下。

| 用途 | 数量 |
|---|---:|
| development：开发及选规则 | 59,188 |
| calibration：规则固定后校准 | 7,345 |
| reserved_evaluation：预留评测 | 7,564 |
| excluded_by_team：团队另行保留的 ID | 0 |

配置见 [冻结清单](../configs/splits/answer_repair_nq_train_dev_v1.json)，完整 ID 表保存在本地 `data/processed/answer_repair/nq_train_v1/partition_ids.json`。之前已预览的 `-1000144997787636808` 强制归入开发部分。预留评测部分不能用于 E 选规则或阈值；校准部分一旦用于调参，也应如实记录。

这些样本都来自官方 **train**，上述划分只是 E 的内部用途，尚未与 Diagnosis 的未来训练/测试用途协调，也未按 Wikipedia 页面隔离，不能直接当作整个系统的未见测试集。团队确定共同保留 ID 后，可用 `--reserved-ids` 提供 JSON 字符串列表，并指定全套新输出路径生成下一版。同一归一化问题的所有 ID 应一起排除；跨用途时程序拒绝继续。本版冻结文件不覆盖。

## 首批审核

开发部分有 19,010 条筛查候选及 22,730 条 EM 匹配记录。分别用固定 seed 的 ID 哈希排序，选 80 条候选和 20 条对照，按每四条候选后一条对照排列。100 是首批人工工作量，全量 74,097 条仍保留。

| 文件 | 用途 |
|---|---|
| [审核 TXT](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.txt) | 编号、问题、C1 答案和待审状态 |
| [完整审核 JSONL](../artifacts/answer_repair/nq_train_dev_v1/review/review_queue.jsonl) | Top-5、离线参考、指标、来源和人工字段 |
| [C1 兼容输入](../artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_c1.jsonl) | E runner 开发输入，含离线评分字段 |
| [纯运行时输入](../artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_runtime.jsonl) | 仅 sample_id、dataset、query、initial_answer、evidence |
| [生成记录](../artifacts/answer_repair/nq_train_dev_v1/preparation.json) | 全量与审核包的数量、SHA-256、状态 |
| [入口检查](../artifacts/answer_repair/nq_train_dev_v1/preflight.json) | 已有 v7 策略读取成功，没有调用模型 |

审核分别判断 C1 是否正确、证据是否充分、证据是否容易定位，以及参考是否有问题。不能因为 gold 在 Top-5 就认定 utilisation，也不能把 EM=1 对照自动当成正确且有证据支持。可参考 [上游标注指南](NQ_TRAIN_DIAGNOSIS_ANNOTATION_GUIDE.md)。辅助判断和用户确认另存至 `data/annotations/assisted/`、`data/annotations/human/`，以 `review_set_id + dataset + sample_id` 关联，保留原队列为冻结材料。这不代表 Diagnosis 负责人已完成全量标注。

本队列只审核初始 C1 和证据；新的 [v8 输出审核](../artifacts/answer_repair/v8_nq75/review/changed_outputs_pending.jsonl) 单独保存，沿用 NQ Train 首批编号，并区分策略。与旧 [v7 输出审核](../artifacts/answer_repair/v7_combined_review/review_all.txt) 是不同工作，编号不能混用。首批用于识别自然目标错误、正确对照与参考争议，支持 v8 规则设计。80:20 是筛查比例，不能据此估计总体 utilisation 比例或整集准确率；后续评测应另设不按正确性筛选的队列。

## 复现

使用 Python 3.11+。从 Drive 下载 15 个 JSONL，保持文件名，放入 `data/processed/external/nq_train_delivery_20261017/`。原 ZIP 本地文件名与整体哈希在审计中；重新打包后 ZIP 哈希可能不同，以分片清单为数据一致性依据。

```sh
python scripts/prepare_nq_train_answer_repair.py
python -m unittest discover -s tests -q
python scripts/run_answer_repair.py --dataset nq --c1 artifacts/answer_repair/nq_train_dev_v1/inputs/nq_train_review_c1.jsonl --all --strategy short_answer_quote --check-only --out results/answer_repair/nq_train_dev_v1_preflight_only.jsonl
```

准备命令校验来源、转换全部记录、生成划分和审核包；现有输出字节一致时允许重跑，任何不同均拒绝覆盖。新增 10 项测试覆盖 gold 隔离、模型设置、证据完整性、划分隔离、顺序不变性、重复 ID、损坏及覆盖保护，完整 112 项测试通过。最后一条只是入口检查，`remaining_call_upper_bound=100` 是潜在调用上限，不是实际调用数。

本地全量文件为 `data/processed/answer_repair/nq_train_v1/c1_scored.jsonl`，约 283 MB，包含各用途记录及离线参考。**不要直接对全量文件用 `--all` 选策略或跑开发实验**。开发先使用上面的固定 100 条；扩大开发批次时需按冻结 ID/用途筛选并保存新清单。原始分片、全量转换和完整划分 ID 受 Git 忽略；代码、审计、配置和小型审核包纳入版本管理。

实现见 [适配器](../src/compensation/nq_train_adapter.py) 与 [准备入口](../scripts/prepare_nq_train_answer_repair.py)。运行时字段通过白名单导出，gold、EM/F1、来源和人工标签均不进入 E 提示词。
