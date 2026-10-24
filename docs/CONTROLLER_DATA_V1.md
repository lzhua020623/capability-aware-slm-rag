# Controller 数据冻结 v1 · T2

日期：2026-10-23。对应 [T1 实验协议](CONTROLLER_EXPERIMENT_PROTOCOL_V1.md)。

2026-10-24 C1 补齐：新 FEVER 900 条已统一采集并通过输入/输出与环境验收，runtime 与指纹见 [fever_c1_completed_v1.json](../configs/controller/fever_c1_completed_v1.json)，完成报告见 [completed_v3](../artifacts/controller/fever_c1_v1/prefill_r2/completed_v3/README.md)。下文及原 T2 manifest 保留冻结时的状态，不覆盖其哈希；此新增登记只证明 FEVER C1 可用，正式 Controller 比较仍须通过 T3 执行验收。NQ 既有 C1、原划分和完整保留池不变。

2026-10-24 资源进展：已从团队 Drive 取回 NQ Train 原语料和原索引，并[恢复 ID/offsets 与恢复记录](NQ_TRAIN_INDEX_RECOVERY.md)，64 处独立向量比对通过。本文及 `data_v1/asset_inventory.json` 保留 T2 冻结时的历史状态；当前 NQ 资源以新增恢复归档为准。FEVER 资源包内容、正式执行环境及 T3 provider 适配仍待验收，T2 不因此标为全部就绪。

**已冻结样本、完整评测保留池、配对参考和曝光记录。T2 尚未全部验收：新 FEVER C1 和匹配检索资源仍待补齐。** 本次没有推理、训练、改答案或计算新模型准确率。

## 1. 两个数据集都保留 10%–15% 用于评测

| 数据集 | 本次来源总体 | 完整评测保留池 | 比例 | 本轮固定评测子集 | 其余继续封存 |
|---|---:|---:|---:|---:|---:|
| NQ Train | 74,097 | **7,564** | **10.21%** | 500 | 7,064 |
| FEVER | 13,229 条有有效 gold evidence 的二分类候选 | **1,886** | **14.26%** | 500 | 1,386 |

**500 是本轮预登记的实际评测子集，不是全部保留量。所有保留 ID 都禁止用于训练、prompt 修改、阈值调参和候选选型。** 不能把未抽中的其余保留样本重新放回训练池；也不能看过 500 条结果后临时扩样，直到得到想要的显著性。若要改为完整保留池评测，须在首次输出前登记新的运行范围与预算。

完整保留 ID 在 [evaluation_reserve_ids.json](../artifacts/controller/data_v1/evaluation_reserve_ids.json)，团队统一入口在 [controller_data_v1.json](../configs/splits/controller_data_v1.json)。保留池是用途限制；并不表示整池已做人工语义审核、页面隔离或 C1 资源验收。被质量规则隔离的保留样本仍不得移作训练。

NQ 沿用原始内部用途：development 59,188、calibration 7,345、reserved_evaluation 7,564；历史划分不改。Controller 的 `selection` 对应原 calibration。FEVER 在有效候选上按问题哈希分配 70/15/15 用途，再排除已曝光 ID；得到开发候选 8,956、选型候选 1,869、完整保留池 1,886。FEVER 的比例以 13,229 条有效候选为分母，**不是完整 FEVER train 的规模**。

## 2. 本轮运行清单

| 用途 | NQ | FEVER | 可否用于优化 |
|---|---:|---:|---|
| development | 200 | 200 | 可用于定位错误、设计唯一候选 |
| selection | 200 | 200 | 只用于冻结候选与基线的选型比较，不继续逐题改提示 |
| final_candidate | 500 | 500 | 仅最终版本评测；该目录名表示执行前仍需 C1/资源封存 |
| 合计 | 900 | 900 | 不是全量训练或全量评测 |

全部 ID、顺序和 SHA 在 [manifest.json](../artifacts/controller/data_v1/manifest.json) 与 [selection_ids.json](../artifacts/controller/data_v1/selection_ids.json)。900 条是三个用途的合计，不能合并报作 900 条未见测试成绩。也不能把 74,097 条源数据规模写作实验样本数。

500 条是控制本轮推理规模的预登记选择，不能保证检出 1–2 个百分点的小收益。作为规划示例，单一正确率在最不利分布下的 95% 正态近似半宽约 4.38 个百分点；配对差的区间取决于两模型不一致率，假设不一致率 20% 时约 3.92 个百分点。这些不是实验结果，也不是功效保证。最终以 T1 配对区间和检验为准。

## 3. 筛选与曝光隔离

- NQ 的 15 个原始分片逐一通过交付 SHA 和数量核对，共 74,097 条；使用原始交付的短答案参考，不读取初标标签作为抽样分数。
- FEVER 从本地 13,332 条二分类记录中按既有“有效 gold evidence”条件排除 103 条，剩余 13,229；不按模型对错或预期修复能力选题。
- 检查参考提交 `30738f0` 中 525 个历史 JSON/JSONL 文件，覆盖原 C1/增强结果、审核记录、Diagnosis 划分和已有曝光登记。NQ Train 池命中 426 个已曝光 ID；FEVER 有效池命中 518 个。全部从本轮三个样本组排除。
- NQ 隔离旧审计的 35 条问题变体，并按 NFKC、casefold、去标点、空白归一化隔离另外 46 条跨用途变体。原 split 文件和报告保留。
- 按 seed 42 的 SHA 排序抽取，先保护最终子集，再取选型与开发。候选与已曝光问题、已选问题之间再做 content-token Jaccard ≥0.60 的直接词面近似检查；不做语义推断，不读取 gold/C1 正确性来排序或替补。
- 用户在本轮明确确认团队没有在仓库记录之外使用这些数据做训练、调参或错误分析，该声明保存在 policy/exposure audit。这里的“未曝光”限项目记录与此声明，不声称排除了模型预训练、语义近似或相同 Wikipedia 页面。

完整记录见 [exposure_audit.json](../artifacts/controller/data_v1/exposure_audit.json) 和 [exclusions.jsonl](../artifacts/controller/data_v1/exclusions.jsonl)。词面筛选改变了候选分布，报告应注明这是本项目控制后的评测子集，而不是整个官方数据集的无条件成绩。

## 4. 已有 C1 与仍缺的内容

NQ 导出 900 条严格运行时记录，只有 `sample_id,dataset,query,initial_answer,evidence`，gold 单独保存。每题保留原 C1 和有序 Top5。原 C1 的 Qwen revision/完整环境未导出，因此本轮明确采用“历史 C1 后处理”比较，字段保持 unknown；不会编造为已核验的新 revision。

FEVER 导出 900 条 **仅含题目与 ID 的 C1 生成任务**，其 gold 也单独保存。没有为它们伪造 `initial_answer` 或 Top5。需要用 T1 固定 Qwen 和匹配的 FEVER 索引生成新 C1，封存后再开始六组增强比较。不能以现有 500 条旧 C1 替换这批新的选型/最终题目。

| 路径 | 内容 |
|---|---|
| `artifacts/controller/data_v1/nq/<role>/runtime.jsonl` | NQ 纯运行时输入，可供 Controller 使用 |
| `artifacts/controller/data_v1/fever/<role>/c1_tasks.jsonl` | FEVER 待生成 C1 的任务，尚不能作为 DiagnosisInput |
| `artifacts/controller/data_v1/offline/<dataset>/<role>/gold.jsonl` | 原数据参考答案/标签，只用于独立评测进程 |
| `evaluation_reserve_ids.json` | 两数据集完整保留池；未来训练准备必须排除全部 ID |
| `policy.json` / `manifest.json` | 抽样、用途、哈希、来源与仍未完成的条件 |

选择只经过自动结构检查，没有新增人工 gold，也没有打开最终题目逐题讨论。人工类型触发对照仍需可靠类型标注；不能将新 AI 初标自动当作该对照的真值。这不妨碍先完成集成工程。

## 5. 已重新检索全部远端分支

本次检查了 10 个原远端分支及新增 `controller`，共 11 个分支的文件树。记录见 [branch_resource_inventory.json](../artifacts/controller/data_v1_validation/branch_resource_inventory.json)。

| 已在分支中找到 | 路径 / 说明 |
|---|---|
| NQ Train C1 生成与导出代码 | `scripts/run_nq_train_c1_generation.py`，在 main 等分支，已合入 controller |
| NQ Train C1 全量统计及交付清单 | `data/annotations/nq_train_annotation_candidates_stats.json`、`data/annotations/nq_train_delivery/manifest.json`；记录 74,097 条 |
| FEVER C1 运行器 | `scripts/run_fever_c1_preliminary.py`，来自 feature/fever-c1 |
| FEVER 旧 500 条预测及 Top5 | `results/preliminary/fever_c1_7b.jsonl`、`fever_c1_top5.jsonl` |
| FEVER 索引描述文件 | `results/retrieval/fever_bge_ivfpq_meta.json`；描述 5,396,106 条、IVFPQ，文件本身不是索引 |
| NQ Train 索引构建及读取代码 | `scripts/build_nq_train_bge_index.py`、`scripts/run_nq_train_bge_retriever.py` |

全部分支中没有跟踪 `.index`/`.npy` 文件；[SETUP](SETUP.md) 和 [预实验协议](PRELIMINARY_EXPERIMENT.md) 明确说明大型语料/索引不入 Git，需要从原持有者取得或按协议重建。NQ 完整交付已存在本机并通过清单核验，不能因它受 Git 忽略就称“没有 C1 数据”。

仍需要原运行环境中的以下资源：

| NQ Train（1,224,119 passages） | FEVER（5,396,106 pages） |
|---|---|
| `nq_train_bge_base.index` | `fever_bge_ivfpq.index` |
| `nq_train_bge_base_ids.jsonl` | `fever_bge_ivfpq_ids.jsonl` |
| `nq_train_bge_base_offsets.npy` | `fever_corpus.jsonl` |
| `nq_train_corpus.jsonl` | 已有 metadata，但须与实际文件核对 |
| `nq_train_bge_metadata.json` | 与 C1 匹配的 BGE 权重快照/revision |
| 与 C1 匹配的 BGE 权重快照/revision | |

本机 FEVER corpus 为约 3.52 GB 的云占位文件，未将占位状态算作正文完整可用。NQ Train 资源采用 ID/offset/corpus 组合，当前 Controller provider 接口期待带全文的 metadata JSONL；T3 要补无损适配，不能把 ID 文件直接冒充正文 metadata，也不能换用旧 NQ dev 的 99,922 段索引。

本轮 Google Drive 搜索返回 `ACCESS_TOKEN_SCOPE_INSUFFICIENT`，浏览器回退返回 `No browser is available`；这是远端资源发现尚未完成的原因，不能据此断言同学没有保存这些文件。资源到位前，索引匹配、BGE revision 和全库质量仍为未核验。

## 6. 验证与后续

```bash
python scripts/check_controller_data.py --check-sources
python scripts/check_controller_data.py --require-full-t2
```

第一条检查已登记 manifest、所有产物 SHA、来源 SHA、配对 ID、严格字段隔离、保留比例以及保留 ID 不进入优化集。第二条当前按预期返回 2，表示新 FEVER C1/资源 seal 未完成。数据结构检查不等于质量提升或索引验收。

全量 CPU 回归 **665 项通过**（本次新增 17 项），见 [验证目录](../artifacts/controller/data_v1_validation/README.md)。原始人工审核文件、历史数据、T1 协议和算法提示不改。

下一步可继续 T3 的 Full、No Probe、人工类型接口、数据资源适配与评测验收工程；真正开始 GPU 配对比较前，必须取得匹配资源，生成并封存 FEVER C1，且通过独立执行清单验收。本次不要求先重做人工标注。
