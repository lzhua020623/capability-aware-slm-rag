# Controller 资源接入 · 2026-10-24

当前完成的是 T2/T3 的资源与 NQ Train 接口接入。它解决“向量命中后如何取到对应原文”，不改变 Diagnosis、Probe、路由阈值、增强提示或评分标准。没有新的问答质量提升结论，完整比较仍需执行 seal 和 GPU 验收。

## 已完成的检查

| 对象 | 已完成 | 范围限制 |
|---|---|---|
| NQ Train 检索库 | 1,224,119 条 ID、offset、原文逐条核对，逐条实际 seek 回读；绑定原索引 SHA-256 | 本轮不重新编码全部向量；此前 64 段独立 BGE 比对仍单独记录 |
| NQ Controller provider | 显式资源注册表、文件指纹和全量核对报告验收；按 offset 读取原文，缓存最多 128 段 | 真实 Qwen 路由和答案生成尚需 CUDA；CPU replay 只验收检索 |
| FEVER 原资源 | 从团队 Drive 下载两个原 ZIP；5,396,106 条 corpus/ID 全量顺序核对，生成 offsets | 原始 BGE revision 未记录；不能把结构检查称为语义向量验证 |
| FEVER 原索引 | 元数据与已入库记录字节一致；实际 FAISS 属性和原文搜索回读检查单独归档 | 尚未接入 FEVER 动态补检索，新的 900 条 C1 也未生成 |

两种数量都是**知识库段落数**。NQ 问题仍来自原 74,097 条子集。冻结的开发/选型/最终候选仍为每个数据集 200/200/500；完整保留池 NQ 7,564、FEVER 1,886 不变。

NQ 的独立向量验证快照为 `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`。这是本次验证和新执行显式固定的快照，历史 C1 的 revision 仍为未知。检索 replay 使用已曝光的开发集前 10 题，对比真实 BGE 查询和原索引产生的 Top5 ID，同时检查 Top20 的前五项一致；不读取离线 gold，不生成答案，不按 EM 选样或修改算法。

实际 replay **10/10 题的 Top5 ID 及顺序与原 C1 一致**，Top20 的前五项也一致。报告见 [retrieval_replay.json](../artifacts/controller/nq_mapping_full_v1/retrieval_replay.json)。这是检索资源/接口一致性结果，不能转换成回答正确率或 E 的收益。

## NQ Train 接入方式

```bash
python scripts/run_controller_integration.py \
  --input artifacts/controller/data_v1/nq/development/runtime.jsonl \
  --nq-resources configs/controller/nq_train_resources_v1.json \
  --preflight --output-dir /path/to/new_preflight
```

注册表列出 corpus、index、ID、offset、恢复元数据和全量核对报告的路径、大小和 SHA-256。preflight 不加载模型，校验全部指纹；不指定资源时，旧的输入格式预检查仍可运行，但不能据此称资源就绪。真实执行也会重新校验资源。

`--nq-resources` 与旧的 `--nq-index/--nq-metadata` 二选一。旧接口继续用于含全文的 JSONL metadata；不能把只有 ID 的 sidecar 传给旧接口。新 provider 复用原 NQFaissBackend、原 BGE 查询前缀、L2 normalization、IP 分数和 Top20 上限，传给后续模块的段落仅含 passage_id/text/score。

本机验证使用 FAISS CPU 1.15.1 的 Flat read-only mmap，BGE 独立进程编码，避免此前 macOS OpenMP 冲突。完整 CUDA 环境还需独立锁定和实测，不能用本地 CPU 环境替代 GPU seal。新 provider 显式检查 mmap 支持；缺失时会报错，防止静默换成占用更高的加载方式。

资源检查与重放：

```bash
python scripts/check_nq_train_index_mapping.py --output /path/to/new_mapping.json
python scripts/replay_nq_controller_retrieval.py --output /path/to/new_replay.json
```

## FEVER 资源位置与恢复

两个输入 ZIP 分别为 `for_fever_c1.zip`（424,325,031 bytes）和 `fever_corpus.zip`（1,265,030,053 bytes），来源为团队 ELEC5623 Drive 目录。校验码见 [import_validation.json](../artifacts/controller/fever_resources_v1/import_validation.json)。

本机解包到新的、被 Git 忽略的 `data/processed/controller_resources/fever_v1/`，不覆盖原云端占位文件：

| 文件 | 作用 |
|---|---|
| `fever_corpus.jsonl` | 原文正文 |
| `fever_bge_ivfpq.index` | 原 IndexIVFPQ |
| `fever_bge_ivfpq_ids.jsonl` | 向量序号到 passage ID |
| `fever_bge_ivfpq_offsets.npy` | 向量序号到原文文件位置，本轮生成 |
| `fever_bge_ivfpq_meta.json` | 原建库元数据，与仓库记录相同 |

ZIP 中自带的历史 `fever_dev.jsonl` 只记录哈希，未安装或覆盖 T2 的来源数据。它的字节大小与冻结来源不同；本次没有根据该副本重新选题、替换标签或修改保留池。

```bash
python scripts/import_fever_controller_resources.py \
  --c1-zip /path/to/for_fever_c1.zip --corpus-zip /path/to/fever_corpus.zip \
  --resource-dir /path/to/new_fever_resources --report /path/to/new_import.json
python scripts/check_fever_controller_resources.py \
  --resource-dir /path/to/new_fever_resources --import-report /path/to/new_import.json \
  --output /path/to/new_structure_check.json
```

只提取明确指定的文件到明确目的地；拒绝覆盖、符号链接和重复成员。完整读取 ZIP 成员会校验 CRC。生成 offsets 前核对全部 ID 顺序、重复 ID、文本和数量。FAISS 检查使用三条 PQ 重建向量确认搜索结果能回读原文，不用这一步证明 BGE 语义匹配或 QA 收益。

## 后续执行顺序

1. 补 FEVER 动态检索 provider，并核验固定 BGE 查询快照与原检索结果的兼容性。
2. 为冻结 FEVER 900 条生成原协议 C1 Top5/答案并封存；开发、选型、最终输出分开，不以答题结果删样。
3. 补 T3 Full、No Probe、人工类型离线对照、各组预算和全样本配对验收，登记实际 CUDA 环境并创建 execution seal。
4. 在同批输入上比较修复、误修、路由损失和成本，再决定集中优化；不提前使用保留池做调参。

完整 CPU 回归 **706 项全部通过**。完整验证和源码差异登记见 [本轮验收](../artifacts/controller/resource_adapter_v1/README.md)。T1 协议及 T2 原冻结记录保持原字节；新进度是追加记录，不回写历史验收结论。
