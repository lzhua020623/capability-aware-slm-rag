# NQ Train 索引配套文件恢复

本次沿用团队原有的 74,097 条问题子集及其检索库，不扩大 NQ 样本、不重建完整向量索引。语料和原索引来自团队 [nq_train_document](https://drive.google.com/drive/u/0/folders/1ZF9i_M-PJSzJSOW34rKPcAB1Ft-0QCQA) 共享目录。

## 恢复什么

| 本地路径（相对仓库） | 作用 |
|---|---|
| `data/processed/retrieval/nq_train_corpus.jsonl` | 从 Drive 取回的原语料正文 |
| `data/processed/indexes/nq_train_bge_base.index` | 从 Drive 取回的原 FAISS IndexFlatIP |
| `data/processed/indexes/nq_train_bge_base_ids.jsonl` | 第 i 个向量对应的 passage ID |
| `data/processed/indexes/nq_train_bge_base_offsets.npy` | 第 i 段原文在 UTF-8 corpus 文件中的位置，int64 数组 |
| `data/processed/indexes/nq_train_bge_metadata.json` | 本次恢复的指纹和验证记录，明确区别于原始建库记录 |

文件与原检索程序的 `CorpusLookup.seek` 读取方式一致。大型语料、索引及生成的 sidecars 位于被 Git 忽略的 `data/processed/`；可复现脚本、测试、哈希和验证摘要入库。

2026-10-24 的实际恢复已经成功：1,224,119 个段落的 ID 和 offsets 均已生成，64 个位置的原文定位和独立 BGE 向量比对通过。最小余弦相似度为 0.9999998808，最大分量差为 3.39×10⁻⁷。这里统计的是原检索库的段落数量，问题来源仍为原 74,097 条子集。实际模型快照及完整文件指纹见验证目录。

## 为什么可以恢复

[原建库代码](../scripts/build_nq_train_bge_index.py) 按语料文件中非空行的原始顺序编码，并按 shard 编号顺序将向量追加进索引。因此在原语料与原索引确实配对时，可以扫描语料恢复 ID 与位置，保留每一条记录的原顺序，不需要重新编码全部语料。

恢复工具不会自动筛除、重排或修正记录。重复 ID、空正文、无效 JSON、数量或维度不一致都会报错，且不发布半成品；已有配套文件不会被覆盖。

## 核验方式与边界

[恢复脚本](../scripts/recover_nq_train_index_sidecars.py) 执行以下检查：

1. 原索引为 IndexFlatIP、768 维、1,224,119 个向量，全部向量有限且已 L2 归一化。
2. 非空语料行与索引数量一致、passage ID 唯一、正文非空；保存原文件 SHA-256。
3. 在固定 seed 42 抽取的 64 个位置（含首尾）上，用原检索程序相同的文本文件 `seek` 方式读取，并核对 ID。
4. 指定 `--verify-embeddings` 后，解析并记录一个确定的 BGE revision，在 CPU 上重编码这 64 段正文，与原索引对应行的向量逐一比较。余弦相似度须 ≥0.9999，最大分量差须 ≤0.001；不一致时不发布配套文件。

第 4 项只检验抽样位置及当前快照的数值兼容性，不能证明全部向量逐项相同，也不能将本次 BGE revision 声称为原 C1 的历史 revision。原始运行环境和编码耗时没有找回，metadata 中保持 unknown/null。

## 崩溃核对与低内存运行

原合并进程的 macOS 崩溃报告确认 `SIGSEGV / EXC_BAD_ACCESS`，故障栈位于 `libomp.dylib`，进程内存在三个同名运行库镜像。报告没有记录 OOM 终止；这些证据支持运行库兼容问题这一排查方向，但尚不能单凭报告确定是哪两个包发生冲突。随后应用重启导致的执行会话丢失也不等于已确认相同原因。

当前工具默认使用只读磁盘映射加载 FlatIP，依赖 FAISS 的 [`IO_FLAG_MMAP_IFC`](https://github.com/facebookresearch/faiss/blob/main/faiss/index_io.h) 支持；BGE 抽样编码在独立 CPU 子进程运行，并将 OpenMP、OpenBLAS、Accelerate 和 Torch 编码线程限制为一。这样不需要在同一个进程同时持有大索引和 Torch 编码运行库。该方式已完成实际恢复；无须为本次 sidecar 生成更换 CPU 或 GPU。

不支持 Flat mmap 的 FAISS 会明确报错，不能静默退回大堆内存加载。`--index-read-mode heap` 是显式兼容选项，需自行确认内存余量。原语料和索引不被改写。完整系统 Qwen 实验仍应使用项目已选的 CUDA GPU 平台。

这些检查恢复的是检索资源的可用性，不计算 QA 指标，不表示模型准确率提升、全部语料经过人工审核或 Controller 全部执行路径已经验收。FEVER 资源、NQ Train Controller provider 适配和正式执行 seal 继续按 T2/T3 清单处理；原评测保留池和最终样本不改动。

## 执行

需先将原 corpus 和 index 放到表中的路径，并安装项目要求的 NumPy、FAISS CPU、Sentence Transformers、PyTorch 等依赖。恢复本身使用 CPU；重编码抽样段落会在本机缓存缺失时下载 BGE 权重。

```bash
python scripts/recover_nq_train_index_sidecars.py --verify-embeddings --sample-count 64 --index-read-mode mmap
python -m unittest tests.test_nq_index_sidecar_recovery -v
```

工具拒绝覆盖已有 sidecars。`--verify-embeddings` 未启用的记录会明确写 `performed=false`，不能当作通过向量顺序核验。

本次实际运行及测试结果保存于 [验证目录](../artifacts/controller/nq_index_recovery_v1/README.md)。

后续已对全部 1,224,119 个段落逐条检查 ID/offset/原文，并执行实际 seek 回读；这与此前 64 段向量比对是两个不同检查。Controller 的新资源注册及 offset provider 见 [资源接入说明](CONTROLLER_RESOURCE_ADAPTER_V1.md)，全量定位原始报告见 [validation.json](../artifacts/controller/nq_mapping_full_v1/validation.json)。
