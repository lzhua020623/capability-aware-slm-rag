# FEVER 检索与新 C1 采集 · 2026-10-24

已实现 Controller 的 FEVER P3-20 检索路径，并完成新 900 条题目的 Top5 准备及冻结 GPU 包。完整 CPU 回归 **726 项通过**。Qwen 对这 900 条的新答案须由独立 Kaggle/Colab 采集任务生成；打包、CPU 检索或测试通过均不等于已生成答案，也不代表增强收益。

采集进度：2026-10-24 核查 [Kaggle 后台 Version #1](https://www.kaggle.com/code/yearo111/controller-fever-c1-900-frozen-20261024) 已 Failed，44.5 秒，在 CPU preflight 因 `src.retrieval.__init__` 提前加载未安装的 FAISS 中止。77 个输入文件校验通过，Torch 2.11.0+cu128 / CUDA 12.8 / T4×2；尚未调用 Qwen，生成 **0/900**。失败日志和本地复现在 `artifacts/controller/fever_c1_v1/attempt1_failure/`。新的 900 条 C1 仍待完成和输出验收；NQ C1 不重跑。

修正版 `transport_r1` 只把 retrieval 包公开 API 改为延迟导入；原有 73 个冻结源码仅此一处变化，不安装或调用额外检索模型。原 900 条题目、Top5、顺序、划分、模型/提示/预算逐项保持。通过独立登记脚本生成派生清单，不覆盖原包；入口为 `notebooks/controller_fever_c1_transport_r1.ipynb`，与新私有输入配套。此修正发生在首次 Qwen 调用前，不属于看过 QA 结果后的策略优化。733 项 CPU 回归及从实际归档解压的无 FAISS/编码器 preflight 均通过，凭证在 `transport_r1/validation/`。

**Version #2（`scriptVersionId=361668830`）已因 CUDA OOM 失败**，总耗时 1817.1 秒，保留 645 条完整输出：开发 200、选型 200、最终候选 245。原始 ZIP、11 个输出文件、日志、输入长度清单和覆盖验收在 `attempt2_oom/`。输入/输出逐条顺序、证据全文、原提示及原始生成关系通过程序核对；没有逐题展示最终候选答案或计算 QA 分数。

第 646 条有 12,911 个输入 token；日志确认 OOM 发生于 `scaled_dot_product_attention` 的预填充阶段，申请额外 17.39 GiB，而 GPU 0 总容量为 14.56 GiB。实际模型参数在 `cuda:0`，T4×2 不是合并的单卡显存。日志不含具体 SDPA kernel profiler 结果，因此后端回退只能视为可能机制。PyTorch 的 [SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) 说明其多种实现及 math 后端中间精度；冻结 Transformers 5.16.1 的 [`_prefill` 实现](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py) 提供原生分块预填充。

`prefill_r2` 是明确登记的新计算配置：全部 900 条统一采用 `prefill_chunk_size=256`，保留所有输入 token、原 Top5、模型 revision、量化、原提示、greedy 和 64 输出 token。首次真实采集前运行原两条虚构检查及一条至少 13,000 token 的虚构长输入。分块可能影响数值和时延，故不把旧 645 条与新输出拼接；保留旧失败记录，新建完整运行。相对 r1 仅修改 C1 runner 的冻结源码。741 项 CPU 回归与实际解压归档的依赖隔离 preflight 通过，入口为 `notebooks/controller_fever_c1_prefill_r2.ipynb`；云端启动和 GPU 长输入验证见最新 `cloud_status.json`。

## 本轮复用与新增

| 数据 | 处理 |
|---|---|
| NQ Train 既有 C1 | 本轮 900 条直接复用，不重新生成 |
| FEVER 旧 500 条 C1 | 保留历史结果；固定前 10 条仅用于检索资源兼容验收 |
| FEVER 新 900 条 | 与旧 500 条 ID 无重合；开发 200、选型 200、最终候选 500，按原冻结清单补 C1 |

新 FEVER 抽样与原保留池保持 T2 版本，未根据预测对错、失败类型或预期修复能力筛选。10 条历史查询的 Top5 ID 及顺序全部匹配，Top20 前五项与 Top5 一致。查询快照固定 `BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`；历史建库 revision 未记录，这次检索回放不补写历史 provenance，也不代表所有向量均经过独立语义验证。

## Controller 路径

`--fever-resources configs/controller/fever_resources_v1.json` 显式加载原 IVFPQ、ID、offset 和语料，验证原包/全量顺序/结构/查询回放的哈希链。NQ 与 FEVER 后端严格按 dataset 分派，不能把 NQ Train 索引用于 FEVER。

FEVER 按 T1 规定执行原 claim Top20 → 固定 reranker → 等权 RRF(k=60) → Top5 → 原 FEVER C1 提示重答。NQ 原默认 API 继续仅支持 NQ，FEVER 需要显式启用；正式 runner 的 FEVER adaptive/always_retrieve 在缺少资源时提前报错，不把缺资源回退计作实现覆盖。

FEVER 接受既有 parser 可解析的 SUPPORTS/REFUTES，并保存原始生成文本；不可解析或 provider 出错则保留进入阶段的答案。NQ 原 span grounding 规则不用于 FEVER 二分类标签。标签可解析不证明判断正确，这仍须独立评测。这里使用固定 Qwen，没有引入 NLI 或新提示。

```bash
python scripts/run_controller_integration.py \
  --input /path/to/fever_development_runtime.jsonl \
  --fever-resources configs/controller/fever_resources_v1.json \
  --preflight --output-dir /path/to/new_preflight
```

两数据集混合执行时另外提供 `--nq-resources configs/controller/nq_train_resources_v1.json`。新 C1 runtime 未生成前不能用 query-only c1_tasks 冒充 Controller 输入。

## 900 条采集包

本机 CPU 已为全部 900 条生成真实 Top5，严格字段为 `sample_id,dataset,query,evidence`；不读 gold、不生成答案。位置为 [collection](../artifacts/controller/fever_c1_v1/collection/manifest.json)，三个用途文件分开保存。模型查询编码在独立 Torch 子进程执行，父进程使用 FAISS，避免此前本机 OpenMP 混载问题。

GPU 包 [controller_fever_c1_gpu_v1.tar.gz](../artifacts/controller/fever_c1_v1/controller_fever_c1_gpu_v1.tar.gz) 为 2,027,671 bytes，含 77 个校验文件，不包含 gold、标注、模型权重或大索引。代码与输入逐文件 SHA-256 固定，支持 Kaggle 自动解压输入。入口为 [controller_fever_c1_v1.ipynb](../notebooks/controller_fever_c1_v1.ipynb)。

GPU 仅运行固定 Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`、NF4、FP16、double quant、greedy、64 输出 token、seed 42，沿用原 FEVER C1 提示。先运行两条虚构输入的工程检查，再一次性采集 900 条。工程检查只验输出可解析及加载契约，不调参、不以真实题答案决定是否继续。

```bash
python scripts/prepare_controller_fever_c1.py --output-dir /path/to/new_collection
python scripts/build_controller_fever_c1_bundle.py \
  --collection /path/to/new_collection --archive /path/to/new_bundle.tar.gz \
  --notebook /path/to/new_notebook.ipynb
python scripts/run_controller_fever_c1.py \
  --collection /path/to/new_collection --output-dir /path/to/new_outputs --preflight
```

生成输出按用途包含 `runtime.jsonl`、`generations.jsonl`，另有环境、pip freeze、工程检查、输入指纹和完成状态。原始输出保留；空输出显式记为 UNKNOWN，不猜二分类答案；模型异常则标记失败并保留部分结果，不能删除失败题或挑题重跑。runtime 不含 gold，参考标签仍只在 T2 离线目录。

每条保存 Qwen 调用耗时和 token 数。当前成本范围为 C1 生成，CPU Top5 准备未逐题计时，不能据此声称已取得完整端到端 C1 成本。

最终候选输出应封存，不逐题查看或据此继续优化。Full、No Probe、人工类型对照、分组预算、完整配对统计和正式执行 seal 仍按 T3 推进；此采集任务用于补齐 T2 基线，不是正式六组增强比较。

验收记录见 [fever_provider_v1](../artifacts/controller/fever_provider_v1/README.md)，GPU 任务状态见 [cloud_status.json](../artifacts/controller/fever_c1_v1/cloud_status.json)。
