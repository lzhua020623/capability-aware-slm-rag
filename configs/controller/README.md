# Controller 配置索引

截至 2026-10-31，默认主线是 **B + BM25 + 自适应答案选择**。
默认值集中在 `src/pipeline/registrations.py`；历史 JSON 是冻结记录，保留原文件和哈希。

| 登记文件 | 用途 | 是否默认 |
|---|---|---|
| `quality_adaptive_v1.json` | B + BM25 + 自适应答案选择，当前 NQ 主线 | 是 |
| `quality_bm25_v1.json` | B + BM25 对照 | 否，显式选择 |
| `quality_baseline_v1.json` | B 路由对照 | 否，显式选择 |
| `current_baseline.json` | 历史 controller_2 四动作基线 | 否；文件名保留用于历史复现 |

其余 JSON 包括数据／模型资源登记、实验协议和组件配置，不能直接代替上述完整流水线登记。
`recoverability_oracle_v1.json` 是使用金证据的能力上限测量协议，不属于线上产品。

## 当前入口

```bash
python scripts/check_controller_current.py
```

返回的 `registration` 应为 `configs/controller/quality_adaptive_v1.json`，
`p3_bm25_enabled` 和 `answer_selection_enabled` 均应为 `true`。
此检查不启动模型，只检查已登记的代码、模型指纹和数据边界；检索索引、语料、CUDA 等运行资源仍需准备。

```python
from pathlib import Path
from src.pipeline.current import load_current_pipeline

pipeline, environment = load_current_pipeline(
    nq_resources=Path('configs/controller/nq_train_resources_v1.json'),
    data_role='validation',
)
result = pipeline.run('your question here')
print(result.final_answer)
```

确认集使用仅含 `sample_id`、`query` 的完整登记输入；输出目录必须是本次运行的独立目录：

```bash
python scripts/run_quality_eval.py \
  --role confirm --queries /path/to/confirm_queries.jsonl \
  --output-dir /path/to/new_adaptive_confirm
```

默认采用 `quality_adaptive_v1.json`。历史 B 或 B+BM25 对照分别显式传入
`--registration configs/controller/quality_baseline_v1.json` 或
`--registration configs/controller/quality_bm25_v1.json`。
Test 的显式解封开关与数据边界继续生效；切换默认配置不授权新增 Test 运行。

## 续跑和历史复现

续跑前检查登记路径及 SHA-256、问题文件、数据角色、批次大小、分片和题目范围。
任一项不同就停止，不截断或覆盖旧结果，也不加载 GPU。旧 B 运行续跑时必须显式选择 B。
有结果但没有运行清单的目录也不能直接续跑。

`precompute_quality_sweep.py`、`precompute_p3_coverage.py`、`l2_generate.py` 和
`smoke_production_e2e.py` 固定使用历史 `current_baseline.json`，保留原实验的 backend；
最后一个脚本专门检查历史四动作覆盖。
团队 `baselines/run_product_evaluation.py` 使用其配置的 `production.baseline_registration`，
不跟随最新主线默认值。历史结果的回放测试按各自归档登记核验。

## 结果范围和运行成本

当前 NQ Test 11,114 条归档判官准确率为 62.76%，相对 B+BM25 增加 2.14 个百分点。
Test 已经曝光多次，使用次数及比较方法见
[归档报告](../../artifacts/controller/adaptive_selection_eval_v1/README.md)。
接入后的产品在确认集 2,440 条上复现全部最终答案，见
[工程等价性记录](../../artifacts/controller/adaptive_product_equivalence_v1/README.md)。
这里没有新增 GPU 运行或新得分。

选定策略 θ=0，每题都进入选择器；一致性阈值 τ=0.8 控制在 3 / 5 / 8 个上下文中的停止阶段。
确认集平均额外生成约 3 次，整条流水线在该次 RTX 5090 D 上约 2.6 秒／题，B+BM25 约 1.3 秒／题。
这些结果和边界仅针对 NQ，不能据此宣布 FEVER 已完成本主线评测。

## 跨平台字节要求

部分冻结 SHA-256 对应 CRLF 文本，Git blob 存储为 LF。根目录 `.gitattributes` 按路径固定当前主线所需文件的 CRLF 检出字节，Linux / macOS / Windows 均适用，无需修改全局 Git 换行设置。
新克隆会应用这些规则；已有混合换行工作区须核对就绪检查后再运行。打包使用检出后的文件，不能直接把 LF blob 当成冻结运行包。
哈希不匹配仍停止加载，不改登记哈希、不跳过检查。
模型权重、FAISS / BM25 索引和完整语料仍按资源登记在 Git 外提供。
