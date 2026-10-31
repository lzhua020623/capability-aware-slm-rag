# 自适应答案选择：产品代码等价性检查（确认集，2026-10-31）

用接入后的产品代码（登记 `configs/controller/quality_adaptive_v1.json`）在确认集 2,440 题上真实运行一次，逐题对比离线评测（`artifacts/controller/adaptive_selection_eval_v1`）按选定策略回放得到的结果。只做工程一致性检查，不判分，不用于任何选择。

| 检查 | 一致 / 总数 |
|---|---:|
| C1 答案 | 2440 / 2440 |
| 控制器给出的答案（B+BM25） | 2440 / 2440 |
| 停止阶段（3 / 5 / 8） | 2440 / 2440 |
| 每个候选答案组的 15 个特征 | 2440 / 2440 |
| 额外生成次数 | 2438 / 2440 |
| **最终答案** | **2440 / 2440** |

- 产品改了 372 题的答案，和离线回放相同；运行中没有错误（0 次）。
- 两题的生成次数比离线统计多 1 次：某个上下文的 Top-5 与该题已经作答过的 Top-5 段落相同、顺序不同（P3-20 在 CPU 上重排，这里在 GPU 上重排，分数极接近的两段排序互换）。离线统计按"已经答过"计成本，产品重新生成了一次，答案相同。
- 因此 `artifacts/controller/adaptive_selection_eval_v1` 中离线得到的确认集和 Test 结果可以代表产品代码的表现。
- 成本：答案选择阶段每题平均 3.00 次生成，平均 1.29 秒（NVIDIA GeForce RTX 5090 D）。整条流水线每题约 2.6 秒，B+BM25 约 1.3 秒。
- 运行：正式运行、未续跑；完整结果（压缩后约 142 MB）留在本地 `data/processed/quality_sweep_v1/adaptive_product_confirm/`，哈希记在 `provenance.json`。
- 测试 `tests/test_adaptive_product_equivalence.py` 回放这些对比。
