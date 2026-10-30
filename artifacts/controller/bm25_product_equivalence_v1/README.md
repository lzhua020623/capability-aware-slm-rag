# P3-BM25 产品代码等价性检查（确认集，2026-10-31）

用接入后的产品代码（commit afb1818，登记 `configs/controller/quality_bm25_v1.json`）在确认集 2,440 题上真实运行一次，逐题对比"路线 1 结果 + 离线 BM25 规则"的期望答案。只做工程一致性检查，不判分，不用于任何选择。

| 检查 | 一致 / 总数 |
|---|---:|
| C1 答案 | 2440 / 2440 |
| 路由动作 | 2440 / 2440 |
| P3-20 Top-5（路由题） | 1206 / 1206 |
| BM25 是否触发（路由题） | 1206 / 1206 |
| L2 Top-5（触发的题） | 333 / 334 |
| **最终答案** | **2440 / 2440** |

- 唯一一处 L2 Top-5 不同来自 GPU 重排的微小数值差（产品对约 35 段、离线对约 150 段分批打分），触发判断和最终答案都相同。
- 因此 `artifacts/controller/bm25_gate_test_v1` 中离线得到的 Test 结果可以代表产品代码的表现。
- 运行：NVIDIA GeForce RTX 5090 D，16 线程，正式运行、未续跑；完整结果留在本地 `data/processed/quality_sweep_v1/bm25_product_confirm/`。
- 测试 `tests/test_bm25_product_equivalence.py` 回放这些对比。
