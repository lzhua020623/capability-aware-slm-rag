# L2 与采纳门控：确认集结果（2026-10-30）

确认集 2,440 题（Validation）上两项按登记的检验，都没有通过。路线 1（`quality_round1_B_final`）仍是产品，不跑 Test。

| 检验 | 登记 | 修好 / 改坏（相对路线 1） | 净值 | 结果 |
|---|---|---:|---:|---|
| L2 `bge20_bm2520_ce_only` 替换 P3-20（第三次使用确认集） | `l2_round_v1.json`：净值 > 0 且 McNemar p < 0.05 | 98 / 99 | -1 | 不通过 |
| 采纳门控：L2 上下文回到 C1 时保留 C1（第四次使用） | `acceptance_gate_v1.json`：净值 > 0 才进 Test | 19 / 20 | -1 | 不通过 |

- 扫描集上（样本内）L2 比 P3-20 多 +40，门控比路线 1 多 +26（57 / 31）；两者在确认集上都是 −1。路线 1 本身从扫描 +99 到确认 +45，量级一致。
- 门控在确认集上改回 C1 的 93 题中，挡住的改坏 19、损失的修好 20。
- GPU 重排对照：路线 1 送去 P3 的 1206 题中 1205 题 Top-5 与路线 1 相同。

## 文件

- `results.tar.gz`：`items.jsonl`（每题 C1、路线 1、L2 答案和两组 Top-5）、`labels.jsonl`（用到的每个答案的对错和来源）、`generations/`、`candidates/manifest.json`、`judge/l2_confirm/`、`reports/`（L2 采纳检验、门控确认检验、门控扫描回放、L2 扫描选择报告）。
- 原始候选列表留在本地 `data/processed/quality_sweep_v1/l2_confirm/`，`provenance.json` 记录哈希。
- 测试 `tests/test_l2_confirm_release.py` 用归档内容回放两项检验。
