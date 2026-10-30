# L2 扫描集结果（2026-10-30）

登记文件 `configs/controller/l2_round_v1.json` 第 1、2 步在扫描集 6,000 题（Validation）上的结果。路由沿用路线 1（`quality_round1_B_final`）：送去 P3 的 3,013 题由 L2 设置替换 P3-20。

## 选择结果

按登记规则（扫描集上判官口径净值最高；差 3 题以内选候选更少的，再优先 rrf），选中 **`bge20_bm2520_ce_only`**：BGE 前 20 加 BM25 前 20 合并，按重排分数取 Top-5。

| 设置 | 送去 P3 的题：修好 / 改坏 / 净值 | 全部 6,000 题：修好 / 改坏 / 净值 |
|---|---:|---:|
| `bge20_bm250_rrf`（对照，即 P3-20） | 239 / 140 / **+99** | 330 / 246 / **+84** |
| `bge20_bm2520_ce_only`（选中） | 338 / 199 / **+139** | 480 / 353 / **+127** |

- 对照在送去 P3 的题上逐题等于路线 1 的最终答案（3,013 / 3,013）。
- 18 种设置的完整结果见 `score.json`。所有对照均相对 C1；答案不变的题记 0。
- 选中设置随后在确认集 2,440 题上和路线 1 配对比较（登记：净值 > 0 且精确 McNemar p < 0.05），**未通过**：净 −1（98 / 99），见 `artifacts/controller/l2_confirm_v1`。
- 按机制拆分：Top-5 含 BM25 带来的段落的题，扫描集 +39、确认集 +16；只在 BGE 前 20 内重新排序的题，扫描集 +1、确认集 −17。据此登记的后续规则见 `configs/controller/bm25_gate_v1.json`。

## 文件

- `results.tar.gz`：`items.jsonl`（每题 C1、路线 1 答案，以及各设置下与 C1 不同的答案）、`labels.jsonl`（评分用到的每个答案的对错和来源）、`coverage/`（第 1 步 P3-20 覆盖结果，不含段落正文）、`candidates/`（候选清单、GPU 重排一致性检查、BM25 索引回执）、`generations/`（L2 生成）、`judge/`（两批判官输入输出）、`reports/`（两份评分报告）。
- 原始候选列表和覆盖记录（压缩后约 41 MB）留在本地 `data/processed/quality_sweep_v1/`，`provenance.json` 记录哈希。
- 测试 `tests/test_l2_sweep_release.py` 用归档内容逐项回放两份报告和选择。
